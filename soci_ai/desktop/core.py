"""Freshness-gated, explainable prototype assessment, independent of the UI.

Scores describe communication cues, not a diagnosis or proof of hostile intent.
The fast attack makes changes visible without inventing score jitter. No trained
microaggression classifier or validation accuracy is claimed for these rules.
"""
from collections import deque
import math

from .advice import live_guidance, meeting_advice, key_moments, ACTION_NAMES, ReadableGuidance, language_feedback, cumulative_focus
from .language import analyze_text, classify_text, corpus_status


def clamp(value):
    return max(0.0, min(1.0, float(value)))


class RealtimeAssessment:
    def __init__(self):
        self.last_at = None
        self.risk = 0.0
        self.vision = {}
        self.audio = {}
        self.semantic = (0.0, -100.0, "", "")
        self.semantic_details = {}
        self.readable_guidance = ReadableGuidance()
        self.confirmed_language = None
        self.pending_language = None
        self.session_focus = {'items':[], 'status':'正在积累有效信号'}
        self.last_focus_at = -10.
        self.showing_language = False
        # These are derived records for the whole current session, not raw
        # audio/video queues. Do not silently drop the opening of a long run.
        self.transcripts = deque()
        self.events = deque()
        self.history = deque()
        self.last_history_at = -1
        self.score_sources = {}
        self.last_event_at = -2
        self.current_text = ""
        self.current_final = False
        self.last_final = None
        self.peak_sbi = 0.0
        self.score_sum = 0.0
        self.score_count = 0
        self.last_smile_event_at = -5
        self.observations = {"face_seconds": 0.0, "tension_seconds": 0.0, "smile_seconds": 0.0,
                             "audio_seconds": 0.0, "pressure_seconds": 0.0,
                             "action_seconds": {key: 0.0 for key in ACTION_NAMES},
                             "peaks": {key: 0.0 for key in (*ACTION_NAMES, "smile")}}

    def observe_text(self, text, at, is_final):
        if not text.strip():
            return
        details = analyze_text(text)
        strength, evidence = details["score"], details["explanation"]
        self.semantic_details = details
        self.semantic = (strength * (1 if is_final else .8), at, text, evidence)
        self.current_text, self.current_final = text, is_final
        feedback = language_feedback(details, text, at, is_final) if strength or details.get('review_required') else None
        if is_final:
            self.pending_language = None
            if feedback:
                self.confirmed_language = feedback
        else:
            self.pending_language = feedback
        if is_final and (text, at) != self.last_final:
            self.transcripts.append({"at_ms": int(at * 1000), "text": text, "is_final": True})
            self.last_final = (text, at)
            if strength:
                self.events.append({"at_ms": int(at * 1000), "modality": "semantic", "value": strength, "text": text,
                                    "evidence": evidence + "：" + text, "category": details.get("category"),
                                    "category_label": details.get('category_label'), 'tags': details.get('tags', []),
                                    "example_id": details.get("example_id"), "rewrite": details.get("rewrite", "")})

    def invalidate_partial(self):
        if not self.current_final:
            self.semantic = (0.0, -100.0, "", "")
            self.semantic_details = {}
            self.current_text = ""
            self.pending_language = None

    def focus(self, at, *, force=False):
        if force or at - self.last_focus_at >= 10:
            self.last_focus_at = at
            summary = {**self.observations, 'transcript_count':len(self.transcripts)}
            self.session_focus = cumulative_focus(summary, self.events)
        return self.session_focus

    def update(self, at, *, vision=None, audio=None):
        if vision is not None:
            self.vision = vision
        if audio is not None:
            self.audio = audio
        elapsed = 0 if self.last_at is None else max(0, at - self.last_at)
        observed_dt = elapsed if elapsed <= .6 else 0
        dt = .1 if self.last_at is None else min(.4, elapsed)
        self.last_at = at
        vision_age = at - self.vision.get("captured_at", -100)
        face = bool(self.vision.get("face_detected")) and 0 <= vision_age < .6
        features = self.vision.get("features", {}) if face else {}
        ready = face and bool(features.get("baseline_ready"))
        # Raw actions are displayed instantly; sustained calibrated evidence has
        # higher weight. A blink alone is deliberately not an offence signal.
        raw = max((clamp(features.get(k, 0)) for k in ACTION_NAMES), default=0)
        visual = max(clamp(features.get("micro_expression", 0)), raw * .25) if ready else 0
        audio_age = at - self.audio.get("captured_at", -100)
        audio_live = 0 <= audio_age < .8
        pressure = clamp(self.audio.get("pressure", 0)) if audio_live else 0
        semantic, semantic_at, text, evidence = self.semantic
        semantic_age = at - semantic_at
        semantic_live = 0 <= semantic_age < 12
        semantic = semantic * math.exp(-max(0, semantic_age - 2) / 4) if semantic_live else 0
        has_evidence = ready or semantic_live or audio_live
        smile = clamp(features.get("smile", 0)) if ready else 0
        if ready:
            self.observations["face_seconds"] += observed_dt
            self.observations["tension_seconds"] += observed_dt if visual >= .3 else 0
            self.observations["smile_seconds"] += observed_dt if smile >= .25 else 0
            for key in ACTION_NAMES:
                self.observations['action_seconds'][key] += observed_dt if clamp(features.get(key, 0)) >= .3 else 0
            for key in self.observations["peaks"]:
                self.observations["peaks"][key] = max(self.observations["peaks"][key], clamp(features.get(key, 0)))
        if audio_live:
            self.observations["audio_seconds"] += observed_dt
            self.observations["pressure_seconds"] += observed_dt if pressure >= .45 else 0
        # Single facial cue cannot establish offence; language carries stronger
        # weight. Acoustic loudness is only a small supporting signal.
        target = clamp(max(.65 * visual, .9 * semantic) + .12 * min(visual, semantic) + .10 * pressure)
        tau = .18 if target > self.risk else 1.4
        smoothing = 1 - math.exp(-dt / tau)
        self.risk += (target - self.risk) * smoothing
        if not has_evidence:
            self.risk = 0.0
        sbi = round(100 * self.risk, 2) if has_evidence else None
        if sbi is None:
            mode = "observing"
        elif sbi >= 60:
            mode = "risk"
        elif sbi >= 25:
            mode = "signal"
        else:
            mode = "friendly"
        suggestion_title, suggestion = live_guidance(
            available=has_evidence, features=features if ready else {}, visual=visual,
            pressure=pressure, semantic=semantic, text=text, evidence=evidence,
            rewrite=self.semantic_details.get("rewrite", ""))
        # Reading time is independent of evidence freshness/decay. The card
        # explicitly refers to the last utterance; no score is kept artificially high.
        language_alert = None
        for item, lifetime in ((self.pending_language, 2), (self.confirmed_language, 10)):
            if item and 0 <= at - item['at_ms'] / 1000 < lifetime:
                language_alert = item
                break
        if language_alert:
            self.showing_language = True
            qualifier = '请核对字幕' if language_alert.get('review_required') else '语言提醒'
            suggestion_title = '疑似' + language_alert['category_label'] + ' · ' + qualifier
            suggestion = '建议：' + language_alert['rewrite']
        else:
            if self.showing_language or not has_evidence:
                self.readable_guidance.clear()
            self.showing_language = False
            suggestion_title, suggestion = self.readable_guidance.update(at, suggestion_title, suggestion)
        if visual > .3 and at - self.last_event_at >= 1:
            self.last_event_at = at
            action = max(ACTION_NAMES, key=lambda key: features.get(key, 0))
            self.events.append({"at_ms": int(at * 1000), "modality": "vision", "value": visual,
                                "title": ACTION_NAMES[action] + "增强",
                                "evidence": f"{ACTION_NAMES[action]}相对中性基线增强，动作强度 {features.get(action, 0)*100:.0f}/100；可先放松、停顿，再说明具体问题。这不是对意图的判断。"})
        elif smile >= .25 and at - self.last_smile_event_at >= 5:
            self.last_smile_event_at = at
            self.events.append({"at_ms": int(at * 1000), "modality": "smile", "value": smile,
                                "evidence": f"微笑动作强度 {smile*100:.0f}/100；可以自然保持，给对方留出回应时间。"})
        # Explanations follow the SAME smoothing as the score, including changes
        # between history samples. Split the max term to its winning channel,
        # the cross-modal term equally, and retain each label's decaying share.
        # These bookkeeping values never feed back into the assessment score.
        action = max(ACTION_NAMES, key=lambda key: features.get(key, 0))
        action_name = '眉部紧张' if action == 'brow_tension' else ACTION_NAMES[action]
        visual_label = action_name + '增强' if features.get(action, 0) > 0 else '面部动作增强'
        category = self.semantic_details.get('category_label')
        semantic_label = '疑似' + category if category else '语言概括倾向'
        synergy = .06 * min(visual, semantic)
        visual_share = synergy + (.65 * visual if .65 * visual > .9 * semantic else 0)
        semantic_share = synergy + (.9 * semantic if .9 * semantic >= .65 * visual else 0)
        source_targets = {visual_label: visual_share, semantic_label: semantic_share,
                          '声音压力升高': .10 * pressure}
        total = sum(source_targets.values())
        source_targets = {key: value * target / total for key, value in source_targets.items() if value > 0} if total else {}
        if sbi is None:
            self.score_sources = {}
        else:
            self.score_sources = {
                key: previous + (source_targets.get(key, 0) - previous) * smoothing
                for key in self.score_sources.keys() | source_targets.keys()
                for previous in (self.score_sources.get(key, 0),)
            }
            self.score_sources = {key: value for key, value in self.score_sources.items() if value > .00001}
        # Omit negligible contributors, and keep at most three concise labels.
        cutoff = max(.015, .10 * self.risk)
        cues = [key for key, value in sorted(self.score_sources.items(), key=lambda item: (-item[1], item[0]))
                if value >= cutoff][:3]
        cue_phases = {key: 'recovering' if source_targets.get(key, 0) + .00001 < self.score_sources[key] else 'current'
                      for key in cues}
        phase = ('unavailable' if sbi is None else 'recovering'
                 if cues and all(value == 'recovering' for value in cue_phases.values()) else 'current')
        if at - self.last_history_at >= .5:
            self.last_history_at = at
            self.history.append({"at_ms": int(at * 1000), "sbi": sbi,
                                 "friendliness": None if sbi is None else round(100 - sbi, 2),
                                 "cues": cues, "cue_phase": phase, "cue_phases": cue_phases})
        if sbi is not None:
            self.peak_sbi = max(self.peak_sbi, sbi)
            self.score_sum += sbi
            self.score_count += 1
        return {
            "sbi": sbi, "friendliness": None if sbi is None else round(100 - sbi, 2),
            "tree": {"mode": mode, "health": .55 if sbi is None else 1 - self.risk * .85, "risk": self.risk, "bloom": 0 if sbi is None else max(0, (.6 + .4 * smile) * (1 - self.risk * 2)), "wind": pressure, "smile": smile},
            "vision": {**self.vision, "face_detected": face, "box": self.vision.get("box") if face else None, "features": features},
            "audio": self.audio if audio_live else {"rms": 0, "peak": 0, "status": "音频未更新"},
            "metrics": {"visual_tension": visual, "tone_pressure": pressure, "semantic_bias": semantic},
            "suggestion": suggestion,
            "suggestion_title": suggestion_title,
            'session_focus': self.focus(at),
            'language_alert': dict(language_alert) if language_alert else {},
            "explanation": evidence if semantic else "依据本人校准后的动作与新鲜信号。单一表情/大音量不能证明微冒犯。",
            "elapsed_ms": int(at * 1000),
        }

    def report(self, at, telemetry, model_status):
        summary = {key: round(value, 2) if isinstance(value, float) else dict(value) for key, value in self.observations.items()}
        summary.update(semantic_hits=sum(event.get("modality") == "semantic" for event in self.events), transcript_count=len(self.transcripts))
        return {"elapsed_seconds": round(at, 1), "peak_sbi": round(self.peak_sbi, 2) if self.score_count else None,
                "average_sbi": round(self.score_sum / self.score_count, 2) if self.score_count else None,
                "transcripts": list(self.transcripts), "events": list(self.events),
                "history": list(self.history), "telemetry": telemetry, "model_status": model_status,
                "observation_summary": summary, "advice": meeting_advice(summary, self.events, self.transcripts),
                "moments": key_moments(self.events),
                'session_focus': self.session_focus,
                "language_corpus": corpus_status(),
                "provenance": "measured_inputs_heuristic_assessment", "limitations": "动作与语音来自本机真实采集；SBI为未校验的原型规则评分，不是准确率或意图诊断。"}
