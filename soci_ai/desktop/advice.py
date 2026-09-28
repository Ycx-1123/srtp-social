"""Small, deterministic coaching summaries derived only from observed cues."""
from collections import Counter


ACTION_NAMES = {"brow_tension": "眉部紧张", "mouth_downturn": "嘴角下压"}

_SHORT_REWRITES = {
    'gender_ability': '按兴趣、经验和实际表现讨论能力。',
    'gender_roles': '先问本人意愿，再依据能力分配角色。',
    'age_bias': '先了解相关经验，不用年龄推断能力。',
    'region_accent': '关注具体内容，不凭地域或口音评价人。',
    'education_bias': '用作品和经验评价，不凭学历预设能力。',
    'appearance_bias': '关注行为与成果，不从外貌推断性格。',
    'socioeconomic_bias': '依据实际表现，不以家庭经济条件判断人。',
    'disability_bias': '先询问需要的支持，再讨论具体任务。',
    'family_status': '先了解时间安排，不用婚育身份推断投入。',
    'identity_othering': '按一致标准参与，给每个人表达机会。',
    'ability_dismissal': '说明具体困难，并询问需要什么支持。',
    'emotional_invalidation': '先听清感受，再讨论事实与解决办法。',
}


def language_feedback(details, text, at, is_final):
    action = ('按经验、能力和本人意愿分工。' if '任务分配' in details.get('tags', [])
              else _SHORT_REWRITES.get(details.get('category'), '讨论具体行为、影响与可执行请求。'))
    return {**details, 'text': text, 'at_ms': int(at * 1000), 'is_final': is_final,
            'full_rewrite': details.get('rewrite', ''), 'rewrite': action}


class ReadableGuidance:
    """Hold presentation only, never evidence or assessment scores."""
    def __init__(self):
        self.current = None
        self.until = 0.0

    def update(self, at, title, body):
        new_action = (self.current is not None and title != self.current[0]
                      and title != '保持自然、具体的表达')
        if self.current is None or at >= self.until or new_action:
            self.current = title, body
            self.until = at + 8
        return self.current

    def clear(self):
        self.current = None
        self.until = 0.0


def live_guidance(*, available, features, visual, pressure, semantic, text, evidence, rewrite=""):
    if not available:
        return "等待有效信号", "面向镜头，确认麦克风电平。"
    if semantic >= .35:
        replacement = rewrite or "按具体表现、经验和任务要求讨论，不用群体身份推断个人能力。"
        return "留意语言中的概括", replacement
    brow = features.get('brow_tension', 0) >= .3
    downturn = features.get('mouth_downturn', 0) >= .3
    if brow and downturn:
        return '留意眉部与嘴角变化', '舒展眉头，放松嘴角，再说明具体担心。'
    if brow:
        return '留意眉部紧张', '舒展眉头，放松眉间，再说明具体担心。'
    if downturn:
        return '留意嘴角下压', '放松嘴角，回到自然表情，再回应对方。'
    if pressure >= .45:
        return "放缓语气，留出回应", "降低音量，停顿一下，问“你怎么看？”"
    if features.get("smile", 0) >= .25:
        return "观察到微笑动作", "保持自然微笑，给对方回应时间。"
    return "保持自然、具体的表达", "说具体事实，听完再回应。"


_FOCUS_LANGUAGE = {
    'gender_ability':'不凭性别判断能力', 'gender_roles':'按意愿与能力分工',
    'age_bias':'不凭年龄预设能力', 'region_accent':'不凭地域口音评价',
    'education_bias':'避免学历概括', 'appearance_bias':'不凭外貌评价',
    'socioeconomic_bias':'不凭经济背景评价', 'disability_bias':'关注能力与所需支持',
    'family_status':'不凭婚育判断投入', 'identity_othering':'给每个人表达机会',
    'ability_dismissal':'具体指出困难', 'emotional_invalidation':'先回应对方感受',
}


def cumulative_focus(summary, events):
    """Brief priorities from whole-session coverage, never a live-frame verdict.

    Only confirmed semantic events enter this view. Facial/volume priorities
    require both observed duration and a meaningful share of valid coverage;
    silence and missing inputs cannot become 'raise your volume' advice.
    """
    ranked = []
    language = [event for event in events if event.get('modality') == 'semantic']
    if language:
        counts = Counter(event.get('category') for event in language)
        category = counts.most_common(1)[0][0]
        labels = Counter(event.get('category_label') or '语言概括' for event in language)
        detail = '本场确认文字中的提醒：' + '、'.join(
            f'疑似{name} {count} 次' for name, count in labels.most_common(3)) + '；请结合原话与语境核对。'
        ranked.append((2, {'key':'language', 'text':_FOCUS_LANGUAGE.get(category, '依据具体事实表达'),
                           'detail':detail, 'tone':'amber'}))
    face_seconds = summary.get('face_seconds', 0)
    durations = summary.get('action_seconds', {})
    for key, text, tone in (('brow_tension', '舒展眉头', 'amber'),
                            ('mouth_downturn', '放松嘴角', 'blue')):
        duration = durations.get(key, 0)
        ratio = duration / max(face_seconds, .01)
        if face_seconds >= 2 and duration >= 1 and ratio >= .15:
            detail = f'本场有效面部观测中，{ACTION_NAMES[key]}约占 {ratio:.0%}。'
            detail += '可先放松眉间，再表达具体担心。' if key == 'brow_tension' else '可回到自然表情；无需刻意保持笑容。'
            ranked.append((ratio, {'key':key,'text':text,'detail':detail,'tone':tone}))
    audio_seconds, pressure_seconds = summary.get('audio_seconds', 0), summary.get('pressure_seconds', 0)
    ratio = pressure_seconds / max(audio_seconds, .01)
    if audio_seconds >= 2 and pressure_seconds >= 1 and ratio >= .15:
        ranked.append((ratio, {'key':'pressure','text':'降低音量', 'tone':'amber',
            'detail':f'本场有效声音观测中，音量压力升高约占 {ratio:.0%}。适当放低音量，句间留出停顿；不是语速或音高测量。'}))
    if not ranked:
        available = face_seconds >= 2 or audio_seconds >= 2 or summary.get('transcript_count', 0) > 0
        if not available:
            return {'items':[], 'status':'正在积累有效信号'}
        smile = face_seconds >= 2 and summary.get('smile_seconds', 0) / face_seconds >= .15
        ranked.append((0, {'key':'steady', 'text':'保持自然微笑' if smile else '保持自然表达', 'tone':'mint',
            'detail':'截至目前的可用观测未形成累计重点提醒；不代表所有措辞都已被完整识别。'}))
        ranked.append((0, {'key':'pause', 'text':'句间留出停顿', 'tone':'blue',
            'detail':'这是一般交流建议，给对方回应时间，不是检测到了语速过快。'}))
    return {'items':[item for _, item in sorted(ranked, key=lambda pair:pair[0], reverse=True)[:3]],
            'status':'截至目前 · 累计重点'}


def _main_action(summary):
    durations, peaks = summary.get('action_seconds', {}), summary.get('peaks', {})
    key = max(ACTION_NAMES, key=lambda item: (durations.get(item, 0), peaks.get(item, 0)))
    duration = durations.get(key)
    if duration is None:
        pattern = '有增强记录，独立持续时长未记录'
    elif duration >= 1 and duration / max(summary.get('face_seconds', 0), .01) >= .15:
        pattern = '在较多观测时段增强'
    else:
        pattern = '有短暂或偶发增强'
    return key, pattern


def meeting_advice(summary, events, transcripts):
    cards = []
    language = [event for event in events if event.get("modality") == "semantic"]
    observations, priorities = [], []
    face_seconds = summary.get('face_seconds', 0)
    if face_seconds > 0:
        peaks = summary.get('peaks', {})
        key, pattern = _main_action(summary)
        if peaks.get(key, 0) >= .3:
            observations.append(ACTION_NAMES[key] + pattern)
            priorities.append('表达分歧时先放松眉嘴，再说明具体担心')
        else:
            observations.append('面部动作整体较平稳')
    else:
        observations.append('有效面部观测不足')
    labels = Counter(event.get('category_label') or '语言概括' for event in language)
    if labels:
        observations.append('确认字幕中出现疑似' + '、'.join(label for label, _ in labels.most_common(3)) + '倾向')
        priorities.insert(0, '优先把身份判断改成基于经验、表现和任务要求的表达')
    elif transcripts:
        observations.append('已确认的文字未触发当前语言库提醒')
    else:
        observations.append('确认转写不足，暂不评价措辞')
    audio_seconds = summary.get('audio_seconds', 0)
    pressure_seconds = summary.get('pressure_seconds', 0)
    if audio_seconds > 0:
        if pressure_seconds > .2:
            pattern = '较多时段' if pressure_seconds / audio_seconds >= .15 else '少数时段'
            observations.append(pattern + '音量压力升高')
            priorities.append('放低音量，给句间留出停顿')
        else:
            observations.append('未观察到持续较高的音量压力')
    else:
        observations.append('有效声音观测不足')
    cards.append({'kind': 'overview', 'title': '本场总结 · 重点与改进方向',
        'body': '；'.join(observations) + '。' +
                ('建议：' + '；'.join(priorities[:2]) + '。' if priorities else
                 '下一次继续以具体事实回应，并给对方表达空间。')})
    if language:
        strongest = max(language, key=lambda event: event.get("value", 0))
        phrase = strongest.get("text") or strongest.get("evidence", "")
        rewrite = strongest.get("rewrite") or "我们依据这项任务所需的经验与实际表现来分工，也听听本人意愿。"
        category_summary = '、'.join(f'{name} {count} 次' for name, count in labels.most_common(3))
        cards.append({"kind": "language", "title": "语言 · 留意具体的偏见倾向",
                      "body": f"本场确认字幕中的主要提醒为：{category_summary}。代表表达：“{phrase}”。建议改为：“{rewrite}”。优先避免用身份预设个人能力；请结合字幕与语境核对，不据此判断本人意图。"})
    elif transcripts:
        cards.append({"kind": "language", "title": "语言 · 保持事实导向，也照顾感受",
                      "body": f"记录到 {len(transcripts)} 段确认转写，未命中当前样例库。可以继续使用“我观察到……，希望……”来代替“你们都……”。这只是有限样例与规则的结果，不是全语境无偏见的证明。"})
    else:
        cards.append({"kind": "availability", "title": "语言 · 本场尚无确认转写",
                      "body": "没有可供核对的完整转写，因此不评价本场措辞。下次开始后先说一句完整短句并确认字幕出现，再进行会议测试。"})

    peaks = summary.get("peaks", {})
    face_seconds = summary.get("face_seconds", 0)
    if face_seconds > 0:
        key, pattern = _main_action(summary)
        if peaks.get(key, 0) >= .3:
            cards.append({"kind": "expression", "title": f"表情 · 留意{ACTION_NAMES[key]}的时刻",
                          "body": f"主要变化是{ACTION_NAMES[key]}，{pattern}。表达不同意见前先放松眉嘴，用“我担心的是……”说明具体问题。可结合曲线上的原因标签回看；动作变化不等于情绪或意图。"})
        else:
            cards.append({"kind": "expression", "title": "表情 · 自然表达，无需刻意保持笑容",
                          "body": f"本场有约 {face_seconds:.1f} 秒有效面部观测，未记录到明显的校准后眉嘴收紧。继续保持自然表情；如画面角度或光线明显改变，可重新校准。"})
        if summary.get("smile_seconds", 0) > .2:
            cards.append({"kind": "positive", "title": "互动 · 微笑之外，也给出明确回应",
                          "body": f"观测到约 {summary['smile_seconds']:.1f} 秒微笑动作。可搭配点头、简短复述或“我理解你的意思”，确认理解；微笑不能抵消语言中的偏见，也不代表已知道对方感受。"})
    else:
        cards.append({"kind": "availability", "title": "表情 · 有效面部观测不足",
                      "body": "本场缺少基线就绪后的清晰面部信号，不能对表情作总结。下次面向镜头、避免强逆光，保持自然表情约 1 秒完成基线后再测试。"})

    if summary.get("audio_seconds", 0) > 0:
        duration = summary.get("pressure_seconds", 0)
        cards.append({"kind": "voice", "title": "声音 · 音量与表达节奏",
                      "body": ("部分时段音量压力偏高。出现分歧时先停顿，适当降低音量，再询问对方的看法。" if duration > .2 else "本场未记录到持续较高的音量压力。保持清晰音量，句间适当停顿。") + "本次没有可靠的语速与音高测量，不将音量升高写成‘语速过快’或‘音调过高’，也不据此断定打断他人。"})
    else:
        cards.append({"kind": "availability", "title": "声音 · 本场缺少有效音频",
                      "body": "请检查应用所选麦克风，并在下一次开始后确认电平随说话变化。没有音频时不根据画面推测你的语气。"})
    return cards


def key_moments(events, limit=3):
    """Condense actual events into distinct episodes; retain raw export data.

    Repeated language is one example, not a new incident on each utterance.
    Same-action observations no more than 15 seconds apart share a display
    episode. This does not claim the action was continuous between samples.
    """
    groups, language, recent_actions = [], {}, {}
    for event in sorted(events, key=lambda item: item.get('at_ms', 0)):
        kind = event.get('modality')
        if kind not in ('semantic', 'vision'):
            continue
        at = event.get('at_ms', 0)
        if kind == 'semantic':
            key = ''.join(str(event.get('text') or event.get('evidence') or '').split())
            group = language.get(key)
        else:
            key = event.get('title') or '面部动作变化'
            group = recent_actions.get(key)
            if group and at - group['end_ms'] > 15000:
                group = None
        if group is None:
            group = {'event': event, 'start_ms': at, 'end_ms': at, 'count': 1}
            groups.append(group)
            (language if kind == 'semantic' else recent_actions)[key] = group
        else:
            group['end_ms'] = at
            group['count'] += 1
            if event.get('value', 0) > group['event'].get('value', 0):
                group['event'] = event
    ranked = sorted(groups, key=lambda group: (group['event'].get('modality') == 'semantic',
                                               group['event'].get('value', 0)), reverse=True)
    selected, deferred, actions = [], [], set()
    capacity = max(0, min(3, limit))
    for group in ranked:
        event = group['event']
        action = event.get('title') or '面部动作变化'
        if event['modality'] == 'vision' and action in actions:
            deferred.append(group)
            continue
        selected.append(group)
        if event['modality'] == 'vision':
            actions.add(action)
    # Distinct language/action types come first. Only unused slots may show
    # another well-separated episode of the same action.
    selected = (selected[:capacity] + deferred)[:capacity]
    output = []
    for group in sorted(selected, key=lambda item: item['event'].get('at_ms', 0)):
        event = group['event']
        kind = event['modality']
        title = event.get('title') or ('语言表达提醒' if kind == 'semantic' else '面部动作变化')
        if kind == 'semantic':
            phrase = event.get('text') or event.get('evidence', '')
            detail = '“' + phrase + '”'
            if event.get('rewrite'):
                detail += '  建议：' + event['rewrite']
        else:
            detail = title + '；先放松眉嘴，再说明具体担心。'
            if group['count'] > 1:
                detail += ' 同类变化已合并。'
        output.append({'at_ms': event.get('at_ms', 0), 'start_ms': group['start_ms'],
                       'end_ms': group['end_ms'], 'title': title, 'detail': detail, 'kind': kind})
    return output
