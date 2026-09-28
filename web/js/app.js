import { connectState, control, request } from "./api.js";
import { SbiChart } from "./charts.js";
import { formatLiveScore, LiveController } from "./live.js";
import { appState } from "./state.js";
import { LivingTree } from "./tree.js";
import {
  renderConnection,
  renderDatasetAudit,
  renderEvidence,
  renderExplanation,
  renderInterventionComparison,
  renderProfile,
  renderSnapshot
} from "./render.js";

const $ = id => document.getElementById(id);
const chart = new SbiChart($("sbi-chart"));
const tree = new LivingTree($("living-tree"));
tree.start();

let toastTimer;
function toast(message) {
  const element = $("toast");
  element.textContent = message;
  element.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove("show"), 3200);
}

async function action(name, payload) {
  try { renderReplayState(await control(name, payload)); }
  catch (error) { toast(error.message); }
}

async function loadScenarios() {
  const payload = await request("/api/scenarios");
  $("scenario-select").innerHTML = payload.scenarios.map(item => `<option value="${item.id}">${item.title}</option>`).join("");
}

function renderReplayState(snapshot) {
  renderSnapshot(snapshot, chart);
  renderExplanation(snapshot);
  renderInterventionComparison(snapshot);
  renderProfile(snapshot);
}

function clearSessionView(duration) {
  appState.dialogueKeys.clear(); appState.history=[]; chart.reset(duration);
  $("dialogue-stream").innerHTML = '<div class="empty-state">等待备用回放开始<br><small>该区域的数据均标记为 SIMULATED</small></div>';
}

const stateCopy = {
  observing: ["生命树正在观察", "等待足够的本人视觉、声音与语义证据。"],
  friendly: ["当前表达友好平稳", "树冠正在舒展，继续保持事实导向与平和语气。"],
  signal: ["检测到细微偏差信号", "部分非语言或语音线索发生变化，请留意自己的表达方式。"],
  risk: ["微冒犯风险正在累积", "建议先停顿，再把概括性判断改成可核对的事实。"],
  recovering: ["表达方式正在改善", "系统识别到主动修正，生命树正在重新生长。"]
};

function setMeter(id, value) {
  const safe = Math.max(0, Math.min(100, Number(value) || 0));
  $(id).style.width = `${safe}%`;
  return safe;
}

function renderLiveState(state) {
  appState.liveState = state;
  const liveSignals = state.liveSignals || { visionFresh: false, audioFresh: false, transcriptFresh: false };
  const metrics = state.metrics || {};
  const friendliness = Math.max(0, Math.min(100, Number(state.friendliness) || 0));
  const hasEvaluableSignal = liveSignals.visionFresh && Boolean(state.vision?.face_detected)
    || liveSignals.audioFresh && state.audio?.status === "speech"
    || liveSignals.transcriptFresh && state.transcript?.status === "completed" && Boolean(state.transcript?.text);
  const copy = stateCopy[state.tree?.mode] || stateCopy.observing;
  $("live-sbi").textContent = hasEvaluableSignal ? Number(state.sbi || 0).toFixed(1) : "—";
  $("live-friendliness").textContent = hasEvaluableSignal ? formatLiveScore(friendliness) : "—";
  $("friendliness-label").textContent = hasEvaluableSignal ? "社交友好度" : "等待有效信号";
  $("live-sequence").textContent = `#${Number(state.signal_sequence || 0)}`;
  $("live-activity").textContent = `${Number(state.signal_activity || 0).toFixed(0)}%`;
  $("friendliness-ring").style.strokeDashoffset = hasEvaluableSignal ? `${320.44 * (1 - friendliness / 100)}` : "320.44";
  $("friendliness-ring").style.stroke = !hasEvaluableSignal ? "#55758c" : state.sbi >= 70 ? "#ff735d" : state.sbi >= 35 ? "#f7c66a" : "#54efbd";
  $("live-state-title").textContent = copy[0]; $("live-state-summary").textContent = copy[1];
  $("live-suggestion").textContent = state.suggestion || "系统保持中性观察。";
  const tone = setMeter("tone-meter", liveSignals.audioFresh ? metrics.tone_pressure : 0);
  const visual = setMeter("visual-meter", liveSignals.visionFresh ? metrics.visual_tension : 0);
  const semantic = setMeter("semantic-meter", liveSignals.transcriptFresh ? metrics.semantic_bias : 0);
  $("tone-value").textContent = tone.toFixed(0); $("visual-value").textContent = visual.toFixed(0); $("semantic-value").textContent = semantic.toFixed(0);
  const frameReceived = state.last_frame_at_ms !== null && state.last_frame_at_ms !== undefined;
  $("face-status").textContent = state.status === "stopped" ? "本次自检已结束"
    : !liveSignals.visionFresh && frameReceived ? "视觉追踪已过期 · 正在恢复"
      : liveSignals.visionFresh && state.vision?.face_detected ? `已锁定本人 · ${(liveSignals.visionAgeMs / 1000).toFixed(1)}s`
        : frameReceived ? "已收到画面，等待清晰人脸"
          : state.status === "running" ? "等待摄像头首帧" : "等待人脸";
  $("camera-placeholder").hidden = state.status === "running" && frameReceived;
  $("camera-placeholder").textContent = state.status === "stopped" ? "本次自检已结束" : frameReceived ? "已收到画面 · 等待人脸" : state.status === "running" ? "等待摄像头输出画面" : "开始自检后显示本人画面";
  $("vision-confidence").textContent = liveSignals.visionFresh && state.vision?.face_detected ? `${Math.round(state.vision.confidence * 100)}%` : "—";
  const facial = state.vision?.features || {};
  const facialReadout = $("facial-cue-readout");
  if (!liveSignals.visionFresh && state.status === "running") {
    facialReadout.textContent = "视觉信号暂未更新 · 正在恢复采集";
    facialReadout.dataset.state = "stale";
  } else if (!state.vision?.face_detected) {
    facialReadout.textContent = frameReceived ? "等待清晰人脸进入检测区域" : "眉间与唇部动作线索将在本机分析";
    facialReadout.dataset.state = "idle";
  } else if (!facial.facial_cue_available) {
    facialReadout.textContent = "基础表情分类可用 · 面部动作模型未就绪";
    facialReadout.dataset.state = "unavailable";
  } else if (!facial.baseline_ready) {
    facialReadout.textContent = "正在建立个人静息基线 · 请自然放松面部";
    facialReadout.dataset.state = "calibrating";
  } else {
    const brow = Math.round(Math.max(0, Math.min(1, Number(facial.brow_tension) || 0)) * 100);
    const lip = Math.round(Math.max(0, Math.min(1, Number(facial.lip_tension) || 0)) * 100);
    const duration = Math.round(Number(facial.cue_duration_ms) || 0);
    const active = Number(facial.micro_expression) > 0;
    facialReadout.textContent = active
      ? `眉间收紧 ${brow}% · 唇部紧绷 ${lip}% · 持续线索 ${Math.round(duration / 100) / 10}s`
      : `眉间收紧 ${brow}% · 唇部紧绷 ${lip}% · 未形成持续线索`;
    facialReadout.dataset.state = active ? "active" : "ready";
  }
  const measuredActive = liveSignals.visionFresh && state.vision?.face_detected
    || liveSignals.audioFresh && ["speech", "silent"].includes(state.audio?.status)
    || liveSignals.transcriptFresh;
  const evidenceLabel = !measuredActive ? "等待新的实时证据"
    : state.evidence_status === "sufficient" ? "多模态证据充分"
      : state.evidence_status === "partial" ? "单模态信号变化" : "实时信号已接入";
  $("live-evidence").textContent = `${evidenceLabel} · MEASURED`;
  $("live-evidence").className = `evidence-pill ${measuredActive ? state.evidence_status : "insufficient"}`;
  const running = state.status === "running";
  $("live-session-state").classList.toggle("running", running);
  $("live-session-state").lastChild.textContent = running ? frameReceived ? "实时自检运行中" : "等待摄像头首帧" : state.status === "stopped" ? "本次自检已结束" : "设备待命";
  $("start-live").disabled = running; $("stop-live").disabled = !running;
  $("detail-vision").textContent = !liveSignals.visionFresh && state.status === "running"
    ? "视觉结果已过期，正在恢复采集"
    : state.vision?.face_detected
      ? `${state.vision.expression || "表情分类"} / ${Math.round(state.vision.confidence * 100)}% · 眉间 ${Math.round((Number(facial.brow_tension) || 0) * 100)}% · 唇部 ${Math.round((Number(facial.lip_tension) || 0) * 100)}% · 持续 ${Math.round(Number(facial.cue_duration_ms) || 0)}ms`
      : "未检测到清晰人脸";
  $("detail-audio").textContent = !liveSignals.audioFresh && state.status === "running"
    ? "音频信号暂未更新"
    : state.audio?.status === "speech" ? `声学唤醒 ${Math.round((state.audio.arousal || 0) * 100)}%` : "等待有效语音";
  $("detail-semantic").textContent = liveSignals.transcriptFresh ? state.transcript.text : "等待新的实时转写";
}

const live = new LiveController({
  tree, video: $("camera-video"), overlay: $("face-overlay"), waveCanvas: $("audio-wave"),
  speechState: $("speech-state"), transcript: $("live-transcript"), faceStatus: $("face-status"), onState: renderLiveState,
  onError: error => toast(error.message)
});

$("start-live").addEventListener("click", async () => {
  $("start-live").disabled = true;
  try { await live.start(); } catch { $("start-live").disabled = false; }
});
$("stop-live").addEventListener("click", () => live.stop().catch(error => toast(error.message)));
$("live-detail-toggle").addEventListener("click", event => {
  const drawer = $("live-detail-drawer"); drawer.hidden = !drawer.hidden;
  event.currentTarget.setAttribute("aria-expanded", String(!drawer.hidden));
});

document.querySelectorAll(".nav-item").forEach(button => button.addEventListener("click", () => {
  const target = button.dataset.view; appState.activeView = target;
  document.querySelectorAll(".nav-item").forEach(element => element.classList.toggle("active", element === button));
  document.querySelectorAll(".view").forEach(element => element.classList.toggle("active", element.id === `view-${target}`));
  const titles = { live: "实时微冒犯自检", profile: "会后复盘", explain: "机制解释", evidence: "研究证据", replay: "备用场景回放" };
  $("page-title").textContent = titles[target];
  $("live-primary-controls").style.display = target === "live" ? "flex" : "none";
}));

$("play-button").addEventListener("click", () => action("start"));
$("pause-button").addEventListener("click", () => action("pause"));
$("reset-button").addEventListener("click", () => { clearSessionView(appState.snapshot?.duration_ms); action("reset"); });
$("speed-select").addEventListener("change", event => action("speed", { value: Number(event.target.value) }));
$("scenario-select").addEventListener("change", event => { clearSessionView(); action(`load/${event.target.value}`); });
$("time-scrubber").addEventListener("change", event => action("seek", { at_ms: Number(event.target.value) }));
window.addEventListener("resize", () => { chart.draw(); tree.resize(); });

try {
  await loadScenarios();
  const [initial, evidence, audit] = await Promise.all([request("/api/runtime/state"), request("/api/evidence"), request("/api/evidence/dataset")]);
  renderEvidence(evidence); renderDatasetAudit(audit); renderReplayState(initial);
  connectState(renderReplayState, renderConnection);
} catch (error) { renderConnection("error"); toast(error.message); }
