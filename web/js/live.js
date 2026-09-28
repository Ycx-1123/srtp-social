import {
  createLiveSession,
  getLiveState,
  sendAudioChunk,
  sendAudioFeatures,
  sendLiveFrame,
  stopLiveSession
} from "./live-api.js";
import { MediaCapture } from "./media.js";

export function formatLiveScore(value) {
  return Math.max(0, Math.min(100, Number(value) || 0)).toFixed(2);
}

export function drawFaceBox(canvas, video, box) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  canvas.setAttribute("data-mirrored", "true");
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  if (!box || !video.videoWidth) return;
  const x = (1 - box.x - box.width) * rect.width;
  const y = box.y * rect.height;
  const width = box.width * rect.width;
  const height = box.height * rect.height;
  context.strokeStyle = "#54f0bd";
  context.lineWidth = 2;
  context.shadowColor = "rgba(84,240,189,.9)";
  context.shadowBlur = 12;
  context.strokeRect(x, y, width, height);
  context.shadowBlur = 0;
  context.fillStyle = "rgba(5,17,28,.82)";
  context.fillRect(x, Math.max(0, y - 23), Math.min(width, 150), 21);
  context.fillStyle = "#baf9e7";
  context.font = "11px ui-monospace, monospace";
  context.fillText("SELF / FACE LOCK", x + 7, Math.max(14, y - 8));
}

class FaceBoxTracker {
  constructor(canvas, video) {
    this.canvas = canvas;
    this.video = video;
    this.current = null;
    this.target = null;
    this.raf = null;
  }

  setTarget(box) {
    this.target = box ? { ...box } : null;
    if (!this.target) {
      this.current = null;
      drawFaceBox(this.canvas, this.video, null);
      return;
    }
    if (!this.current) this.current = { ...this.target };
    if (!this.raf) this.raf = requestAnimationFrame(() => this.step());
  }

  step() {
    this.raf = null;
    if (!this.target || !this.current) return;
    let remaining = 0;
    for (const key of ["x", "y", "width", "height"]) {
      this.current[key] += (this.target[key] - this.current[key]) * 0.32;
      remaining = Math.max(remaining, Math.abs(this.target[key] - this.current[key]));
    }
    drawFaceBox(this.canvas, this.video, this.current);
    if (remaining > 0.001 && !this.raf) this.raf = requestAnimationFrame(() => this.step());
  }
}

export class LiveController {
  constructor({ tree, video, overlay, waveCanvas, speechState, transcript, faceStatus, onState, onError }) {
    this.tree = tree;
    this.video = video;
    this.overlay = overlay;
    this.speechState = speechState;
    this.transcript = transcript;
    this.faceStatus = faceStatus;
    this.onState = onState;
    this.onError = onError;
    this.faceTracker = new FaceBoxTracker(overlay, video);
    this.sessionId = null;
    this.framePending = false;
    this.featurePending = false;
    this.transcriptPending = false;
    this.queuedUtterance = null;
    this.stateTimer = null;
    this.statePollPending = false;
    this.lastAppliedSequence = -1;
    this.lastErrorAt = 0;
    this.signalClocks = new Map();
    this.media = new MediaCapture({
      video, waveCanvas, fps: 4,
      onFrame: (blob, atMs) => this.submitFrame(blob, atMs),
      onAudioFeatures: features => this.submitFeatures(features),
      onUtterance: (pcm, atMs) => this.submitUtterance(pcm, atMs),
      onStatus: status => this.renderSpeechState(status),
      onCameraStatus: status => this.renderCameraStatus(status),
      onError: error => this.reportError(error)
    });
  }

  async start() {
    if (this.sessionId) return;
    try {
      this.signalClocks.clear();
      this.renderCameraStatus("starting");
      const mediaReady = this.media.prepare().then(
        stream => ({ stream }),
        error => ({ error })
      );
      const state = await createLiveSession();
      this.sessionId = state.session_id;
      this.lastAppliedSequence = -1;
      this.applyState(state);
      this.startStatePolling();
      const preparation = await mediaReady;
      if (preparation.error) throw preparation.error;
      await this.media.start();
    } catch (error) {
      this.renderSpeechState("unavailable");
      this.onError?.(error);
      await this.media.stop().catch(() => {});
      if (this.sessionId) await stopLiveSession(this.sessionId).catch(() => {});
      this.stopStatePolling();
      this.sessionId = null;
      throw error;
    }
  }

  async stop() {
    this.stopStatePolling();
    await this.media.stop();
    this.faceTracker.setTarget(null);
    if (!this.sessionId) return;
    try { this.applyState(await stopLiveSession(this.sessionId)); }
    finally { this.sessionId = null; }
  }

  async submitFrame(blob, atMs) {
    if (!this.sessionId || this.framePending) return;
    this.framePending = true;
    try { this.applyState(await sendLiveFrame(this.sessionId, blob, atMs)); }
    catch (error) { this.reportError(error); throw error; }
    finally { this.framePending = false; }
  }

  async submitFeatures(features) {
    if (!this.sessionId || this.featurePending) return;
    this.featurePending = true;
    try { this.applyState(await sendAudioFeatures(this.sessionId, features)); }
    catch (error) { this.reportError(error); }
    finally { this.featurePending = false; }
  }

  startStatePolling() {
    this.stopStatePolling();
    this.stateTimer = window.setInterval(() => { void this.refreshState(); }, 1000);
    void this.refreshState();
  }

  stopStatePolling() {
    if (this.stateTimer) window.clearInterval(this.stateTimer);
    this.stateTimer = null;
    this.statePollPending = false;
  }

  async refreshState() {
    const sessionId = this.sessionId;
    if (!sessionId || this.statePollPending) return;
    this.statePollPending = true;
    try {
      const state = await getLiveState(sessionId);
      if (this.sessionId === sessionId) this.applyState(state);
    } catch (error) {
      if (this.sessionId !== sessionId) return;
      if (error.status === 404 || error.code === "unknown_live_session") {
        this.stopStatePolling();
        this.sessionId = null;
        this.signalClocks.clear();
        await this.media.stop().catch(() => {});
        this.faceTracker.setTarget(null);
        this.tree?.setTarget({ mode: "observing", health: 0.55, risk: 0, bloom: 0.08, wind: 0.12 });
        this.onState?.({
          session_id: sessionId,
          status: "error",
          elapsed_ms: 0,
          last_frame_at_ms: null,
          signal_sequence: 0,
          signal_activity: 0,
          sbi: 0,
          friendliness: 0,
          evidence_status: "insufficient",
          tree: { mode: "observing" },
          vision: { face_detected: false, confidence: 0, features: {} },
          audio: { status: "idle" },
          transcript: { status: "idle", text: "" },
          metrics: {},
          suggestion: "实时会话已失效，请重新开始。",
          liveSignals: { visionFresh: false, audioFresh: false, transcriptFresh: false },
        });
        this.renderCameraStatus("error");
        this.renderSpeechState("unavailable");
        this.reportError(new Error("实时会话已失效（服务刚重启或页面过期），音视频采集已停止，请重新点击开始。"));
      } else {
        this.reportError(error);
      }
    } finally {
      this.statePollPending = false;
    }
  }

  async submitUtterance(pcm, atMs) {
    if (!this.sessionId) return;
    if (this.transcriptPending) {
      this.queuedUtterance = { pcm, atMs };
      return;
    }
    this.transcriptPending = true;
    this.renderSpeechState("transcribing");
    try {
      const state = await sendAudioChunk(this.sessionId, pcm, atMs);
      this.renderSpeechState("analyzing");
      this.applyState(state);
      window.setTimeout(() => this.renderSpeechState("listening"), 520);
    } catch (error) {
      this.renderSpeechState("unavailable");
      this.reportError(error);
    } finally {
      this.transcriptPending = false;
      const queued = this.queuedUtterance;
      this.queuedUtterance = null;
      if (queued && this.sessionId) void this.submitUtterance(queued.pcm, queued.atMs);
    }
  }

  applyState(state) {
    if (state.session_id === this.sessionId) {
      const sequence = Number(state.signal_sequence || 0);
      if (sequence < this.lastAppliedSequence) return;
      this.lastAppliedSequence = sequence;
    }
    const now = performance.now();
    const visionAge = this.observeSignalAge("vision", state.vision?.at_ms, state.elapsed_ms, now);
    const audioAge = this.observeSignalAge("audio", state.audio?.at_ms, state.elapsed_ms, now);
    const transcriptAge = this.observeSignalAge("transcript", state.transcript?.at_ms, state.elapsed_ms, now);
    const liveSignals = {
      visionFresh: state.status !== "stopped" && visionAge <= 1800,
      visionAgeMs: Number.isFinite(visionAge) ? Math.round(visionAge) : null,
      audioFresh: audioAge <= 1500,
      audioAgeMs: Number.isFinite(audioAge) ? Math.round(audioAge) : null,
      transcriptFresh: state.transcript?.status === "completed"
        && Boolean(state.transcript?.text)
        && transcriptAge <= 10000
    };
    const box = liveSignals.visionFresh && state.vision?.face_detected ? state.vision.box : null;
    this.faceTracker.setTarget(box);
    const hasCurrentEvidence = liveSignals.visionFresh && state.vision?.face_detected
      || liveSignals.audioFresh && state.audio?.status === "speech"
      || liveSignals.transcriptFresh;
    const displayState = hasCurrentEvidence || state.status === "stopped"
      ? state
      : { ...state, tree: { mode: "observing", health: 0.55, risk: 0, bloom: 0.08, wind: 0.12 } };
    this.tree?.setTarget(displayState.tree);
    this.renderCameraState(state, liveSignals);
    if (this.transcript) {
      this.transcript.textContent = liveSignals.transcriptFresh
        ? state.transcript.text
        : state.status === "running" && state.transcript?.text ? "等待新的实时转写" : "开始后，你说的话会在这里实时转写。";
    }
    if (state.status === "running" && !liveSignals.audioFresh) this.renderSpeechState("stale");
    else if (state.listening_state) this.renderSpeechState(state.listening_state);
    this.onState?.({ ...displayState, liveSignals });
  }

  observeSignalAge(kind, atMs, elapsedMs, now) {
    const timestamp = Number(atMs);
    if (!Number.isFinite(timestamp) || timestamp <= 0) return Infinity;
    const eventAge = Math.max(0, Number(elapsedMs || 0) - timestamp);
    let clock = this.signalClocks.get(kind);
    if (!clock || clock.atMs !== timestamp) {
      clock = { atMs: timestamp, baseAge: eventAge, receivedAt: now };
      this.signalClocks.set(kind, clock);
    } else {
      clock.baseAge = Math.max(clock.baseAge, eventAge);
    }
    return Math.max(eventAge, clock.baseAge + Math.max(0, now - clock.receivedAt));
  }

  renderSpeechState(status) {
    if (!this.speechState) return;
    const labels = {
      idle: "等待开始", listening: "正在监听", transcribing: "正在转写",
      analyzing: "正在分析", unavailable: "转写不可用", stale: "音频信号暂未更新"
    };
    this.speechState.textContent = labels[status] || status;
    this.speechState.dataset.state = status;
  }

  reportError(error) {
    const now = Date.now();
    if (now - this.lastErrorAt < 5000) return;
    this.lastErrorAt = now;
    this.onError?.(error);
  }

  renderCameraState(state, liveSignals) {
    if (!this.faceStatus) return;
    if (state.status === "stopped") {
      this.renderCameraStatus("stopped");
      return;
    }
    const received = state.last_frame_at_ms !== null && state.last_frame_at_ms !== undefined;
    if (!received) {
      this.renderCameraStatus(state.status === "running" ? "waiting" : "idle");
      return;
    }
    if (!liveSignals?.visionFresh) {
      this.renderCameraStatus("stale");
      return;
    }
    const status = state.vision?.face_detected ? "face" : "no_face";
    this.renderCameraStatus(status);
    if (status === "face" && liveSignals.visionAgeMs !== null) {
      this.faceStatus.textContent = `已锁定本人 · ${(liveSignals.visionAgeMs / 1000).toFixed(1)}s`;
    }
  }

  renderCameraStatus(status) {
    if (!this.faceStatus) return;
    const labels = {
      idle: "等待启动", starting: "正在启动摄像头", waiting: "等待摄像头首帧",
      sending: "正在读取画面", live: "画面已接入", face: "已锁定本人",
      no_face: "已收到画面，等待清晰人脸", stale: "视觉追踪已过期 · 正在恢复",
      error: "摄像头画面异常", stopped: "本次自检已结束"
    };
    this.faceStatus.textContent = labels[status] || status;
    this.faceStatus.dataset.state = status;
  }
}
