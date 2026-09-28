const TARGET_RATE = 16000;
const MAX_UTTERANCE_SECONDS = 3;
const SILENCE_HOLD_MS = 450;
const CALIBRATION_SAMPLES = TARGET_RATE / 2;
const MIN_RMS_GATE = 0.0012;
const MIN_PEAK_GATE = 0.005;

export function encodePcm16(samples) {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  for (let index = 0; index < samples.length; index += 1) {
    const value = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(index * 2, value < 0 ? value * 32768 : value * 32767, true);
  }
  return buffer;
}

function downsample(input, sourceRate, targetRate = TARGET_RATE) {
  if (sourceRate === targetRate) return new Float32Array(input);
  const ratio = sourceRate / targetRate;
  const length = Math.max(1, Math.floor(input.length / ratio));
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    const from = Math.floor(index * ratio);
    const to = Math.max(from + 1, Math.floor((index + 1) * ratio));
    let sum = 0;
    for (let cursor = from; cursor < Math.min(to, input.length); cursor += 1) sum += input[cursor];
    output[index] = sum / Math.max(1, Math.min(to, input.length) - from);
  }
  return output;
}

function mergeChunks(chunks, count) {
  const merged = new Float32Array(count);
  let offset = 0;
  chunks.forEach(chunk => { merged.set(chunk, offset); offset += chunk.length; });
  return merged;
}

function normalizeForTranscription(samples) {
  let peak = 0;
  for (const value of samples) peak = Math.max(peak, Math.abs(value));
  if (peak < 0.0001 || peak >= 0.16) return samples;
  const gain = Math.min(12, 0.16 / peak);
  const normalized = new Float32Array(samples.length);
  for (let index = 0; index < samples.length; index += 1) {
    normalized[index] = Math.max(-1, Math.min(1, samples[index] * gain));
  }
  return normalized;
}

export class MediaCapture {
  constructor({ video, waveCanvas, fps = 3, onFrame, onAudioFeatures, onUtterance, onStatus, onCameraStatus, onError }) {
    this.video = video;
    this.waveCanvas = waveCanvas;
    this.fps = fps;
    this.onFrame = onFrame;
    this.onAudioFeatures = onAudioFeatures;
    this.onUtterance = onUtterance;
    this.onStatus = onStatus;
    this.onCameraStatus = onCameraStatus;
    this.onError = onError;
    this.stream = null;
    this.preparePromise = null;
    this.cancelled = false;
    this.frameTimer = null;
    this.audioContext = null;
    this.audioNodes = [];
    this.frameBusy = false;
    this.utteranceChunks = [];
    this.utteranceSamples = 0;
    this.speaking = false;
    this.silentSamples = 0;
    this.noiseFloor = 0.0005;
    this.noisePeak = 0.001;
    this.calibrationSamples = 0;
    this.startedAt = 0;
    this.lastFeatureAt = 0;
  }

  elapsed() { return Math.max(0, performance.now() - this.startedAt); }

  prepare() {
    if (this.preparePromise) return this.preparePromise;
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("当前浏览器不支持摄像头和麦克风采集");
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) throw new Error("当前浏览器不支持 Web Audio");
    this.cancelled = false;
    this.audioContext = new AudioContextClass();
    const audioReady = this.audioContext.resume();
    const cameraReady = navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 }
    });
    this.preparePromise = Promise.all([cameraReady, audioReady]).then(([stream]) => {
      if (this.cancelled) {
        stream.getTracks().forEach(track => track.stop());
        throw new Error("实时自检已取消");
      }
      this.stream = stream;
      return stream;
    }).catch(error => {
      cameraReady.then(stream => stream.getTracks().forEach(track => track.stop())).catch(() => {});
      this.audioContext?.close().catch(() => {});
      this.audioContext = null;
      this.preparePromise = null;
      throw error;
    });
    return this.preparePromise;
  }

  async start() {
    if (!this.stream) await this.prepare();
    if (!this.stream.getVideoTracks().length) throw new Error("没有取得摄像头视频轨道");
    if (!this.stream.getAudioTracks().length) throw new Error("没有取得麦克风音频轨道");
    this.startedAt = performance.now();
    this.video.srcObject = this.stream;
    this.video.muted = true;
    this.video.playsInline = true;
    this.onCameraStatus?.("starting");
    await this.video.play();
    await this.waitForVideoFrame();
    this.onStatus?.("listening");
    this.startFrameLoop(1000 / this.fps);
    await this.startAudioGraph(TARGET_RATE);
  }

  async waitForVideoFrame(timeoutMs = 7000) {
    const track = this.stream?.getVideoTracks()[0];
    if (!track) throw new Error("没有取得摄像头视频轨道，请检查浏览器摄像头权限");
    const deadline = performance.now() + timeoutMs;
    while (performance.now() < deadline) {
      if (track.readyState !== "live") throw new Error("摄像头视频轨道已结束，请重新连接摄像头后重试");
      if (this.video.readyState >= 2 && this.video.videoWidth > 0 && this.video.videoHeight > 0) return;
      await new Promise(resolve => window.setTimeout(resolve, 80));
    }
    throw new Error("摄像头已授权，但 7 秒内没有输出画面。请关闭其他正在使用摄像头的 SOCI-AI 标签页后重试。");
  }

  async stop() {
    this.cancelled = true;
    if (this.frameTimer) clearInterval(this.frameTimer);
    this.frameTimer = null;
    this.audioNodes.forEach(node => { try { node.disconnect(); } catch {} });
    this.audioNodes = [];
    if (this.audioContext) await this.audioContext.close().catch(() => {});
    this.audioContext = null;
    this.stream?.getTracks().forEach(track => track.stop());
    this.stream = null;
    this.preparePromise = null;
    this.video.srcObject = null;
    this.resetUtterance();
    this.onStatus?.("idle");
  }

  startFrameLoop(intervalMs) {
    const canvas = document.createElement("canvas");
    canvas.width = 480;
    canvas.height = 360;
    const context = canvas.getContext("2d", { alpha: false });
    let previousVideoTime = -1;
    let unchangedVideoTicks = 0;
    const capture = async () => {
      if (this.frameBusy || !this.stream) return;
      if (this.video.readyState < 2 || this.video.videoWidth <= 0) {
        this.onCameraStatus?.("waiting");
        return;
      }
      const videoTime = Number(this.video.currentTime);
      if (previousVideoTime >= 0 && videoTime <= previousVideoTime + 0.001) {
        unchangedVideoTicks += 1;
        if (unchangedVideoTicks >= 4) {
          this.onCameraStatus?.("stale");
          return;
        }
      } else {
        previousVideoTime = videoTime;
        unchangedVideoTicks = 0;
      }
      this.frameBusy = true;
      try {
        const width = this.video.videoWidth || 640;
        const height = this.video.videoHeight || 480;
        const scale = Math.min(1, 480 / Math.max(width, height));
        canvas.width = Math.max(1, Math.round(width * scale));
        canvas.height = Math.max(1, Math.round(height * scale));
        context.drawImage(this.video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.78));
        if (!blob) throw new Error("无法从摄像头画面生成图像帧");
        this.onCameraStatus?.("sending");
        await this.onFrame?.(blob, this.elapsed());
      } catch (error) {
        this.onCameraStatus?.("error");
        this.onError?.(error);
      } finally {
        this.frameBusy = false;
      }
    };
    this.frameTimer = setInterval(capture, intervalMs);
    capture();
  }

  async startAudioGraph() {
    if (!this.audioContext) throw new Error("麦克风音频通道没有初始化");
    if (this.audioContext.state !== "running") await this.audioContext.resume();
    if (this.audioContext.state !== "running") throw new Error("麦克风音频通道没有启动，请重新点击开始自检");
    const source = this.audioContext.createMediaStreamSource(this.stream);
    this.audioNodes.push(source);
    try {
      await this.startWorklet(source);
    } catch {
      this.startScriptProcessor(source);
    }
  }

  async startWorklet(source) {
    if (!this.audioContext.audioWorklet || typeof AudioWorkletNode === "undefined") throw new Error("worklet-unavailable");
    const processor = `class SociPcmProcessor extends AudioWorkletProcessor { process(inputs) { const channel=inputs[0]&&inputs[0][0]; if(channel) this.port.postMessage(channel.slice()); return true; } } registerProcessor("soci-pcm", SociPcmProcessor);`;
    const url = URL.createObjectURL(new Blob([processor], { type: "text/javascript" }));
    try { await this.audioContext.audioWorklet.addModule(url); } finally { URL.revokeObjectURL(url); }
    const node = new AudioWorkletNode(this.audioContext, "soci-pcm");
    const mute = this.audioContext.createGain(); mute.gain.value = 0;
    node.port.onmessage = event => this.handleSamples(event.data, this.audioContext.sampleRate);
    source.connect(node); node.connect(mute); mute.connect(this.audioContext.destination);
    this.audioNodes.push(node, mute);
  }

  startScriptProcessor(source) {
    const node = this.audioContext.createScriptProcessor(2048, 1, 1);
    const mute = this.audioContext.createGain(); mute.gain.value = 0;
    node.onaudioprocess = event => this.handleSamples(event.inputBuffer.getChannelData(0), this.audioContext.sampleRate);
    source.connect(node); node.connect(mute); mute.connect(this.audioContext.destination);
    this.audioNodes.push(node, mute);
  }

  handleSamples(raw, sourceRate) {
    const samples = downsample(raw, sourceRate);
    let sum = 0, peak = 0;
    for (const value of samples) { sum += value * value; peak = Math.max(peak, Math.abs(value)); }
    const rms = Math.sqrt(sum / Math.max(1, samples.length));
    const calibrationComplete = this.calibrationSamples >= CALIBRATION_SAMPLES;
    if (!calibrationComplete) {
      this.noiseFloor = this.noiseFloor * 0.86 + rms * 0.14;
      this.noisePeak = this.noisePeak * 0.86 + peak * 0.14;
      this.calibrationSamples += samples.length;
    }
    const adaptiveActive = calibrationComplete && (
      rms > Math.max(MIN_RMS_GATE, this.noiseFloor * 2.2)
      || peak > Math.max(MIN_PEAK_GATE, this.noisePeak * 2.4)
    );
    const active = rms >= 0.006 || peak >= 0.025 || adaptiveActive;
    if (!active && calibrationComplete) {
      this.noiseFloor = this.noiseFloor * 0.985 + rms * 0.015;
      this.noisePeak = this.noisePeak * 0.985 + peak * 0.015;
    }
    this.drawWave(samples, active);
    if (active) {
      this.speaking = true;
      this.silentSamples = 0;
    } else if (this.speaking) {
      this.silentSamples += samples.length;
    }
    if (this.speaking) {
      this.utteranceChunks.push(samples);
      this.utteranceSamples += samples.length;
    }
    const now = this.elapsed();
    if (now - this.lastFeatureAt >= 320) {
      this.lastFeatureAt = now;
      this.onAudioFeatures?.({
        at_ms: Math.round(now), rms: Math.min(1, rms * 4), peak: Math.min(1, peak),
        speech_ratio: active ? 1 : 0, pace: 0
      });
    }
    const silenceHeld = this.silentSamples >= TARGET_RATE * SILENCE_HOLD_MS / 1000;
    const shortUtteranceReady = silenceHeld && this.utteranceSamples >= 3200;
    const rollingSegmentReady = this.utteranceSamples >= TARGET_RATE * MAX_UTTERANCE_SECONDS;
    if (shortUtteranceReady || rollingSegmentReady) {
      const utterance = mergeChunks(this.utteranceChunks, this.utteranceSamples);
      this.onUtterance?.(encodePcm16(normalizeForTranscription(utterance)), Math.round(now));
      this.resetUtterance();
    }
  }

  resetUtterance() {
    this.utteranceChunks = [];
    this.utteranceSamples = 0;
    this.speaking = false;
    this.silentSamples = 0;
  }

  drawWave(samples, active) {
    const canvas = this.waveCanvas;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    const context = canvas.getContext("2d");
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, rect.width, rect.height);
    const gradient = context.createLinearGradient(0, 0, rect.width, 0);
    gradient.addColorStop(0, "#45e6c1"); gradient.addColorStop(1, active ? "#f7c66a" : "#63b9ff");
    context.strokeStyle = gradient; context.lineWidth = 1.5; context.beginPath();
    const stride = Math.max(1, Math.floor(samples.length / Math.max(1, rect.width)));
    for (let x = 0; x < rect.width; x += 1) {
      const value = samples[Math.min(samples.length - 1, x * stride)] || 0;
      const y = rect.height / 2 + value * rect.height * 0.42;
      if (x === 0) context.moveTo(x, y); else context.lineTo(x, y);
    }
    context.stroke();
  }
}
