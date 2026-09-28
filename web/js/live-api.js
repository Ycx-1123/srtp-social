async function parseResponse(response) {
  if (response.ok) return response.json();
  const payload = await response.json().catch(() => ({}));
  const error = new Error(payload.detail?.message || `实时请求失败 (${response.status})`);
  error.code = payload.detail?.code || "live_request_failed";
  error.status = response.status;
  throw error;
}

async function liveRequest(url, options = {}, timeoutMs = 5000, requestLabel = "本地实时请求") {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await parseResponse(await fetch(url, { ...options, signal: controller.signal }));
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error(`${requestLabel}超时（${Math.round(timeoutMs / 1000)} 秒）；后续采样将继续尝试`);
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export const getLiveCapabilities = () => liveRequest("/api/live/capabilities", {}, 5000);

export const createLiveSession = () => liveRequest("/api/live/sessions", {
  method: "POST",
  headers: { "Content-Type": "application/json" }
}, 10000);

export const getLiveState = sessionId =>
  liveRequest(`/api/live/sessions/${sessionId}/state`, {}, 2500, "实时状态查询");

export const sendLiveFrame = (sessionId, blob, atMs) => liveRequest(
  `/api/live/sessions/${sessionId}/frame?at_ms=${Math.max(0, Math.round(atMs))}`,
  { method: "POST", headers: { "Content-Type": blob.type || "image/jpeg" }, body: blob },
  2500,
  "摄像头画面分析"
);

export const sendAudioFeatures = (sessionId, features) => liveRequest(
  `/api/live/sessions/${sessionId}/audio-features`,
  { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(features) },
  2000,
  "麦克风信号上传"
);

export const sendAudioChunk = (sessionId, pcm, atMs) => liveRequest(
  `/api/live/sessions/${sessionId}/audio-chunk?sample_rate=16000&at_ms=${Math.max(0, Math.round(atMs))}`,
  { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: pcm },
  30000,
  "本地语音转写"
);

export const stopLiveSession = sessionId => liveRequest(
  `/api/live/sessions/${sessionId}/stop`,
  { method: "POST", headers: { "Content-Type": "application/json" } },
  5000
);
