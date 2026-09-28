export async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail?.message || `请求失败 (${response.status})`);
  }
  return response.json();
}

export const control = (name, payload) => request(`/api/runtime/${name}`, {
  method: "POST",
  body: payload === undefined ? undefined : JSON.stringify(payload),
});

export function connectState(onState, onStatus) {
  let attempts = 0;
  let stopped = false;
  let socket;
  let pollTimer;
  const startPolling = () => {
    if (pollTimer || stopped) return;
    onStatus("polling");
    pollTimer = setInterval(async () => {
      try { onState(await request("/api/runtime/state")); }
      catch { onStatus("reconnecting"); }
    }, 1000);
  };
  const stopPolling = () => { if (pollTimer) clearInterval(pollTimer); pollTimer = undefined; };
  const open = () => {
    if (stopped) return;
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${protocol}://${location.host}/ws/state`);
    socket.onopen = () => { attempts = 0; stopPolling(); onStatus("connected"); };
    socket.onmessage = event => {
      try { onState(JSON.parse(event.data)); } catch { onStatus("invalid"); }
    };
    socket.onerror = () => onStatus("error");
    socket.onclose = () => {
      if (stopped) return;
      onStatus(pollTimer ? "polling" : "reconnecting");
      attempts += 1;
      if (attempts >= 2) startPolling();
      setTimeout(open, Math.min(5000, 500 * (2 ** (attempts - 1))));
    };
  };
  open();
  return () => { stopped = true; stopPolling(); socket?.close(); };
}
