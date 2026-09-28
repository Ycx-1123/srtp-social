export const MODALITIES = {
  semantic: { label: "语义微偏差", sub: "COLD / CBBQ", icon: "Tx" },
  acoustic: { label: "声学张力", sub: "Prosody / ASR", icon: "Au" },
  vision: { label: "本人面部信号", sub: "YOLOv8-FER", icon: "Vi" },
  pose: { label: "姿态互动", sub: "Pose / Turn", icon: "Po" },
};

export const RISK_LABELS = { safe: "安全基线", observe: "持续观察", warning: "建议关注", high: "高风险累积", critical: "关键偏差" };
export const ACTION_LABELS = { observe: "静默观察", ambient: "环境提示", nudge: "柔性建议", pause: "保护性暂停" };
export const EVIDENCE_LABELS = { insufficient: "证据不足", partial: "部分证据", sufficient: "证据充分" };

export const appState = {
  snapshot: null,
  liveState: null,
  dialogueKeys: new Set(),
  activeView: "live",
  history: [],
};

export function formatTime(ms = 0) {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}
