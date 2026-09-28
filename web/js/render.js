import { ACTION_LABELS, EVIDENCE_LABELS, MODALITIES, RISK_LABELS, appState, formatTime } from "./state.js";

const $ = id => document.getElementById(id);
const safeText = value => value === undefined || value === null || value === "" ? "证据不足" : String(value);
const featureSummary = features => {
  const entries = Object.entries(features || {}).slice(0, 2);
  if (!entries.length) return "等待通道信号";
  return entries.map(([key, value]) => `${key} ${typeof value === "number" ? value.toFixed(2) : value}`).join(" · ");
};

export function renderModalityGrid(snapshot) {
  const html = Object.entries(MODALITIES).map(([key, meta]) => {
    const contribution = snapshot.fusion.contributions[key];
    const reliability = snapshot.fusion.reliabilities[key];
    const known = Number.isFinite(contribution) && Number.isFinite(reliability) && reliability > 0;
    const percent = known ? Math.min(100, Math.max(0, contribution * 100)) : 0;
    return `<div class="modality-card">
      <div class="modality-top"><span class="modality-icon">${meta.icon}</span><span class="modality-name"><b>${meta.label}</b><small>${meta.sub}</small></span><span class="modality-score">${known ? percent.toFixed(0)+"%" : "—"}</span></div>
      <div class="meter"><i style="width:${percent}%"></i></div><div class="modality-detail">${safeText(featureSummary(snapshot.modality_features[key]))}</div>
    </div>`;
  }).join("");
  $("modality-grid").innerHTML = html;
}

function renderDialogue(snapshot) {
  if (!snapshot.dialogue) return;
  const key = `${snapshot.elapsed_ms}:${snapshot.speaker}:${snapshot.dialogue}`;
  if (appState.dialogueKeys.has(key)) return;
  if (!appState.dialogueKeys.size) $("dialogue-stream").innerHTML = "";
  appState.dialogueKeys.add(key);
  $("dialogue-stream").querySelectorAll(".dialogue-item.active").forEach(el => el.classList.remove("active"));
  const item = document.createElement("div"); item.className = "dialogue-item active";
  const meta = document.createElement("div"); meta.className = "dialogue-meta"; meta.textContent = `${formatTime(snapshot.elapsed_ms)} / ${snapshot.speaker || "发言者"}`;
  const text = document.createElement("p"); text.textContent = snapshot.dialogue;
  const stream = $("dialogue-stream");
  item.append(meta, text); stream.append(item);
  while (stream.children.length > 8) stream.firstElementChild.remove();
  stream.scrollTop = stream.scrollHeight;
}

export function renderSnapshot(snapshot, chart) {
  appState.snapshot = snapshot;
  const fusion = snapshot.fusion, intervention = snapshot.intervention;
  $("sbi-value").textContent = Number(fusion.sbi).toFixed(1);
  $("risk-label").textContent = RISK_LABELS[fusion.risk_level] || fusion.risk_level;
  $("energy-field").querySelector(".energy-core").className = `energy-core ${fusion.risk_level}`;
  $("instant-risk").textContent = Number(fusion.instantaneous_risk).toFixed(2);
  $("energy-value").textContent = Number(fusion.energy).toFixed(2);
  $("dominant-modalities").textContent = fusion.dominant_modalities.length ? fusion.dominant_modalities.map(k=>MODALITIES[k]?.label||k).join(" × ") : "等待证据";
  $("evidence-status").textContent = `${EVIDENCE_LABELS[fusion.evidence_status] || fusion.evidence_status} · ${String(fusion.provenance).toUpperCase()}`;
  $("evidence-status").className = `evidence-pill ${fusion.evidence_status}`;
  $("phase-label").textContent = snapshot.phase || "准备阶段";
  $("explanation-text").textContent = fusion.explanation;
  $("intervention-action").textContent = ACTION_LABELS[intervention.action] || intervention.action;
  $("intervention-message").textContent = intervention.message;
  $("intervention-reason").textContent = `${intervention.reason} · ${String(intervention.provenance).toUpperCase()}`;
  $("intervention-signal").dataset.action = intervention.action;
  $("elapsed-time").textContent = formatTime(snapshot.elapsed_ms); $("duration-time").textContent = formatTime(snapshot.duration_ms);
  $("time-scrubber").max = snapshot.duration_ms; $("time-scrubber").value = snapshot.elapsed_ms;
  document.querySelectorAll(".sensor").forEach(el=>el.classList.remove("active"));
  fusion.dominant_modalities.forEach(key=>document.querySelector(`.sensor-${key}`)?.classList.add("active"));
  renderDialogue(snapshot); renderModalityGrid(snapshot); chart.duration = snapshot.duration_ms; chart.push(snapshot.elapsed_ms, fusion.sbi, intervention.action);
}

export function renderExplanation(snapshot) {
  const rows = Object.entries(snapshot.fusion.contributions)
    .sort((a,b)=>b[1]-a[1])
    .map(([key,value],index)=>`<div class="rank-row"><span class="rank-number">0${index+1}</span><span>${MODALITIES[key]?.label||key}</span><span class="rank-bar"><i style="width:${Math.min(100,value*100)}%"></i></span><span class="rank-value">${(value*100).toFixed(1)}%</span></div>`).join("");
  $("contribution-ranking").innerHTML = rows;
  $("reliability-grid").innerHTML = Object.entries(snapshot.fusion.reliabilities).map(([key,value])=>`<div class="reliability-cell"><span>${MODALITIES[key]?.label||key}</span><b>${(value*100).toFixed(0)}%</b><small>${value>.7?"高质量信号":value>0?"谨慎纳入":"证据不足"}</small></div>`).join("");
  $("explain-sbi").textContent = `SBI ${snapshot.fusion.sbi.toFixed(1)}`;
  $("explain-reasoning").textContent = snapshot.fusion.explanation;
}

export function renderInterventionComparison(snapshot) {
  const current = snapshot.intervention.action;
  const dialogue = snapshot.dialogue || "当前对话仍处于基线状态。";
  const policies = [
    {action:"observe",index:"A / PASSIVE",title:"静默观察",condition:"证据不足或风险尚未持续",quote:"保持静默，仅继续积累跨模态证据。",tone:"#557386"},
    {action:"ambient",index:"B / AMBIENT",title:"环境提示",condition:"出现趋势，但不适合打断交流",quote:"通过柔和视觉提示提醒放慢节奏、留出发言空间。",tone:"#4ee7ff"},
    {action:"nudge",index:"C / SEMANTIC",title:"语义建议",condition:"高置信度风险跨通道持续",quote:snapshot.intervention.message || "建议先复述对方约束，再讨论交付边界。",tone:"#9a7cff"},
  ];
  $("policy-comparison").innerHTML = policies.map(item=>`<article class="policy-card ${current===item.action?"selected":""}" style="--tone:${item.tone}"><span class="policy-index">${item.index}</span><h3>${item.title}</h3><p class="policy-condition">${item.condition}</p><blockquote>${item.quote}</blockquote><footer>输入语境：${dialogue}<br>策略来源：SIMULATED COUNTERFACTUAL</footer></article>`).join("");
}

function drawProfile(history) {
  const canvas = $("profile-trend"); if (!canvas) return;
  const dpr=Math.min(devicePixelRatio||1,2),rect=canvas.getBoundingClientRect(),w=Math.max(300,rect.width),h=Math.max(180,rect.height);canvas.width=w*dpr;canvas.height=h*dpr;
  const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.strokeStyle="rgba(125,170,199,.11)";
  for(let i=0;i<5;i++){const y=15+(h-30)*i/4;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}
  if(history.length<2)return;const maxAt=Math.max(...history.map(p=>p.at),1);const xy=p=>[w*p.at/maxAt,15+(h-30)*(1-p.value/100)];
  const grad=ctx.createLinearGradient(0,0,0,h);grad.addColorStop(0,"rgba(154,124,255,.34)");grad.addColorStop(1,"rgba(78,231,255,0)");ctx.beginPath();history.forEach((p,i)=>{const[x,y]=xy(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.lineTo(w,h-15);ctx.lineTo(0,h-15);ctx.fillStyle=grad;ctx.fill();ctx.beginPath();history.forEach((p,i)=>{const[x,y]=xy(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle="#9a7cff";ctx.lineWidth=2;ctx.shadowColor="#9a7cff";ctx.shadowBlur=8;ctx.stroke();ctx.shadowBlur=0;
}

export function renderProfile(snapshot) {
  const last=appState.history.at(-1);if(!last||last.at!==snapshot.elapsed_ms)appState.history.push({at:snapshot.elapsed_ms,value:snapshot.fusion.sbi});
  if(appState.history.length>300)appState.history.shift();const values=appState.history.map(p=>p.value);
  $("profile-peak").textContent=Math.max(...values).toFixed(1);$("profile-mean").textContent=(values.reduce((a,b)=>a+b,0)/values.length).toFixed(1);
  $("profile-dominant").textContent=snapshot.fusion.dominant_modalities.map(key=>MODALITIES[key]?.label||key).join(" / ")||"证据不足";
  $("profile-recovery").textContent=snapshot.phase==="recovery"?"表达修正":snapshot.fusion.risk_level==="safe"?"稳定":"仍在累积";drawProfile(appState.history);
}

export function renderEvidence(payload) {
  $("adapter-health-grid").innerHTML=payload.runtime_adapters.map(item=>`<div class="adapter-card"><span>${item.name}</span><b class="${item.available?"":"optional"}">${item.active?"ACTIVE":item.available?"AVAILABLE":"OPTIONAL / NOT LOADED"}</b></div>`).join("");
  $("model-card-grid").innerHTML=payload.model_cards.map(card=>`<article class="model-card"><div class="model-card-head"><h3>${card.name}</h3><span class="stage">${card.stage}</span></div><p class="model-role">${card.role}</p><div class="io-row"><div><span>INPUT</span><small>${card.input}</small></div><div><span>OUTPUT</span><small>${card.output}</small></div></div><div class="metric-list">${card.metrics.map(metric=>`<div class="metric-row"><span>${metric.name}</span><b>${metric.value} ${metric.unit}</b><span class="metric-origin ${metric.provenance}">${metric.provenance.toUpperCase()}</span></div>`).join("")}</div><ul class="limitations">${card.limitations.map(item=>`<li>${item}</li>`).join("")}</ul></article>`).join("");
  $("privacy-boundary-text").textContent=payload.privacy.boundary;
}

export function renderDatasetAudit(audit) {
  $("audit-verdict").textContent=audit.independent_evaluation_valid?"独立评估条件有效":"发现跨划分内容泄漏";
  $("audit-conclusion").textContent=audit.conclusion;
  $("audit-total").textContent=audit.total_files;$("audit-train").textContent=audit.split_counts.train??"—";$("audit-val").textContent=audit.split_counts.val??audit.split_counts.validation??"—";$("audit-leaked").textContent=audit.cross_split_exact_duplicates;
  $("dataset-audit-card").classList.toggle("valid",audit.independent_evaluation_valid);
}

export function renderConnection(status) {
  const labels = { connected: "实时通道已连接", polling: "兼容轮询已连接", reconnecting: "连接恢复中", error: "通道异常", invalid: "数据格式异常" };
  $("connection-status").textContent = labels[status] || status;
  document.querySelector(".status-dot").classList.toggle("connected", ["connected", "polling"].includes(status));
}
