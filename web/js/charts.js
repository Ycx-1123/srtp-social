function setup(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(300, rect.width);
  const height = Math.max(90, rect.height);
  canvas.width = width * dpr; canvas.height = height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

export class SbiChart {
  constructor(canvas) { this.canvas = canvas; this.points = []; this.duration = 45000; this.interventions = []; }
  reset(duration = 45000) { this.points = []; this.interventions = []; this.duration = duration; this.draw(); }
  push(atMs, value, action = "observe") {
    const last = this.points[this.points.length - 1];
    if (last && last.at === atMs) last.value = value;
    else this.points.push({ at: atMs, value });
    if (!["observe", "ambient"].includes(action) && !this.interventions.includes(atMs)) this.interventions.push(atMs);
    if (this.points.length > 320) this.points.shift();
    this.draw();
  }
  draw() {
    const { ctx, width: w, height: h } = setup(this.canvas);
    ctx.clearRect(0, 0, w, h);
    const pad = { l: 28, r: 8, t: 8, b: 15 }, iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    ctx.strokeStyle = "rgba(132,174,202,.11)"; ctx.lineWidth = 1;
    [0, 25, 50, 75, 100].forEach(v => { const y = pad.t + ih * (1 - v / 100); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w-pad.r, y); ctx.stroke(); });
    const warningY = pad.t + ih * .45;
    ctx.setLineDash([4,5]); ctx.strokeStyle = "rgba(255,189,102,.55)"; ctx.beginPath(); ctx.moveTo(pad.l,warningY);ctx.lineTo(w-pad.r,warningY);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle = "#607b8d";ctx.font="8px ui-monospace, monospace";ctx.fillText("100",2,pad.t+4);ctx.fillText("55",8,warningY+3);ctx.fillText("0",14,h-12);
    if (this.points.length < 1) return;
    const xy = p => [pad.l + iw * Math.min(1, p.at / this.duration), pad.t + ih * (1 - p.value/100)];
    const gradient = ctx.createLinearGradient(0,pad.t,0,h);gradient.addColorStop(0,"rgba(78,231,255,.34)");gradient.addColorStop(1,"rgba(78,231,255,0)");
    ctx.beginPath(); this.points.forEach((p,i)=>{const [x,y]=xy(p); i?ctx.lineTo(x,y):ctx.moveTo(x,y)}); const [lastX]=xy(this.points.at(-1));ctx.lineTo(lastX,h-pad.b);ctx.lineTo(pad.l,h-pad.b);ctx.closePath();ctx.fillStyle=gradient;ctx.fill();
    ctx.beginPath();this.points.forEach((p,i)=>{const [x,y]=xy(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle="#4ee7ff";ctx.lineWidth=2;ctx.shadowColor="#4ee7ff";ctx.shadowBlur=7;ctx.stroke();ctx.shadowBlur=0;
    this.interventions.forEach(at=>{const x=pad.l+iw*(at/this.duration);ctx.strokeStyle="#9a7cff";ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,h-pad.b);ctx.stroke();ctx.fillStyle="#9a7cff";ctx.beginPath();ctx.arc(x,pad.t+4,3,0,Math.PI*2);ctx.fill()});
  }
}
