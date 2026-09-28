const TAU = Math.PI * 2;
const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, Number(value) || 0));
const damp = (from, to, factor) => from + (to - from) * factor;

function mulberry32(seed) {
  return () => {
    let value = seed += 0x6D2B79F5;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}

function mixColor(a, b, amount) {
  const parse = color => color.match(/[\da-f]{2}/gi).map(value => parseInt(value, 16));
  const left = parse(a), right = parse(b), t = clamp(amount);
  return `rgb(${left.map((value, index) => Math.round(value + (right[index] - value) * t)).join(",")})`;
}

function sanitizeTreeState(state = {}) {
  const modes = ["observing", "friendly", "signal", "risk", "recovering"];
  return {
    mode: modes.includes(state.mode) ? state.mode : "observing",
    health: clamp(state.health ?? .55), risk: clamp(state.risk),
    bloom: clamp(state.bloom ?? .08), wind: clamp(state.wind ?? .12)
  };
}

const PALETTES = {
  observing: { trunk: "#416f76", leaf: "#72d6c9", glow: "#51c8ba", bloom: "#c5e6dc" },
  friendly: { trunk: "#507e68", leaf: "#49e0a6", glow: "#43f0bd", bloom: "#f3dd8d" },
  signal: { trunk: "#8c7450", leaf: "#e2bd64", glow: "#ffc968", bloom: "#ffe4a1" },
  risk: { trunk: "#86504b", leaf: "#ef795f", glow: "#ff735d", bloom: "#f3a06f" },
  recovering: { trunk: "#4f8074", leaf: "#56e5c6", glow: "#6ff7d2", bloom: "#fff0a8" }
};

export class LivingTree {
  constructor(canvas, { seed = 20260925 } = {}) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.seed = seed;
    this.random = mulberry32(seed);
    this.depth = 9;
    this.current = sanitizeTreeState();
    this.target = sanitizeTreeState();
    this.fallingLeaves = [];
    this.frameTimes = [];
    this.particleBudget = 1200;
    this.particles = Array.from({ length: 1200 }, (_, index) => ({
      angle: this.random() * TAU,
      radius: .08 + this.random() * .48,
      speed: .08 + this.random() * .35,
      size: .35 + this.random() * 1.35,
      alpha: .12 + this.random() * .55,
      phase: this.random() * TAU,
      index
    }));
    this.reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.running = false;
    this.raf = null;
    this.lastFrameAt = 0;
    this.resizeObserver = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => this.resize()) : null;
    this.resizeObserver?.observe(canvas);
    window.addEventListener("resize", () => this.resize(), { passive: true });
    this.resize();
  }

  setTarget(next) { this.target = sanitizeTreeState(next); }

  start() {
    if (this.running) return;
    this.running = true;
    this.lastFrameAt = performance.now();
    this.raf = requestAnimationFrame(now => this.frame(now));
  }

  stop() {
    this.running = false;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = null;
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.width = Math.max(320, rect.width || 800);
    this.height = Math.max(320, rect.height || 650);
    this.canvas.width = Math.round(this.width * dpr);
    this.canvas.height = Math.round(this.height * dpr);
    this.context.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  frame(now) {
    if (!this.running) return;
    const elapsed = Math.min(48, Math.max(0, now - this.lastFrameAt));
    this.lastFrameAt = now;
    this.frameTimes.push(elapsed);
    if (this.frameTimes.length > 60) this.frameTimes.shift();
    if (this.frameTimes.length === 60) {
      const average = this.frameTimes.reduce((sum, value) => sum + value, 0) / 60;
      this.particleBudget = average > 24 ? 360 : average > 17 ? 700 : 1200;
    }
    const motionScale = this.reducedMotion ? .12 : 1;
    this.current.health = damp(this.current.health, this.target.health, .055);
    this.current.risk = damp(this.current.risk, this.target.risk, .045);
    this.current.bloom = damp(this.current.bloom, this.target.bloom, .035);
    this.current.wind = damp(this.current.wind, this.target.wind, .05);
    this.current.mode = this.target.mode;
    this.drawBackdrop(now);
    this.drawParticles(now, motionScale);
    this.drawGlow();
    this.drawRoots(now, motionScale);
    const baseLength = Math.min(this.height * .205, this.width * .18);
    this.drawBranch(this.width * .5, this.height * .91, baseLength, -Math.PI / 2, this.depth, 1, now, motionScale);
    this.updateFallingLeaves(now, elapsed, motionScale);
    this.raf = requestAnimationFrame(time => this.frame(time));
  }

  palette() {
    const selected = PALETTES[this.current.mode] || PALETTES.observing;
    if (this.current.mode === "signal") {
      return { ...selected, leaf: mixColor("#49e0a6", "#e2bd64", this.current.risk * 1.35) };
    }
    return selected;
  }

  drawBackdrop(now) {
    const ctx = this.context;
    ctx.clearRect(0, 0, this.width, this.height);
    const sky = ctx.createRadialGradient(this.width * .5, this.height * .48, 20, this.width * .5, this.height * .52, this.width * .72);
    sky.addColorStop(0, "rgba(18,47,57,.52)");
    sky.addColorStop(.45, "rgba(6,22,34,.38)");
    sky.addColorStop(1, "rgba(3,10,18,.96)");
    ctx.fillStyle = sky; ctx.fillRect(0, 0, this.width, this.height);
    ctx.globalAlpha = .22;
    for (let index = 0; index < 46; index += 1) {
      const x = (index * 83.13 % this.width);
      const y = (index * 47.73 % (this.height * .78));
      const pulse = .5 + .5 * Math.sin(now * .0007 + index);
      ctx.fillStyle = this.palette().glow;
      ctx.beginPath(); ctx.arc(x, y, .35 + pulse * .8, 0, TAU); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  drawGlow() {
    const ctx = this.context, palette = this.palette();
    const radius = Math.min(this.width, this.height) * (.29 + this.current.health * .08);
    const glow = ctx.createRadialGradient(this.width / 2, this.height * .54, 0, this.width / 2, this.height * .54, radius);
    glow.addColorStop(0, palette.glow.replace("#", "#") + "38");
    glow.addColorStop(.46, palette.glow + "12");
    glow.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = glow; ctx.fillRect(0, 0, this.width, this.height);
  }

  drawParticles(now, motionScale) {
    const ctx = this.context, palette = this.palette();
    const centerX = this.width / 2, centerY = this.height * .50;
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    for (let index = 0; index < this.particleBudget; index += 1) {
      const particle = this.particles[index];
      const sway = Math.sin(now * .00025 * particle.speed + particle.phase) * 12 * motionScale;
      const x = centerX + Math.cos(particle.angle) * particle.radius * this.width * .72 + sway;
      const y = centerY + Math.sin(particle.angle) * particle.radius * this.height * .48;
      ctx.globalAlpha = particle.alpha * (.36 + this.current.health * .64);
      ctx.fillStyle = index % 7 === 0 ? palette.bloom : palette.leaf;
      ctx.beginPath(); ctx.arc(x, y, particle.size, 0, TAU); ctx.fill();
    }
    ctx.restore();
  }

  drawRoots(now, motionScale) {
    const ctx = this.context, palette = this.palette();
    const rootY = this.height * .91;
    ctx.save(); ctx.strokeStyle = palette.trunk; ctx.globalAlpha = .44; ctx.lineWidth = 1;
    for (let index = 0; index < 13; index += 1) {
      const side = index % 2 ? 1 : -1;
      const reach = this.width * (.08 + (index % 6) * .026);
      const wobble = Math.sin(now * .0004 + index) * 4 * motionScale;
      ctx.beginPath(); ctx.moveTo(this.width / 2, rootY);
      ctx.bezierCurveTo(this.width / 2 + side * reach * .25, rootY + 8, this.width / 2 + side * reach * .7, rootY + 22 + wobble, this.width / 2 + side * reach, rootY + 17);
      ctx.stroke();
    }
    ctx.restore();
  }

  drawBranch(x, y, length, angle, depth, branchSeed, now, motionScale) {
    const ctx = this.context, palette = this.palette();
    const riskCurl = this.current.risk * .19;
    const wind = Math.sin(now * .0014 + branchSeed * 1.77) * this.current.wind * .055 * motionScale;
    const lean = depth < 6 ? wind + riskCurl * (branchSeed % 2 ? 1 : -1) : wind * .25;
    const nextAngle = angle + lean;
    const endX = x + Math.cos(nextAngle) * length;
    const endY = y + Math.sin(nextAngle) * length + this.current.risk * (this.depth - depth) * .55;
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineWidth = Math.max(.55, depth * depth * .115);
    ctx.strokeStyle = depth > 5 ? palette.trunk : mixColor(palette.trunk, palette.leaf, .34);
    ctx.shadowColor = palette.glow; ctx.shadowBlur = depth > 6 ? 5 : 2.5;
    ctx.beginPath(); ctx.moveTo(x, y);
    const bend = Math.sin(branchSeed * 2.31) * length * .07;
    ctx.quadraticCurveTo((x + endX) / 2 + bend, (y + endY) / 2, endX, endY); ctx.stroke();
    ctx.restore();
    if (depth <= 0) {
      this.drawLeaf(endX, endY, nextAngle, branchSeed, now);
      if (this.current.bloom > .14 && branchSeed % 4 === 0) this.drawBloom(endX, endY, branchSeed, now);
      return;
    }
    const spread = .34 + (branchSeed % 5) * .018 + this.current.health * .07;
    const shrink = .71 + (branchSeed % 3) * .015;
    this.drawBranch(endX, endY, length * shrink, nextAngle - spread, depth - 1, branchSeed * 2, now, motionScale);
    this.drawBranch(endX, endY, length * (shrink - .025), nextAngle + spread, depth - 1, branchSeed * 2 + 1, now, motionScale);
    if (depth === 5 && branchSeed % 3 === 0) {
      this.drawBranch(endX, endY, length * .53, nextAngle + Math.sin(branchSeed) * .15, depth - 2, branchSeed * 3 + 5, now, motionScale);
    }
  }

  drawLeaf(x, y, angle, seed, now) {
    const ctx = this.context, palette = this.palette();
    const pulse = 1 + Math.sin(now * .002 + seed) * .08;
    const size = (2.2 + (seed % 5) * .45) * pulse * (.5 + this.current.health * .65);
    ctx.save(); ctx.translate(x, y); ctx.rotate(angle + Math.sin(seed) * .8);
    ctx.fillStyle = palette.leaf; ctx.globalAlpha = .48 + this.current.health * .48;
    ctx.shadowColor = palette.glow; ctx.shadowBlur = 7;
    ctx.beginPath(); ctx.ellipse(0, 0, size * 1.8, size * .72, 0, 0, TAU); ctx.fill();
    ctx.restore();
  }

  drawBloom(x, y, seed, now) {
    const ctx = this.context, palette = this.palette();
    const size = (1.6 + this.current.bloom * 2.4) * (1 + Math.sin(now * .0018 + seed) * .08);
    ctx.save(); ctx.translate(x, y); ctx.globalAlpha = this.current.bloom;
    ctx.fillStyle = palette.bloom; ctx.shadowColor = palette.bloom; ctx.shadowBlur = 10;
    for (let petal = 0; petal < 5; petal += 1) {
      ctx.rotate(TAU / 5); ctx.beginPath(); ctx.ellipse(0, -size, size * .55, size, 0, 0, TAU); ctx.fill();
    }
    ctx.fillStyle = "#fff5bd"; ctx.beginPath(); ctx.arc(0, 0, size * .45, 0, TAU); ctx.fill(); ctx.restore();
  }

  updateFallingLeaves(now, elapsed, motionScale) {
    const palette = this.palette();
    if (!this.reducedMotion && this.current.risk > .28 && this.fallingLeaves.length < 110) {
      const chance = this.current.risk * elapsed * .004;
      if (Math.random() < chance) this.fallingLeaves.push({
        x: this.width * (.25 + Math.random() * .5), y: this.height * (.25 + Math.random() * .4),
        vx: (Math.random() - .5) * .55, vy: .28 + Math.random() * .65,
        spin: Math.random() * TAU, life: 1
      });
    }
    const ctx = this.context;
    this.fallingLeaves = this.fallingLeaves.filter(leaf => {
      leaf.x += leaf.vx * elapsed * motionScale; leaf.y += leaf.vy * elapsed * motionScale;
      leaf.spin += elapsed * .002; leaf.life -= elapsed * .00022;
      ctx.save(); ctx.translate(leaf.x, leaf.y); ctx.rotate(leaf.spin);
      ctx.globalAlpha = clamp(leaf.life); ctx.fillStyle = palette.leaf;
      ctx.beginPath(); ctx.ellipse(0, 0, 3.2, 1.2, 0, 0, TAU); ctx.fill(); ctx.restore();
      return leaf.life > 0 && leaf.y < this.height * .96;
    });
  }
}
