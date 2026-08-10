(() => {
  "use strict";

  const canvas = document.getElementById("simCanvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;

  const ui = {
    play: document.getElementById("playBtn"),
    reset: document.getElementById("resetBtn"),
    reverse: document.getElementById("reverseBtn"),
    direction: document.getElementById("directionStatus"),
    time: document.getElementById("timeValue"),
    particleSlider: document.getElementById("particleSlider"),
    particleValue: document.getElementById("particleValue"),
    noiseSlider: document.getElementById("noiseSlider"),
    noiseValue: document.getElementById("noiseValue"),
    driftSlider: document.getElementById("driftSlider"),
    driftValue: document.getElementById("driftValue"),
    speedSlider: document.getElementById("speedSlider"),
    speedValue: document.getElementById("speedValue"),
    lessonButtons: document.getElementById("lessonButtons"),
    lessonText: document.getElementById("lessonText"),
    sdeEquation: document.getElementById("sdeEquation"),
    keyIdea: document.getElementById("keyIdea")
  };

  const lessons = [
    {
      short: "Brown運動",
      subtitle: "random walk",
      text: "<strong>1粒子の運動</strong>から始めます。driftはなく、各粒子は独立なノイズだけを受けます。粒子数を増やしても、SDE自体は1粒子ごとの式です。",
      idea: "各粒子は独立なBrown運動をします。",
      drift: 0,
      noise: 0.85,
      init: "point",
      showDensity: false,
      showScore: false,
      allowReverse: false
    },
    {
      short: "Langevin",
      subtitle: "drift + noise",
      text: "ノイズに<strong>決定論的なdrift</strong>を加えます。ここでは f(x)=−kx。ランダムに揺れながら、粒子は原点へ戻されます。",
      idea: "Langevin方程式 = driftによる系統的な運動 + Brownノイズ。",
      drift: 0.65,
      noise: 0.7,
      init: "wide",
      showDensity: false,
      showScore: false,
      allowReverse: false
    },
    {
      short: "確率分布",
      subtitle: "ensemble → pₜ(x)",
      text: "同じSDEを<strong>多数の粒子</strong>に適用します。下段の曲線は粒子から推定した経験分布 pₜ(x)。1粒子では見えない「集団の形」が現れます。",
      idea: "SDEは粒子を動かす。多数の粒子の集団が pₜ(x) を作る。",
      drift: 0,
      noise: 0.85,
      init: "point",
      showDensity: true,
      showScore: false,
      allowReverse: false
    },
    {
      short: "score",
      subtitle: "∂ₓ log pₜ(x)",
      text: "scoreは<strong>粒子の履歴ではなく、現在の分布</strong>から決まります。矢印は密度が高くなる向き。二峰分布なら、場所によって向きが切り替わります。",
      idea: "同じ位置 x にいる粒子には、同じ score sₜ(x) が作用します。",
      drift: 0,
      noise: 0.72,
      init: "mixture",
      showDensity: true,
      showScore: true,
      allowReverse: false
    },
    {
      short: "Reverse",
      subtitle: "scoreで戻す",
      text: "まずForwardで十分に拡散させます。次にReverseを押すと、<strong>forward時に集団から得たscore</strong>を使って粒子を高密度側へ戻します。軌跡の逆再生ではありません。",
      idea: "Reverse diffusion は『元の粒子を覚える』のではなく、時刻ごとの score を使う。",
      drift: 0,
      noise: 0.78,
      init: "mixture",
      showDensity: true,
      showScore: true,
      allowReverse: true
    }
  ];

  const state = {
    lesson: 0,
    particles: [],
    running: false,
    direction: "forward",
    t: 0,
    dt: 0.012,
    maxT: 4.2,
    xMin: -6,
    xMax: 6,
    density: [],
    score: [],
    history: [],
    historyCursor: -1,
    stepsSinceRecord: 0,
    plotBins: 181,
    rngSpare: null
  };

  function randn() {
    if (state.rngSpare !== null) {
      const out = state.rngSpare;
      state.rngSpare = null;
      return out;
    }
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    const mag = Math.sqrt(-2 * Math.log(u));
    const z0 = mag * Math.cos(2 * Math.PI * v);
    const z1 = mag * Math.sin(2 * Math.PI * v);
    state.rngSpare = z1;
    return z0;
  }

  function lesson() { return lessons[state.lesson]; }
  function particleCount() { return Number(ui.particleSlider.value); }
  function sigma() { return Number(ui.noiseSlider.value); }
  function driftK() { return Number(ui.driftSlider.value); }
  function speed() { return Number(ui.speedSlider.value); }

  function initParticle(mode) {
    if (mode === "point") return 0.10 * randn();
    if (mode === "wide") return 2.7 * randn();
    if (mode === "mixture") return (Math.random() < 0.5 ? -2.25 : 2.25) + 0.42 * randn();
    return randn();
  }

  function resetSimulation() {
    state.running = false;
    state.direction = "forward";
    state.t = 0;
    state.history = [];
    state.historyCursor = -1;
    state.stepsSinceRecord = 0;
    state.rngSpare = null;
    const mode = lesson().init;
    state.particles = Array.from({ length: particleCount() }, () => initParticle(mode));
    computeDensityAndScore();
    recordSnapshot(true);
    syncUI();
    draw();
  }

  function applyLesson(index) {
    state.lesson = index;
    const L = lesson();
    ui.driftSlider.value = String(L.drift);
    ui.noiseSlider.value = String(L.noise);
    renderLessonButtons();
    ui.lessonText.innerHTML = L.text;
    resetSimulation();
  }

  function renderLessonButtons() {
    ui.lessonButtons.innerHTML = "";
    lessons.forEach((L, i) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "lesson-btn" + (i === state.lesson ? " active" : "");
      button.innerHTML = `<small>STEP ${i + 1}</small>${L.short}<small>${L.subtitle}</small>`;
      button.addEventListener("click", () => applyLesson(i));
      ui.lessonButtons.appendChild(button);
    });
  }

  function xToCanvas(x, left, width) {
    return left + (x - state.xMin) / (state.xMax - state.xMin) * width;
  }

  function computeDensityAndScore() {
    const n = state.plotBins;
    const counts = new Float64Array(n);
    const dx = (state.xMax - state.xMin) / (n - 1);

    for (const x of state.particles) {
      const f = (x - state.xMin) / dx;
      const i = Math.floor(f);
      const a = f - i;
      if (i >= 0 && i < n) counts[i] += 1 - a;
      if (i + 1 >= 0 && i + 1 < n) counts[i + 1] += a;
    }

    // Discrete Gaussian smoothing approximates a KDE and is much faster than O(N * bins).
    const radius = 7;
    const bandwidthBins = 2.8;
    const kernel = [];
    let kernelSum = 0;
    for (let j = -radius; j <= radius; j++) {
      const v = Math.exp(-0.5 * (j / bandwidthBins) ** 2);
      kernel.push(v);
      kernelSum += v;
    }

    const density = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = -radius; j <= radius; j++) {
        const k = i + j;
        if (k >= 0 && k < n) s += counts[k] * kernel[j + radius];
      }
      density[i] = s / kernelSum / Math.max(1, state.particles.length) / dx;
    }

    const score = new Float64Array(n);
    const eps = 1e-5;
    for (let i = 1; i < n - 1; i++) {
      const logL = Math.log(density[i - 1] + eps);
      const logR = Math.log(density[i + 1] + eps);
      score[i] = (logR - logL) / (2 * dx);
    }
    score[0] = score[1];
    score[n - 1] = score[n - 2];

    state.density = Array.from(density);
    state.score = Array.from(score);
  }

  function recordSnapshot(force = false) {
    if (!force && state.stepsSinceRecord < 4) return;
    computeDensityAndScore();
    state.history.push({
      t: state.t,
      density: Float32Array.from(state.density),
      score: Float32Array.from(state.score)
    });
    state.stepsSinceRecord = 0;
    if (state.history.length > 260) state.history.shift();
    state.historyCursor = state.history.length - 1;
  }

  function interpolateGrid(grid, x) {
    const n = grid.length;
    const u = (x - state.xMin) / (state.xMax - state.xMin) * (n - 1);
    const i = Math.floor(u);
    const a = u - i;
    if (i < 0) return grid[0];
    if (i >= n - 1) return grid[n - 1];
    return grid[i] * (1 - a) + grid[i + 1] * a;
  }

  function forwardStep() {
    const dt = state.dt;
    const s = sigma();
    const k = driftK();
    const sq = s * Math.sqrt(dt);
    for (let i = 0; i < state.particles.length; i++) {
      const x = state.particles[i];
      const drift = -k * x;
      state.particles[i] = x + drift * dt + sq * randn();
    }
    state.t = Math.min(state.maxT, state.t + dt);
    state.stepsSinceRecord++;
    recordSnapshot(false);
    if (state.t >= state.maxT) state.running = false;
  }

  function reverseStep() {
    if (state.historyCursor <= 0) {
      state.t = 0;
      state.running = false;
      return;
    }
    const dt = state.dt * 4; // one reverse step corresponds approximately to one recorded score interval
    const s = sigma();
    const k = driftK();
    const snap = state.history[state.historyCursor];
    const sq = s * Math.sqrt(dt);

    for (let i = 0; i < state.particles.length; i++) {
      const x = state.particles[i];
      const score = Math.max(-9, Math.min(9, interpolateGrid(snap.score, x)));
      // Backward-time discretization: -f + sigma^2 * score.
      const backwardDrift = k * x + s * s * score;
      state.particles[i] = x + backwardDrift * dt + sq * randn();
    }

    state.historyCursor--;
    state.t = state.history[state.historyCursor].t;
    // For display, show the distribution produced by the current reverse particles,
    // while arrows come from the forward-time score snapshot.
    computeDensityAndScore();
    state.score = Array.from(state.history[state.historyCursor].score);
  }

  function stepSimulation() {
    const iterations = speed();
    for (let j = 0; j < iterations; j++) {
      if (!state.running) break;
      if (state.direction === "forward") forwardStep();
      else reverseStep();
    }
    if (state.direction === "forward") computeDensityAndScore();
  }

  function switchReverse() {
    const L = lesson();
    if (!L.allowReverse) return;
    if (state.direction === "forward") {
      // Ensure the latest state has a matching score snapshot.
      recordSnapshot(true);
      if (state.history.length < 8 || state.t < 0.18) return;
      state.direction = "reverse";
      state.historyCursor = state.history.length - 1;
    } else {
      state.direction = "forward";
      // Starting forward again creates a new score history from this point.
      state.history = [];
      state.stepsSinceRecord = 0;
      recordSnapshot(true);
    }
    syncUI();
  }

  function syncUI() {
    const L = lesson();
    ui.play.textContent = state.running ? "❚❚ 一時停止" : "▶ 再生";
    ui.time.textContent = state.t.toFixed(2);
    ui.particleValue.textContent = particleCount();
    ui.noiseValue.textContent = sigma().toFixed(2);
    ui.driftValue.textContent = driftK().toFixed(2);
    ui.speedValue.textContent = `${speed()}×`;
    ui.reverse.disabled = !L.allowReverse || (state.direction === "forward" && (state.history.length < 8 || state.t < 0.18));
    ui.reverse.textContent = state.direction === "forward" ? "⇠ Reverse" : "⇢ Forwardへ";
    ui.direction.textContent = state.direction === "forward" ? "Forward" : "Reverse";
    ui.direction.classList.toggle("reverse", state.direction === "reverse");
    ui.keyIdea.textContent = L.idea;
    const k = driftK();
    ui.sdeEquation.textContent = k > 0.001 ? "dXₜ = −kXₜ dt + σ dWₜ" : "dXₜ = σ dWₜ";
  }

  function panelFrame(y, h, title, subtitle, tint) {
    ctx.save();
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1.2;
    roundedRect(18, y, W - 36, h, 18);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = tint;
    ctx.font = "700 18px system-ui, sans-serif";
    ctx.fillText(title, 42, y + 38);
    ctx.fillStyle = "#7c8798";
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText(subtitle, 42, y + 61);
    ctx.restore();
  }

  function roundedRect(x, y, w, h, r) {
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  function drawAxis(y, left, width) {
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + width, y); ctx.stroke();
    ctx.fillStyle = "#98a2b3";
    ctx.font = "12px system-ui, sans-serif";
    [-6,-4,-2,0,2,4,6].forEach(v => {
      const x = xToCanvas(v, left, width);
      ctx.beginPath(); ctx.moveTo(x, y - 4); ctx.lineTo(x, y + 4); ctx.stroke();
      ctx.fillText(String(v), x - 5, y + 18);
    });
  }

  function drawParticles(y, h) {
    const left = 78, width = W - 120;
    const axisY = y + h - 44;
    drawAxis(axisY, left, width);

    // Illustrate drift field for Langevin.
    if (driftK() > 0.001 && state.lesson === 1) {
      ctx.save();
      ctx.strokeStyle = "rgba(234,88,12,.55)";
      ctx.fillStyle = "rgba(234,88,12,.65)";
      ctx.lineWidth = 2;
      for (let v = -5; v <= 5; v += 1) {
        if (v === 0) continue;
        const x = xToCanvas(v, left, width);
        const dir = v > 0 ? -1 : 1;
        arrow(x, axisY - 58, x + dir * 28, axisY - 58, 6);
      }
      ctx.fillStyle = "#c2410c";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText("drift f(x)=−kx", left + width - 122, y + 35);
      ctx.restore();
    }

    ctx.fillStyle = "rgba(37,99,235,.58)";
    const n = state.particles.length;
    for (let i = 0; i < n; i++) {
      const x = xToCanvas(state.particles[i], left, width);
      if (x < left - 3 || x > left + width + 3) continue;
      const jitter = ((i * 47) % 71) / 71;
      const py = axisY - 10 - jitter * 88;
      ctx.beginPath(); ctx.arc(x, py, n > 900 ? 2.2 : 2.8, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = "#475467";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText(`N = ${n} particles`, left, y + 36);
  }

  function drawDensity(y, h) {
    const left = 78, width = W - 120;
    const baseY = y + h - 38;
    drawAxis(baseY, left, width);
    const density = state.density;
    const maxD = Math.max(0.15, ...density);
    const amp = h - 100;

    if (!lesson().showDensity) {
      ctx.save();
      ctx.fillStyle = "#98a2b3";
      ctx.font = "600 14px system-ui, sans-serif";
      ctx.fillText("このレッスンでは、まず粒子そのものに注目します。", left, y + h / 2);
      ctx.restore();
      return;
    }

    ctx.beginPath();
    density.forEach((d, i) => {
      const x = left + i / (density.length - 1) * width;
      const py = baseY - (d / maxD) * amp;
      if (i === 0) ctx.moveTo(x, py); else ctx.lineTo(x, py);
    });
    ctx.lineTo(left + width, baseY); ctx.lineTo(left, baseY); ctx.closePath();
    const grad = ctx.createLinearGradient(0, y, 0, baseY);
    grad.addColorStop(0, "rgba(8,145,178,.24)");
    grad.addColorStop(1, "rgba(8,145,178,.04)");
    ctx.fillStyle = grad; ctx.fill();

    ctx.beginPath();
    density.forEach((d, i) => {
      const x = left + i / (density.length - 1) * width;
      const py = baseY - (d / maxD) * amp;
      if (i === 0) ctx.moveTo(x, py); else ctx.lineTo(x, py);
    });
    ctx.strokeStyle = "#0891b2"; ctx.lineWidth = 3; ctx.stroke();

    ctx.fillStyle = "#0e7490";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText("粒子から推定した pₜ(x)", left, y + 35);
  }

  function drawScore(y, h) {
    const left = 78, width = W - 120;
    const axisY = y + h - 58;
    drawAxis(axisY, left, width);
    if (!lesson().showScore) {
      ctx.fillStyle = "#98a2b3";
      ctx.font = "600 14px system-ui, sans-serif";
      ctx.fillText("score は STEP 4 で登場します。", left, y + h / 2);
      return;
    }

    const grid = state.score;
    const arrowY = axisY - 52;
    const points = 25;
    for (let j = 0; j < points; j++) {
      const xVal = state.xMin + (j + 0.5) / points * (state.xMax - state.xMin);
      const s = interpolateGrid(grid, xVal);
      const len = Math.min(42, 7 + Math.abs(s) * 11);
      const x = xToCanvas(xVal, left, width);
      const dir = s >= 0 ? 1 : -1;
      ctx.strokeStyle = `rgba(124,58,237,${0.35 + Math.min(.55, Math.abs(s) * .10)})`;
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = 2.1;
      arrow(x - dir * len / 2, arrowY, x + dir * len / 2, arrowY, 6);
    }

    ctx.fillStyle = "#6d28d9";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText("sₜ(x)=∂ₓ log pₜ(x)  —  矢印は高密度方向", left, y + 35);

    if (state.direction === "reverse") {
      ctx.fillStyle = "rgba(124,58,237,.08)";
      roundedRect(left + width - 210, y + 16, 190, 42, 10); ctx.fill();
      ctx.fillStyle = "#6d28d9";
      ctx.font = "700 12px system-ui, sans-serif";
      ctx.fillText("Reverse: 保存したscoreを使用中", left + width - 196, y + 42);
    }
  }

  function arrow(x1, y1, x2, y2, head) {
    const angle = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(angle - Math.PI / 6), y2 - head * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x2 - head * Math.cos(angle + Math.PI / 6), y2 - head * Math.sin(angle + Math.PI / 6));
    ctx.closePath(); ctx.fill();
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#f8fbff"; ctx.fillRect(0, 0, W, H);

    panelFrame(18, 218, "1. Individual particles", "SDE が直接動かしている対象", "#2563eb");
    panelFrame(254, 238, "2. Distribution  pₜ(x)", "粒子集団から現れる経験分布", "#0891b2");
    panelFrame(510, 232, "3. Score  ∂ₓ log pₜ(x)", "分布が教える「高密度方向」", "#7c3aed");

    drawParticles(18, 218);
    drawDensity(254, 238);
    drawScore(510, 232);
  }

  let lastFrame = performance.now();
  function animate(now) {
    const elapsed = now - lastFrame;
    if (elapsed > 16) {
      if (state.running) stepSimulation();
      syncUI();
      draw();
      lastFrame = now;
    }
    requestAnimationFrame(animate);
  }

  ui.play.addEventListener("click", () => {
    state.running = !state.running;
    syncUI();
  });
  ui.reset.addEventListener("click", resetSimulation);
  ui.reverse.addEventListener("click", switchReverse);

  ui.particleSlider.addEventListener("change", resetSimulation);
  ui.noiseSlider.addEventListener("input", syncUI);
  ui.noiseSlider.addEventListener("change", () => {
    if (state.t === 0) computeDensityAndScore();
  });
  ui.driftSlider.addEventListener("input", syncUI);
  ui.speedSlider.addEventListener("input", syncUI);

  renderLessonButtons();
  ui.lessonText.innerHTML = lesson().text;
  resetSimulation();
  requestAnimationFrame(animate);
})();
