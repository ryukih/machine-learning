/**
 * =============================================================================
 * Machine Learning トップページ — 背景の粒子場
 * =============================================================================
 *
 * 【このファイルの目的】
 *   ページ上部に、ゆっくり拡散する粒子を薄く描く装飾レイヤ。
 *   このサイトの主題が拡散過程であることを、読ませる文字を増やさずに示す。
 *   情報は一切持たないため canvas は aria-hidden であり、
 *   このスクリプトが読み込まれなくてもページの機能は損なわれない。
 *
 * 【運動モデル】
 *   各粒子は等方な2次元Brown運動をする。
 *
 *       x ← x + s ξₓ,   y ← y + s ξ_y,   ξ ~ N(0,1)
 *
 *   Langevin & Score Lab のレッスン1と同じ運動だが、こちらは見た目だけが
 *   目的なので、歩幅 s はモデル単位ではなく「1フレームあたりの CSSピクセル」
 *   として直接与えている。
 *
 *   純粋なBrown運動は時間とともに広がって画面外へ出ていくため、画面端は
 *   ラップアラウンドさせる。こうすると粒子の総数と密度が一定に保たれる。
 *
 * 【奥行きの表現】
 *   大きい粒子ほど不透明で歩幅も大きくする。手前のものほど速く見えるため、
 *   視差のような奥行きが出る。
 *
 * 【単位と規約】
 *   - 座標・半径・歩幅はすべて CSSピクセル。
 *   - バッキングストアのみ devicePixelRatio 倍し、描画コードは CSSピクセルで書く。
 *   - prefers-reduced-motion が有効な環境ではアニメーションせず1枚だけ描く。
 * =============================================================================
 */

(() => {
  "use strict";

  const canvas = document.getElementById("particleField");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // ===========================================================================
  // 定数定義
  // ===========================================================================

  /** 粒子場の見た目を決める設定 */
  const FIELD = Object.freeze({
    /** 粒子1個あたりが受け持つ面積 [CSSピクセル²]。小さいほど密になる。 */
    areaPerParticle: 13000,
    /** 粒子数の下限・上限 [個]。極端な画面サイズでの破綻を防ぐ。 */
    minParticles: 34,
    maxParticles: 130,
    /** 半径の範囲 [CSSピクセル] */
    minRadius: 0.9,
    maxRadius: 2.8,
    /**
     * 1フレームあたりの歩幅の範囲 [CSSピクセル]。
     * 最小半径の粒子が minStep、最大半径の粒子が maxStep で動く。
     */
    minStep: 0.10,
    maxStep: 0.34,
    /** 不透明度の範囲。大きい粒子ほど濃くする。 */
    minOpacity: 0.06,
    maxOpacity: 0.26,
    /** ラップアラウンド時の余白 [CSSピクセル]。端で粒子が瞬間移動して見えるのを防ぐ。 */
    wrapMargin: 6,
  });

  /** 粒子の色。ページの violet と blue を混ぜて奥行きを出す。 */
  const PARTICLE_COLORS = Object.freeze(["124, 58, 237", "37, 99, 235"]);

  /** バッキングストア倍率の上限。3倍端末で塗り面積が9倍になるのを防ぐ。 */
  const MAX_DEVICE_PIXEL_RATIO = 2;

  /** アニメーションを望まないかどうか（OS/ブラウザの設定） */
  const prefersReducedMotion =
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ===========================================================================
  // 状態
  // ===========================================================================

  const state = {
    /** CSSピクセルでの描画領域 */
    width: 0,
    height: 0,
    /** {x, y, radius, step, color} の配列 */
    particles: [],
    /** Box–Muller法で余った2つ目の標準正規乱数 */
    gaussianSpare: null,
  };

  /** 標準正規分布 N(0,1) に従う乱数（Box & Muller, 1958） */
  function standardNormalRandom() {
    if (state.gaussianSpare !== null) {
      const cached = state.gaussianSpare;
      state.gaussianSpare = null;
      return cached;
    }
    let radiusSeed = 0;
    let angleSeed = 0;
    while (radiusSeed === 0) radiusSeed = Math.random();
    while (angleSeed === 0) angleSeed = Math.random();
    const magnitude = Math.sqrt(-2 * Math.log(radiusSeed));
    const angle = 2 * Math.PI * angleSeed;
    state.gaussianSpare = magnitude * Math.sin(angle);
    return magnitude * Math.cos(angle);
  }

  /** 範囲 [low, high] を一様にサンプリングする */
  function randomBetween(low, high) {
    return low + Math.random() * (high - low);
  }

  // ===========================================================================
  // 粒子の生成と更新
  // ===========================================================================

  /**
   * 半径から歩幅と不透明度を決める。
   * 大きい粒子ほど速く濃くすることで、手前にあるように見せる。
   */
  function depthFromRadius(radius) {
    const depth = (radius - FIELD.minRadius) / (FIELD.maxRadius - FIELD.minRadius);
    return {
      step: FIELD.minStep + depth * (FIELD.maxStep - FIELD.minStep),
      opacity: FIELD.minOpacity + depth * (FIELD.maxOpacity - FIELD.minOpacity),
    };
  }

  /** 描画領域いっぱいに粒子を一様配置する */
  function createParticles() {
    const target = Math.round((state.width * state.height) / FIELD.areaPerParticle);
    const count = Math.max(FIELD.minParticles, Math.min(FIELD.maxParticles, target));

    state.particles = Array.from({ length: count }, () => {
      const radius = randomBetween(FIELD.minRadius, FIELD.maxRadius);
      const depth = depthFromRadius(radius);
      const colorIndex = Math.floor(Math.random() * PARTICLE_COLORS.length);
      return {
        x: Math.random() * state.width,
        y: Math.random() * state.height,
        radius,
        step: depth.step,
        color: `rgba(${PARTICLE_COLORS[colorIndex]}, ${depth.opacity.toFixed(3)})`,
      };
    });
  }

  /** 座標を描画領域の反対側へ回り込ませる */
  function wrap(value, limit) {
    const margin = FIELD.wrapMargin;
    if (value < -margin) return limit + margin;
    if (value > limit + margin) return -margin;
    return value;
  }

  /** 全粒子をBrown運動で1ステップ進める */
  function stepParticles() {
    for (const particle of state.particles) {
      particle.x = wrap(particle.x + particle.step * standardNormalRandom(), state.width);
      particle.y = wrap(particle.y + particle.step * standardNormalRandom(), state.height);
    }
  }

  // ===========================================================================
  // 描画とサイズ調整
  // ===========================================================================

  function draw() {
    ctx.clearRect(0, 0, state.width, state.height);
    for (const particle of state.particles) {
      ctx.fillStyle = particle.color;
      ctx.beginPath();
      ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  /**
   * 表示サイズに合わせて canvas を作り直す。
   * setTransform は絶対指定なので、繰り返し呼んでも倍率は累積しない。
   */
  function resize() {
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    const pixelRatio = Math.min(window.devicePixelRatio || 1, MAX_DEVICE_PIXEL_RATIO);
    state.width = rect.width;
    state.height = rect.height;
    canvas.width = Math.round(rect.width * pixelRatio);
    canvas.height = Math.round(rect.height * pixelRatio);
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

    createParticles();
    draw();
  }

  function animate() {
    stepParticles();
    draw();
    requestAnimationFrame(animate);
  }

  // ===========================================================================
  // 起動
  // ===========================================================================

  resize();
  // 幅の変化にだけ追従する。モバイルでは縦スクロールでアドレスバーが伸縮し
  // 高さだけが変わるが、その都度粒子を作り直すとちらつくため無視する。
  let lastWidth = window.innerWidth;
  window.addEventListener("resize", () => {
    if (window.innerWidth === lastWidth) return;
    lastWidth = window.innerWidth;
    resize();
  });

  if (!prefersReducedMotion) requestAnimationFrame(animate);
})();
