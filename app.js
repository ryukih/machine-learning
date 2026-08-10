/**
 * =============================================================================
 * Langevin & Score Lab — 1次元確率微分方程式(SDE)の対話的教材
 * =============================================================================
 *
 * 【このファイルの目的】
 *   Brown運動 → Langevin方程式 → 確率分布 pₜ(x) → score ∂ₓ log pₜ(x) →
 *   reverse diffusion という概念の連鎖を、同一の粒子集団を3つの視点
 *   （粒子・分布・score）で同時に描画することによって示す。
 *
 *   教材として伝えたい主張は次の2点である。
 *     - SDEが直接記述しているのは、個々の粒子 Xₜ⁽ⁱ⁾ の運動である。
 *     - score が参照しているのは粒子集団が作る確率分布 pₜ(x) であり、
 *       個々の粒子の履歴は一切参照しない。
 *
 * 【物理モデル】
 *   Ornstein–Uhlenbeck 型の Langevin 方程式（Uhlenbeck & Ornstein, 1930）
 *
 *       dXₜ = f(Xₜ) dt + σ dWₜ,     f(x) = −k x
 *
 *   drift強度 k = 0 のとき、純粋な Brown 運動（Wiener過程）に帰着する。
 *
 * 【数値解法】
 *   1. 前進積分 — Euler–Maruyama 法
 *          X_{t+Δt} = Xₜ + f(Xₜ)Δt + σ√Δt ξ,    ξ ~ N(0,1)
 *
 *   2. 密度推定 — 粒子を線形補間でビンへ配分したヒストグラムに、離散
 *      Gaussianカーネルを畳み込む。素朴なKDEの計算量 O(N × bins) を
 *      O(N + bins × radius) へ落とすための近似である。
 *
 *   3. score推定 — log pₜ(x) の中心差分
 *          sₜ(xᵢ) ≈ [log p(x_{i+1}) − log p(x_{i−1})] / (2Δx)
 *
 *   4. 逆時間積分 — reverse-time SDE
 *      （Anderson, 1982, Stoch. Proc. Appl. 12(3), 313–326;
 *        Song et al., ICLR 2021, "Score-Based Generative Modeling through SDEs"）
 *
 *          dXₜ = [f(Xₜ) − σ² sₜ(Xₜ)] dt + σ dW̄ₜ     (t が T→0 へ進む)
 *
 *      を、前進向きの離散化として書き下すと次式になる。
 *
 *          X_{t−Δt} ≈ Xₜ + [−f(Xₜ) + σ² sₜ(Xₜ)]Δt + σ√Δt ξ
 *
 *      ここで sₜ は forward 実行中に粒子集団から推定して保存した score場である。
 *      各粒子の軌跡を逆再生しているのではない、という点がこの教材の要点である。
 *
 * 【単位系と符号の規約】
 *   本デモは無次元化されたモデル単位系で完結している。
 *     - 位置 x : モデル長さ単位 [L]。描画範囲は x ∈ [−6, 6]。
 *     - 時刻 t : モデル時間単位 [T]。
 *     - σ      : ノイズ強度 [L·T^(−1/2)]。Wiener増分が √Δt に比例するための次元。
 *     - k      : drift強度 [T⁻¹]。f(x) = −kx が原点へ引き戻す向きを正とする。
 *   Canvas座標は右向きが +x、下向きが +y（HTML Canvas の標準）。
 *   モデル座標 → Canvas座標の変換は modelXToCanvasX() に集約している。
 * =============================================================================
 */

(() => {
  "use strict";

  // ===========================================================================
  // 定数定義
  //
  // 計算ロジック中にマジックナンバーを置かないため、物理的意味と単位を持つ量は
  // すべてここに集約する。
  // ===========================================================================

  /** 時間積分・履歴記録に関する設定 */
  const SIMULATION = Object.freeze({
    /** 前進積分の時間刻み Δt [モデル時間単位] */
    timeStep: 0.012,
    /**
     * score場スナップショットを記録する間隔 [前進ステップ数]。
     * reverse の1ステップはこの間隔ぶんの時間を一気に戻すため、
     * reverse側の時間刻みもこの値から導出する（両者は必ず一致する必要がある）。
     */
    stepsPerSnapshot: 4,
    /** Reverse を許可するために最低限必要なスナップショット数 */
    minSnapshotsToReverse: 8,
    /** Reverse を許可するために最低限必要な経過時刻 [モデル時間単位] */
    minTimeToReverse: 0.18,
    /**
     * reverse時に score へ課す絶対値の上限 [L⁻¹]。
     * 分布の裾では推定密度がゼロに近づき score が発散するため、
     * 数値的な吹き飛びを防ぐクリップを入れる。
     */
    maxAbsScore: 9.0,
    /**
     * drift が「有効」とみなす閾値 [T⁻¹]。
     * スライダーが浮動小数のため、厳密なゼロ比較を避ける。
     */
    driftActiveThreshold: 0.001,
  });

  /** 空間離散化（描画範囲とビン分割） */
  const DOMAIN = Object.freeze({
    /** 表示・推定を行う左端 [L] */
    xMin: -6,
    /** 表示・推定を行う右端 [L] */
    xMax: 6,
    /** [xMin, xMax] を分割するグリッド点数。奇数にして中央を x=0 に合わせる。 */
    binCount: 181,
  });

  /** 密度推定（ビン化 + Gaussian平滑化）と score 推定の設定 */
  const DENSITY_ESTIMATION = Object.freeze({
    /** 畳み込みカーネルの打ち切り半径 [ビン数] */
    kernelRadiusInBins: 7,
    /** Gaussianカーネルの標準偏差 [ビン数]。KDEのバンド幅に相当する。 */
    kernelBandwidthInBins: 2.8,
    /**
     * log を取る前に密度へ加える下駄 [L⁻¹]。
     * 粒子が存在しないビンで log(0) = −∞ となるのを防ぐ。
     */
    densityFloorForLog: 1e-5,
    /**
     * 密度カーブを描画するときの縦軸スケールの下限 [L⁻¹]。
     * 分布が平坦なときにノイズが画面いっぱいに拡大されるのを防ぐ。
     */
    minDensityForPlotScale: 0.15,
  });

  /**
   * レッスンごとの初期粒子配置。
   * standardDeviation / modeOffset はいずれもモデル長さ単位 [L]。
   *
   * 二峰配置が2種類あるのは、レッスン4とレッスン5で求めるものが逆だからである。
   * 等重み2峰Gaussian混合が単峰化する条件は「半間隔 modeOffset ≤ 峰の幅」であり、
   * 時刻 t での峰の幅は √(standardDeviation² + σ²t) で与えられる。
   *   - レッスン4(score) は、終了時刻まで二峰が残っていないと
   *     「位置によって score の向きが切り替わる」という主題が成り立たない。
   *   - レッスン5(Reverse) は逆に、forward で確実に単峰まで潰れてくれないと
   *     「score が構造を復元する」という主題が弱くなる。
   */
  const INITIAL_DISTRIBUTIONS = Object.freeze({
    /** ほぼ一点に集中（デルタ関数の近似） */
    point: Object.freeze({ kind: "gaussian", standardDeviation: 0.10 }),
    /** 広く散らばった単峰分布 */
    wide: Object.freeze({ kind: "gaussian", standardDeviation: 2.7 }),
    /**
     * レッスン4用の二峰配置。峰を十分に離し、終了時刻 t=4.2（σ=0.72）でも
     * 半間隔2.25 > 峰の幅1.53 を保って二峰性が残るようにしてある。
     */
    separatedModes: Object.freeze({ kind: "bimodal", modeOffset: 2.25, standardDeviation: 0.42 }),
    /**
     * レッスン5用の二峰配置。終了時刻 t=6.0（σ=0.92）で
     * 半間隔1.95 < 峰の幅2.29 となり、単峰へ潰れきる。
     */
    mergingModes: Object.freeze({ kind: "bimodal", modeOffset: 1.95, standardDeviation: 0.42 }),
  });

  /**
   * Canvas レイアウト（すべて論理座標系のピクセル値）。
   * 論理座標系のサイズは index.html の canvas の width / height 属性を唯一の
   * 情報源とし、下記のパネル配置はその高さに収まるよう調整してある。
   */
  const LAYOUT = Object.freeze({
    /** 縦に並ぶ3パネルの上端と高さ */
    panels: Object.freeze({
      particles: Object.freeze({ top: 18, height: 218 }),
      distribution: Object.freeze({ top: 254, height: 238 }),
      score: Object.freeze({ top: 510, height: 232 }),
    }),
    /** パネル枠の左右マージンと角丸半径 */
    panelInsetX: 18,
    panelCornerRadius: 18,
    /** パネル内テキストの左位置と、上端からのベースライン距離 */
    panelTitleOffsetX: 42,
    panelTitleOffsetY: 38,
    panelSubtitleOffsetY: 61,
    /** プロット領域の左端と、論理幅から差し引く総マージン */
    plotLeft: 78,
    plotWidthInset: 120,
    /** パネル内キャプションのベースライン距離（上端から） */
    captionOffsetY: 35,
    particleCaptionOffsetY: 36,
    /** 各パネルの横軸位置（パネル下端からの距離） */
    particleAxisOffsetFromBottom: 44,
    densityAxisOffsetFromBottom: 38,
    scoreAxisOffsetFromBottom: 58,
    /** 軸目盛の描画 */
    axisTickValues: Object.freeze([-6, -4, -2, 0, 2, 4, 6]),
    axisTickHalfLength: 4,
    axisLabelOffsetY: 18,
    axisLabelOffsetX: -5,
    /** 粒子ドットの描画 */
    particleDotBaseOffset: 10,
    particleJitterSpan: 88,
    particleDotRadiusSmall: 2.2,
    particleDotRadiusLarge: 2.8,
    /** ドット半径を小さくする粒子数の閾値 */
    particleCountForSmallDots: 900,
    /** 描画範囲外の粒子を捨てる際の余白 [px] */
    particleCullMargin: 3,
    /** drift場を示す矢印 */
    driftArrowRowOffset: 58,
    driftArrowLength: 28,
    driftArrowHeadSize: 6,
    driftArrowPositions: Object.freeze([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]),
    driftLabelOffsetFromRight: 122,
    /** 密度カーブの縦方向の余白（パネル高さから差し引く） */
    densityVerticalInset: 100,
    /** score矢印 */
    scoreArrowCount: 25,
    scoreArrowRowOffset: 52,
    scoreArrowHeadSize: 6,
    scoreArrowMinLength: 7,
    scoreArrowMaxLength: 42,
    scoreArrowLengthPerUnit: 11,
    /** Reverse中に表示するバッジ */
    reverseBadgeWidth: 190,
    reverseBadgeHeight: 42,
    reverseBadgeOffsetFromRight: 210,
    reverseBadgeOffsetY: 16,
    reverseBadgeCornerRadius: 10,
    reverseBadgeTextOffsetX: 196,
    reverseBadgeTextOffsetY: 42,
  });

  /** 配色。Canvas は CSS 変数を参照できないため、ここで一元管理する。 */
  const PALETTE = Object.freeze({
    canvasBackground: "#f8fbff",
    panelFill: "#ffffff",
    panelBorder: "#e2e8f0",
    panelSubtitle: "#7c8798",
    placeholderText: "#98a2b3",
    axisLine: "#cbd5e1",
    axisLabel: "#98a2b3",
    particle: "rgba(37, 99, 235, .58)",
    particleTitle: "#2563eb",
    particleCaption: "#475467",
    driftArrowStroke: "rgba(234, 88, 12, .55)",
    driftArrowFill: "rgba(234, 88, 12, .65)",
    driftLabel: "#c2410c",
    densityTitle: "#0891b2",
    densityStroke: "#0891b2",
    densityCaption: "#0e7490",
    densityFillTop: "rgba(8, 145, 178, .24)",
    densityFillBottom: "rgba(8, 145, 178, .04)",
    scoreTitle: "#7c3aed",
    scoreCaption: "#6d28d9",
    /** score矢印は score の大きさで不透明度を変えるため RGB 成分だけ保持する */
    scoreArrowRgb: "124, 58, 237",
    reverseBadgeFill: "rgba(124, 58, 237, .08)",
  });

  /** score矢印の不透明度（|score| に応じて濃くする） */
  const SCORE_ARROW_OPACITY = Object.freeze({
    base: 0.35,
    maxBonus: 0.55,
    perUnit: 0.10,
  });

  /** Canvas 上のフォント指定 */
  const TYPOGRAPHY = Object.freeze({
    panelTitle: "700 18px system-ui, sans-serif",
    panelSubtitle: "13px system-ui, sans-serif",
    caption: "12px system-ui, sans-serif",
    placeholder: "600 14px system-ui, sans-serif",
    badge: "700 12px system-ui, sans-serif",
  });

  /** 線幅 */
  const LINE_WIDTH = Object.freeze({
    panelBorder: 1.2,
    axis: 1.2,
    driftArrow: 2,
    densityCurve: 3,
    scoreArrow: 2.1,
  });

  /**
   * 粒子ドットの縦位置に与える決定論的なばらつき。
   * 乱数を使わないのは、フレームごとにドットが上下にちらつくのを避けるため。
   * 互いに素な2数を使うことで、連番の添字が縦方向に均等に散る。
   */
  const PARTICLE_JITTER = Object.freeze({ multiplier: 47, modulus: 71 });

  /** アニメーションループの最小フレーム間隔 [ms]（およそ60fpsに制限する） */
  const MIN_FRAME_INTERVAL_MS = 16;

  // ===========================================================================
  // レッスン定義
  //
  // 各レッスンが drift / noise の初期値、初期配置、終了時刻、表示フラグを持つ。
  // ここがスライダー初期値の唯一の情報源であり、起動時にも applyLesson() 経由で
  // 適用される。
  //
  // maxTime をレッスンごとに持たせているのは、レッスンによって「どこまで拡散
  // させたいか」が異なるためである。レッスン4は二峰性を保ったまま終わらせたい
  // ので短く、レッスン5は単峰まで潰しきりたいので長くとる。
  // ===========================================================================

  const lessons = Object.freeze([
    {
      short: "Brown運動",
      subtitle: "random walk",
      text: "<strong>1粒子の運動</strong>から始めます。driftはなく、各粒子は独立なノイズだけを受けます。粒子数を増やしても、SDE自体は1粒子ごとの式です。",
      idea: "各粒子は独立なBrown運動をします。",
      driftStrength: 0,
      noiseAmplitude: 0.85,
      initialDistribution: "point",
      maxTime: 4.2,
      showDensity: false,
      showScore: false,
      showDriftField: false,
      allowReverse: false,
    },
    {
      short: "Langevin",
      subtitle: "drift + noise",
      text: "ノイズに<strong>決定論的なdrift</strong>を加えます。ここでは f(x)=−kx。ランダムに揺れながら、粒子は原点へ戻されます。",
      idea: "Langevin方程式 = driftによる系統的な運動 + Brownノイズ。",
      driftStrength: 0.65,
      noiseAmplitude: 0.7,
      initialDistribution: "wide",
      maxTime: 4.2,
      showDensity: false,
      showScore: false,
      showDriftField: true,
      allowReverse: false,
    },
    {
      short: "確率分布",
      subtitle: "ensemble → pₜ(x)",
      text: "同じSDEを<strong>多数の粒子</strong>に適用します。下段の曲線は粒子から推定した経験分布 pₜ(x)。1粒子では見えない「集団の形」が現れます。",
      idea: "SDEは粒子を動かす。多数の粒子の集団が pₜ(x) を作る。",
      driftStrength: 0,
      noiseAmplitude: 0.85,
      initialDistribution: "point",
      maxTime: 4.2,
      showDensity: true,
      showScore: false,
      showDriftField: false,
      allowReverse: false,
    },
    {
      short: "score",
      subtitle: "∂ₓ log pₜ(x)",
      text: "scoreは<strong>粒子の履歴ではなく、現在の分布</strong>から決まります。矢印は密度が高くなる向き。二峰分布なら、場所によって向きが切り替わります。",
      idea: "同じ位置 x にいる粒子には、同じ score sₜ(x) が作用します。",
      driftStrength: 0,
      noiseAmplitude: 0.72,
      initialDistribution: "separatedModes",
      maxTime: 4.2,
      showDensity: true,
      showScore: true,
      showDriftField: false,
      allowReverse: false,
    },
    {
      short: "Reverse",
      subtitle: "scoreで戻す",
      text: "まずForwardで拡散させ、二峰構造が完全に潰れるまで進めます。止まったら<strong>「⇠ score で戻す」</strong>を押すと、<strong>forward時に集団から得たscore</strong>を使って粒子を高密度側へ戻します。軌跡の逆再生ではありません。",
      idea: "Reverse diffusion は『元の粒子を覚える』のではなく、時刻ごとの score を使う。",
      driftStrength: 0,
      // σ と maxTime を他レッスンより大きくしてあるのは、forward で二峰構造を
      // 確実に潰しきるため。ここが単峰まで崩れていないと、reverse が構造を
      // 復元して見せる意味が薄れる。
      noiseAmplitude: 0.92,
      initialDistribution: "mergingModes",
      maxTime: 6.0,
      showDensity: true,
      showScore: true,
      showDriftField: false,
      allowReverse: true,
    },
  ]);

  /**
   * score場スナップショット履歴の上限 [スナップショット数]。
   *
   * 固定値にしてはいけない。この上限を超えると recordSnapshot() が最も古い
   * スナップショットから捨てていくため、reverse が初期分布まで戻れなくなる。
   * しかも reverseStep() は historyCursor が 0 に達した時点で state.time = 0 と
   * 表示だけ書き換えて打ち切るので、UI上は「t=0 まで戻った」ように見えたまま
   * 分布が復元されないという分かりにくい壊れ方をする。
   *
   * そこで、最も長いレッスンが必要とする枚数から導出する。+2 は初期記録
   * （resetSimulation 時の force 記録）と端数ステップぶんの余裕。
   */
  const MAX_SNAPSHOTS = (() => {
    const snapshotInterval = SIMULATION.timeStep * SIMULATION.stepsPerSnapshot;
    const longestMaxTime = Math.max(...lessons.map((lesson) => lesson.maxTime));
    return Math.ceil(longestMaxTime / snapshotInterval) + 2;
  })();

  // ===========================================================================
  // DOM参照とCanvasの初期化
  // ===========================================================================

  const canvas = document.getElementById("simCanvas");
  const ctx = canvas.getContext("2d");

  /** 描画コードが使う論理座標系のサイズ。HTML の属性値を情報源とする。 */
  const logicalWidth = canvas.width;
  const logicalHeight = canvas.height;

  const ui = {
    primary: document.getElementById("primaryBtn"),
    reset: document.getElementById("resetBtn"),
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
    keyIdea: document.getElementById("keyIdea"),
  };

  /**
   * 高DPIディスプレイでの描画を鮮明にする。
   * バッキングストアを devicePixelRatio 倍に拡大したうえで座標系を同じ倍率で
   * スケールするため、以降の描画コードは論理座標のまま書ける。
   * setTransform は絶対指定なので、繰り返し呼んでも倍率は累積しない。
   */
  function configureCanvasForDisplayDensity() {
    const pixelRatio = window.devicePixelRatio || 1;
    canvas.width = Math.round(logicalWidth * pixelRatio);
    canvas.height = Math.round(logicalHeight * pixelRatio);
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  }

  // ===========================================================================
  // シミュレーション状態
  // ===========================================================================

  const state = {
    /** 現在のレッスン添字 */
    lessonIndex: 0,
    /** 各粒子の位置 Xₜ⁽ⁱ⁾ [L] */
    particles: [],
    /** アニメーションが進行中か */
    running: false,
    /** "forward" | "reverse" */
    direction: "forward",
    /** 現在時刻 [モデル時間単位] */
    time: 0,
    /** 現在の粒子集団から推定した密度 pₜ(x)（グリッド上の値） */
    density: new Float64Array(DOMAIN.binCount),
    /** 表示中の score場 sₜ(x)（グリッド上の値） */
    score: new Float64Array(DOMAIN.binCount),
    /** forward中に記録した score場スナップショットの列 */
    history: [],
    /** reverse再生で参照中のスナップショット添字 */
    historyCursor: -1,
    /** 直近のスナップショット記録からの経過ステップ数 */
    stepsSinceSnapshot: 0,
    /** Box–Muller法で余った2つ目の標準正規乱数のキャッシュ */
    gaussianSpare: null,
  };

  // ===========================================================================
  // 乱数生成
  // ===========================================================================

  /**
   * 標準正規分布 N(0,1) に従う乱数を返す（Box & Muller, 1958）。
   * 1回の計算で2つの独立な標準正規乱数が得られるため、片方をキャッシュして
   * 次回の呼び出しで返す。
   */
  function standardNormalRandom() {
    if (state.gaussianSpare !== null) {
      const cached = state.gaussianSpare;
      state.gaussianSpare = null;
      return cached;
    }
    // log(0) と偏角の縮退を避けるため、0 は引き直す。
    let radiusSeed = 0;
    let angleSeed = 0;
    while (radiusSeed === 0) radiusSeed = Math.random();
    while (angleSeed === 0) angleSeed = Math.random();

    const magnitude = Math.sqrt(-2 * Math.log(radiusSeed));
    const angle = 2 * Math.PI * angleSeed;
    state.gaussianSpare = magnitude * Math.sin(angle);
    return magnitude * Math.cos(angle);
  }

  // ===========================================================================
  // UI値のアクセサ
  //
  // スライダーの生値を、物理的意味の明確な名前で取り出す層。
  // ===========================================================================

  /** 現在のレッスン定義 */
  function currentLesson() {
    return lessons[state.lessonIndex];
  }

  /** 粒子数 N [個] */
  function particleCount() {
    return Number(ui.particleSlider.value);
  }

  /** ノイズ強度 σ [L·T^(−1/2)] */
  function noiseAmplitude() {
    return Number(ui.noiseSlider.value);
  }

  /** drift強度 k [T⁻¹]（f(x) = −kx） */
  function driftStrength() {
    return Number(ui.driftSlider.value);
  }

  /** 1フレームあたりに進める積分ステップ数 [ステップ/フレーム] */
  function stepsPerFrame() {
    return Number(ui.speedSlider.value);
  }

  /** drift が有効か（浮動小数の厳密ゼロ比較を避ける） */
  function isDriftActive(strength) {
    return strength > SIMULATION.driftActiveThreshold;
  }

  // ===========================================================================
  // 初期化とレッスン切り替え
  // ===========================================================================

  /**
   * 指定モードに従って粒子1個の初期位置 [L] をサンプリングする。
   * bimodal は左右いずれかのモードを等確率で選び、その周りに正規分布させる。
   */
  function sampleInitialPosition(distributionName) {
    const distribution = INITIAL_DISTRIBUTIONS[distributionName];
    if (!distribution) return standardNormalRandom();

    if (distribution.kind === "bimodal") {
      const modeSign = Math.random() < 0.5 ? -1 : 1;
      return modeSign * distribution.modeOffset
        + distribution.standardDeviation * standardNormalRandom();
    }
    return distribution.standardDeviation * standardNormalRandom();
  }

  /** 粒子・時刻・履歴をレッスンの初期状態へ戻す */
  function resetSimulation() {
    state.running = false;
    state.direction = "forward";
    state.time = 0;
    state.history = [];
    state.historyCursor = -1;
    state.stepsSinceSnapshot = 0;
    state.gaussianSpare = null;

    const distributionName = currentLesson().initialDistribution;
    state.particles = Array.from(
      { length: particleCount() },
      () => sampleInitialPosition(distributionName)
    );

    updateDensityAndScore();
    recordSnapshot(true);
    syncUI();
    draw();
  }

  /** レッスンを適用し、スライダー初期値をレッスン定義から復元する */
  function applyLesson(index) {
    state.lessonIndex = index;
    const lesson = currentLesson();
    ui.driftSlider.value = String(lesson.driftStrength);
    ui.noiseSlider.value = String(lesson.noiseAmplitude);
    renderLessonButtons();
    ui.lessonText.innerHTML = lesson.text;
    resetSimulation();
  }

  function renderLessonButtons() {
    ui.lessonButtons.innerHTML = "";
    lessons.forEach((lesson, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "lesson-btn" + (index === state.lessonIndex ? " active" : "");
      button.innerHTML =
        `<small>STEP ${index + 1}</small>${lesson.short}<small>${lesson.subtitle}</small>`;
      button.addEventListener("click", () => applyLesson(index));
      ui.lessonButtons.appendChild(button);
    });
  }

  // ===========================================================================
  // 密度と score の推定
  // ===========================================================================

  /** グリッドの刻み幅 Δx [L] */
  function binWidth() {
    return (DOMAIN.xMax - DOMAIN.xMin) / (DOMAIN.binCount - 1);
  }

  /**
   * 粒子を線形補間で隣接2ビンへ配分し、重み付きヒストグラムを作る。
   *
   * 注意: 描画範囲 [xMin, xMax] の外へ出た粒子は、どちらのビンにも加算されず
   * 単純に捨てられる。一方で正規化は全粒子数 N で行うため、粒子が範囲外へ
   * 流出すると ∫p dx < 1 となる。これは意図した挙動であり、「確率質量が視野の
   * 外へ拡散していく」ことを裾の減衰として可視化するためである。
   */
  function accumulateParticleHistogram(particles, dx) {
    const counts = new Float64Array(DOMAIN.binCount);
    for (let i = 0; i < particles.length; i++) {
      const gridPosition = (particles[i] - DOMAIN.xMin) / dx;
      const lowerBin = Math.floor(gridPosition);
      const upperWeight = gridPosition - lowerBin;
      if (lowerBin >= 0 && lowerBin < DOMAIN.binCount) {
        counts[lowerBin] += 1 - upperWeight;
      }
      if (lowerBin + 1 >= 0 && lowerBin + 1 < DOMAIN.binCount) {
        counts[lowerBin + 1] += upperWeight;
      }
    }
    return counts;
  }

  /** 打ち切り半径つき離散Gaussianカーネルを作る（重みと総和を返す） */
  function buildGaussianKernel(radiusInBins, bandwidthInBins) {
    const weights = new Float64Array(2 * radiusInBins + 1);
    let total = 0;
    for (let offset = -radiusInBins; offset <= radiusInBins; offset++) {
      const weight = Math.exp(-0.5 * (offset / bandwidthInBins) ** 2);
      weights[offset + radiusInBins] = weight;
      total += weight;
    }
    return { weights, total };
  }

  /**
   * ヒストグラムにGaussianカーネルを畳み込み、確率密度 [L⁻¹] に正規化する。
   * 正規化はカーネル総和・粒子数・ビン幅の3つで割ることで行う。
   */
  function smoothHistogramToDensity(counts, kernel, radiusInBins, particleTotal, dx) {
    const density = new Float64Array(DOMAIN.binCount);
    // 粒子数0でのゼロ除算を避ける
    const effectiveParticleCount = Math.max(1, particleTotal);

    for (let bin = 0; bin < DOMAIN.binCount; bin++) {
      let weightedSum = 0;
      for (let offset = -radiusInBins; offset <= radiusInBins; offset++) {
        const neighbor = bin + offset;
        if (neighbor >= 0 && neighbor < DOMAIN.binCount) {
          weightedSum += counts[neighbor] * kernel.weights[offset + radiusInBins];
        }
      }
      // 3つの因子を先に掛けてから1回で割る形にしないのは、IEEE754では
      // a/b/c/d と a/(b*c*d) の最下位ビットが一致せず、score推定と
      // 矢印の濃淡にごく僅かな差が出るためである（既存の描画結果との一致を優先）。
      density[bin] = weightedSum / kernel.total / effectiveParticleCount / dx;
    }
    return density;
  }

  /**
   * score sₜ(x) = ∂ₓ log pₜ(x) を log密度の中心差分で推定する。
   * 端点は片側差分を避け、隣接点の値をコピーして勾配を平坦に扱う。
   */
  function estimateScoreFromDensity(density, dx) {
    const score = new Float64Array(DOMAIN.binCount);
    const floor = DENSITY_ESTIMATION.densityFloorForLog;
    for (let bin = 1; bin < DOMAIN.binCount - 1; bin++) {
      const logLeft = Math.log(density[bin - 1] + floor);
      const logRight = Math.log(density[bin + 1] + floor);
      score[bin] = (logRight - logLeft) / (2 * dx);
    }
    score[0] = score[1];
    score[DOMAIN.binCount - 1] = score[DOMAIN.binCount - 2];
    return score;
  }

  /** 現在の粒子集団から state.density と state.score を更新する */
  function updateDensityAndScore() {
    const dx = binWidth();
    const radius = DENSITY_ESTIMATION.kernelRadiusInBins;
    const counts = accumulateParticleHistogram(state.particles, dx);
    const kernel = buildGaussianKernel(radius, DENSITY_ESTIMATION.kernelBandwidthInBins);

    state.density = smoothHistogramToDensity(
      counts, kernel, radius, state.particles.length, dx
    );
    state.score = estimateScoreFromDensity(state.density, dx);
  }

  /**
   * グリッド上の値を任意の位置 x [L] で線形補間する。
   * 範囲外は端点の値で外挿（クランプ）する。
   */
  function interpolateOnGrid(grid, x) {
    const lastIndex = grid.length - 1;
    const gridPosition = (x - DOMAIN.xMin) / (DOMAIN.xMax - DOMAIN.xMin) * lastIndex;
    const lowerIndex = Math.floor(gridPosition);
    if (lowerIndex < 0) return grid[0];
    if (lowerIndex >= lastIndex) return grid[lastIndex];
    const upperWeight = gridPosition - lowerIndex;
    return grid[lowerIndex] * (1 - upperWeight) + grid[lowerIndex + 1] * upperWeight;
  }

  // ===========================================================================
  // score場スナップショットの記録
  // ===========================================================================

  /**
   * 現在時刻の score場を履歴へ記録する。
   *
   * σ と k も一緒に保存するのが重要である。reverse-time SDE は forward を
   * 生成したときと同じ係数を使わなければ逆過程にならないため、reverse中に
   * スライダーが動かされても、記録時点の値を使い続ける必要がある。
   *
   * @param {boolean} force 記録間隔を待たずに即座に記録するか
   */
  function recordSnapshot(force = false) {
    if (!force && state.stepsSinceSnapshot < SIMULATION.stepsPerSnapshot) return;

    updateDensityAndScore();
    state.history.push({
      time: state.time,
      score: Float32Array.from(state.score),
      noiseAmplitude: noiseAmplitude(),
      driftStrength: driftStrength(),
    });
    state.stepsSinceSnapshot = 0;

    if (state.history.length > MAX_SNAPSHOTS) state.history.shift();
    state.historyCursor = state.history.length - 1;
  }

  // ===========================================================================
  // 時間発展
  // ===========================================================================

  /**
   * Euler–Maruyama法による前進1ステップ。
   *     X_{t+Δt} = Xₜ + f(Xₜ)Δt + σ√Δt ξ,  f(x) = −kx
   */
  function forwardStep() {
    const dt = SIMULATION.timeStep;
    const sigma = noiseAmplitude();
    const k = driftStrength();
    const noiseScale = sigma * Math.sqrt(dt);

    for (let i = 0; i < state.particles.length; i++) {
      const x = state.particles[i];
      const drift = -k * x;
      state.particles[i] = x + drift * dt + noiseScale * standardNormalRandom();
    }

    // 終了時刻はレッスンごとに異なる。到達時は端数を切って厳密に maxTime へ揃える。
    const maxTime = currentLesson().maxTime;
    state.time = Math.min(maxTime, state.time + dt);
    state.stepsSinceSnapshot++;
    recordSnapshot(false);
    if (state.time >= maxTime) state.running = false;
  }

  /**
   * reverse-time SDE による逆向き1ステップ。
   *     X_{t−Δt} ≈ Xₜ + [−f(Xₜ) + σ² sₜ(Xₜ)]Δt + σ√Δt ξ
   * f(x) = −kx なので −f(x) = +kx となる。
   *
   * 時間刻みはスナップショット間隔ぶん（stepsPerSnapshot × timeStep）であり、
   * reverse の1ステップが記録済み score場1枚ぶんの時間に対応する。
   * σ と k は「現在のスライダー値」ではなく「そのscore場を記録したときの値」を使う。
   */
  function reverseStep() {
    if (state.historyCursor <= 0) {
      state.time = 0;
      state.running = false;
      return;
    }

    const dt = SIMULATION.timeStep * SIMULATION.stepsPerSnapshot;
    const snapshot = state.history[state.historyCursor];
    const sigma = snapshot.noiseAmplitude;
    const k = snapshot.driftStrength;
    const noiseScale = sigma * Math.sqrt(dt);
    const scoreLimit = SIMULATION.maxAbsScore;

    for (let i = 0; i < state.particles.length; i++) {
      const x = state.particles[i];
      const rawScore = interpolateOnGrid(snapshot.score, x);
      const score = Math.max(-scoreLimit, Math.min(scoreLimit, rawScore));
      const backwardDrift = k * x + sigma * sigma * score;
      state.particles[i] = x + backwardDrift * dt + noiseScale * standardNormalRandom();
    }

    state.historyCursor--;
    state.time = state.history[state.historyCursor].time;
    // 分布は「reverseで戻ってきた現在の粒子」から描き、矢印は forward時に
    // 保存した score場から描く。両者を並べることで、scoreが粒子を高密度側へ
    // 引き戻している様子が見える。
    updateDensityAndScore();
    state.score = state.history[state.historyCursor].score;
  }

  /** 1フレームぶんの時間発展を進める */
  function advanceSimulation() {
    const iterations = stepsPerFrame();
    for (let i = 0; i < iterations; i++) {
      if (!state.running) break;
      if (state.direction === "forward") forwardStep();
      else reverseStep();
    }
    // forward中はスナップショット間の中間時刻でも表示を追従させる。
    // reverse は reverseStep() 内で更新済みのため再計算しない。
    if (state.direction === "forward") updateDensityAndScore();
  }

  /** Reverse を開始できる状態か（履歴が十分に溜まっているか） */
  function canStartReverse() {
    return state.history.length >= SIMULATION.minSnapshotsToReverse
      && state.time >= SIMULATION.minTimeToReverse;
  }

  /** forward を reverse へ切り替え、そのまま戻り始める */
  function startReverse() {
    if (!currentLesson().allowReverse) return;

    // 現在時刻に対応する score場を確実に持っておく。
    recordSnapshot(true);
    if (!canStartReverse()) return;

    state.direction = "reverse";
    state.historyCursor = state.history.length - 1;
    // forward は maxTime 到達時に自動停止しているため、ここで再生を復帰させる。
    // そうしないと「戻す」を押したのに何も起きない、という状態になる。
    state.running = true;
  }

  // ===========================================================================
  // 主ボタンの状態機械
  //
  // レッスンは Brown運動 → … → reverse という一方向の流れを持つため、
  // 「次に何をすべきか」はシミュレーション状態から一意に決まる。そこで操作を
  // 1つのボタンに集約し、常時 disabled のボタンを並べないようにしている。
  // ===========================================================================

  /**
   * 主ボタンが今どの意味を持つかを解決する。
   * @returns {{kind: "play"|"pause"|"reverse"|"nextLesson"|"restart", label: string}}
   */
  function resolvePrimaryAction() {
    const lesson = currentLesson();
    if (state.running) return { kind: "pause", label: "❚❚ 一時停止" };

    // reverse を一時停止した場合は続きから再開する。t=0 まで戻り切ったら終了。
    if (state.direction === "reverse") {
      if (state.time <= 0) return { kind: "restart", label: "↺ もう一度" };
      return { kind: "play", label: "▶ 再生" };
    }

    // forward が終了時刻まで到達したときにだけ、次の一手を提示する。
    if (state.time >= lesson.maxTime) {
      if (lesson.allowReverse) return { kind: "reverse", label: "⇠ score で戻す" };
      const nextLessonIndex = state.lessonIndex + 1;
      if (nextLessonIndex < lessons.length) {
        return { kind: "nextLesson", label: `STEP ${nextLessonIndex + 1} へ →` };
      }
      return { kind: "restart", label: "↺ もう一度" };
    }

    return { kind: "play", label: "▶ 再生" };
  }

  /** 解決済みの action に従って主ボタンの動作を実行する */
  function runPrimaryAction() {
    const action = resolvePrimaryAction();
    switch (action.kind) {
      case "pause": state.running = false; break;
      case "play": state.running = true; break;
      case "reverse": startReverse(); break;
      case "nextLesson": applyLesson(state.lessonIndex + 1); return;  // 内部でsyncUI済み
      case "restart": resetSimulation(); return;                      // 内部でsyncUI済み
    }
    syncUI();
  }

  // ===========================================================================
  // UI同期
  // ===========================================================================

  function syncUI() {
    const lesson = currentLesson();
    const isForward = state.direction === "forward";
    const primaryAction = resolvePrimaryAction();

    ui.primary.textContent = primaryAction.label;
    // 「score で戻す」だけは score パネルと同じ violet 系に切り替え、
    // 押した結果どのビューの話になるのかを色で示す。
    ui.primary.classList.toggle("accent", primaryAction.kind === "reverse");
    ui.primary.classList.toggle("primary", primaryAction.kind !== "reverse");

    ui.time.textContent = state.time.toFixed(2);
    ui.particleValue.textContent = particleCount();
    ui.noiseValue.textContent = noiseAmplitude().toFixed(2);
    ui.driftValue.textContent = driftStrength().toFixed(2);
    ui.speedValue.textContent = `${stepsPerFrame()}×`;

    ui.direction.textContent = isForward ? "Forward" : "Reverse";
    ui.direction.classList.toggle("reverse", !isForward);

    ui.keyIdea.textContent = lesson.idea;
    ui.sdeEquation.textContent = isDriftActive(driftStrength())
      ? "dXₜ = −kXₜ dt + σ dWₜ"
      : "dXₜ = σ dWₜ";
  }

  // ===========================================================================
  // 描画 — 共通部品
  // ===========================================================================

  /** プロット領域の水平方向のジオメトリ */
  function plotGeometry() {
    return { left: LAYOUT.plotLeft, width: logicalWidth - LAYOUT.plotWidthInset };
  }

  /** モデル座標 x [L] を Canvas の水平座標 [px] へ変換する */
  function modelXToCanvasX(x, geometry) {
    const normalized = (x - DOMAIN.xMin) / (DOMAIN.xMax - DOMAIN.xMin);
    return geometry.left + normalized * geometry.width;
  }

  function roundedRectPath(x, y, width, height, radius) {
    const clamped = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + clamped, y);
    ctx.arcTo(x + width, y, x + width, y + height, clamped);
    ctx.arcTo(x + width, y + height, x, y + height, clamped);
    ctx.arcTo(x, y + height, x, y, clamped);
    ctx.arcTo(x, y, x + width, y, clamped);
    ctx.closePath();
  }

  /** 始点から終点へ、三角形の矢尻をつけた矢印を描く */
  function drawArrow(fromX, fromY, toX, toY, headSize) {
    const angle = Math.atan2(toY - fromY, toX - fromX);
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(toX, toY);
    ctx.lineTo(toX - headSize * Math.cos(angle - Math.PI / 6),
      toY - headSize * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(toX - headSize * Math.cos(angle + Math.PI / 6),
      toY - headSize * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  }

  /** パネルの枠と見出しを描く */
  function drawPanelFrame(panel, title, subtitle, titleColor) {
    ctx.save();
    ctx.fillStyle = PALETTE.panelFill;
    ctx.strokeStyle = PALETTE.panelBorder;
    ctx.lineWidth = LINE_WIDTH.panelBorder;
    roundedRectPath(
      LAYOUT.panelInsetX, panel.top,
      logicalWidth - 2 * LAYOUT.panelInsetX, panel.height,
      LAYOUT.panelCornerRadius
    );
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = titleColor;
    ctx.font = TYPOGRAPHY.panelTitle;
    ctx.fillText(title, LAYOUT.panelTitleOffsetX, panel.top + LAYOUT.panelTitleOffsetY);
    ctx.fillStyle = PALETTE.panelSubtitle;
    ctx.font = TYPOGRAPHY.panelSubtitle;
    ctx.fillText(subtitle, LAYOUT.panelTitleOffsetX, panel.top + LAYOUT.panelSubtitleOffsetY);
    ctx.restore();
  }

  /** 横軸と目盛を描く */
  function drawHorizontalAxis(axisY, geometry) {
    ctx.strokeStyle = PALETTE.axisLine;
    ctx.lineWidth = LINE_WIDTH.axis;
    ctx.beginPath();
    ctx.moveTo(geometry.left, axisY);
    ctx.lineTo(geometry.left + geometry.width, axisY);
    ctx.stroke();

    ctx.fillStyle = PALETTE.axisLabel;
    ctx.font = TYPOGRAPHY.caption;
    for (const tickValue of LAYOUT.axisTickValues) {
      const tickX = modelXToCanvasX(tickValue, geometry);
      ctx.beginPath();
      ctx.moveTo(tickX, axisY - LAYOUT.axisTickHalfLength);
      ctx.lineTo(tickX, axisY + LAYOUT.axisTickHalfLength);
      ctx.stroke();
      ctx.fillText(String(tickValue),
        tickX + LAYOUT.axisLabelOffsetX, axisY + LAYOUT.axisLabelOffsetY);
    }
  }

  /** そのレッスンではまだ扱わない要素の代わりに案内文を描く */
  function drawPlaceholder(message, panel, geometry) {
    ctx.save();
    ctx.fillStyle = PALETTE.placeholderText;
    ctx.font = TYPOGRAPHY.placeholder;
    ctx.fillText(message, geometry.left, panel.top + panel.height / 2);
    ctx.restore();
  }

  // ===========================================================================
  // 描画 — パネル1: 粒子
  // ===========================================================================

  /** drift場 f(x) = −kx が原点へ向かうことを矢印で示す */
  function drawDriftFieldArrows(axisY, panel, geometry) {
    ctx.save();
    ctx.strokeStyle = PALETTE.driftArrowStroke;
    ctx.fillStyle = PALETTE.driftArrowFill;
    ctx.lineWidth = LINE_WIDTH.driftArrow;
    const arrowY = axisY - LAYOUT.driftArrowRowOffset;

    for (const position of LAYOUT.driftArrowPositions) {
      const startX = modelXToCanvasX(position, geometry);
      // 原点へ引き戻される向き。x>0 なら左向き、x<0 なら右向き。
      const direction = position > 0 ? -1 : 1;
      drawArrow(startX, arrowY, startX + direction * LAYOUT.driftArrowLength, arrowY,
        LAYOUT.driftArrowHeadSize);
    }

    ctx.fillStyle = PALETTE.driftLabel;
    ctx.font = TYPOGRAPHY.caption;
    ctx.fillText("drift f(x)=−kx",
      geometry.left + geometry.width - LAYOUT.driftLabelOffsetFromRight,
      panel.top + LAYOUT.captionOffsetY);
    ctx.restore();
  }

  /** 各粒子を1つの点として描く */
  function drawParticleDots(axisY, geometry) {
    const total = state.particles.length;
    const radius = total > LAYOUT.particleCountForSmallDots
      ? LAYOUT.particleDotRadiusSmall
      : LAYOUT.particleDotRadiusLarge;
    const leftBound = geometry.left - LAYOUT.particleCullMargin;
    const rightBound = geometry.left + geometry.width + LAYOUT.particleCullMargin;

    ctx.fillStyle = PALETTE.particle;
    for (let i = 0; i < total; i++) {
      const dotX = modelXToCanvasX(state.particles[i], geometry);
      if (dotX < leftBound || dotX > rightBound) continue;
      const jitter = ((i * PARTICLE_JITTER.multiplier) % PARTICLE_JITTER.modulus)
        / PARTICLE_JITTER.modulus;
      const dotY = axisY - LAYOUT.particleDotBaseOffset - jitter * LAYOUT.particleJitterSpan;
      ctx.beginPath();
      ctx.arc(dotX, dotY, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function drawParticlePanel(panel) {
    const geometry = plotGeometry();
    const axisY = panel.top + panel.height - LAYOUT.particleAxisOffsetFromBottom;
    drawHorizontalAxis(axisY, geometry);

    if (currentLesson().showDriftField && isDriftActive(driftStrength())) {
      drawDriftFieldArrows(axisY, panel, geometry);
    }
    drawParticleDots(axisY, geometry);

    ctx.fillStyle = PALETTE.particleCaption;
    ctx.font = TYPOGRAPHY.caption;
    ctx.fillText(`N = ${state.particles.length} particles`,
      geometry.left, panel.top + LAYOUT.particleCaptionOffsetY);
  }

  // ===========================================================================
  // 描画 — パネル2: 確率分布
  // ===========================================================================

  /** 縦軸スケール用の密度の最大値。平坦な分布の過剰な拡大を下限で防ぐ。 */
  function findDensityPlotScale(density) {
    let maximum = DENSITY_ESTIMATION.minDensityForPlotScale;
    for (let i = 0; i < density.length; i++) {
      if (density[i] > maximum) maximum = density[i];
    }
    return maximum;
  }

  /**
   * 密度カーブの描画に必要な幾何量をひとまとめにする。
   * 塗りと線描で同じカーブを2度なぞるため、スケールの再計算を避ける狙いもある。
   */
  function buildDensityCurveSpec(panel, geometry, baselineY) {
    const density = state.density;
    return {
      density,
      geometry,
      baselineY,
      plotScale: findDensityPlotScale(density),
      amplitude: panel.height - LAYOUT.densityVerticalInset,
    };
  }

  /** 密度カーブに沿ったパスを開始する（塗りと線描の両方から使う） */
  function traceDensityCurve(spec) {
    const lastIndex = spec.density.length - 1;
    ctx.beginPath();
    for (let i = 0; i <= lastIndex; i++) {
      const curveX = spec.geometry.left + (i / lastIndex) * spec.geometry.width;
      const curveY = spec.baselineY - (spec.density[i] / spec.plotScale) * spec.amplitude;
      if (i === 0) ctx.moveTo(curveX, curveY);
      else ctx.lineTo(curveX, curveY);
    }
  }

  /** カーブを右下→左下と結んで閉じ、上ほど濃いグラデーションで塗る */
  function fillUnderDensityCurve(spec, panelTop) {
    traceDensityCurve(spec);
    ctx.lineTo(spec.geometry.left + spec.geometry.width, spec.baselineY);
    ctx.lineTo(spec.geometry.left, spec.baselineY);
    ctx.closePath();

    const gradient = ctx.createLinearGradient(0, panelTop, 0, spec.baselineY);
    gradient.addColorStop(0, PALETTE.densityFillTop);
    gradient.addColorStop(1, PALETTE.densityFillBottom);
    ctx.fillStyle = gradient;
    ctx.fill();
  }

  /** 塗りの上に密度カーブ本体の線を重ねる */
  function strokeDensityCurve(spec) {
    traceDensityCurve(spec);
    ctx.strokeStyle = PALETTE.densityStroke;
    ctx.lineWidth = LINE_WIDTH.densityCurve;
    ctx.stroke();
  }

  function drawDistributionPanel(panel) {
    const geometry = plotGeometry();
    const baselineY = panel.top + panel.height - LAYOUT.densityAxisOffsetFromBottom;
    drawHorizontalAxis(baselineY, geometry);

    if (!currentLesson().showDensity) {
      drawPlaceholder("このレッスンでは、まず粒子そのものに注目します。", panel, geometry);
      return;
    }

    const spec = buildDensityCurveSpec(panel, geometry, baselineY);
    fillUnderDensityCurve(spec, panel.top);
    strokeDensityCurve(spec);

    ctx.fillStyle = PALETTE.densityCaption;
    ctx.font = TYPOGRAPHY.caption;
    ctx.fillText("粒子から推定した pₜ(x)", geometry.left, panel.top + LAYOUT.captionOffsetY);
  }

  // ===========================================================================
  // 描画 — パネル3: score
  // ===========================================================================

  /** score の大きさに応じて矢印の長さ [px] を決める */
  function scoreArrowLength(score) {
    return Math.min(
      LAYOUT.scoreArrowMaxLength,
      LAYOUT.scoreArrowMinLength + Math.abs(score) * LAYOUT.scoreArrowLengthPerUnit
    );
  }

  /** score の大きさに応じた矢印の色（強いほど濃い） */
  function scoreArrowColor(score) {
    const bonus = Math.min(SCORE_ARROW_OPACITY.maxBonus,
      Math.abs(score) * SCORE_ARROW_OPACITY.perUnit);
    return `rgba(${PALETTE.scoreArrowRgb}, ${SCORE_ARROW_OPACITY.base + bonus})`;
  }

  /** score場を等間隔サンプリングし、密度が高くなる向きの矢印として描く */
  function drawScoreArrows(arrowY, geometry) {
    const count = LAYOUT.scoreArrowCount;
    for (let i = 0; i < count; i++) {
      // ビンの中心でサンプリングするため 0.5 をずらす。
      const modelX = DOMAIN.xMin + (i + 0.5) / count * (DOMAIN.xMax - DOMAIN.xMin);
      const score = interpolateOnGrid(state.score, modelX);
      const length = scoreArrowLength(score);
      const centerX = modelXToCanvasX(modelX, geometry);
      const direction = score >= 0 ? 1 : -1;

      ctx.strokeStyle = scoreArrowColor(score);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = LINE_WIDTH.scoreArrow;
      drawArrow(centerX - direction * length / 2, arrowY,
        centerX + direction * length / 2, arrowY, LAYOUT.scoreArrowHeadSize);
    }
  }

  /** reverse中であることと、保存済みscoreを使っていることを示すバッジ */
  function drawReverseBadge(panel, geometry) {
    const badgeX = geometry.left + geometry.width - LAYOUT.reverseBadgeOffsetFromRight;
    ctx.fillStyle = PALETTE.reverseBadgeFill;
    roundedRectPath(badgeX, panel.top + LAYOUT.reverseBadgeOffsetY,
      LAYOUT.reverseBadgeWidth, LAYOUT.reverseBadgeHeight, LAYOUT.reverseBadgeCornerRadius);
    ctx.fill();

    ctx.fillStyle = PALETTE.scoreCaption;
    ctx.font = TYPOGRAPHY.badge;
    ctx.fillText("Reverse: 保存したscoreを使用中",
      geometry.left + geometry.width - LAYOUT.reverseBadgeTextOffsetX,
      panel.top + LAYOUT.reverseBadgeTextOffsetY);
  }

  function drawScorePanel(panel) {
    const geometry = plotGeometry();
    const axisY = panel.top + panel.height - LAYOUT.scoreAxisOffsetFromBottom;
    drawHorizontalAxis(axisY, geometry);

    if (!currentLesson().showScore) {
      drawPlaceholder("score は STEP 4 で登場します。", panel, geometry);
      return;
    }

    drawScoreArrows(axisY - LAYOUT.scoreArrowRowOffset, geometry);

    ctx.fillStyle = PALETTE.scoreCaption;
    ctx.font = TYPOGRAPHY.caption;
    ctx.fillText("sₜ(x)=∂ₓ log pₜ(x)  —  矢印は高密度方向",
      geometry.left, panel.top + LAYOUT.captionOffsetY);

    if (state.direction === "reverse") drawReverseBadge(panel, geometry);
  }

  // ===========================================================================
  // 描画 — 全体
  // ===========================================================================

  function draw() {
    ctx.clearRect(0, 0, logicalWidth, logicalHeight);
    ctx.fillStyle = PALETTE.canvasBackground;
    ctx.fillRect(0, 0, logicalWidth, logicalHeight);

    const panels = LAYOUT.panels;
    drawPanelFrame(panels.particles,
      "1. Individual particles", "SDE が直接動かしている対象", PALETTE.particleTitle);
    drawPanelFrame(panels.distribution,
      "2. Distribution  pₜ(x)", "粒子集団から現れる経験分布", PALETTE.densityTitle);
    drawPanelFrame(panels.score,
      "3. Score  ∂ₓ log pₜ(x)", "分布が教える「高密度方向」", PALETTE.scoreTitle);

    drawParticlePanel(panels.particles);
    drawDistributionPanel(panels.distribution);
    drawScorePanel(panels.score);
  }

  // ===========================================================================
  // アニメーションループ
  // ===========================================================================

  let lastFrameTimestamp = performance.now();

  function animate(now) {
    if (now - lastFrameTimestamp > MIN_FRAME_INTERVAL_MS) {
      if (state.running) advanceSimulation();
      syncUI();
      draw();
      lastFrameTimestamp = now;
    }
    requestAnimationFrame(animate);
  }

  // ===========================================================================
  // イベント登録と起動
  // ===========================================================================

  ui.primary.addEventListener("click", runPrimaryAction);
  ui.reset.addEventListener("click", resetSimulation);

  // 粒子数の変更は粒子集団の作り直しを伴うため、確定時のみ反映する。
  ui.particleSlider.addEventListener("change", resetSimulation);
  // σ・k・表示速度は次の積分ステップから自然に効くため、表示の更新だけでよい。
  ui.noiseSlider.addEventListener("input", syncUI);
  ui.driftSlider.addEventListener("input", syncUI);
  ui.speedSlider.addEventListener("input", syncUI);

  configureCanvasForDisplayDensity();
  applyLesson(0);
  requestAnimationFrame(animate);
})();
