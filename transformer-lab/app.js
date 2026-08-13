/*
 * =============================================================================
 * Transformer Lab — scaled dot-product self-attention の数値計算と描画
 * =============================================================================
 *
 * 【このファイルの目的】
 *   "I like playing football" という4トークンの文に対して、self-attention を
 *   最初から最後まで実際に数値計算し、その途中経過をすべて画面に出す。
 *   数式の各記号が、どの数字に対応しているかを目で追えるようにするのが狙いである。
 *
 * 【描画の順序 — index.html の一本道に合わせる】
 *   ページは 01 → 09 の一方向に進む構成であり、このファイルもその順に描く。
 *
 *       05  renderProjectionSection 使う行列（X, W^K, W^Q, W^V）を、平面の図と交互に開示する
 *       06  renderQueryStep         STEP1 : x の特徴で W^Q の行を混ぜ、q を1本作る
 *           renderScoreStep         STEP2 : q と4つの k を1枚の平面に重ね、内積を取る
 *           renderSoftmaxStep       STEP3 : softmax の中身（exp と正規化）を分解して見せる
 *           renderValueMixing       STEP4 : 配分で v を混ぜ、最後に z として足し合わせる
 *           renderVectorComparison  RESULT: x と z を並べる
 *       07  renderAttentionMatrix   06 を4語ぶん繰り返した結果として行列を出す
 *           renderReadingList       4行それぞれの読み方（数値から生成する）
 *
 *   STEP2 と STEP3 を別の表に分けているのは、「内積を取る」ことと
 *   「それを足して1になる比率に直す」ことが別の操作だからである。
 *   1つの表に詰めると、softmax が単なる列の変換に見えてしまう。
 *
 *   4×4 の行列を 07 に置くのも同じ理由で、行列は 06 の結果であって前提ではない。
 *
 * 【実装している数値モデル】
 *   Vaswani et al. (2017) "Attention Is All You Need" の式(1)、
 *   scaled dot-product attention をそのまま実装している。
 *
 *       Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
 *
 *   self-attention なので Q, K, V はすべて同じ入力列 X の線形射影である。
 *
 *       Q = X W^Q,   K = X W^K,   V = X W^V
 *
 * 【埋め込みの規約 — 軸に意味を与えていること】
 *   埋め込み X を密なベクトルにし、d_model の4軸に読める意味を与えている。
 *
 *       軸 = [主体, 動作, もの, スポーツ]
 *       x_playing = [0.0, 0.6, 0.2, 0.9]   動作でもあり、スポーツでもある動名詞
 *
 *   これは W の役割を見せるための選択である。W は「入力のどの特徴を、どれだけ拾って
 *   別の空間へ写すか」を決める行列であり、その働きを数字で追うには、拾われる側に
 *   名前が要る。X を one-hot にすると x_t W は「W の第 t 行を引く」表引きに退化し、
 *   特徴を混ぜる操作そのものが画面から消えてしまう。
 *
 *   ただし、本物の埋め込みの軸にこういう意味は無い。意味は特定の軸ではなく多数の軸に
 *   分散して載っており、1軸を取り出しても人間に読める概念には対応しない。
 *   この断りは index.html の 03 に明記している。
 *
 *   意味を持たせているのは d_model の軸だけである。d_k（query/key の平面）と
 *   d_v（value 空間）の軸には意味を置かず、「軸1」「軸2」と番号でだけ呼ぶ。
 *
 * 【行列の向きの規約】
 *   すべての行列は「行 = トークン」で持つ。したがって x_t, q_t, k_t, v_t は
 *   いずれも行ベクトルであり、射影は右から掛ける（q_t = x_t W^Q）。
 *   重み行列 W の形は (入力次元 × 出力次元) で、W[入力軸][出力軸] と添字を取る。
 *
 *   スコア行列 S は S[i][j] = 「i 番目の語（query 側）から見た j 番目の語（key 側）」
 *   と定める。softmax は行方向に取るので、各行の和が 1 になる。
 *   つまり「i 番目の語が、自分の文脈ベクトルを作るために、どの語をどれだけ
 *   参照するか」の配分が第 i 行である。行と列の役割は非対称なので注意する。
 *
 * 【単位について】
 *   物理量ではないため単位はない。意味を持つのは次元（軸の本数）だけなので、
 *   次元は MODEL_DIMENSIONS に、各軸が表す意味は *_AXIS_LABELS に集約している。
 *
 * 【重みの出どころ — 重要】
 *   QUERY_WEIGHTS / KEY_WEIGHTS / VALUE_WEIGHTS は学習で得た値ではない。
 *   「主語は述語を探す」「動詞は目的語を探す」という関係が現れるように人手で
 *   書き込んだ値である。本物のモデルはこれをデータから学習する。
 *   ここで示したいのは値そのものではなく、値がどう流れて文脈ベクトルになるかである。
 *
 *   値を「きれいに」はしていない。key を直交させたり等間隔に並べたりすれば内積が
 *   0 や ±1 になって表は読みやすくなるが、attention はその性質を要求していないし、
 *   学習で得られる本物の key もそうはならない。将来この値を整えたくなっても、
 *   一般の配置のままにしておくこと。
 *
 * 【q / k の平面の軸に名前を付けていない理由】
 *   attention がこの空間から取り出す情報は内積 q·k だけである。すべての q と k を
 *   同じ回転で回しても内積は1つも変わらないので、出力 z も変わらない。つまり座標の
 *   取り方は任意で、意味を持つのはベクトルどうしの相対的な向きと長さだけである。
 *   したがって軸は「軸1」「軸2」と番号でだけ呼ぶ。
 *   この事情は目で見たほうが早いので、d_k = 2 の平面そのものを SVG で描いている
 *   （renderVectorPlane）。q と k を矢印として並べると、表の数値が矢印の座標として、
 *   内積の大小が矢印どうしの向きの近さとして読める。
 *
 * 【index.html との対応】
 *   参照する要素の id は ELEMENT_IDS に集約している。HTML 側で id を変えるときは
 *   ここも合わせて直すこと。描画は DOM（表と div）で行い、Canvas は使わない。
 *   行列と数表が主役であり、数値をテキストとして読ませたいためである。
 *
 *   本文に出る数値はこのファイルが生成する。解説文へ数字を直接書くと、
 *   トグル（√d_k・因果マスク）を切り替えたときに本文だけが古い値のまま残るためである。
 * =============================================================================
 */

'use strict';

/* ===========================================================================
 * 1. モデルの定数
 *    計算ロジックの中には数値を直接書かない。すべてここで名前を付ける。
 * ========================================================================= */

/** 各表現の次元数。d_model は埋め込み、d_k は query/key、d_v は value の次元。 */
const MODEL_DIMENSIONS = {
  embedding: 4, // d_model : 入力埋め込みの次元。意味の特徴を4つ立てている
  key: 2,       // d_k     : query と key が住む空間の次元（両者は同じでなければ内積が取れない）
  value: 4,     // d_v     : value の次元。ここでは比較しやすいよう d_model と同じにしている
};

/** 例文のトークン列。実際のトークナイザは "." も1トークンにするが、ここでは4語に簡略化する。 */
const SENTENCE_TOKENS = ['I', 'like', 'playing', 'football'];

/**
 * 埋め込み空間 d_model の各軸のラベル。
 *
 * ここだけは軸に人間が読める意味を与えている。W が「入力のどの特徴を、どれだけ拾うか」を
 * 決める行列であることを見せるには、拾われる側の特徴に名前が要るためである。
 * 本物の埋め込みの軸にこういう意味は無い（意味は軸ではなく分散して載っている）。
 * その断りは index.html の 03 に書いてある。
 */
const EMBEDDING_AXIS_LABELS = ['主体', '動作', 'もの', 'スポーツ'];

/** query と key が共有する空間 d_k の各軸。両者は同じ空間に住まないと内積が取れない。
 *  この軸に人間が読める意味はないので、番号でだけ呼ぶ。 */
const SCORE_SPACE_AXIS_LABELS = ['軸1', '軸2'];

/** value 空間 d_v の各軸。ここも意味を持たせず番号で呼ぶ。 */
const VALUE_AXIS_LABELS = ['軸1', '軸2', '軸3', '軸4'];

/**
 * 埋め込み X の各行。軸は EMBEDDING_AXIS_LABELS の [主体, 動作, もの, スポーツ]。
 *
 * 密なベクトルにしてあることが重要である。one-hot にすると x_t W が
 * 「W の第 t 行を引く」表引きに退化してしまい、W が入力の成分を混ぜる様子、
 * すなわち特徴抽出そのものが画面から消える。
 *
 * 値は「その語がその特徴をどれだけ持つか」を手で書いたものである。
 * playing が動作 0.6 とスポーツ 0.9 を併せ持つ動名詞であること、
 * football がもの 1.0 でスポーツ 0.3 を帯びること、が下の計算をそのまま動かす。
 */
const TOKEN_EMBEDDINGS = {
  //          主体  動作  もの  スポーツ
  I:        [1.0, 0.0, 0.1, 0.0],
  like:     [0.3, 1.0, 0.1, 0.0],
  playing:  [0.0, 0.6, 0.2, 0.9],
  football: [0.1, 0.1, 1.0, 0.3],
};

/**
 * W^K : 埋め込み → key。行が「埋め込みの軸」、列が「key の平面の軸」である。
 *
 * 各行は「その意味特徴を持つ語を、平面のどちらへどれだけ押すか」を表す。
 * 4つの特徴を平面の4つの違う向きへ割り当ててあるので、特徴の配合が違う語は
 * 平面の違う場所に置かれ、互いに見分けられるようになる。
 *
 *     主体   → 約  27°     もの     → 約 295°
 *     動作   → 約  90°     スポーツ → 約 196°
 *
 * 結果として k_I 20°, k_like 75°, k_playing 170°, k_football 285° に落ちる。
 * 間隔は 55° / 95° / 115° / 95° で等間隔ではなく、どの2本も直交していない
 * （k_I·k_like = 0.76, k_like·k_playing = -0.10, k_playing·k_football = -0.46）。
 * 長さも 1 に揃えていない（0.95〜1.25）。
 * key に効いてくる制約は「2語がほとんど同じ向きにならないこと」だけである。
 */
const KEY_WEIGHTS = [
  [ 0.93,  0.48], // 主体     : 平面の右上へ押す（約  27°）
  [-0.01,  1.18], // 動作     : 平面の真上へ押す（約  90°）
  [ 0.55, -1.17], // もの     : 平面の右下へ押す（約 295°）
  [-1.15, -0.34], // スポーツ : 平面の左へ押す  （約 196°）
];
/**
 * W^Q : 埋め込み → query。行が「埋め込みの軸」、列が「query の平面の軸」である。
 *
 * 各行は「その意味特徴を持つ語は、平面のどの向きの key を探しに行くか」を表す。
 * 語の query は、その語が持つ特徴ぶんだけ各行を足し合わせたものになる。
 *
 *     主体   → 約  62°  ≈ k_like の向き(75°)      （主体らしい語は動作を探す）
 *     動作   → 約 195°  ≈ k_playing の向き(170°)  （動作らしい語は目的語を探す）
 *     もの   → 約 174°  ≈ k_playing の向き(170°)  （もの らしい語は動作を探す）
 *     スポーツ → 約 318°  ≈ k_football の向き(285°) （スポーツらしい語は競技名を探す）
 *
 * 各行は狙う key の向きと完全には一致していない。一致していなくても、
 * 足し合わせた q が他の3つより大きい内積を返せば配分は最大になる。
 *
 * playing の query が k_football を向くのは、この足し合わせの結果である。
 * 動作 0.6 ぶんの左向きと、スポーツ 0.9 ぶんの右下向きが合成されて下を向く。
 * 特徴の配合が向きを決める、という様子がいちばん見えるのがこの語である。
 *
 * 向きが「どの語の内積が最も大きくなるか」を、長さが「その差をどれだけ広げるか」を決める。
 * 各行の長さは、出来上がる q の長さが 2.4〜2.7 に収まるように取ってある。
 */
const QUERY_WEIGHTS = [
  [ 1.29,  2.38], // 主体     : k_like    の向きを指す
  [-2.57, -0.70], // 動作     : k_playing の向きを指す
  [-3.12,  0.35], // もの     : k_playing の向きを指す
  [ 2.88, -2.57], // スポーツ : k_football の向きを指す
];
/**
 * W^V : 埋め込み → value。行が「埋め込みの軸」、列が「value 空間の軸」である。
 *
 * value に課される幾何的な制約は無い。key と揃える必要も、query と内積が取れる必要もない。
 * ここでは意味特徴を軽く混ぜて別の4次元へ写しているだけである。
 * 重要なのは、出来上がる v_t が x_t とは別のベクトルになること。
 * これが「重みが掛かる先は V であって埋め込み X ではない」ということの中身である。
 */
const VALUE_WEIGHTS = [
  [1.0, 0.2, 0.0, 0.1], // 主体
  [0.1, 1.0, 0.3, 0.0], // 動作
  [0.0, 0.2, 1.0, 0.4], // もの
  [0.2, 0.0, 0.3, 1.0], // スポーツ
];

/**
 * 各語の query が「何を探しているか」を日本語で言い直したもの。
 * QUERY_WEIGHTS をそう書いたことの説明であって、計算には一切使わない。
 * 実際にどの語を最も強く参照したかは毎回 attention 重みから求めるので、
 * 因果マスクを掛けて参照先が変わっても、この文と食い違うことはない。
 */
const QUERY_INTENTS = {
  I:        '述語（主語 → 述語）',
  like:     '目的語（動詞 → 目的語）',
  playing:  '目的語（動詞 → 目的語）',
  football: '述語（目的語 → 述語）',
};

/** 各トークンに割り当てる表示色。CSS の :root と手動で同期させている。 */
const TOKEN_COLORS = ['#2563eb', '#0891b2', '#7c3aed', '#ea580c'];

/** 重み行列の行（意味の特徴）を平面に描くときの色。語ではないので彩色しない。 */
const AXIS_ARROW_COLOR = '#94a3b8';

/** マスクされたスコアに入れる値。softmax の指数が 0 になり、その語は完全に無視される。 */
const MASKED_SCORE = Number.NEGATIVE_INFINITY;

/** 文中で「無視できる」と見なす重みのしきい値。読み方の文から 0 の項を落とすのに使う。 */
const NEGLIGIBLE_WEIGHT = 0.0005;

/** 表示上の丸め桁数。計算には一切使わず、文字列化のときだけ使う。 */
const DISPLAY_DIGITS = {
  vector: 2,
  score: 3,
  weight: 3,
};

/** index.html から参照する要素の id。 */
const ELEMENT_IDS = {
  inputTable: 'inputTable',
  keyPlane: 'keyPlane',
  keyTable: 'keyTable',
  queryPlane: 'queryPlane',
  queryTable: 'queryTable',
  valueTable: 'valueTable',
  walkthrough: 'walkthrough',
  tokenButtons: 'tokenButtons',
  focusHeadline: 'focusHeadline',
  queryStep: 'queryStep',
  scaleToggle: 'scaleToggle',
  scorePlane: 'scorePlane',
  scoreTableBody: 'scoreTableBody',
  scaledScoreHead: 'scaledScoreHead',
  softmaxTableBody: 'softmaxTableBody',
  softmaxTableFoot: 'softmaxTableFoot',
  valueMixing: 'valueMixing',
  vectorCompare: 'vectorCompare',
  resultReading: 'resultReading',
  maskToggle: 'maskToggle',
  attentionMatrix: 'attentionMatrix',
  matrixCaption: 'matrixCaption',
  readingList: 'readingList',
};

/* ===========================================================================
 * 2. 数値ユーティリティ（純関数）
 * ========================================================================= */

/** 2つのベクトルの内積。query と key の噛み合いを測る演算そのもの。 */
function dotProduct(leftVector, rightVector) {
  return leftVector.reduce((sum, value, index) => sum + value * rightVector[index], 0);
}

/** 行ベクトルに重み行列を右から掛ける。weightMatrix[入力軸][出力軸] の向きを前提とする。 */
function multiplyRowVectorByMatrix(rowVector, weightMatrix) {
  const outputDimension = weightMatrix[0].length;
  const projected = new Array(outputDimension).fill(0);
  rowVector.forEach((component, inputAxis) => {
    for (let outputAxis = 0; outputAxis < outputDimension; outputAxis += 1) {
      projected[outputAxis] += component * weightMatrix[inputAxis][outputAxis];
    }
  });
  return projected;
}

/** 要素ごとの和。value を配分どおりに足し込むのに使う。 */
function addVectors(firstVector, secondVector) {
  return firstVector.map((value, index) => value + secondVector[index]);
}

/** ベクトルのスカラー倍。attention 重み a_ij を value v_j に掛けるのに使う。 */
function scaleVector(vector, factor) {
  return vector.map((value) => value * factor);
}

/**
 * softmax の途中経過をすべて返す版。STEP3 で exp と正規化を分けて見せるため、
 * 最終結果だけでなく max と exp の配列も返す。
 * 最大値を引いてから指数を取るのは、指数のオーバーフローを避けるためであり、
 * 引き算しても結果は変わらない（分子分母に同じ定数が掛かるだけ）。
 * マスクされた要素は -∞ なので exp が 0 になり、配分から完全に外れる。
 */
function softmaxWithDetail(scores) {
  const maximumScore = Math.max(...scores);
  const exponentials = scores.map((score) => Math.exp(score - maximumScore));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return {
    maximumScore,
    exponentials,
    total,
    weights: exponentials.map((value) => value / total),
  };
}

/* ===========================================================================
 * 3. attention の計算（純関数）
 * ========================================================================= */

/**
 * 入力行列 X を作る。各行は語の意味特徴を並べた密なベクトルである。
 * 本物のモデルはここへさらに positional encoding を足すが、このページでは足していない。
 * 足さない影響は「語順の情報が入らない」ことで、その意味は解説側（09）で述べている。
 */
function buildInputVectors() {
  return SENTENCE_TOKENS.map((token) => TOKEN_EMBEDDINGS[token].slice());
}

/**
 * 因果マスク（decoder / GPT 型）。i 番目の語が自分より後ろ（j > i）を見るのを禁じる。
 * 「次の語を予測する」訓練で答えを先読みさせないための仕掛けであり、
 * encoder 型（BERT など）では掛けないので文全体を双方向に参照できる。
 */
function applyCausalMask(scoreMatrix) {
  return scoreMatrix.map((row, queryIndex) =>
    row.map((score, keyIndex) => (keyIndex > queryIndex ? MASKED_SCORE : score))
  );
}

/** attention 重み a_i: で value を混ぜ、文脈ベクトル z_i を作る。z_i = Σ_j a_ij v_j。 */
function mixValues(attentionRow, valueVectors) {
  const zeroVector = new Array(MODEL_DIMENSIONS.value).fill(0);
  return attentionRow.reduce(
    (accumulated, weight, keyIndex) => addVectors(accumulated, scaleVector(valueVectors[keyIndex], weight)),
    zeroVector
  );
}

/**
 * self-attention を1回ぶん通す。途中経過をすべて返すのは、画面で追わせるためである。
 * options.useScaling : false にすると √d_k で割らない（softmax が鋭くなるのを見せる）
 * options.useCausalMask : true で因果マスクを掛ける
 */
function computeAttentionPass(inputVectors, options) {
  const queryVectors = inputVectors.map((x) => multiplyRowVectorByMatrix(x, QUERY_WEIGHTS));
  const keyVectors = inputVectors.map((x) => multiplyRowVectorByMatrix(x, KEY_WEIGHTS));
  const valueVectors = inputVectors.map((x) => multiplyRowVectorByMatrix(x, VALUE_WEIGHTS));

  const rawScores = queryVectors.map((query) => keyVectors.map((key) => dotProduct(query, key)));
  const scaleDivisor = options.useScaling ? Math.sqrt(MODEL_DIMENSIONS.key) : 1;
  const scaledScores = rawScores.map((row) => row.map((score) => score / scaleDivisor));
  const maskedScores = options.useCausalMask ? applyCausalMask(scaledScores) : scaledScores;

  const softmaxDetails = maskedScores.map(softmaxWithDetail);
  const attentionWeights = softmaxDetails.map((detail) => detail.weights);
  const contextVectors = attentionWeights.map((row) => mixValues(row, valueVectors));

  return {
    inputVectors, queryVectors, keyVectors, valueVectors,
    rawScores, scaleDivisor, scaledScores, maskedScores,
    softmaxDetails, attentionWeights, contextVectors,
  };
}

/* ===========================================================================
 * 4. 表示用のフォーマットと小さな DOM ヘルパ
 * ========================================================================= */

/** 数値を固定桁で文字列にする。マスクされた -∞ は記号で示す。 */
function formatNumber(value, digits) {
  if (!Number.isFinite(value)) return '−∞';
  return value.toFixed(digits);
}

/** ベクトルを "[1.00, 0.00, 0.50, 0.25]" の形にする。 */
function formatVector(vector, digits) {
  return `[${vector.map((value) => formatNumber(value, digits)).join(', ')}]`;
}

/** 要素を作ってクラスとテキストを設定する。 */
function createElement(tagName, className, textContent) {
  const element = document.createElement(tagName);
  if (className) element.className = className;
  if (textContent !== undefined) element.textContent = textContent;
  return element;
}

/** 表の1行を、セルの中身の配列から作る。第1列だけは行見出し（th）にする。 */
function createTableRow(rowHeadText, cellTexts, options) {
  const settings = options || {};
  const row = createElement('tr', settings.rowClassName);
  const head = createElement('th', settings.headClassName, rowHeadText);
  head.scope = 'row';
  if (settings.tokenColor) head.style.setProperty('--token-color', settings.tokenColor);
  row.appendChild(head);
  cellTexts.forEach((text, columnIndex) => {
    const isLast = columnIndex === cellTexts.length - 1;
    row.appendChild(createElement('td', isLast ? settings.lastCellClassName : undefined, text));
  });
  return row;
}

/** 軸ラベル付きでベクトルを表示する小さなチップ列。数字だけでなく軸の意味も一緒に読ませる。 */
function createVectorChips(vector, axisLabels, highlightMaxValue) {
  const container = createElement('div', 'vector-chips');
  const maximumValue = Math.max(...vector);
  vector.forEach((value, axis) => {
    const chip = createElement('span', 'chip');
    if (highlightMaxValue && value === maximumValue && maximumValue > 0) chip.classList.add('is-peak');
    chip.appendChild(createElement('b', 'chip-label', axisLabels[axis]));
    chip.appendChild(createElement('span', 'chip-value', formatNumber(value, DISPLAY_DIGITS.vector)));
    container.appendChild(chip);
  });
  return container;
}

/** 配列の最大値の位置。どの語が最も強く参照されたかを強調するために使う。 */
function argumentMaximum(values) {
  return values.reduce((bestIndex, value, index) => (value > values[bestIndex] ? index : bestIndex), 0);
}

/**
 * 最大値を取る位置を「すべて」返す。因果マスクを掛けると同率首位が実際に起きるため
 * （例: playing の行は I と playing がともに 0.461）、1つに決め打ちして
 * 「最も強く見ているのは I」と言い切らないようにする。
 * 表示桁で区別できない差は同率として扱う。
 */
function argumentMaxima(values) {
  const maximumValue = Math.max(...values);
  const tolerance = Math.pow(10, -DISPLAY_DIGITS.weight) / 2;
  return values.reduce((indices, value, index) => {
    if (maximumValue - value < tolerance) indices.push(index);
    return indices;
  }, []);
}

/** 「"football"（0.730）」「"I" と "playing"（同率 0.461）」のような語句を作る。 */
function describeTopReferences(weights) {
  const topIndices = argumentMaxima(weights);
  const names = topIndices.map((index) => `"${SENTENCE_TOKENS[index]}"`).join(' と ');
  const amount = formatNumber(weights[topIndices[0]], DISPLAY_DIGITS.weight);
  return `${names}（${topIndices.length > 1 ? '同率 ' : ''}${amount}）`;
}

/**
 * 2つのベクトルの向きの近さ（余弦類似度）。長さの効果を除いた「角度だけ」の指標である。
 * 内積は「向きの近さ × 両者の長さ」なので、key の長さが揃っていない場合、
 * 最も向きが近い key と、内積が最大の key は一致しないことがありうる。
 * 平面図の説明文でそこを言い分けるために使う。計算そのものには使わない。
 */
function cosineSimilarity(leftVector, rightVector) {
  const denominator = Math.hypot(...leftVector) * Math.hypot(...rightVector);
  return denominator === 0 ? 0 : dotProduct(leftVector, rightVector) / denominator;
}

/** そのマスがマスクされているか。表示の分岐に何度も使うので関数にしておく。 */
function isMaskedCell(pass, queryIndex, keyIndex) {
  return !Number.isFinite(pass.maskedScores[queryIndex][keyIndex]);
}

/* ===========================================================================
 * 5. 画面の状態
 * ========================================================================= */

const ui = {};
const state = {
  focusTokenIndex: 2, // 既定は "playing"。目的語 football を引き寄せる様子が最も分かりやすいため
  useScaling: true,
  useCausalMask: false,
};

/* ===========================================================================
 * 6. 描画 — d_k = 2 の平面を SVG で描く
 *
 *   d_k を 2 にしてあるのは、query と key の関係を「平面上の矢印」として
 *   そのまま描けるからである。表の数値は矢印の座標、内積の大小は矢印どうしの
 *   向きの近さに対応する。外部ライブラリは使わず SVG 要素を直接組み立てる。
 * ========================================================================= */

const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';

/** 平面の描画寸法。scale は「1.0 を何ピクセルで描くか」。 */
const PLANE = {
  scale: 38,
  extent: 132,   // viewBox の半径。|q| ≒ 2.5 のラベルまで収まる大きさにする
  headLength: 9, // 矢じりの長さ
};

/** 属性を指定して SVG 要素を作る。HTML 要素とは名前空間が違うので createElementNS を使う。 */
function createSvgElement(tagName, attributes) {
  const element = document.createElementNS(SVG_NAMESPACE, tagName);
  Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, String(value)));
  return element;
}

/** 数学の座標を SVG の座標に直す。SVG は下方向が正なので y だけ符号を反転する。 */
function planeX(value) { return value * PLANE.scale; }
function planeY(value) { return -value * PLANE.scale; }

/** 平面に文字を置く。 */
function appendPlaneText(svg, x, y, text, anchor, className) {
  const element = createSvgElement('text', { x, y, 'text-anchor': anchor, class: className });
  element.textContent = text;
  svg.appendChild(element);
  return element;
}

/** 軸と、長さの目安になる半径 1・2 の円。軸に意味はないので目盛りは振らない。 */
function appendPlaneFrame(svg) {
  [1, 2].forEach((radius) => {
    svg.appendChild(createSvgElement('circle', { cx: 0, cy: 0, r: radius * PLANE.scale, class: 'plane-circle' }));
  });
  svg.appendChild(createSvgElement('line', { x1: -PLANE.extent, y1: 0, x2: PLANE.extent, y2: 0, class: 'plane-axis' }));
  svg.appendChild(createSvgElement('line', { x1: 0, y1: -PLANE.extent, x2: 0, y2: PLANE.extent, class: 'plane-axis' }));
  appendPlaneText(svg, PLANE.extent - 2, 13, SCORE_SPACE_AXIS_LABELS[0], 'end', 'plane-axis-label');
  appendPlaneText(svg, 5, -PLANE.extent + 12, SCORE_SPACE_AXIS_LABELS[1], 'start', 'plane-axis-label');
}

/** 原点から1本の矢印を描き、その先にラベルを置く。 */
function appendPlaneArrow(svg, arrow) {
  const tipX = planeX(arrow.vector[0]);
  const tipY = planeY(arrow.vector[1]);
  const length = Math.hypot(tipX, tipY);
  if (length < 1) return;

  const unitX = tipX / length;
  const unitY = tipY / length;
  const baseX = tipX - unitX * PLANE.headLength;
  const baseY = tipY - unitY * PLANE.headLength;
  const halfWidth = PLANE.headLength * 0.42;

  svg.appendChild(createSvgElement('line', {
    x1: 0, y1: 0, x2: baseX, y2: baseY, stroke: arrow.color, class: arrow.className,
  }));
  svg.appendChild(createSvgElement('polygon', {
    points: [
      `${tipX},${tipY}`,
      `${baseX - unitY * halfWidth},${baseY + unitX * halfWidth}`,
      `${baseX + unitY * halfWidth},${baseY - unitX * halfWidth}`,
    ].join(' '),
    fill: arrow.color,
    class: arrow.className,
  }));

  const labelX = tipX + unitX * 13;
  const labelY = tipY + unitY * 13 + 4;
  const anchor = labelX > 4 ? 'start' : (labelX < -4 ? 'end' : 'middle');
  appendPlaneText(svg, labelX, labelY, arrow.label, anchor, arrow.className + ' plane-label')
    .setAttribute('fill', arrow.color);
}

/** 矢印の集合を1枚の平面として描く。arrows は先に描いたものが下に来る。 */
function renderVectorPlane(container, arrows, description) {
  container.innerHTML = '';
  const figure = createElement('figure', 'vector-plane');
  const svg = createSvgElement('svg', {
    viewBox: `${-PLANE.extent} ${-PLANE.extent} ${PLANE.extent * 2} ${PLANE.extent * 2}`,
    role: 'img',
    'aria-label': description,
  });
  appendPlaneFrame(svg);
  arrows.forEach((arrow) => appendPlaneArrow(svg, arrow));
  figure.appendChild(svg);
  figure.appendChild(createElement('figcaption', '', description));
  container.appendChild(figure);
}

/** 語ごとの矢印の定義をまとめて作る。 */
function buildTokenArrows(vectors, prefix, className) {
  return SENTENCE_TOKENS.map((token, index) => ({
    vector: vectors[index],
    color: TOKEN_COLORS[index],
    label: `${prefix}_${token}`,
    className,
  }));
}

/** 重み行列の各行を矢印にする。行は語ではなく意味の特徴なので、色は付けず灰色でまとめる。 */
function buildAxisArrows(weightMatrix) {
  return EMBEDDING_AXIS_LABELS.map((label, index) => ({
    vector: weightMatrix[index],
    color: AXIS_ARROW_COLOR,
    label,
    className: 'plane-vector is-faint',
  }));
}

/* ===========================================================================
 * 7. 描画 — 05: 使用する行列と、その平面での姿
 *
 *   key を先に出し、query を後に出す。query は「どの key を探すか」で向きが
 *   決まるので、key を知らないと query の数値を読む足場が無いためである。
 *   重み行列（行 = 意味の特徴）と、その結果の Q/K/V（行 = 語）を対にして出す。
 *   両方を並べないと、W の行と語の行が同じものに見えてしまう。
 * ========================================================================= */

function renderProjectionSection(pass) {
  ui.inputTable.innerHTML = '';
  ui.inputTable.appendChild(createMatrixTable(
    'X : 入力の埋め込み — 行が語、列が意味の特徴',
    SENTENCE_TOKENS, EMBEDDING_AXIS_LABELS, pass.inputVectors));

  // 図には W^K の4行（特徴が平面のどちらへ押すか）と、その合成である4つの key を重ねる。
  renderVectorPlane(ui.keyPlane,
    buildAxisArrows(KEY_WEIGHTS).concat(buildTokenArrows(pass.keyVectors, 'k', 'plane-vector')),
    '灰色の細い矢印が W^K の4行（各特徴が平面のどちらへ押すか）。色の矢印は、語の特徴ぶんだけそれらを足した key。');
  ui.keyTable.innerHTML = '';
  ui.keyTable.appendChild(createMatrixTable(
    'W^K : 行が意味の特徴、列が平面の軸（灰色の矢印の座標）',
    EMBEDDING_AXIS_LABELS, SCORE_SPACE_AXIS_LABELS, KEY_WEIGHTS));
  ui.keyTable.appendChild(createMatrixTable(
    'K = X W^K : 行が語（色の矢印の座標）',
    SENTENCE_TOKENS, SCORE_SPACE_AXIS_LABELS, pass.keyVectors));

  renderVectorPlane(ui.queryPlane,
    buildTokenArrows(pass.keyVectors, 'k', 'plane-vector is-faint')
      .concat(buildTokenArrows(pass.queryVectors, 'q', 'plane-vector is-query')),
    '同じ平面に query（破線）を重ねたもの。各 query は、その語が見つけたい相手の key の方を向いている。');
  ui.queryTable.innerHTML = '';
  ui.queryTable.appendChild(createMatrixTable(
    'W^Q : 行が意味の特徴、列が平面の軸',
    EMBEDDING_AXIS_LABELS, SCORE_SPACE_AXIS_LABELS, QUERY_WEIGHTS));
  ui.queryTable.appendChild(createMatrixTable(
    'Q = X W^Q : 行が語（破線の矢印の座標）',
    SENTENCE_TOKENS, SCORE_SPACE_AXIS_LABELS, pass.queryVectors));

  ui.valueTable.innerHTML = '';
  ui.valueTable.appendChild(createMatrixTable(
    'W^V : 行が意味の特徴、列が value 空間の軸',
    EMBEDDING_AXIS_LABELS, VALUE_AXIS_LABELS, VALUE_WEIGHTS));
  ui.valueTable.appendChild(createMatrixTable(
    'V = X W^V : 行が語。X の行とは別のベクトルになっていることに注目',
    SENTENCE_TOKENS, VALUE_AXIS_LABELS, pass.valueVectors));
}

/** 行ラベル・列ラベル付きの行列表を1つ作る。 */
function createMatrixTable(caption, rowLabels, columnLabels, rows) {
  const figure = createElement('figure', 'param-table');
  figure.appendChild(createElement('figcaption', '', caption));
  const table = createElement('table');
  table.appendChild(createTableHeadRow(columnLabels));
  const body = createElement('tbody');
  rows.forEach((row, rowIndex) => {
    const cells = row.map((value) => formatNumber(value, DISPLAY_DIGITS.vector));
    body.appendChild(createTableRow(rowLabels[rowIndex], cells));
  });
  table.appendChild(body);
  figure.appendChild(table);
  return figure;
}

/** 表の見出し行。左上は空セルにする。 */
function createTableHeadRow(columnLabels) {
  const head = createElement('thead');
  const row = createElement('tr');
  row.appendChild(createElement('th', '', ''));
  columnLabels.forEach((label) => {
    const cell = createElement('th', '', label);
    cell.scope = 'col';
    row.appendChild(cell);
  });
  head.appendChild(row);
  return head;
}

/* ===========================================================================
 * 8. 描画 — 06: 1語を最後まで追う
 * ========================================================================= */

/** 注目する語を選ぶボタン列。sticky バーの中にあり、STEP を読みながら切り替えられる。 */
function renderTokenButtons() {
  ui.tokenButtons.innerHTML = '';
  SENTENCE_TOKENS.forEach((token, index) => {
    const button = createElement('button', 'token-button', token);
    button.type = 'button';
    button.style.setProperty('--token-color', TOKEN_COLORS[index]);
    button.setAttribute('aria-pressed', String(index === state.focusTokenIndex));
    if (index === state.focusTokenIndex) button.classList.add('is-active');
    button.addEventListener('click', () => {
      state.focusTokenIndex = index;
      renderAll();
    });
    ui.tokenButtons.appendChild(button);
  });
}

/**
 * STEP1: 注目している語の query を1本作る。
 *
 * STEP4 と同じ4段の階段で見せる。ここは「W が入力のどの特徴をどれだけ拾うか」が
 * 数字として現れる唯一の場所なので、掛け算を畳まずに開く。
 *
 *     q_i = x_i W^Q
 *         = 0.00·(主体の行) + 0.60·(動作の行) + 0.20·(もの の行) + 0.90·(スポーツの行)
 *         = 0.00·[-0.22, 2.58] + 0.60·[-2.15, -0.62] + …
 *         = [0.47, -2.66]
 *
 * 成分が 0 の項も落とさない。「この語はその特徴を持たないので、その行を1ミリも使わない」
 * ということ自体が読ませたい情報だからである。
 */
function renderQueryStep(pass) {
  const index = state.focusTokenIndex;
  const token = SENTENCE_TOKENS[index];
  const features = pass.inputVectors[index];
  const terms = { digits: DISPLAY_DIGITS.vector, keepAll: true };

  ui.queryStep.innerHTML = '';
  ui.queryStep.appendChild(createElement('p', 'step-formula',
    `x_${token} = ${formatVector(features, DISPLAY_DIGITS.vector)}` +
    `   （軸: ${EMBEDDING_AXIS_LABELS.join(' / ')}）`));

  const ladder = createElement('div', 'mix-ladder');
  appendLadderRow(ladder,
    `q_${token} = x_${token} W^Q`,
    '定義。x の各成分で W^Q の行を混ぜる');
  appendLadderRow(ladder,
    `= ${formatWeightedTerms(features, (axis) => `(${EMBEDDING_AXIS_LABELS[axis]} の行)`, terms)}`,
    'どの特徴をどれだけ使うかは x が決める');
  appendLadderRow(ladder,
    `= ${formatWeightedTerms(features,
      (axis) => formatVector(QUERY_WEIGHTS[axis], DISPLAY_DIGITS.vector), terms)}`,
    'W^Q の行を開く');
  appendLadderRow(ladder,
    `= ${formatVector(pass.queryVectors[index], DISPLAY_DIGITS.vector)}`,
    `これが ${token} の query`);
  ui.queryStep.appendChild(ladder);

  ui.queryStep.appendChild(createVectorChips(pass.queryVectors[index], SCORE_SPACE_AXIS_LABELS, false));
}

/**
 * STEP2 の平面と表。
 * 平面には4つの key と、いま選んでいる語の query だけを重ねる。
 * 「q に向きが近い k ほど内積が大きい」という、表の数値の由来をそのまま絵にしたものである。
 */
function renderScoreStep(pass) {
  const queryIndex = state.focusTokenIndex;
  const token = SENTENCE_TOKENS[queryIndex];
  const rawScores = pass.rawScores[queryIndex];
  const queryVector = pass.queryVectors[queryIndex];
  const nearestIndex = argumentMaximum(pass.keyVectors.map((key) => cosineSimilarity(queryVector, key)));
  const largestIndex = argumentMaximum(rawScores);
  // 向きが最も近い key と、内積が最大の key は、key の長さが揃っていなければ食い違いうる。
  // 食い違ったときはそれ自体が「内積は向きと長さの両方で決まる」ことの実例なので、そう述べる。
  const planeDescription = nearestIndex === largestIndex
    ? `q_${token}（破線）に最も向きが近いのは k_${SENTENCE_TOKENS[nearestIndex]} で、` +
      `内積も ${formatNumber(rawScores[largestIndex], DISPLAY_DIGITS.score)} で最大になります。`
    : `q_${token}（破線）に最も向きが近いのは k_${SENTENCE_TOKENS[nearestIndex]} ですが、` +
      `内積が最大なのは k_${SENTENCE_TOKENS[largestIndex]}（${formatNumber(rawScores[largestIndex], DISPLAY_DIGITS.score)}）です。` +
      `内積は向きの近さと長さの両方で決まるためです。`;

  renderVectorPlane(ui.scorePlane,
    buildTokenArrows(pass.keyVectors, 'k', 'plane-vector').concat([{
      vector: queryVector,
      color: TOKEN_COLORS[queryIndex],
      label: `q_${token}`,
      className: 'plane-vector is-query',
    }]),
    planeDescription);

  ui.scoreTableBody.innerHTML = '';
  SENTENCE_TOKENS.forEach((token, keyIndex) => {
    const masked = isMaskedCell(pass, queryIndex, keyIndex);
    ui.scoreTableBody.appendChild(createTableRow(token, [
      formatVector(pass.keyVectors[keyIndex], DISPLAY_DIGITS.vector),
      formatNumber(pass.rawScores[queryIndex][keyIndex], DISPLAY_DIGITS.score),
      masked ? '−∞（マスク）' : formatNumber(pass.scaledScores[queryIndex][keyIndex], DISPLAY_DIGITS.score),
    ], { headClassName: 'score-token', tokenColor: TOKEN_COLORS[keyIndex] }));
  });
  ui.scaledScoreHead.textContent = state.useScaling
    ? 's_ij = q·k ⁄ √d_k'
    : 's_ij = q·k（割らない）';
}

/** STEP3 の表。softmax を「exp を取る」「和で割る」の2列に開いて見せる。 */
function renderSoftmaxStep(pass) {
  const queryIndex = state.focusTokenIndex;
  const detail = pass.softmaxDetails[queryIndex];
  const topIndex = argumentMaximum(pass.attentionWeights[queryIndex]);

  ui.softmaxTableBody.innerHTML = '';
  SENTENCE_TOKENS.forEach((token, keyIndex) => {
    const masked = isMaskedCell(pass, queryIndex, keyIndex);
    ui.softmaxTableBody.appendChild(createTableRow(token, [
      masked ? '−∞' : formatNumber(pass.maskedScores[queryIndex][keyIndex], DISPLAY_DIGITS.score),
      formatNumber(detail.exponentials[keyIndex], DISPLAY_DIGITS.score),
      formatNumber(pass.attentionWeights[queryIndex][keyIndex], DISPLAY_DIGITS.weight),
    ], {
      rowClassName: keyIndex === topIndex ? 'is-top' : undefined,
      headClassName: 'score-token',
      tokenColor: TOKEN_COLORS[keyIndex],
      lastCellClassName: 'score-weight',
    }));
  });

  // 合計行。exp の和が「割る数」そのものであり、割った結果の和が 1 になることを同じ行で見せる。
  ui.softmaxTableFoot.innerHTML = '';
  ui.softmaxTableFoot.appendChild(createTableRow('合計', [
    `max = ${formatNumber(detail.maximumScore, DISPLAY_DIGITS.score)}`,
    formatNumber(detail.total, DISPLAY_DIGITS.score),
    formatNumber(detail.weights.reduce((sum, weight) => sum + weight, 0), DISPLAY_DIGITS.weight),
  ]));
}

/**
 * STEP4: value をどう混ぜたか。
 *
 * 最初に4段の階段を出す。記号 → 配分 → v の中身 → 結果 の順に1段ずつ具体化する。
 *
 *     z_i = Σ_j a_ij · v_j                        定義
 *         = 0.125·v_I + … + 0.730·v_football      配分を入れた形。係数が混合比そのもの
 *         = 0.125·[1.00, 0.20, 0.00, 0.00] + …    v の中身を開いた形。v ≠ x が見える
 *         = [0.51, 0.09, 0.28, 0.75]              足し合わせた結果
 *
 * 2段目を必ず経由させるのが要点である。いきなり数値ベクトルへ展開すると、
 * 「どの語から何割来たのか」という混合比が数字の中に埋もれてしまう。
 * 3段目を省かないのも同じ理由で、v を記号のまま置くと
 * 「重みが掛かる先は V であって埋め込み X ではない」ことが確認できない。
 *
 * 階段のあとに置く棒は、同じ足し算を語ごとの寄与として視覚化したものである。
 */
function renderValueMixing(pass) {
  const index = state.focusTokenIndex;
  const token = SENTENCE_TOKENS[index];
  const weights = pass.attentionWeights[index];
  ui.valueMixing.innerHTML = '';

  const ladder = createElement('div', 'mix-ladder');
  appendLadderRow(ladder,
    `z_${token} = Σ_j a_${token},j · v_j`,
    '定義。STEP3 で得た配分で v を混ぜる');
  appendLadderRow(ladder,
    `= ${formatWeightedTerms(weights, (keyIndex) => `v_${SENTENCE_TOKENS[keyIndex]}`)}`,
    'STEP3 の配分を入れる。係数が混合比そのもの');
  appendLadderRow(ladder,
    `= ${formatWeightedTerms(weights, (keyIndex) => formatVector(pass.valueVectors[keyIndex], DISPLAY_DIGITS.vector))}`,
    'v の中身を開く。入力の x とは別のベクトル');
  appendLadderRow(ladder,
    `= ${formatVector(pass.contextVectors[index], DISPLAY_DIGITS.vector)}`,
    `足し合わせた結果が文脈ベクトル z_${token}`);
  ui.valueMixing.appendChild(ladder);

  ui.valueMixing.appendChild(createElement('p', 'mix-bars-title', '同じ足し算を、語ごとの寄与として見る'));
  weights.forEach((weight, keyIndex) => {
    const contribution = scaleVector(pass.valueVectors[keyIndex], weight);
    const row = createElement('div', 'mix-row');
    row.style.setProperty('--token-color', TOKEN_COLORS[keyIndex]);
    row.appendChild(createElement('span', 'mix-token', SENTENCE_TOKENS[keyIndex]));
    row.appendChild(createMixBar(weight));
    row.appendChild(createElement('code', 'mix-vector',
      `${formatNumber(weight, DISPLAY_DIGITS.weight)} × ${formatVector(pass.valueVectors[keyIndex], DISPLAY_DIGITS.vector)}` +
      ` = ${formatVector(contribution, DISPLAY_DIGITS.vector)}`));
    ui.valueMixing.appendChild(row);
  });
}

/** 階段の1段。左に式、右にその段が何をしたかの短い注釈を置く。 */
function appendLadderRow(container, expressionText, noteText) {
  const row = createElement('div', 'ladder-row');
  row.appendChild(createElement('code', 'ladder-code', expressionText));
  row.appendChild(createElement('span', 'ladder-note', noteText));
  container.appendChild(row);
}

/**
 * "0.125·v_I + 0.730·v_like + …" の形の線形結合を作る。
 * describeTerm に何を返させるかで、記号のままの形にも、v の中身を開いた形にもなる。
 * 因果マスクで 0 になった項は落とす。落とすこと自体が「その語は見ていない」の表示である。
 */
function formatWeightedTerms(weights, describeTerm, options) {
  const settings = options || {};
  const digits = settings.digits === undefined ? DISPLAY_DIGITS.weight : settings.digits;
  return weights
    .map((weight, keyIndex) => ({ weight, keyIndex }))
    .filter((term) => settings.keepAll || term.weight >= NEGLIGIBLE_WEIGHT)
    .map((term) => `${formatNumber(term.weight, digits)}·${describeTerm(term.keyIndex)}`)
    .join(' + ');
}

/** attention 重みを長さで示す棒。 */
function createMixBar(weight) {
  const track = createElement('span', 'mix-bar');
  const fill = createElement('span', 'mix-bar-fill');
  fill.style.width = `${weight * 100}%`;
  track.appendChild(fill);
  return track;
}

/** RESULT: 入力ベクトル x_i と文脈ベクトル z_i を並べ、何が変わったかを見せる。 */
function renderVectorComparison(pass) {
  const index = state.focusTokenIndex;
  ui.vectorCompare.innerHTML = '';
  ui.vectorCompare.appendChild(createComparisonBlock(
    `入力 x_${SENTENCE_TOKENS[index]}（この語だけの情報）`,
    pass.inputVectors[index], EMBEDDING_AXIS_LABELS));
  ui.vectorCompare.appendChild(createElement('div', 'compare-arrow', '↓  self-attention'));
  ui.vectorCompare.appendChild(createComparisonBlock(
    `文脈 z_${SENTENCE_TOKENS[index]}（周りを取り込んだ情報）`,
    pass.contextVectors[index], VALUE_AXIS_LABELS));
}

/** 比較ブロック1つぶん。軸の意味が違うので、ラベルも一緒に渡す。 */
function createComparisonBlock(title, vector, axisLabels) {
  const block = createElement('div', 'compare-block');
  block.appendChild(createElement('p', 'compare-title', title));
  block.appendChild(createVectorChips(vector, axisLabels, true));
  return block;
}

/** RESULT の読み方を1文で述べる。何をどれだけ参照し、その結果 z が何の混合になったかを言う。 */
function renderResultReading(pass) {
  const index = state.focusTokenIndex;
  const focusToken = SENTENCE_TOKENS[index];
  ui.resultReading.textContent =
    `x_${focusToken} は "${focusToken}" という語そのものの特徴だけを並べた、周りを何も知らないベクトルでした。` +
    `attention を通すと ${describeTopReferences(pass.attentionWeights[index])}を最も強く参照し、` +
    `z_${focusToken} は value をその配分で混ぜたベクトルになります。` +
    `他の3語も選んでみてください。同じ計算が、違う配分で走ります。`;
}

/* ===========================================================================
 * 9. 描画 — 07: 4語ぶんまとめた行列
 * ========================================================================= */

/** 4×4 の attention 行列。行が query 側の語、列が key 側の語。 */
function renderAttentionMatrix(pass) {
  ui.attentionMatrix.innerHTML = '';
  ui.attentionMatrix.style.setProperty('--token-count', String(SENTENCE_TOKENS.length));
  ui.attentionMatrix.appendChild(createElement('div', 'matrix-corner', 'query \\ key'));
  SENTENCE_TOKENS.forEach((token, index) => {
    const header = createElement('div', 'matrix-head', token);
    header.style.setProperty('--token-color', TOKEN_COLORS[index]);
    ui.attentionMatrix.appendChild(header);
  });

  pass.attentionWeights.forEach((row, queryIndex) => {
    ui.attentionMatrix.appendChild(createMatrixRowHeader(queryIndex));
    row.forEach((weight, keyIndex) => {
      ui.attentionMatrix.appendChild(createMatrixCell(
        weight, isMaskedCell(pass, queryIndex, keyIndex), queryIndex === state.focusTokenIndex));
    });
  });
}

/**
 * 行見出し。クリックするとその語で 06 を計算し直し、06 の先頭までスクロールして戻す。
 * 行列は 06 の下にあるので、戻さないと「押したのに何も起きない」ように見えるためである。
 */
function createMatrixRowHeader(queryIndex) {
  const header = createElement('button', 'matrix-row-head', SENTENCE_TOKENS[queryIndex]);
  header.type = 'button';
  header.style.setProperty('--token-color', TOKEN_COLORS[queryIndex]);
  if (queryIndex === state.focusTokenIndex) header.classList.add('is-active');
  header.addEventListener('click', () => {
    state.focusTokenIndex = queryIndex;
    renderAll();
    ui.walkthrough.scrollIntoView({ block: 'start' });
  });
  return header;
}

/** 行列の1セル。attention 重みの大きさを背景の濃さで示す。 */
function createMatrixCell(weight, isMasked, isFocusedRow) {
  const cell = createElement('div', 'matrix-cell');
  if (isMasked) {
    cell.classList.add('is-masked');
    cell.textContent = '—';
    return cell;
  }
  if (isFocusedRow) cell.classList.add('in-focus-row');
  cell.style.setProperty('--fill', String(weight));
  cell.textContent = formatNumber(weight, DISPLAY_DIGITS.weight);
  return cell;
}

/** 行列の下に置く説明文。トグルの状態で読み方が変わるので、そのつど作り直す。 */
function renderMatrixCaption(pass) {
  const divisorText = state.useScaling
    ? `√d_k = √${MODEL_DIMENSIONS.key} ≈ ${formatNumber(pass.scaleDivisor, DISPLAY_DIGITS.vector)} で割っています`
    : '√d_k で割っていません（内積をそのまま softmax に入れています）';
  const maskText = state.useCausalMask
    ? '因果マスクが有効なので、各語は自分より後ろを見られません（decoder / GPT 型）。'
    : 'マスクなしなので、各語は文全体を双方向に見ています（encoder / BERT 型）。';
  ui.matrixCaption.textContent =
    `各行の和は 1 です。第 i 行が、06 で1語ぶん計算した配分そのものです。${divisorText}。${maskText}`;
}

/**
 * 4行それぞれの読み方。数値は必ず計算結果から作る。
 * 解説文に数字を直接書くと、トグルを切り替えたときにこの節だけが古い値のまま残る。
 */
function renderReadingList(pass) {
  ui.readingList.innerHTML = '';
  SENTENCE_TOKENS.forEach((token, queryIndex) => {
    const weights = pass.attentionWeights[queryIndex];
    const item = createElement('li');
    item.appendChild(createElement('b', '', token));
    item.appendChild(document.createTextNode(
      ` は${QUERY_INTENTS[token]}を探すので、いま最も強く見ているのは ${describeTopReferences(weights)}です。`));
    item.appendChild(createElement('code', '', formatMixtureExpression(token, weights)));
    ui.readingList.appendChild(item);
  });
}

/** "z_I = 0.125·v_I + 0.730·v_like + …" の形の式を作る。STEP4 の階段の2段目と同じ形。 */
function formatMixtureExpression(token, weights) {
  return `z_${token} = ${formatWeightedTerms(weights, (keyIndex) => `v_${SENTENCE_TOKENS[keyIndex]}`)}`;
}

/* ===========================================================================
 * 10. まとめて描き直す
 * ========================================================================= */

/** 状態から計算し直して全体を描き直す。入力は毎回作り直すが、量が小さいので問題にならない。 */
function renderAll() {
  const pass = computeAttentionPass(buildInputVectors(), {
    useScaling: state.useScaling,
    useCausalMask: state.useCausalMask,
  });

  renderProjectionSection(pass);

  ui.focusHeadline.textContent = `いま追っている語: "${SENTENCE_TOKENS[state.focusTokenIndex]}"`;
  renderTokenButtons();
  renderQueryStep(pass);
  renderScoreStep(pass);
  renderSoftmaxStep(pass);
  renderValueMixing(pass);
  renderVectorComparison(pass);
  renderResultReading(pass);

  renderAttentionMatrix(pass);
  renderMatrixCaption(pass);
  renderReadingList(pass);
}

/* ===========================================================================
 * 11. 起動
 * ========================================================================= */

function bindUserInterface() {
  Object.entries(ELEMENT_IDS).forEach(([name, id]) => {
    ui[name] = document.getElementById(id);
  });
  ui.scaleToggle.checked = state.useScaling;
  ui.maskToggle.checked = state.useCausalMask;
  ui.scaleToggle.addEventListener('change', () => {
    state.useScaling = ui.scaleToggle.checked;
    renderAll();
  });
  ui.maskToggle.addEventListener('change', () => {
    state.useCausalMask = ui.maskToggle.checked;
    renderAll();
  });
}

bindUserInterface();
renderAll();
