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
 * 【埋め込みの規約 — このページの中心的な単純化】
 *   埋め込み X を単位行列（one-hot）にしている。すなわち d_model の第 t 軸は
 *   「その語が t 番目の語かどうか」だけを表し、それ以外の意味を持たない。
 *
 *       x_I = [1,0,0,0]  x_like = [0,1,0,0]  x_playing = [0,0,1,0]  x_football = [0,0,0,1]
 *
 *   こうすると x_t W は「W の第 t 行を取り出す」ことと等しくなるので、
 *   W^Q / W^K / W^V の各行が、そのまま語 t の query / key / value になる。
 *   実際のモデルの埋め込み表引きも、内部的にはこれと同じ演算である。
 *   軸に「人」「動作」のような人工的な意味を割り当てないための選択でもある。
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
 *   QUERY_WEIGHTS / KEY_WEIGHTS / VALUE_WEIGHTS は
 *   学習で得た値ではない。読者が電卓で追える大きさに収め、かつ
 *   「主語は述語を探す」「動詞は目的語を探す」という関係が現れるように、
 *   人手で書き込んだ値である。本物のモデルはこれをデータから学習する。
 *   ここで示したいのは値そのものではなく、値がどう流れて文脈ベクトルになるかである。
 *
 * 【q / k / v の軸に名前を付けていない理由】
 *   実際のモデルでは、これらの空間の各軸に人間が読める意味はない。意味を持つのは
 *   軸ではなく、内積の大小（＝どれだけ参照するか）と、混ざった結果のベクトルである。
 *   したがって軸は「軸1」「軸2」と番号でだけ呼ぶ。
 *
 * 【index.html との対応】
 *   参照する要素の id は ELEMENT_IDS に集約している。HTML 側で id を変えるときは
 *   ここも合わせて直すこと。描画は DOM（表と div）で行い、Canvas は使わない。
 *   行列と数表が主役であり、数値をテキストとして読ませたいためである。
 * =============================================================================
 */

'use strict';

/* ===========================================================================
 * 1. モデルの定数
 *    計算ロジックの中には数値を直接書かない。すべてここで名前を付ける。
 * ========================================================================= */

/** 各表現の次元数。d_model は埋め込み、d_k は query/key、d_v は value の次元。 */
const MODEL_DIMENSIONS = {
  embedding: 4, // d_model : 入力埋め込みの次元。X を one-hot にしているのでトークン数と等しい
  key: 2,       // d_k     : query と key が住む空間の次元（両者は同じでなければ内積が取れない）
  value: 4,     // d_v     : value の次元。ここでは比較しやすいよう d_model と同じにしている
};

/** 例文のトークン列。実際のトークナイザは "." も1トークンにするが、ここでは4語に簡略化する。 */
const SENTENCE_TOKENS = ['I', 'like', 'playing', 'football'];

/** 埋め込み空間 d_model の各軸のラベル。第 t 軸は「その語が t 番目の語かどうか」を表すので、
 *  軸の名前はトークンそのものになる。抽象的な意味づけを一切置かないための設計である。 */
const EMBEDDING_AXIS_LABELS = SENTENCE_TOKENS;

/** query と key が共有する空間 d_k の各軸。両者は同じ空間に住まないと内積が取れない。
 *  この軸に人間が読める意味はないので、番号でだけ呼ぶ。 */
const SCORE_SPACE_AXIS_LABELS = ['軸1', '軸2'];

/** value 空間 d_v の各軸。ここも意味を持たせず番号で呼ぶ。 */
const VALUE_AXIS_LABELS = ['軸1', '軸2', '軸3', '軸4'];

/**
 * 埋め込み X の各行。単位行列（one-hot）にしている。
 * 語そのもの以外の情報を持たない、最小の埋め込みである。
 * この形にすると x_t W = 「W の第 t 行」となるので、下の3つの重み行列は
 * そのまま「語ごとの query / key / value の一覧表」として読める。
 */
const TOKEN_EMBEDDINGS = {
  I:        [1, 0, 0, 0],
  like:     [0, 1, 0, 0],
  playing:  [0, 0, 1, 0],
  football: [0, 0, 0, 1],
};

/**
 * W^Q : 埋め込み → query。X が one-hot なので、第 t 行がそのまま語 t の query である。
 * 「どの語を探しに行くか」を、その語の key と同じ向きを指すベクトルとして書き込んである。
 * 大きさ 2.5 は softmax の鋭さを決める。大きいほど1語に集中する。
 */
const QUERY_WEIGHTS = [
  [ 0.0,  2.5], // I        : like の key の向きを指す（主語 → 述語）
  [-2.5,  0.0], // like     : playing の key の向きを指す（動詞 → 目的語）
  [ 0.0, -2.5], // playing  : football の key の向きを指す（動詞 → 目的語）
  [-2.2, -0.8], // football : playing を指しつつ like からは離す（目的語 → 述語）
];

/**
 * W^K : 埋め込み → key。第 t 行がそのまま語 t の key である。
 * 4語を平面上の 0°, 90°, 180°, 270° に置いて、互いに区別できるようにしている。
 * 直交する組の内積は 0、反対向きの組は負になる。
 */
const KEY_WEIGHTS = [
  [ 1.0,  0.0], // I        :   0°
  [ 0.0,  1.0], // like     :  90°
  [-1.0,  0.0], // playing  : 180°
  [ 0.0, -1.0], // football : 270°
];

/**
 * W^V : 埋め込み → value。第 t 行がそのまま語 t の value である。
 * 単位行列にしていないことが重要で、v_t は one-hot の x_t とは別のベクトルになる。
 * これが「重みが掛かる先は V であって埋め込み X ではない」ということの中身である。
 */
const VALUE_WEIGHTS = [
  [1.0, 0.2, 0.0, 0.0], // I
  [0.0, 1.0, 0.4, 0.0], // like
  [0.0, 0.3, 1.0, 0.5], // playing
  [0.6, 0.0, 0.2, 1.0], // football
];

/** 各トークンに割り当てる表示色。CSS の :root と手動で同期させている。 */
const TOKEN_COLORS = ['#2563eb', '#0891b2', '#7c3aed', '#ea580c'];

/** マスクされたスコアに入れる値。softmax の指数が 0 になり、その語は完全に無視される。 */
const MASKED_SCORE = Number.NEGATIVE_INFINITY;

/** 表示上の丸め桁数。計算には一切使わず、文字列化のときだけ使う。 */
const DISPLAY_DIGITS = {
  vector: 2,
  score: 3,
  weight: 3,
};

/** index.html から参照する要素の id。 */
const ELEMENT_IDS = {
  tokenButtons: 'tokenButtons',
  scaleToggle: 'scaleToggle',
  maskToggle: 'maskToggle',
  attentionMatrix: 'attentionMatrix',
  matrixCaption: 'matrixCaption',
  focusHeadline: 'focusHeadline',
  queryStep: 'queryStep',
  scoreTableBody: 'scoreTableBody',
  scoreTableHead: 'scoreTableHead',
  valueMixing: 'valueMixing',
  vectorCompare: 'vectorCompare',
  resultReading: 'resultReading',
  parameterTables: 'parameterTables',
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

/** 要素ごとの和。token embedding と positional encoding を足すのに使う。 */
function addVectors(firstVector, secondVector) {
  return firstVector.map((value, index) => value + secondVector[index]);
}

/** ベクトルのスカラー倍。attention 重み a_ij を value v_j に掛けるのに使う。 */
function scaleVector(vector, factor) {
  return vector.map((value) => value * factor);
}

/**
 * softmax。最大値を引いてから指数を取るのは、指数のオーバーフローを避けるため。
 * 引き算しても結果は変わらない（分子分母に同じ定数が掛かるだけ）。
 * マスクされた要素は -∞ なので exp が 0 になり、配分から完全に外れる。
 */
function softmax(scores) {
  const maximumScore = Math.max(...scores);
  const exponentials = scores.map((score) => Math.exp(score - maximumScore));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

/* ===========================================================================
 * 3. attention の計算（純関数）
 * ========================================================================= */

/**
 * 入力行列 X を作る。各行は語そのものを表す one-hot ベクトルである。
 * 本物のモデルはここへさらに positional encoding を足すが、このページでは足していない。
 * 足さない影響は「語順の情報が入らない」ことで、その意味は解説側で述べている。
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

  const attentionWeights = maskedScores.map(softmax);
  const contextVectors = attentionWeights.map((row) => mixValues(row, valueVectors));

  return {
    inputVectors, queryVectors, keyVectors, valueVectors,
    rawScores, scaleDivisor, scaledScores, maskedScores,
    attentionWeights, contextVectors,
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
 * 6. 描画
 * ========================================================================= */

/** 注目する語を選ぶボタン列。 */
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
      const isMasked = !Number.isFinite(pass.maskedScores[queryIndex][keyIndex]);
      ui.attentionMatrix.appendChild(createMatrixCell(weight, isMasked, queryIndex === state.focusTokenIndex));
    });
  });
}

/** 行見出し。クリックでその語に注目を移せるようボタンにしている。 */
function createMatrixRowHeader(queryIndex) {
  const header = createElement('button', 'matrix-row-head', SENTENCE_TOKENS[queryIndex]);
  header.type = 'button';
  header.style.setProperty('--token-color', TOKEN_COLORS[queryIndex]);
  if (queryIndex === state.focusTokenIndex) header.classList.add('is-active');
  header.addEventListener('click', () => {
    state.focusTokenIndex = queryIndex;
    renderAll();
  });
  return header;
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
    `各行の和は 1 です。行 i は「i 番目の語が文脈ベクトルを作るときの配分」を表します。${divisorText}。${maskText}`;
}

/** STEP1: 注目している語の query を表示する。 */
function renderQueryStep(pass) {
  const token = SENTENCE_TOKENS[state.focusTokenIndex];
  const index = state.focusTokenIndex;
  ui.queryStep.innerHTML = '';
  ui.queryStep.appendChild(createElement('p', 'step-formula',
    `x_${token} = ${formatVector(pass.inputVectors[index], DISPLAY_DIGITS.vector)}` +
    `   （軸: ${EMBEDDING_AXIS_LABELS.join(' / ')}） ← ${index + 1}番目の軸だけが 1`));
  ui.queryStep.appendChild(createElement('p', 'step-formula',
    `q_${token} = x_${token} W^Q = ${formatVector(pass.queryVectors[index], DISPLAY_DIGITS.vector)}` +
    `   ← W^Q の ${token} 行そのもの`));
  ui.queryStep.appendChild(createVectorChips(pass.queryVectors[index], SCORE_SPACE_AXIS_LABELS, false));
}

/** STEP2/3 の表。1行が「相手の語 j」に対応し、内積から配分までを横に並べる。 */
function renderScoreTable(pass) {
  const index = state.focusTokenIndex;
  ui.scoreTableBody.innerHTML = '';
  pass.attentionWeights[index].forEach((weight, keyIndex) => {
    const row = createElement('tr');
    if (keyIndex === argumentMaximum(pass.attentionWeights[index])) row.classList.add('is-top');
    appendScoreCells(row, pass, index, keyIndex, weight);
    ui.scoreTableBody.appendChild(row);
  });
  ui.scoreTableHead.textContent = state.useScaling
    ? `q·k ÷ √d_k`
    : `q·k（割らない）`;
}

/** 表の1行ぶんのセルを詰める。 */
function appendScoreCells(row, pass, queryIndex, keyIndex, weight) {
  const isMasked = !Number.isFinite(pass.maskedScores[queryIndex][keyIndex]);
  const tokenCell = createElement('th', 'score-token', SENTENCE_TOKENS[keyIndex]);
  tokenCell.scope = 'row';
  tokenCell.style.setProperty('--token-color', TOKEN_COLORS[keyIndex]);
  row.appendChild(tokenCell);
  row.appendChild(createElement('td', '', formatVector(pass.keyVectors[keyIndex], DISPLAY_DIGITS.vector)));
  row.appendChild(createElement('td', '', formatNumber(pass.rawScores[queryIndex][keyIndex], DISPLAY_DIGITS.score)));
  row.appendChild(createElement('td', '', isMasked
    ? '−∞'
    : formatNumber(pass.scaledScores[queryIndex][keyIndex], DISPLAY_DIGITS.score)));
  row.appendChild(createElement('td', 'score-weight', formatNumber(weight, DISPLAY_DIGITS.weight)));
}

/** 配列の最大値の位置。どの語が最も強く参照されたかを強調するために使う。 */
function argumentMaximum(values) {
  return values.reduce((bestIndex, value, index) => (value > values[bestIndex] ? index : bestIndex), 0);
}

/** STEP4: value をどう混ぜたか。棒の長さが a_ij、右の数字が実際に足された a_ij·v_j。 */
function renderValueMixing(pass) {
  const index = state.focusTokenIndex;
  ui.valueMixing.innerHTML = '';
  pass.attentionWeights[index].forEach((weight, keyIndex) => {
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

/** attention 重みを長さで示す棒。 */
function createMixBar(weight) {
  const track = createElement('span', 'mix-bar');
  const fill = createElement('span', 'mix-bar-fill');
  fill.style.width = `${weight * 100}%`;
  track.appendChild(fill);
  return track;
}

/** 結果: 入力ベクトル x_i と文脈ベクトル z_i を並べ、何が変わったかを見せる。 */
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

/** 結果の読み方を1文で述べる。何をどれだけ参照し、その結果 z が何の混合になったかを言う。 */
function renderResultReading(pass) {
  const index = state.focusTokenIndex;
  const focusToken = SENTENCE_TOKENS[index];
  const topIndex = argumentMaximum(pass.attentionWeights[index]);
  const topWeight = pass.attentionWeights[index][topIndex];
  const topPercent = Math.round(topWeight * 100);
  ui.resultReading.textContent =
    `x_${focusToken} は "${focusToken}" の軸だけが 1 の、周りを何も知らないベクトルでした。` +
    `attention を通すと "${SENTENCE_TOKENS[topIndex]}" を最も強く（${topPercent}%）参照し、` +
    `z_${focusToken} は4本の value をその配分で混ぜたベクトルになります。` +
    `因果マスクを入れて見比べてください。参照できる語が変われば z も変わります。`;
}

/** 使っているパラメータをすべて表にして最後に置く。値を隠さないことが教材の前提である。 */
function renderParameterTables(pass) {
  ui.parameterTables.innerHTML = '';
  ui.parameterTables.appendChild(createMatrixTable(
    'X : 入力（one-hot 埋め込み）', SENTENCE_TOKENS, EMBEDDING_AXIS_LABELS, pass.inputVectors));
  ui.parameterTables.appendChild(createMatrixTable(
    'W^Q : 語ごとの query 一覧', EMBEDDING_AXIS_LABELS, SCORE_SPACE_AXIS_LABELS, QUERY_WEIGHTS));
  ui.parameterTables.appendChild(createMatrixTable(
    'W^K : 語ごとの key 一覧', EMBEDDING_AXIS_LABELS, SCORE_SPACE_AXIS_LABELS, KEY_WEIGHTS));
  ui.parameterTables.appendChild(createMatrixTable(
    'W^V : 語ごとの value 一覧', EMBEDDING_AXIS_LABELS, VALUE_AXIS_LABELS, VALUE_WEIGHTS));
  ui.parameterTables.appendChild(createMatrixTable(
    'Q = X W^Q（X が one-hot なので W^Q と一致）', SENTENCE_TOKENS, SCORE_SPACE_AXIS_LABELS, pass.queryVectors));
  ui.parameterTables.appendChild(createMatrixTable(
    'K = X W^K（同上）', SENTENCE_TOKENS, SCORE_SPACE_AXIS_LABELS, pass.keyVectors));
  ui.parameterTables.appendChild(createMatrixTable(
    'V = X W^V（同上。x とは別のベクトルであることに注意）', SENTENCE_TOKENS, VALUE_AXIS_LABELS, pass.valueVectors));
}

/** 行ラベル・列ラベル付きの行列表を1つ作る。 */
function createMatrixTable(caption, rowLabels, columnLabels, rows) {
  const figure = createElement('figure', 'param-table');
  figure.appendChild(createElement('figcaption', '', caption));
  const table = createElement('table');
  table.appendChild(createTableHeadRow(columnLabels));
  const body = createElement('tbody');
  rows.forEach((row, rowIndex) => {
    const tableRow = createElement('tr');
    const rowHead = createElement('th', '', rowLabels[rowIndex]);
    rowHead.scope = 'row';
    tableRow.appendChild(rowHead);
    row.forEach((value) => tableRow.appendChild(createElement('td', '', formatNumber(value, DISPLAY_DIGITS.vector))));
    body.appendChild(tableRow);
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

/** 状態から計算し直して全体を描き直す。入力は毎回作り直すが、量が小さいので問題にならない。 */
function renderAll() {
  const pass = computeAttentionPass(buildInputVectors(), {
    useScaling: state.useScaling,
    useCausalMask: state.useCausalMask,
  });
  ui.focusHeadline.textContent = `注目している語: "${SENTENCE_TOKENS[state.focusTokenIndex]}"`;
  renderTokenButtons();
  renderAttentionMatrix(pass);
  renderMatrixCaption(pass);
  renderQueryStep(pass);
  renderScoreTable(pass);
  renderValueMixing(pass);
  renderVectorComparison(pass);
  renderResultReading(pass);
  renderParameterTables(pass);
}

/* ===========================================================================
 * 7. 起動
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
