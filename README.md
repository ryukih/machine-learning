# Machine Learning

機械学習の概念を、式を眺めるだけでなく動かして確かめるための対話的な教材集です。
すべて外部ライブラリなし・ビルド不要の静的ページで、ブラウザだけで完結します。

公開URL: <https://ryukih.github.io/machine-learning/>

## Contents

| ラボ | 内容 | URL |
| --- | --- | --- |
| [Langevin & Score Lab](langevin-score-lab/) | 拡散モデルの score を1次元粒子シミュレーションで可視化 | [/langevin-score-lab/](https://ryukih.github.io/machine-learning/langevin-score-lab/) |
| [Transformer Lab](transformer-lab/) | self-attention の Q・K・V を4語の文で最後まで数値計算 | [/transformer-lab/](https://ryukih.github.io/machine-learning/transformer-lab/) |

## Structure

ラボは1つにつき1ディレクトリとし、その中で完結させます。

```text
.
├── index.html                 トップページ（トピック一覧）
├── style.css                  トップページ用
├── langevin-score-lab/
│   ├── index.html
│   ├── app.js
│   └── style.css
└── transformer-lab/
    ├── index.html
    ├── app.js
    └── style.css
```

各ラボは自前の `index.html` / `app.js` / `style.css` を持ち、他のラボやトップページの
CSSに依存しません。ラボ間で見た目が壊し合わないことを優先した構成です。
配色だけは各ページの `:root` で同じ値を定義して揃えています。

## Local preview

Pythonがあれば、リポジトリのルートで次を実行してください。

```bash
python3 -m http.server 8000
```

その後ブラウザで `http://localhost:8000` を開きます。

各ラボの `index.html` を直接開くこともできます。

## GitHub Pages

このリポジトリには `.github/workflows/pages.yml` が含まれています。

1. Repository → Settings → Pages を開きます。
2. Build and deployment の Source を **GitHub Actions** に設定します。
3. `main` ブランチへpushするとPages workflowが実行されます。

ワークフローはリポジトリのルートをそのまま配信するため、ラボを
ディレクトリとして追加すれば設定変更なしで公開されます。

---

## Langevin & Score Lab

Brown運動 → Langevin方程式 → 確率分布 → score → reverse diffusion を、
1次元粒子シミュレーションで順番に学ぶ教材です。

中心となるメッセージは次の2点です。

- SDEが直接記述しているのは、個々の粒子 `X_t^(i)` の運動。
- score `s_t(x) = ∂_x log p_t(x)` が参照するのは、粒子集団が作る確率分布 `p_t(x)`。

### Features

- Pure HTML + CSS + Canvas + JavaScript
- Brown運動とOrnstein–Uhlenbeck型Langevin過程
- 粒子、経験分布、scoreを同時表示
- 粒子からヒストグラム + Gaussian smoothing で `p_t(x)` を推定
- `∂_x log p_t(x)` を有限差分で推定
- Forward中に時刻ごとのscoreを記録し、reverse-time stepで利用
- 状態に応じてラベルと動作が変わる単一の操作ボタン
- レスポンシブUI、devicePixelRatio対応

### Educational notes

reverse diffusion のデモは「forwardで使ったランダムノイズを保存して逆再生」しているのでは
ありません。Forward中に粒子集団から推定した時刻別score fieldを保存し、reverse-time SDEの
Euler–Maruyama型離散化で現在の粒子に作用させています。

また、このデモが score を求める方法は本物の拡散モデルとは異なります。ここでは粒子集団から
周辺分布 `p_t(x)` を直接推定して `log p_t(x)` を数値微分していますが、これは1次元だから
できる方法です。実際の拡散モデルは密度を推定せず、閉じた形で書ける条件付きスコア
`∇_x log p_t(x|x_0) = -ε/σ_t` を教師信号とした回帰（denoising score matching）で
score をニューラルネットに学習させます。

各STEPは独立して動くので、順番に実行する必要はありません。STEPの並びは概念が
積み上がる順序を示しているだけです。ただしSTEP5の内部では、reverse がそのレッスン自身の
forward で作った score 履歴を必要とします。

---

## Transformer Lab

`I like playing football` の4語だけを使って、self-attention を最初から最後まで
実際に数値計算する教材です。

中心となるメッセージは次の2点です。

- `Q` と `K` の内積が決めるのは**どの語をどれだけ見るか**だけ。
- 実際に混ざって運ばれる中身は `V` であり、埋め込み `X` そのものではない。

計算しているのは Vaswani et al. (2017) の式(1)、scaled dot-product attention です。

```text
Q = X W^Q,  K = X W^K,  V = X W^V
s_ij = (q_i · k_j) / √d_k
a_i: = softmax(s_i:)            Σ_j a_ij = 1
z_i  = Σ_j a_ij v_j             ← 語 i の文脈ベクトル
```

`I`, `like`, `playing`, `football` の4語それぞれについて `z_i` が1本ずつでき、
`softmax(QKᵀ/√d_k)` は 4×4 の行列になります。その第 `i` 行が語 `i` の重み配分です。

### Features

- Pure HTML + CSS + JavaScript（Canvasも外部ライブラリも使わない）
- `d_model = 4`, `d_k = 2`, `d_v = 4`, head は1つ
- 4×4 の attention 行列をヒートマップ表示
- 語を選ぶと `q` の生成 → 4つの `k` との内積 → softmax → `v` の合成が数表で開く
- 入力ベクトル `x_i` と文脈ベクトル `z_i` の比較
- `√d_k` で割るかどうかの切り替え（割らないと softmax が鋭くなる）
- 因果マスク（GPT型）の切り替え（右上が消え、配分が計算し直される）
- 使用した行列 `X` / `W^Q` / `W^K` / `W^V` / `Q` / `K` / `V` をすべて表示

### Educational notes

`W^Q`, `W^K`, `W^V` は**学習で得た値ではありません**。読者が電卓で追える大きさに収め、
かつ「主語は述語を探す」「動詞は目的語を探す」という関係が現れるように手で書いた値です。
本物のモデルはこの配線をデータから学習します。軸に「人」「動作」などの名前を付けているのも
同じ理由で、実際のモデルの各軸に人間が読める意味はありません。

そのため文法どおりにならない行もあります。`like` の行は最も「対象らしい」`football` を
選び、`playing` を選びません。1つの head に全部の関係を担わせていないためで、
これが head を複数持つ（multi-head attention）理由でもあります。ページ側でも
この点は隠さずに書いてあります。

トークン化も簡略化しています。実際のトークナイザは `I like playing football.` を
5トークン以上に分割しますが（句点も1トークン、語によってはサブワードに割れる）、
このページは self-attention の計算だけに集中するため4語として扱っています。

positional encoding も同様で、本来は全次元へ異なる周期の正弦波を書き込みますが、
ここでは位置専用の軸を1本だけ置き、位置が1つ進むごとに `0.25` を足す形にしています。
