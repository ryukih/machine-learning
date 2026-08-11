# Machine Learning

機械学習の概念を、式を眺めるだけでなく動かして確かめるための対話的な教材集です。
すべて外部ライブラリなし・ビルド不要の静的ページで、ブラウザだけで完結します。

公開URL: <https://ryukih.github.io/machine-learning/>

## Contents

| ラボ | 内容 | URL |
| --- | --- | --- |
| [Langevin & Score Lab](langevin-score-lab/) | 拡散モデルの score を1次元粒子シミュレーションで可視化 | [/langevin-score-lab/](https://ryukih.github.io/machine-learning/langevin-score-lab/) |
| Transformer Lab | 準備中 | — |

## Structure

ラボは1つにつき1ディレクトリとし、その中で完結させます。

```text
.
├── index.html                 トップページ（トピック一覧）
├── style.css                  トップページ用
└── langevin-score-lab/
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
