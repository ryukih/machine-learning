# Langevin & Score Lab

Brown運動 → Langevin方程式 → 確率分布 → score → reverse diffusion を、1次元粒子シミュレーションで順番に学ぶ静的HTML教材です。

中心となるメッセージは次の2点です。

- SDEが直接記述しているのは、個々の粒子 `X_t^(i)` の運動。
- score `∂_x log p_t(x)` が参照するのは、粒子集団が作る確率分布 `p_t(x)`。

## Features

- Pure HTML + CSS + Canvas + JavaScript
- 外部ライブラリなし、ビルド不要
- Brown運動とOrnstein–Uhlenbeck型Langevin過程
- 粒子、経験分布、scoreを同時表示
- 粒子からヒストグラム + Gaussian smoothing で `p_t(x)` を推定
- `∂_x log p_t(x)` を有限差分で推定
- Forward中に時刻ごとのscoreを記録し、reverse-time stepで利用
- レスポンシブUI

## Local preview

Pythonがあれば、リポジトリのルートで次を実行してください。

```bash
python3 -m http.server 8000
```

その後ブラウザで `http://localhost:8000` を開きます。

`index.html` を直接開くこともできます。

## GitHub Pages

このリポジトリには `.github/workflows/pages.yml` が含まれています。

1. GitHubにPublicリポジトリを作成して、このフォルダの内容をpushします。
2. Repository → Settings → Pages を開きます。
3. Build and deployment の Source を **GitHub Actions** に設定します。
4. `main` ブランチへpushするとPages workflowが実行されます。

通常の公開URLは次の形式です。

```text
https://<username>.github.io/<repository-name>/
```

## Educational notes

reverse diffusion のデモは「forwardで使ったランダムノイズを保存して逆再生」しているのではありません。Forward中に粒子集団から推定した時刻別score fieldを保存し、reverse-time SDEのEuler–Maruyama型離散化で現在の粒子に作用させています。

教材としての見通しを優先した1次元デモなので、本格的なscore-based generative modelingで使う学習済みscore networkや高精度SDE solverは実装していません。
