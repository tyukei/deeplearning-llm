# 自作LLM教材（marimo notebook）

Python初心者・機械学習初心者向けに、トークナイザーから GPT までを
ゼロから手を動かして作る教材です。各章は **marimo** のリアクティブノートブック（`.py`）です。

> 元ネタ: 『ゼロから作るDeep Learning ❻』(オライリー・ジャパン) の章構成を参考に、
> 初心者向けの解説・雑学・インタラクティブ要素を加えて再構成しています。

## 章立て

| ファイル | 内容 |
|----------|------|
| `01_tokenizer.py` | 文字／バイト／BPE トークナイザー（特殊トークン・事前トークン化・保存・評価まで） |
| `02_attention.py` | Attention の直感 → Q/K/V → マルチヘッド → GPT 本体の組み立て |
| `03_pretrain_chat.py` | 事前学習 → 生成 → 指示チューニング(SFT) → チャット（学習はボタン実行、対話はUIで） |

> `03_pretrain_chat.py` は ch01 が生成した `data/` を再利用します（無ければ `data/sample_codes.txt` から自動生成）。
> 重い学習は **ボタンを押したときだけ** 走り、チャットは marimo のUIから対話できます。
> 本書サイズのフル学習は付録セル（Colab GPU）を参照。

`codebot/` は本と同じパッケージ構成（`tokenizer.py` / `model.py` / `utils.py`）、
`data/` は学習用サンプルや生成物（`.bin` / `.pkl`）を置きます。

## セットアップ（uv）

```bash
# 依存をインストール（.venv を自動作成）
uv sync

# ノートブックを編集モードで開く
uv run marimo edit 01_tokenizer.py

# アプリとして閲覧（読み取り専用）
uv run marimo run 02_attention.py
```

ライブラリを追加するとき:

```bash
uv add <package>
```

## Google Colab で動かす（marimoを維持／プロキシ方式）

marimo は `.ipynb` に変換しなくても Colab で動かせます。
Colab上で marimo を headless 起動し、Colabのポートプロキシ経由でUIを開きます。
**marimoのインタラクティブ要素（スライダー等）を保ったまま GPU を使えます。**

1. このリポジトリを GitHub に push（または Colab にファイルを直接アップロード）
2. [`colab_launcher.ipynb`](colab_launcher.ipynb) を Google Colab で開く
3. ランタイムを **GPU（T4）** にして、上から順にセルを実行
4. 最後に表示される URL を **新しいタブ** で開く → marimo が起動

> 仕組み: `marimo edit --headless --token-password <TOKEN>` をバックグラウンド起動し、
> `google.colab.kernel.proxyPort(8888)` で発行したURLに `?access_token=<TOKEN>` を付けてアクセスします。

### （参考）Jupyter形式に変換したい場合

ネイティブな `.ipynb` が必要なら変換もできます（ただしリアクティブ機能・`mo.ui`は静的になります）:

```bash
uv run marimo export ipynb 01_tokenizer.py -o 01_tokenizer.ipynb
```
