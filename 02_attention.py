import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    return F, nn, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第2章 Attention と GPT

    第1章では「文章 → トークンID列」への変換器（トークナイザー）を作りました。
    第2章では、そのトークンID列を受け取って **次のトークンを予測する** モデル本体、
    すなわち **GPT（Transformer）** を組み立てます。

    GPT の心臓部が **Attention（注意機構）** です。
    Attention をひとことで言うと——

    > 「いま注目している単語が、**他のどの単語をどれくらい参考にすべきか**」を計算し、
    > 参考にする度合いで情報を混ぜ合わせる仕組み

    です。この章では、身近な「辞書」のたとえから出発して、
    少しずつ本物の Attention、そして GPT までを積み上げます。

    | 節 | テーマ |
    |----|--------|
    | 2-1 | ソフトな辞書（Attentionの直感） |
    | 2-2 | Attention の数式（Q・K・V） |
    | 2-3 | スケーリング（なぜ √d で割るのか） |
    | 2-4 | 因果マスク（未来を見ない） |
    | 2-5 | 出力変換（Value と W_o） |
    | 2-6 | マルチヘッド Attention |
    | 2-7 | LayerNorm・GELU・FFN・Block |
    | 2-8 | GPT 全体の組み立て |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | 豆知識：Transformer と Attention の名前の由来
    - 「**Attention（注意）**」の原型は **2014年、Bahdanau ら** が機械翻訳のために導入しました。
    - **2017年、Google の論文 "Attention Is All You Need"（Vaswani ら）** が、
      それまで主流だった RNN を捨て、**Attention だけで構成する Transformer** を提案。
      いまの LLM は、ほぼ全部この Transformer の子孫です。
    - 論文タイトルはビートルズの "All You Need Is Love" のもじり、とよく言われます。
    - 「**Transformer**」は「変換するもの」の意味。入力の系列を別の表現へ変換することから。
      （変形ロボットや映画とは関係ありません）
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2-1. ソフトな辞書 — Attention の直感

    ふつうの辞書（ディクショナリ）は、**キーが完全一致** したときだけ値を返します。
    """)
    return


@app.cell
def _():
    _d = {"美ら海水族館": 2180, "大石林山": 1200, "ナゴパイナップルパーク": 1200, "ネオパークオキナワ": 1300}
    _query = "美ら海水族館"
    _d[_query]  # 2180（完全一致のみ）
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 「似ているキー」も少しずつ参考にしたい

    では、キーが **数値ベクトル** だったらどうでしょう。
    完全一致を待つのではなく、「**似ているキーほど強く参考にする**」ことができます。

    例として、やんばる（沖縄北部）の観光スポットを
    `(自然の豊かさ, 海の近さ, アクセスの良さ)` の3次元で表し、
    各スポットに「おすすめ度」がついた「辞書」を考えます。
    気になる条件に対して、**条件に似たスポットのおすすめ度で重み付けして予測** します。

    /// note | 3つのステップ
    1. **類似度**: クエリと各キーの **内積**（`np.dot`）。向きが近いほど大きい。
    2. **重み**: 類似度を **ソフトマックス** で「合計1の割合」に変換。
    3. **重み付き和**: 重み × 各バリュー を全部足す。

    これがそのまま Attention の正体です。
    ///
    """)
    return


@app.cell
def _():
    # キー: (自然, 海, アクセス) 各0〜10、 バリュー: おすすめ度(0〜100)
    okinawa_spots = {
        (8, 2, 3): 85,  # やんばるの森（自然たっぷり・山奥）
        (3, 9, 1): 70,  # 古宇利島の海（海メイン）
        (1, 2, 9): 60,  # ナゴパイナップルパーク（街中・アクセス良）
        (5, 5, 5): 75,  # 今帰仁城跡（バランス型）
        (7, 6, 2): 80,  # 辺戸岬（自然と海）
        (2, 7, 6): 65,  # 美ら海水族館（海・アクセス良）
        (9, 1, 1): 90,  # 大石林山（大自然）
    }
    new_spot = (6, 4, 5)  # おすすめ度を予測したい条件（自然そこそこ・海少し）
    return new_spot, okinawa_spots


@app.cell
def _(np):
    def soft_dictionary(query, dictionary):
        # ① 類似度（内積）
        similarity = [np.dot(query, key) for key in dictionary]

        # ② ソフトマックスで重みに変換（合計1）
        exp_similarity = np.exp(similarity)
        weights = exp_similarity / np.sum(exp_similarity)

        # ③ 重み付き和
        result = 0
        for weight, value in zip(weights, dictionary.values()):
            result += weight * value

        return result, weights
    return (soft_dictionary,)


@app.cell
def _(new_spot, okinawa_spots, soft_dictionary):
    _rating, _weights = soft_dictionary(new_spot, okinawa_spots)
    print(f"条件 {new_spot} のおすすめ度予測: {_rating:.2f} 点\n")
    print("各スポットが予測に効いた割合:")
    for _key, _w in zip(okinawa_spots.keys(), _weights):
        print(f"  スポット {_key}: {_w * 100:5.2f}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    内積が大きい（＝似ている）スポットほど重みが大きくなり、予測に強く効いています。
    この「似ているものを重視して混ぜる」のが Attention の本質です。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2-2. Attention の数式（Q・K・V）

    先ほどの計算を、複数のクエリをまとめて **行列演算** で書き直したものが Attention です。
    用語を整理します。

    | 記号 | 名前 | 役割 |
    |------|------|------|
    | **Q** | Query（問い合わせ） | 「何を知りたいか」 |
    | **K** | Key（鍵） | 「各データの見出し」 |
    | **V** | Value（値） | 「各データの中身」 |

    $$\text{Attention}(Q,K,V) = \text{softmax}(QK^\top)\,V$$

    - $QK^\top$ … すべてのクエリ × すべてのキーの類似度（内積）を一気に計算
    - $\text{softmax}$ … 各行を「合計1の重み」に
    - $\cdots V$ … 重み付き和

    /// note | PyTorch の関数
    - `torch.matmul(A, B)` … 行列積（`A @ B` と同じ）
    - `K.t()` … 転置（行と列を入れ替え）
    - `F.softmax(x, dim=1)` … 指定した軸に沿ってソフトマックス
    ///
    """)
    return


@app.cell
def _(torch):
    # キー（各スポットの特性: 自然, 海, アクセス）
    K = torch.tensor([
        [8, 2, 3], [3, 9, 1], [1, 2, 9], [5, 5, 5],
        [7, 6, 2], [2, 7, 6], [9, 1, 1],
    ], dtype=torch.float32)

    # バリュー（おすすめ度）
    V = torch.tensor([[85], [70], [60], [75], [80], [65], [90]], dtype=torch.float32)

    # 気になる条件（複数クエリ）
    Q = torch.tensor([
        [6, 4, 5],  # 自然そこそこ・海少し
        [2, 8, 3],  # 海メイン
        [4, 3, 7],  # アクセス重視
    ], dtype=torch.float32)
    return K, Q, V


@app.cell
def _(F, torch):
    def attention(Q, K, V):
        similarity = torch.matmul(Q, K.t())     # QK^T：類似度
        weights = F.softmax(similarity, dim=1)   # 行ごとにソフトマックス
        output = torch.matmul(weights, V)        # 重み付き和
        return output, weights
    return (attention,)


@app.cell
def _(K, Q, V, attention):
    _ratings, _weights = attention(Q, K, V)
    for _spot, _rating in zip(Q, _ratings):
        print(f"条件 {_spot.numpy()} のおすすめ度予測: {_rating.item():.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2-3. スケーリング — なぜ √d で割るのか

    Attention の式には、実はもう1つ大事な工夫があります。
    $QK^\top$ を **次元数の平方根 $\sqrt{d}$ で割る** のです。

    $$\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

    なぜでしょうか。まず、ソフトマックスは **入力の値が大きすぎると「飽和」** します。
    """)
    return


@app.cell
def _(F, torch):
    _x = torch.tensor([100.0, 200.0, 300.0])
    F.softmax(_x, dim=0)  # ほぼ [0, 0, 1] に振り切れる（=飽和）
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    値の差が大きいと、ソフトマックスは1つの要素だけ1・残り0に振り切れてしまい、
    「いろいろなものを少しずつ混ぜる」という Attention の良さが失われます。

    ところが内積 $q\cdot k$ は、**次元数 $d$ が大きいほど値が大きく（分散が $d$ に比例して）** なります。
    そこで $\sqrt{d}$ で割ると、$d$ によらず分散が一定に保たれ、飽和を防げます。

    下のスライダーで次元数 $d$ を変えて、内積の分布（分散）がどう変わるか観察しましょう。
    """)
    return


@app.cell
def _(mo):
    dim_slider = mo.ui.slider(2, 256, value=10, label="次元数 d", show_value=True)
    dim_slider
    return (dim_slider,)


@app.cell
def _(dim_slider, np, plt):
    _d = dim_slider.value
    _n = 5000
    _dots, _scaled = [], []
    for _ in range(_n):
        _q = np.random.randn(_d)
        _k = np.random.randn(_d)
        _dp = float(np.dot(_q, _k))
        _dots.append(_dp)
        _scaled.append(_dp / np.sqrt(_d))

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.hist(_dots, bins=50, alpha=0.5, label=f"without scaling (var={np.var(_dots):.1f})")
    _ax.hist(_scaled, bins=50, alpha=0.5, label=f"with scaling (var={np.var(_scaled):.2f})")
    _ax.set_title(f"dot-product distribution (d = {_d})")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    「スケーリングなし」の分散は $d$ に比例して広がっていくのに対し、
    「√dで割る」と $d$ を変えても分散がほぼ **1のまま** に保たれます。
    これがスケーリングの効果です。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2-4. 因果マスク — 未来を見ない

    ここからは、文章を扱う実戦的な Attention を `nn.Module` で実装します。
    入力は **`(B, C, E)`** の形のテンソルです。

    | 記号 | 意味 |
    |------|------|
    | B | バッチサイズ（同時に処理する文の数） |
    | C | コンテキスト長（1文のトークン数） |
    | E | 埋め込み次元（各トークンを表すベクトルの長さ） |

    GPT は「**次のトークンを予測**」するモデルなので、
    位置 $i$ のトークンを処理するとき **未来（$i$ より後ろ）を見てはいけません**。
    カンニング防止です。

    そこで $QK^\top$ のうち未来にあたる部分を $-\infty$ に置き換えます。
    ソフトマックスを通すと $e^{-\infty}=0$ になり、未来の重みがゼロになります。
    この三角形の覆いを **因果マスク（causal mask）** と呼びます。

    /// note | 使うもの
    - `nn.Linear(in, out, bias=False)` … 学習可能な線形変換 $W$（QKVを作る）
    - `torch.tril(...)` … 下三角だけ1にする（過去＝OK、未来＝0）
    - `masked_fill(mask==0, -inf)` … 未来の位置を $-\infty$ に
    ///
    """)
    return


@app.cell
def _(torch):
    # 因果マスク（過去＝1、未来＝0）の例。C=5
    torch.tril(torch.ones(5, 5))
    return


@app.cell
def _(F, nn, torch):
    class MaskedAttention(nn.Module):
        def __init__(self, embed_dim, key_dim):
            super().__init__()
            self.W_q = nn.Linear(embed_dim, key_dim, bias=False)
            self.W_k = nn.Linear(embed_dim, key_dim, bias=False)
            self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
            self.key_dim = key_dim

        def forward(self, x):           # x: (B, C, E)
            Q = self.W_q(x)             # (B, C, D)
            K = self.W_k(x)             # (B, C, D)
            V = self.W_v(x)             # (B, C, E)

            K_t = K.transpose(-2, -1)             # (B, D, C)
            scores = torch.matmul(Q, K_t)         # (B, C, C)
            scores = scores / (self.key_dim ** 0.5)  # スケーリング

            # 因果マスク：未来を -inf に
            B, C, E = x.shape
            mask = torch.tril(torch.ones(C, C, device=scores.device))
            scores = scores.masked_fill(mask == 0, float("-inf"))

            weights = F.softmax(scores, dim=-1)   # (B, C, C)
            output = torch.matmul(weights, V)     # (B, C, E)
            return output
    return (MaskedAttention,)


@app.cell
def _(MaskedAttention, torch):
    _attn = MaskedAttention(embed_dim=256, key_dim=64)
    _x = torch.randn(2, 5, 256)  # (B=2, C=5, E=256)
    _y = _attn(_x)
    print("入力形状:", _x.shape)
    print("出力形状:", _y.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2-5. 出力変換 — Value と W_o

    2-4 では Value の次元を埋め込み次元 $E$ に合わせていましたが、
    実際には **Value も小さな次元 $D$ に圧縮** し、最後に
    **出力変換 `W_o`** でもとの $E$ に戻すのが一般的です。

    これにより「Attention で情報を混ぜる空間」と「外に出す空間」を分けられ、
    次のマルチヘッド化（2-6）への布石にもなります。
    """)
    return


@app.cell
def _(F, nn, torch):
    class Attention(nn.Module):
        def __init__(self, embed_dim, key_dim):
            super().__init__()
            self.W_q = nn.Linear(embed_dim, key_dim, bias=False)
            self.W_k = nn.Linear(embed_dim, key_dim, bias=False)
            self.W_v = nn.Linear(embed_dim, key_dim, bias=False)  # V も key_dim に
            self.W_o = nn.Linear(key_dim, embed_dim, bias=False)  # 出力変換
            self.key_dim = key_dim

        def forward(self, x):
            Q = self.W_q(x)
            K = self.W_k(x)
            V = self.W_v(x)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.key_dim ** 0.5)

            B, C, E = x.shape
            mask = torch.tril(torch.ones(C, C, device=scores.device))
            scores = scores.masked_fill(mask == 0, float("-inf"))

            weights = F.softmax(scores, dim=-1)
            hidden = torch.matmul(weights, V)  # (B, C, D)

            output = self.W_o(hidden)          # (B, C, E) に戻す
            return output
    return (Attention,)


@app.cell
def _(Attention, torch):
    _attn = Attention(embed_dim=256, key_dim=64)
    _x = torch.randn(2, 5, 256)
    _y = _attn(_x)
    print("入力形状:", _x.shape)
    print("出力形状:", _y.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2-6. マルチヘッド Attention

    1種類の Attention（1ヘッド）だけだと、注目の仕方が1通りに限られます。
    そこで **Attention を複数（H個）並列に走らせ**、それぞれ別の観点で情報を集めます。
    これが **マルチヘッド Attention** で、Transformer の標準部品です。

    記号: `B`=バッチ, `C`=コンテキスト長, `E`=埋め込み次元, `H`=ヘッド数, `D`=各ヘッドの次元。

    /// note | 効率化のテクニック
    ヘッドごとに小さな行列を H 個用意するのではなく、
    **`(E → H*D)` の大きな行列1つ** でまとめて計算し、
    あとから `view` + `transpose` で `(B, H, C, D)` の形に並べ替えます。
    最後に逆の操作で結合し、`W_o` で `E` に戻します。
    ///

    まずは形（shape）の変化を1ステップずつ追ってみましょう。
    """)
    return


@app.cell
def _(nn, torch):
    # --- 形の変化を1ステップずつ確認（すべて使い捨て変数）---
    _B, _C, _E, _H, _D = 2, 4, 16, 3, 8
    _x = torch.randn(_B, _C, _E)

    _W_q = nn.Linear(_E, _H * _D, bias=False)
    _W_k = nn.Linear(_E, _H * _D, bias=False)
    _W_v = nn.Linear(_E, _H * _D, bias=False)

    _Q = _W_q(_x).view(_B, _C, _H, _D).transpose(1, 2)  # (B, H, C, D)
    _K = _W_k(_x).view(_B, _C, _H, _D).transpose(1, 2)
    _V = _W_v(_x).view(_B, _C, _H, _D).transpose(1, 2)

    print("x         :", tuple(_x.shape), "(B, C, E)")
    print("Q(分割後) :", tuple(_Q.shape), "(B, H, C, D)")

    _scores = torch.matmul(_Q, _K.transpose(-2, -1)) / (_D ** 0.5)  # (B, H, C, C)
    print("scores    :", tuple(_scores.shape), "(B, H, C, C)")

    _hidden = torch.matmul(_scores.softmax(-1), _V)  # (B, H, C, D)
    _hidden = _hidden.transpose(1, 2).contiguous().view(_B, _C, _H * _D)
    print("結合後    :", tuple(_hidden.shape), "(B, C, H*D)")

    _out = nn.Linear(_H * _D, _E, bias=False)(_hidden)
    print("出力      :", tuple(_out.shape), "(B, C, E)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `MultiHeadAttention` クラス

    上の流れをクラスにまとめ、**Dropout**（過学習を防ぐためランダムに一部を0にする）も加えます。
    """)
    return


@app.cell
def _(F, nn, torch):
    class MultiHeadAttention(nn.Module):
        def __init__(self, embed_dim, n_head, head_dim, dropout_rate=0.1):
            super().__init__()
            self.n_head = n_head
            self.head_dim = head_dim
            E, H, D = embed_dim, n_head, head_dim

            self.W_q = nn.Linear(E, H * D, bias=False)
            self.W_k = nn.Linear(E, H * D, bias=False)
            self.W_v = nn.Linear(E, H * D, bias=False)
            self.W_o = nn.Linear(H * D, E, bias=False)

            self.attention_dropout = nn.Dropout(dropout_rate)
            self.output_dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            B, C, E = x.shape
            H, D = self.n_head, self.head_dim

            Q = self.W_q(x)
            K = self.W_k(x)
            V = self.W_v(x)

            # 各ヘッドに分割: (B, C, H*D) → (B, H, C, D)
            Q = Q.view(B, C, H, D).transpose(1, 2)
            K = K.view(B, C, H, D).transpose(1, 2)
            V = V.view(B, C, H, D).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)  # (B, H, C, C)

            mask = torch.tril(torch.ones(C, C, device=scores.device))
            scores = scores.masked_fill(mask == 0, float("-inf"))

            weights = F.softmax(scores, dim=-1)
            weights = self.attention_dropout(weights)
            hidden = torch.matmul(weights, V)  # (B, H, C, D)

            # ヘッドを結合して出力変換
            hidden = hidden.transpose(1, 2).contiguous().view(B, C, H * D)
            output = self.W_o(hidden)
            output = self.output_dropout(output)
            return output
    return (MultiHeadAttention,)


@app.cell
def _(MultiHeadAttention, torch):
    _mha = MultiHeadAttention(embed_dim=512, n_head=8, head_dim=64)
    _x = torch.randn(2, 10, 512)
    _y = _mha(_x)
    print("入力形状:", _x.shape)   # (2, 10, 512)
    print("出力形状:", _y.shape)   # (2, 10, 512)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2-7. LayerNorm・GELU・FFN・Block

    Attention だけでは GPT になりません。残りの部品を用意します。

    - **LayerNorm（層正規化）**: 各トークンのベクトルを平均0・分散1に整える。学習を安定させる。
    - **GELU**: 活性化関数（非線形性を与える）。ReLU を滑らかにしたもの。
    - **FFN（フィードフォワード）**: 各トークンを独立に変換する2層の全結合。
      いったん4倍の次元に広げてから戻す。
    - **Block**: 「LayerNorm → Attention → 残差接続」と「LayerNorm → FFN → 残差接続」をまとめたもの。
      これを何段も積み重ねたものが GPT 本体です。

    /// note | 残差接続（residual connection）
    `x = x + サブ層(x)` のように、入力をそのまま足し込みます。
    層を深くしても勾配が消えにくくなり、深いネットワークの学習を可能にします。
    ///
    """)
    return


@app.cell
def _(nn, torch):
    class LayerNorm(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(embed_dim))   # スケール（学習する）
            self.beta = nn.Parameter(torch.zeros(embed_dim))   # シフト（学習する）
            self.eps = 1e-5

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            return self.gamma * norm_x + self.beta


    class GELU(nn.Module):
        def forward(self, x):
            return 0.5 * x * (1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * torch.pow(x, 3))
            ))
    return GELU, LayerNorm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GELU と ReLU の形を比べる

    ReLU は 0 でカクッと折れますが、GELU は **なめらかに 0 へ近づく** のが特徴です。
    """)
    return


@app.cell
def _(np, plt):
    _x = np.linspace(-3, 3, 500)
    _relu = np.maximum(0, _x)
    _gelu = 0.5 * _x * (1 + np.tanh(np.sqrt(2 / np.pi) * (_x + 0.044715 * _x ** 3)))

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(_x, _relu, "b-", linewidth=2, label="ReLU")
    _ax.plot(_x, _gelu, "r-", linewidth=2, label="GELU")
    _ax.set_xlim(-3, 3)
    _ax.set_ylim(-0.5, 3.0)
    _ax.set_xlabel("x")
    _ax.set_ylabel("f(x)")
    _ax.grid(True, linestyle="--", alpha=0.5)
    _ax.axhline(y=0, color="gray", linewidth=0.5)
    _ax.axvline(x=0, color="gray", linewidth=0.5)
    _ax.legend(loc="upper left")
    _fig
    return


@app.cell
def _(GELU, nn):
    class FFN(nn.Module):
        def __init__(self, x_dim, hidden_dim=None, dropout_rate=0.1):
            super().__init__()
            if hidden_dim is None:
                hidden_dim = int(4 * x_dim)  # 慣例的に4倍に広げる

            self.layers = nn.Sequential(
                nn.Linear(x_dim, hidden_dim),
                GELU(),
                nn.Linear(hidden_dim, x_dim),
                nn.Dropout(dropout_rate),
            )

        def forward(self, x):
            return self.layers(x)
    return (FFN,)


@app.cell
def _(FFN, LayerNorm, MultiHeadAttention, nn):
    class Block(nn.Module):
        def __init__(self, embed_dim, n_head, ff_dim=None, dropout_rate=0.1):
            super().__init__()
            head_dim = embed_dim // n_head
            self.norm1 = LayerNorm(embed_dim)
            self.attn = MultiHeadAttention(embed_dim, n_head, head_dim, dropout_rate)
            self.norm2 = LayerNorm(embed_dim)
            self.ffn = FFN(embed_dim, ff_dim, dropout_rate)

        def forward(self, x):
            x = x + self.attn(self.norm1(x))  # 残差接続①
            x = x + self.ffn(self.norm2(x))   # 残差接続②
            return x
    return (Block,)


@app.cell
def _(Block, torch):
    _block = Block(embed_dim=64, n_head=4)
    _x = torch.randn(2, 10, 64)
    _y = _block(_x)
    print("Block 入力形状:", _x.shape)
    print("Block 出力形状:", _y.shape)  # 形は変わらない（積み重ね可能）
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2-8. GPT 全体の組み立て

    いよいよ全部品を組み合わせて **GPT** を完成させます。流れは次の通りです。

    1. **トークン埋め込み** `embed`: トークンID → ベクトル
    2. **位置埋め込み** `pos_embed`: 「何番目の単語か」の情報を足す
    3. **Block を n_layer 段** 通す
    4. **最終 LayerNorm**
    5. **unembed**: ベクトル → 各トークンの「次に来そうな度合い（ロジット）」

    出力 `logits` は `(B, C, vocab_size)` の形で、
    各位置における「次のトークンの予測スコア」になります。

    /// note | 細かい工夫
    - **重み共有**: 入口の埋め込みと出口の `unembed` で同じ重みを使う（`self.embed.weight = self.unembed.weight`）。パラメータ削減と性能向上。
    - **重みの初期化**: 平均0・標準偏差0.02 の正規分布で初期化（GPT-2 流）。
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | 豆知識：頭文字シリーズ（GPT・GELU・Adam）
    この章に出てきた略語の正体です。

    - **GPT** = **G**enerative **P**re-trained **T**ransformer（生成・事前学習・Transformer）。
      OpenAI が **2018年に初代** を発表しました。
    - **GELU** = **G**aussian **E**rror **L**inear **U**nit（2016年）。
      **ReLU** = **Re**ctified **L**inear **U**nit。どちらも「Linear Unit（線形ユニット）」に
      非線形のひと工夫を加えたもの、という名前です。
    - **Adam** = **Ada**ptive **M**oment estimation（2014年）。学習の進み具合を見て更新幅を自動調整します。
      次章で使う **AdamW** は、それに「重み減衰（**W**eight decay）」を正しく分離した改良版（2017年）。
    ///
    """)
    return


@app.cell
def _(Block, nn, torch):
    class GPT(nn.Module):
        def __init__(self, vocab_size, max_context_len, embed_dim,
                     n_head, n_layer, ff_dim, dropout_rate):
            super().__init__()
            self.vocab_size = vocab_size
            self.max_context_len = max_context_len
            self.embed_dim = embed_dim
            self.n_head = n_head
            self.n_layer = n_layer
            self.ff_dim = ff_dim
            self.dropout_rate = dropout_rate

            # 埋め込み層（トークン＋位置）
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.pos_embed = nn.Embedding(max_context_len, embed_dim)
            self.dropout = nn.Dropout(dropout_rate)

            # Transformer ブロックを n_layer 段
            self.blocks = nn.ModuleList([
                Block(embed_dim, n_head, ff_dim, dropout_rate)
                for _ in range(n_layer)
            ])

            # 出力層
            self.norm = nn.LayerNorm(embed_dim)
            self.unembed = nn.Linear(embed_dim, vocab_size)

            # 重み共有
            self.embed.weight = self.unembed.weight

            # 重みの初期化
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, ids):
            B, C = ids.shape
            device = ids.device

            # 埋め込み（トークン + 位置）
            pos = torch.arange(0, C, dtype=torch.long, device=device)
            x = self.dropout(self.embed(ids) + self.pos_embed(pos))

            # Transformer ブロック
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)

            # 各位置で「次のトークン」のスコア
            logits = self.unembed(x)  # (B, C, vocab_size)
            return logits
    return (GPT,)


@app.cell
def _(GPT, torch):
    # モデル設定（本書の codebot と同じ）
    gpt = GPT(
        vocab_size=1000,
        max_context_len=256,
        embed_dim=384,
        n_head=6,
        n_layer=6,
        ff_dim=4 * 384,
        dropout_rate=0.1,
    )

    # 動作テスト：ダミーのトークンID列を入れてみる
    _dummy = torch.randint(0, 1000, (1, 256))  # (B=1, C=256)
    _logits = gpt(_dummy)

    _params = sum(p.numel() for p in gpt.parameters())
    print("入力 (ids)   :", tuple(_dummy.shape))
    print("出力 (logits):", tuple(_logits.shape), "(B, C, vocab_size)")
    print(f"パラメータ数 : {_params:,}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 第2章のまとめ

    | 節 | 作った部品 |
    |----|-----------|
    | 2-1 | ソフト辞書（Attentionの直感：似てるものを重視して混ぜる） |
    | 2-2 | Attention の式 `softmax(QKᵀ)V` |
    | 2-3 | スケーリング `/√d`（ソフトマックスの飽和を防ぐ） |
    | 2-4 | 因果マスク（未来を見ない） |
    | 2-5 | 出力変換 `W_o` |
    | 2-6 | マルチヘッド Attention |
    | 2-7 | LayerNorm・GELU・FFN・Block |
    | 2-8 | **GPT 本体** |

    第1章のトークナイザーと、第2章の GPT がそろいました。
    出力 `logits` は「次のトークンの予測スコア」です。

    /// tip | 次のステップ
    いまの GPT は **まだ何も学習していない**（重みがランダムな）状態です。
    次の章では、第1章で作った `.bin` データを使って
    **実際に学習（次のトークンを当てる訓練）** させ、文章を生成させていきます。
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
