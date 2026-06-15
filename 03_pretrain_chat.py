import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    import sys
    import json
    import pickle
    from itertools import cycle

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import matplotlib.pyplot as plt

    # 第1章・第2章で作った部品を「ライブラリ」として読み込む
    sys.path.append(".")
    from codebot.model import GPT
    from codebot.tokenizer import train_bpe, BPETokenizer
    from codebot.utils import get_device, generate

    device = get_device()
    return (
        BPETokenizer,
        DataLoader,
        Dataset,
        F,
        GPT,
        cycle,
        device,
        generate,
        json,
        np,
        os,
        pickle,
        plt,
        torch,
        train_bpe,
    )


@app.cell(hide_code=True)
def _(device, mo):
    mo.md(rf"""
    # 第3章 学習させて、対話する

    第1章で **トークナイザー**、第2章で **GPT本体** を作りました。
    でもこの GPT は**まだ何も学習していない**（重みがランダムな）ただの箱です。

    この章では、いよいよモデルを **学習** させて、最後は **チャット** できるようにします。

    | 節 | テーマ | やること |
    |----|--------|----------|
    | 3-1 | データ準備 | 「次のトークン当てクイズ」の形にする |
    | 3-2 | 事前学習 | 大量のコードで「言葉づかい」を覚えさせる |
    | 3-3 | 生成 | 1トークンずつ予測して文章を作る |
    | 3-4 | 指示チューニング(SFT) | 「指示に答える」スタイルを教える |
    | 3-5 | チャット | 対話インターフェイスで話しかける |

    /// warning | このノートブックの方針（軽量デモ）
    本物のLLM学習は **GPU で数時間** かかります。
    ここでは **手元でも数十秒で動く小さなモデル・少ない学習回数** で「仕組み」を体験します。
    そのため生成テキストは拙いものになります（それでOK）。
    本格的な学習は最後の「Colabでフルに動かす」を参照してください。

    いま使えるデバイス: **`{device}`**
    （`cuda`=GPU, `mps`=Apple Silicon, `cpu`=CPU）
    ///

    /// note | 前提
    第1章を実行して `data/sample_codes.bin` と `data/merge_rules.pkl` が出来ていれば、それを使います。
    無い場合は `data/sample_codes.txt` から **自動で作り直す** ので、そのまま進めて大丈夫です。
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3-1. データ準備 — 「次のトークン当てクイズ」

    GPT の学習は、ものすごくシンプルです。

    > **いままでのトークン列を見て、「次の1トークン」を当てる**

    これを大量に繰り返すだけ。だから学習データは「入力 `x`」と「正解 `y`」のペアで、
    `y` は `x` を **1つ後ろにずらしただけ** のものになります。

    ```
    x = [やんばる, は, 沖縄, の, 北部]
    y = [は, 沖縄, の, 北部, です]   ← xを1つ左にずらした列
    ```

    /// note | `Dataset` と `DataLoader`
    - `Dataset` … 「`idx` 番目のデータ（x, y）」を返す係。`__len__` と `__getitem__` を書く。
    - `DataLoader` … Datasetから**ミニbatchをまとめて・シャッフルして**取り出す係。
    ///
    """)
    return


@app.cell
def _(BPETokenizer, Dataset, np, os, pickle, torch, train_bpe):
    # --- トークナイザとトークン化済みデータを用意（無ければ生成） ---
    _txt_path = "data/sample_codes.txt"
    _pkl_path = "data/merge_rules.pkl"
    _bin_path = "data/sample_codes.bin"

    if not os.path.exists(_pkl_path):
        _text = open(_txt_path, encoding="utf-8").read()
        with open(_pkl_path, "wb") as _f:
            pickle.dump(train_bpe(_text, vocab_size=400), _f)

    tokenizer = BPETokenizer.load_from(_pkl_path)

    if not os.path.exists(_bin_path):
        _text = open(_txt_path, encoding="utf-8").read()
        np.array(tokenizer.encode(_text), dtype=np.uint16).tofile(_bin_path)

    token_ids = np.fromfile(_bin_path, dtype=np.uint16)

    # --- 「次トークン当てクイズ」用のDataset ---
    context_len = 128  # 一度に見るトークン数

    class TokenDataset(Dataset):
        def __init__(self, tokens, context_len):
            self.tokens = torch.tensor(tokens, dtype=torch.long)
            self.context_len = context_len

        def __len__(self):
            return len(self.tokens) - self.context_len

        def __getitem__(self, idx):
            x = self.tokens[idx : idx + self.context_len]
            y = self.tokens[idx + 1 : idx + self.context_len + 1]  # 1つずらす
            return x, y

    dataset = TokenDataset(token_ids, context_len)
    return context_len, dataset, tokenizer, token_ids


@app.cell
def _(dataset, mo, tokenizer, token_ids):
    _x, _y = dataset[0]
    mo.md(
        f"""
    - 語彙サイズ: **{tokenizer.vocab_size}**　/　全トークン数: **{len(token_ids):,}**　/　学習サンプル数: **{len(dataset):,}**

    最初のサンプル（先頭20トークンだけ表示）:

    - `x`（入力） : `{_x[:20].tolist()}`
    - `y`（正解） : `{_y[:20].tolist()}`　← xを1つずらした列

    `x` をデコードすると: `{tokenizer.decode(_x[:20].tolist())!r}`
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3-2. 事前学習（Pre-training）

    準備したデータで、GPT に「コードの言葉づかい」を覚えさせます。学習1ステップの流れは:

    1. ミニバッチ `x, y` を取り出す
    2. `logits = model(x)` で「各位置の次トークン予測」を得る
    3. `loss = cross_entropy(logits, y)` で **予測と正解のズレ** を測る
    4. `loss.backward()` で勾配を計算し、`optimizer.step()` で重みを少し更新

    これを何千回も繰り返すと、loss が下がり「それっぽい続き」を書けるようになります。

    /// note | 用語
    - **クロスエントロピー誤差**: 分類問題の定番のloss。正解トークンの確率が低いほど大きくなる。
    - **AdamW**: 代表的な最適化アルゴリズム（重みの賢い更新係）。
    - **logits**: softmax前の生スコア。`(B, C, vocab_size)` の形。
      `cross_entropy` に渡すため `(B*C, vocab_size)` に平らにします。
    ///

    まずモデルとオプティマイザを用意します（軽量デモ用の小さな設定です）。
    """)
    return


@app.cell
def _(GPT, context_len, device, mo, torch, tokenizer):
    # 軽量デモ用のモデル設定（本書のフル設定は最後のセル参照）
    embed_dim = 192
    n_head = 6
    n_layer = 4

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        max_context_len=context_len,
        embed_dim=embed_dim,
        n_head=n_head,
        n_layer=n_layer,
        ff_dim=4 * embed_dim,
        dropout_rate=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    _params = sum(p.numel() for p in model.parameters())
    mo.md(
        f"""
    モデルを作成しました（このセルを再実行すると**重みがリセット**されます）。

    - パラメータ数: **{_params:,}**（約 {_params/1e6:.1f}M）
    - 設定: embed_dim={embed_dim}, n_head={n_head}, n_layer={n_layer}, context_len={context_len}
    - デバイス: **{device}**
    """
    )
    return model, optimizer


@app.cell
def _(mo):
    pretrain_iters = mo.ui.slider(50, 1500, value=400, step=50, label="学習ステップ数", show_value=True)
    pretrain_button = mo.ui.run_button(label="事前学習を実行")
    mo.vstack([pretrain_iters, pretrain_button])
    return pretrain_button, pretrain_iters


@app.cell
def _(
    DataLoader,
    F,
    cycle,
    dataset,
    device,
    mo,
    model,
    optimizer,
    plt,
    pretrain_button,
    pretrain_iters,
):
    mo.stop(
        not pretrain_button.value,
        mo.md("上のスライダーでステップ数を決めて「事前学習を実行」を押してください。"),
    )

    _dl = DataLoader(dataset, batch_size=16, shuffle=True)
    _data_iter = cycle(_dl)  # データを無限ループで供給

    model.train()
    pretrain_losses = []
    with mo.status.progress_bar(total=pretrain_iters.value, title="事前学習中") as _bar:
        for _ in range(pretrain_iters.value):
            _x, _y = next(_data_iter)
            _x, _y = _x.to(device), _y.to(device)

            _logits = model(_x)  # (B, C, vocab_size)
            _loss = F.cross_entropy(
                _logits.view(-1, _logits.size(-1)),  # (B*C, vocab_size)
                _y.view(-1),                          # (B*C,)
            )

            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()

            pretrain_losses.append(_loss.item())
            _bar.update()

    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(pretrain_losses)
    _ax.set_xlabel("iteration")
    _ax.set_ylabel("loss")
    _ax.grid(alpha=0.3)
    mo.vstack([
        mo.md(f"**最終loss: {pretrain_losses[-1]:.3f}**（{len(pretrain_losses)} ステップ）"),
        _fig,
    ])
    return (pretrain_losses,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3-3. 生成（テキスト生成）

    学習したモデルで文章を作ります。GPT の生成は **自己回帰（autoregressive）** です。

    1. プロンプトをトークン化してモデルに入れる
    2. **最後の位置のロジット** から「次トークンの確率」を計算
    3. その確率に従って1トークンサンプリング → 末尾に追加
    4. 2〜3 を繰り返す（`<|endoftext|>` が出たら終了）

    /// note | temperature（温度）
    サンプリングの「ランダムさ」を調整するつまみです。
    - **低い（0に近い）**: 確率最大のトークンを選びがち → かたい・繰り返しがち
    - **高い（1以上）**: いろんなトークンを選ぶ → 多様・暴走しがち

    `temperature=0` のときは確率最大を確定で選びます（argmax）。
    ///

    プロンプトと温度を変えて「生成」を押してみましょう（**事前学習を実行した後**に試してください）。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | 豆知識：「温度（temperature）」は物理学から来た言葉
    なぜ「温度」と呼ぶのでしょう？ これは統計力学の **ボルツマン分布** が由来です。

    物質は **高温ほど粒子が活発にばらつき、低温ほど落ち着く**。
    これと同じで、softmax に温度 $T$ を入れて $\text{softmax}(x / T)$ とすると、
    **高温 → 出力がばらつく（ランダム）／低温 → 一点に集中（かたい）** という振る舞いになります。
    AIの生成における「温度」は、この物理のアナロジーをそのまま借りた名前なのです。
    ///
    """)
    return


@app.cell
def _(mo):
    gen_prompt = mo.ui.text(value="def", label="プロンプト", full_width=True)
    gen_temp = mo.ui.slider(0.0, 1.5, value=1.0, step=0.1, label="temperature", show_value=True)
    gen_len = mo.ui.slider(20, 300, value=120, step=20, label="生成トークン数", show_value=True)
    gen_button = mo.ui.run_button(label="生成")
    mo.vstack([gen_prompt, gen_temp, gen_len, gen_button])
    return gen_button, gen_len, gen_prompt, gen_temp


@app.cell
def _(generate, gen_button, gen_len, gen_prompt, gen_temp, mo, model, tokenizer):
    mo.stop(not gen_button.value, mo.md("「生成」を押すと、いまのモデルで続きを生成します。"))

    _text = generate(
        model,
        tokenizer,
        gen_prompt.value,
        max_new_tokens=gen_len.value,
        temperature=gen_temp.value,
    )
    mo.md(f"**生成結果:**\n```\n{_text}\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3-4. 指示チューニング（SFT: Supervised Fine-Tuning）

    事前学習しただけのモデルは「コードの続き」は書けますが、
    **「質問に答える」「指示に従う」ことはできません**（そういう訓練をしていないから）。

    そこで、**「指示」と「模範解答」のペア** を大量に見せて、
    「指示されたら答える」スタイルを後追いで教えます。これが **SFT** です。

    ### Alpaca 形式

    指示と応答を、決まったテンプレートに流し込みます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | 豆知識：なぜ「アルパカ（Alpaca）」形式？
    - **SFT** = **S**upervised **F**ine-**T**uning（教師ありの微調整）。
    - 「Alpaca 形式」の名前は、**2023年に Stanford が公開した指示追従モデル「Alpaca」** に由来します。
    - その土台が Meta の **LLaMA（ラマ）**。ラマの仲間である**アルパカ**…と動物の名前が続きました。
      その後も **Vicuna（ビクーニャ）** など“ラクダ科”の名前が流行しました。
    - 研究の世界では、こうした**お茶目な名前のつけ方**もちょっとした文化になっています。
    ///
    """)
    return


@app.cell
def _(json, mo, tokenizer):
    with open("data/tiny_codes_sft.json", encoding="utf-8") as _f:
        sft_data = json.load(_f)

    _item = sft_data[0]
    _formatted = (
        f"### Instruction:\n{_item['instruction']}\n\n"
        f"### Response:\n{_item['response']}<|endoftext|>"
    )
    mo.md(
        f"""
    SFTデータ件数: **{len(sft_data):,}**　/　1件目: `{_item}`

    Alpaca形式に変換すると:

    ```
    {_formatted}
    ```

    これをトークン化したものを学習に使います（先頭20トークン: `{tokenizer.encode(_formatted)[:20]}` …）
    """
    )
    return (sft_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `-100` でプロンプト部分を「採点しない」

    SFT の肝は **「応答（Response）部分だけを学習対象にする」** ことです。
    指示（Instruction）部分まで覚えさせると、指示文を**復唱**するようになってしまいます。

    そこで、正解ラベルのうち **プロンプト部分を `-100`** にします。
    `cross_entropy(..., ignore_index=-100)` は **`-100` の位置を採点から除外** してくれます。

    ```
    ids    = [### Instruction: ... ### Response:] + [応答トークン...]
    labels = [-100, -100, ...,        -100      ] + [応答トークン...]
              ┗━━━━━ 採点しない（無視） ━━━━━┛   ┗━ ここだけ学習 ━┛
    ```
    """)
    return


@app.cell
def _(Dataset, json, torch):
    class SFTDataset(Dataset):
        def __init__(self, data_path, tokenizer, context_len):
            self.context_len = context_len
            self.samples = []
            with open(data_path, encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                ids, labels = self._create_sample(
                    item["instruction"], item["response"], tokenizer
                )
                self.samples.append((ids, labels))

        def _create_sample(self, instruction, response, tokenizer):
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            response = f"{response}<|endoftext|>"

            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response)

            ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids  # プロンプトは-100

            # 言語モデル用に入力と正解を1つずらす
            ids = ids[:-1]
            labels = labels[1:]

            # context_len に合わせてパディング / 切り詰め
            pad_len = self.context_len - len(ids)
            if pad_len > 0:
                ids = ids + [0] * pad_len
                labels = labels + [-100] * pad_len
            else:
                ids = ids[: self.context_len]
                labels = labels[: self.context_len]

            return ids, labels

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            ids, labels = self.samples[idx]
            return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    return (SFTDataset,)


@app.cell
def _(SFTDataset, context_len, mo, sft_data, tokenizer):
    sft_dataset = SFTDataset("data/tiny_codes_sft.json", tokenizer, context_len)
    _ids, _labels = sft_dataset[0]
    _trained = (_labels != -100).sum().item()
    mo.md(
        f"""
    SFTデータセットを用意しました（{len(sft_dataset):,} 件）。

    1件目: 全 **{len(_labels)}** トークン中、採点対象（応答部分）は **{_trained}** トークン
    （残り {len(_labels) - _trained} トークンは `-100` で無視）。
    """
    )
    return (sft_dataset,)


@app.cell
def _(mo):
    sft_iters = mo.ui.slider(50, 1000, value=300, step=50, label="SFTステップ数", show_value=True)
    sft_button = mo.ui.run_button(label="指示チューニング(SFT)を実行")
    mo.vstack([sft_iters, sft_button])
    return sft_button, sft_iters


@app.cell
def _(
    DataLoader,
    F,
    cycle,
    device,
    mo,
    model,
    plt,
    sft_button,
    sft_dataset,
    sft_iters,
    torch,
):
    mo.stop(
        not sft_button.value,
        mo.md(
            "**先に「事前学習」を実行してから** このボタンを押してください"
            "（事前学習済みモデルを土台にチューニングします）。"
        ),
    )

    _dl = DataLoader(sft_dataset, batch_size=16, shuffle=True)
    _data_iter = cycle(_dl)

    # SFT専用に新しいオプティマイザを用意
    _sft_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    sft_losses = []
    with mo.status.progress_bar(total=sft_iters.value, title="SFT中") as _bar:
        for _ in range(sft_iters.value):
            _x, _y = next(_data_iter)
            _x, _y = _x.to(device), _y.to(device)

            _logits = model(_x)
            _loss = F.cross_entropy(
                _logits.view(-1, _logits.size(-1)),
                _y.view(-1),
                ignore_index=-100,  # プロンプト部分(-100)は採点しない
            )

            _sft_optimizer.zero_grad()
            _loss.backward()
            _sft_optimizer.step()

            sft_losses.append(_loss.item())
            _bar.update()

    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(sft_losses, color="tab:orange")
    _ax.set_xlabel("iteration")
    _ax.set_ylabel("loss")
    _ax.grid(alpha=0.3)
    mo.vstack([mo.md(f"**SFT最終loss: {sft_losses[-1]:.3f}**"), _fig])
    return (sft_losses,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3-5. チャットしてみる

    SFT 済みのモデルに、Alpaca形式のプロンプトで話しかけます。
    応答（`### Response:` 以降）だけを取り出して表示します。

    /// note | 本書ではCLI、ここではUI
    元コードは `input()` でターミナル対話していましたが、
    marimo では下のテキスト欄で **ブラウザ上から対話** できます。

    軽量デモなので珍回答になります。まともな応答は最後の「Colabでフル学習」で。
    ///
    """)
    return


@app.cell
def _(mo):
    chat_input = mo.ui.text_area(
        value="Hello", label="あなた（指示・質問）", full_width=True, rows=2
    )
    chat_temp = mo.ui.slider(0.0, 1.5, value=0.8, step=0.1, label="temperature", show_value=True)
    chat_button = mo.ui.run_button(label="送信")
    mo.vstack([chat_input, chat_temp, chat_button])
    return chat_button, chat_input, chat_temp


@app.cell
def _(chat_button, chat_input, chat_temp, generate, mo, model, tokenizer):
    mo.stop(not chat_button.value, mo.md("メッセージを入れて「送信」を押してください。"))

    _prompt = f"### Instruction:\n{chat_input.value}\n\n### Response:\n"
    _raw = generate(model, tokenizer, _prompt, max_new_tokens=200, temperature=chat_temp.value)

    # 応答部分だけを抽出
    _resp = _raw.split("### Response:")[-1].strip() if "### Response:" in _raw else _raw

    mo.md(
        f"""
    **You:** {chat_input.value}

    **Bot:**
    ```
    {_resp}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 第3章のまとめ

    | 節 | 学んだこと |
    |----|-----------|
    | 3-1 | 学習データ = 「次トークン当てクイズ」（xを1つずらすとy） |
    | 3-2 | 事前学習 = forward → loss → backward → step の繰り返し |
    | 3-3 | 生成 = 1トークンずつ予測して継ぎ足す（temperatureで多様性調整） |
    | 3-4 | SFT = 応答部分だけ(`-100`でマスク)を学習し「指示に従う」化 |
    | 3-5 | Alpacaプロンプトで対話 |

    **事前学習（言葉づかい）→ SFT（指示への従い方）** という、
    実際のLLM（ChatGPT等）と同じ2段構えを、ミニサイズで一通り体験しました。

    /// tip | この先（発展）
    本書ではこの後、**GRPO（強化学習）** で応答をさらに洗練させます（`ch03/09_grpo.py`）。
    興味があれば次のステップとして挑戦してみてください。
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 付録: Colab でフルに学習する

    手元の軽量デモではなく、**本書サイズ（語彙1000・GPT 約11M・GPU学習）** で動かす手順です。

    **1.** [`colab_launcher.ipynb`](colab_launcher.ipynb) を Colab で開き、ランタイムを **GPU(T4)** に。
    ランチャーの「③ ch03データ取得」を実行すると、
    `codebot/merge_rules.pkl`（語彙1000）・`codebot/tiny_codes.bin`・`codebot/tiny_codes_sft.json` が入ります。

    **2.** このノートブックの設定を本書フル設定に変更:

    ```python
    context_len = 256
    embed_dim   = 384
    n_head      = 6
    n_layer     = 6
    # データ/トークナイザは codebot/ 側（語彙1000）を使う:
    #   tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")
    #   token_ids = np.fromfile("codebot/tiny_codes.bin", dtype=np.uint16)
    ```

    **3.** 事前学習のステップ数を増やす（本書は `max_iters=20000`、SFTは `500`）。

    **4.** 事前学習をスキップして生成だけ試したい場合は、
    ランチャーで `model_pretrain.pt`（学習済み重み）をDLし `GPT.load_from(...)` で読み込めます。

    /// warning
    フル学習はGPUでも時間がかかります。まずは少ないステップ数で loss が下がるのを確認してから増やしましょう。
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
