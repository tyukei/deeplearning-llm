import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第1章 トークナイザー（Tokenizer）

    LLM（大規模言語モデル）は、人間の文章を **数字の列** に変換してから処理します。
    この「文章 → 数字の列」への変換を担当するのが **トークナイザー** です。

    ```
    "Nago"  ─[encode]→  [78, 97, 103, 111]  →[LLM]
    "Nago"  ←[decode]─  [78, 97, 103, 111]  ←[LLM]
    ```

    - **encode（エンコード）**: 文章 → 数字の列（トークンID）
    - **decode（デコード）**: 数字の列 → 文章

    この章では、トークナイザーを少しずつ賢く（実用的に）していきます。

    | 節 | トークナイザー | 1文字 = ? |
    |----|----------------|-----------|
    | 1-1 | 文字トークナイザー | Unicodeコードポイント |
    | 1-2 | バイトトークナイザー | UTF-8バイト |
    | 1-3 | BPEの学習 | （ルールを作る） |
    | 1-4 | BPEトークナイザー | よく出るペアを1トークンに |

    /// tip | marimo の雑学
    marimo はセルを上から順に実行する普通のノートブックとは違い、
    **変数の依存関係を自動で追って必要なセルだけ再実行** します（リアクティブ）。
    そのため **「同じ変数名を2つのセルで定義することはできない」** という制約があります。
    この教材で `char_ids` や `byte_ids` のように名前を分けているのはそのためです。
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1-1. 文字トークナイザー

    いちばん素朴な発想は「**1文字を1つの数字に対応させる**」ことです。

    まずは Python の文字列の基本から見ていきましょう。
    """)
    return


@app.cell
def _():
    SAMPLE_TEXT = "Yanbaru山原"
    SAMPLE_TEXT
    return (SAMPLE_TEXT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `list(文字列)` で1文字ずつに分解

    文字列を `list()` に渡すと、**1文字ずつの要素を持つリスト** になります。
    日本語や絵文字も「1文字」として扱われる点に注目してください。
    """)
    return


@app.cell
def _(SAMPLE_TEXT):
    list(SAMPLE_TEXT)  # ['Y', 'a', 'n', 'b', 'a', 'r', 'u', '山', '原']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `ord()` と `chr()`：文字 ⇄ 数字

    Python には、文字と数字（Unicodeコードポイント）を変換する組み込み関数があります。

    - `ord('文字')` … 文字 → 数字
    - `chr(数字)` … 数字 → 文字

    /// note | Unicodeコードポイントとは？
    世界中のすべての文字に振られた「背番号」のようなものです。
    `'山'` は 23665番、`'原'` は 21407番。
    漢字も「ただの大きな番号」にすぎません。
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | 豆知識：Unicode の歴史
    むかしは国や言語ごとに文字コードがバラバラでした
    （ASCII・Shift_JIS・EUC-JP・ISO-8859 …）。
    同じ番号が環境によって別の文字を指すため、メールやWebでよくUnicode**文字化け（mojibake）**が起きていました。

    そこで **1991年に Unicode Consortium が設立** され、
    「**世界中のすべての文字に、世界共通の番号を1つずつ振る**」という壮大な計画が始まります。

    - 当初は **16ビット（65,536文字）** で足りると思われていた…が、漢字や各国文字で全然足りず、
      1996年に拡張。いまは **U+0000 〜 U+10FFFF（約111万通り、17の「面（plane）」）** まで表せます。
    - 実際に登録済みの文字は約15万。まだまだ空きがあります。
    - コードポイントは `U+` ＋ 16進数で書きます（`U+` の U は Unicode）。
      例: `'A'` = U+0041、`'山'` = U+5C71（= 10進で 23665）。
    ///
    """)
    return


@app.cell
def _():
    print(ord("山"))   # 23665
    print(ord("原"))   # 21407

    print(chr(23665))  # '山'
    print(chr(21407))  # '原'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### リスト内包表記でまとめて変換

    `[ord(char) for char in text]` は **リスト内包表記** という書き方です。

    ```python
    # この2つは同じ意味
    ids = [ord(char) for char in text]

    ids = []
    for char in text:
        ids.append(ord(char))
    ```

    1行で書けて読みやすいので、Python では頻繁に登場します。
    """)
    return


@app.cell
def _(SAMPLE_TEXT):
    ids_by_ord = [ord(char) for char in list(SAMPLE_TEXT)]
    ids_by_ord  # [89, 97, 110, 98, 97, 114, 117, 23665, 21407]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `CharTokenizer` クラスにまとめる

    encode と decode を **クラス** にまとめると、トークナイザーとして再利用しやすくなります。

    /// note | `class` と `self` の話
    - `class` は「データと処理をひとまとめにした設計図」です。
    - `self` は「そのインスタンス（実体）自身」を指します。
      `tokenizer.encode(text)` と呼ぶと、`self` に `tokenizer` が自動で渡されます。
    - `''.join(リスト)` は、文字のリストを連結して1つの文字列に戻すイディオムです。
    ///
    """)
    return


@app.class_definition
class CharTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join([chr(i) for i in ids])


@app.cell
def _(SAMPLE_TEXT):
    char_tokenizer = CharTokenizer()

    char_ids = char_tokenizer.encode(SAMPLE_TEXT)
    char_decoded = char_tokenizer.decode(char_ids)

    print("encode :", char_ids)
    print("decode :", char_decoded)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Python寄り道：メソッドの `self` あり・なし

    クラスのメソッド（クラスの中の関数）は、**第1引数が何か** で役割が変わります。

    | 書き方 | 第1引数 | 呼び出し方 | 役割 |
    |--------|---------|-----------|------|
    | ふつうのメソッド | `self` | `インスタンス.method()` | そのインスタンスのデータを読み書きする |
    | `@staticmethod` | （なし） | `クラス名.method()` | データに触れない“ただの関数”をクラスに同居させる |
    | `@classmethod` | `cls` | `クラス名.method()` | クラス自身を受け取る（別の作り方を提供するなど） |

    - **`self` は呼び出したインスタンスが自動で入ります**（自分では渡しません）。
      つまり `tokenizer.encode(text)` は、裏では `encode(tokenizer, text)` のように動きます。
    - **`self.` を付けた変数はインスタンスごとに別々のデータ**になります。
      `self.` を付けないただの変数は、その場限りの**ローカル変数**でメソッドを抜けると消えます。
    - この教材では、あとで出てくる `BPETokenizer.load_from(...)` が `self` を取らない
      **`@staticmethod`**、第2章の `GPT.load_from(...)` が `cls` を取る **`@classmethod`** の例です。

    /// note | 雑学：`self` は予約語ではない
    `self` は Python の予約語ではなく、**ただの慣習**です。名前は自由に変えられますが
    （`this` でも動く）、**みんなが必ず `self` と書く約束**になっています。
    迷わず `self` にしておきましょう。
    ///
    """)
    return


app._unparsable_cell(
    r"""
    # self あり/なし を実際に動かして比べてみる
    class Counter:
        def __init__(self, start=0):
            self.count = start        # self. を付けた変数 = このインスタンスの持ち物

        def increment(self):          # self あり：自分の count を増やす（インスタンスメソッド）
            self.count += 1
            return self.count
        @staticmethod
        def description():            # self なし：データに触れない“ただの関数”
            return "increment() を呼ぶたびに 1 ずつ増えます"

    class Counter2:
        def __init__(hoge, start=0):
            hoge.count = start

        def increment(hoge):
            hoge.count += 1
            return hoge.count

    a = Counter(10)
    b = Counter(100)
    c = Counter2(1000)

    print(a.increment())  # 11  （a の count だけ増える）
    print(b.increment())  # 101 （b は a と別物なので独立している）
    print(c.increment())　# 1001 （selfでなくても動く）
    # staticmethod はインスタンスを作らなくても、クラスから直接呼べる
    print(Counter.description())
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### やってみよう（インタラクティブ）

    下のボックスに好きな文章を入れると、文字トークナイザーの結果がリアルタイムで変わります。
    （marimo のリアクティブ機能です）
    """)
    return


@app.cell
def _(mo):
    user_text = mo.ui.text(value="やんばるの森", label="文章を入力:", full_width=True)
    user_text
    return (user_text,)


@app.cell
def _(mo, user_text):
    _tok = CharTokenizer()
    _ids = _tok.encode(user_text.value)
    mo.md(
        f"""
    - 入力: `{user_text.value}`
    - 文字数: **{len(user_text.value)}**
    - トークンID: `{_ids}`
    - 復元: `{_tok.decode(_ids)}`
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// warning | 文字トークナイザーの弱点
    文字単位だと、世界中の文字（10万種類以上）すべてに番号を割り当てる必要があり、
    **語彙（ボキャブラリ）が巨大** になります。
    また、見たことのない珍しい文字が来ると困ります。
    → そこで次は「バイト」を使ったトークナイザーを考えます。
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1-2. バイトトークナイザー

    コンピュータは最終的にすべてを **バイト（0〜255の数字）** で扱います。
    文字を **UTF-8** という方式でバイト列に変換すれば、語彙はたった **256種類** で済みます。

    /// note | UTF-8 の雑学
    UTF-8 は「文字によって使うバイト数が変わる」可変長エンコーディングです。

    - 英数字（ASCII）… **1バイト**（例: `'A'` → `[65]`）
    - ひらがな・漢字 … **3バイト**（例: `'森'` → `[230, 163, 174]`）
    - 一部の特殊な文字 … **4バイト**

    だから「文字数」と「バイト数」は一致しません。
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | 豆知識：UTF-8 は「ダイナーの紙ナプキン」から生まれた
    Unicode は「文字にどの**番号**を割り当てるか」を決めるだけ。
    その番号を実際に**バイト列へどう詰めるか**は別問題で、その方式の代表が **UTF-8** です
    （UTF = Unicode Transformation Format、8 は「8ビット単位」の意味）。

    UTF-8 は **1992年、Unix を生んだ Ken Thompson と Rob Pike** が設計しました。
    有名な逸話では、二人は食堂（ダイナー）の**紙ナプキンに走り書き**して仕様を決めたと言われます。

    UTF-8 の賢いところ:

    - **ASCII（0〜127）はそのまま1バイトで互換** … 英語圏の既存データを壊さない
    - **先頭バイトを見れば「その文字が何バイトか」が分かる** … 途中から読んでもズレを復帰できる
      （self-synchronizing）

    いまや **Web ページの約98%** が UTF-8 です。
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | 豆知識：絵文字（emoji）は日本生まれ

    - 「**emoji**」は英語の emotion とは無関係で、**絵＋文字** という日本語そのもの（偶然の一致）。
    - 起源は **1999年、NTTドコモの栗田穣崇（くりた しげたか）** が i-mode 用に
      デザインした **176個・12×12ピクセル** の絵文字。
    - **2010年に Unicode へ採用** され、いまや世界共通の「文字」になりました。
    - 絵文字の多くは **4バイト**（コードポイントが大きい）。
      「絵文字すら、ただの大きな番号にすぎない」というわけです。
    ///
    """)
    return


@app.cell
def _():
    # 'A' の場合（1バイト）
    encoded_A = "A".encode("utf-8")
    print(encoded_A)        # b'A'
    print(list(encoded_A))  # [65]
    return


@app.cell
def _():
    # '森' の場合（3バイト）
    encoded_a = "森".encode("utf-8")
    print(encoded_a)        # b'\xe6\xa3\xae'
    print(list(encoded_a))  # [230, 163, 174]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### バイト列 → 文字へ戻す

    `bytes(数字のリスト).decode("utf-8")` で、バイト列を文字列に戻せます。
    `encode` の逆向きの操作です。
    """)
    return


@app.cell
def _():
    decoded_from_bytes = bytes([65]).decode("utf-8")
    decoded_from_bytes  # 'A'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `ByteTokenizer` クラス

    やることはとてもシンプルです。

    - `encode`: `text.encode("utf-8")` でバイト列にして `list()` で数字のリストに
    - `decode`: `bytes(ids)` でバイト列に戻して `.decode("utf-8")` で文字列に
    """)
    return


@app.class_definition
class ByteTokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8")


@app.cell
def _(SAMPLE_TEXT):
    byte_tokenizer = ByteTokenizer()

    byte_ids = byte_tokenizer.encode(SAMPLE_TEXT)
    byte_decoded = byte_tokenizer.decode(byte_ids)

    print("encode :", byte_ids)
    # [89, 97, 110, 98, 97, 114, 117, 229, 177, 177, 229, 142, 159]
    print("decode :", byte_decoded)
    return (byte_ids,)


@app.cell(hide_code=True)
def _(byte_ids, mo):
    mo.md(f"""
    /// warning | バイトトークナイザーの弱点
    語彙は256種類だけで済む一方、**トークン列が長くなりがち** です。
    例えば `"Yanbaru山原"`（9文字）はバイトだと **{len(byte_ids)}トークン** に増えてしまいました。
    日本語は1文字あたり3バイト（≒3トークン）を消費します。

    → 「短い語彙」と「短いトークン列」を両立したい。
    そこで登場するのが **BPE** です。
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1-3. BPE の学習（train）

    **BPE（Byte Pair Encoding）** は、
    「**よく隣り合って出てくるペアを、新しい1つのトークンにまとめる**」を繰り返す手法です。

    ```
    "aaabdaaabac"
      → "aa"が最頻出 → Zに置換 → "ZabdZabac"
      → "Za"が最頻出 → Yに置換 → "YbdYbac"
      ...
    ```

    こうやって「`is `」「`ing`」のような **よく使う塊** に番号を割り当てていきます。
    実際の GPT 系モデルもこの BPE をベースにしています。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | 豆知識：BPE はもともと「圧縮」の技術だった
    BPE（Byte Pair Encoding）は実は新しい発明ではありません。

    - **1994年、Philip Gage** が提案した**データ圧縮**アルゴリズムが原型。
      「よく出るペアを1つの記号に置き換える」という発想は当時のまま。
    - これを自然言語処理に転用したのが **Sennrich ら（2016年・機械翻訳）**。
      未知の単語も「部分語（subword）」に分解できるのが画期的でした。
    - **GPT-2 以降**は「バイト単位の BPE」が定番になっています。

    ちなみに「**トークン（token）**」の語源は「**しるし・代用貨幣**」。
    文章を区切った最小単位の“しるし”という意味です。
    LLM の利用料が「トークン数」で測られるのも、これが理由です。
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 手順① 隣り合うペアを数える `count_pairs`

    /// note | `defaultdict` と `zip` の話
    - `defaultdict(int)` は「キーが無ければ自動で 0 から始まる辞書」。
      `counts[pair] += 1` を、初期化なしでいきなり書けて便利です。
    - `zip(ids, ids[1:])` は **隣り合う2要素のペア** を作る定番テクニックです。
      `ids = [1, 2, 3]` なら `(1, 2)`, `(2, 3)` が得られます。
    ///
    """)
    return


@app.cell
def _():
    from collections import defaultdict

    def count_pairs(ids):
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    return (count_pairs,)


@app.cell
def _(count_pairs):
    pair_demo_ids = [1, 2, 3, 1, 2]
    dict(count_pairs(pair_demo_ids))  # {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 手順② ペアを新しいIDに置き換える `merge`

    指定したペアを見つけたら、新しいID 1つに置き換えながらリストを作り直します。
    `i += 2`（ペアを置換したら2つ進む）と `i += 1`（それ以外は1つ進む）の使い分けがポイントです。
    """)
    return


@app.function
def merge(ids, pair, new_id):
    merged_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    return merged_ids


@app.cell
def _():
    merge_demo_ids = [1, 2, 3, 1, 2]
    merge(merge_demo_ids, (1, 2), 4)  # [4, 3, 4]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 手順③ 学習ループ `train_bpe`

    ①と②を組み合わせて、決めた回数だけ「最頻出ペアを新トークン化」を繰り返します。

    - 初期語彙は **256**（バイトの 0〜255）
    - `num_merges = vocab_size - 256` … 何回マージするか
    - `max(counts, key=counts.get)` … **値（出現回数）が最大のキー** を取り出すイディオム

    /// tip | 同点（タイ）の決着について
    最頻出ペアが複数あると、`max` の結果が環境によってブレることがあります。
    再現性を厳密にしたいときは、コメントアウトしてある
    `key=lambda pair: (counts[pair], pair[0], pair[1])` のように
    **タイブレーク条件** を足すのが定石です。
    ///
    """)
    return


@app.cell
def _(count_pairs):
    def train_bpe(text, vocab_size):
        # テキストを 0〜255 のID列に変換
        ids = list(text.encode("utf-8"))

        # マージ回数を決定（256は初期の語彙サイズ）
        num_merges = vocab_size - 256
        merge_rules = {}

        for step in range(num_merges):
            # 隣接ペアの統計を取得
            counts = count_pairs(ids)

            # ペアが存在しない場合は終了
            if not counts:
                break

            # 最頻出ペアを選択
            best_pair = max(counts, key=counts.get)
            # best_pair = max(counts, key=lambda pair: (counts[pair], pair[0], pair[1]))

            # 新しいトークンIDを割り当ててマージ
            new_id = 256 + step
            merge_rules[best_pair] = new_id
            ids = merge(ids, best_pair, new_id)

        return merge_rules

    return (train_bpe,)


@app.cell
def _(train_bpe):
    bpe_text = "Northern Okinawa is famous for the Yanbaru forest."
    trained_rules = train_bpe(bpe_text, vocab_size=260)
    trained_rules  # {(111, 114): 256, (32, 102): 257, (116, 104): 258, (258, 101): 259}
    return (trained_rules,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### やってみよう：語彙サイズを動かす

    `vocab_size` を増やすほどマージ回数が増え、学習されるルールも増えます。
    スライダーを動かして、ルールがどう増えるか観察してみましょう。
    """)
    return


@app.cell
def _(mo):
    vocab_slider = mo.ui.slider(256, 300, value=265, label="vocab_size", show_value=True)
    train_area = mo.ui.text_area(
        value="the cat sat on the mat. the cat ran.",
        label="学習テキスト:",
        full_width=True,
    )
    mo.vstack([train_area, vocab_slider])
    return train_area, vocab_slider


@app.cell
def _(mo, train_area, train_bpe, vocab_slider):
    _rules = train_bpe(train_area.value, vocab_size=vocab_slider.value)
    _lines = [
        f"- `{p}` → 新ID **{nid}**" for p, nid in _rules.items()
    ]
    mo.md(
        f"**マージ回数: {len(_rules)} 回**\n\n" + ("\n".join(_lines) or "（マージなし）")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1-4. BPE トークナイザー（学習済みルールを使う）

    1-3 で作った `merge_rules`（学習結果）を使って、実際に encode / decode するクラスです。

    /// note | ポイント
    - `id_to_bytes`: 各トークンIDが「どのバイト列を表すか」の対応表。
      まず 0〜255 を登録し、マージで作られたIDは
      **元の2トークンのバイト列をつなげたもの** として登録します。
    - `encode`: 学習したときと **同じ順番** でマージルールを適用するのが超重要です。
    - `decode` の `errors="replace"`: 途中で切れた不正なバイト列でも
      エラーで止めず、`�`（置換文字）に置き換えて返してくれます。
    ///
    """)
    return


@app.class_definition
class SimpleBPETokenizer:
    def __init__(self, merge_rules):
        self.merge_rules = merge_rules

        # IDからバイト列への対応表（0〜255を登録）
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}

        # マージされたトークンは元トークンのバイト列を連結
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]

        # 語彙サイズ
        self.vocab_size = len(self.id_to_bytes)

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        # 学習時の順序でマージルールを適用
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)
        return ids

    def decode(self, ids):
        byte_list = [self.id_to_bytes[i] for i in ids]
        combined_bytes = b"".join(byte_list)
        return combined_bytes.decode("utf-8", errors="replace")


@app.cell
def _(trained_rules):
    bpe_tokenizer = SimpleBPETokenizer(trained_rules)

    bpe_ids = bpe_tokenizer.encode("north山原")
    bpe_decoded = bpe_tokenizer.decode(bpe_ids)

    print("vocab_size :", bpe_tokenizer.vocab_size)
    print("encode     :", bpe_ids)
    # [110, 256, 258, 229, 177, 177, 229, 142, 159]
    print("decode     :", bpe_decoded)  # north山原
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ここまでのまとめ（基礎編）

    | トークナイザー | 語彙サイズ | トークン列の長さ | 未知の文字 |
    |----------------|-----------|------------------|-----------|
    | 文字（1-1） | 巨大（10万超） | 短い | 弱い |
    | バイト（1-2） | 256で固定 | 長くなりがち | 強い（必ず表現できる） |
    | **BPE（1-3,1-4）** | **自由に調整可** | **短い** | **強い** |

    BPE は「バイトの強さ（何でも表現できる）」と「短いトークン列」を
    両立できるのがポイントです。だから実際の LLM で広く使われています。

    ---

    ここまでで BPE の核は完成です。
    後半（1-5〜1-9）では、これを **実用レベル** に近づけていきます。

    | 節 | テーマ | やること |
    |----|--------|----------|
    | 1-5 | 特殊トークン | `<|endoftext|>` で文書の区切りを表す |
    | 1-6 | 事前トークン化 | 単語の途中でマージしないよう前処理する |
    | 1-7 | 実データで学習 | 学習結果をファイルに保存する（pickle） |
    | 1-8 | 評価 | 学習したトークンと圧縮率を確認する |
    | 1-9 | 一括エンコード | 学習用に全文を数値ファイル化する（numpy） |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1-5. 特殊トークン `<|endoftext|>`

    LLM の学習データは、たくさんの文書をつなげた巨大なテキストです。
    でも「文書Aの終わり」と「文書Bの始まり」がつながっていると、
    モデルが**無関係な文書をまたいで続きを予測**してしまいます。

    そこで、文書の区切りに **特殊トークン** `<|endoftext|>` を入れます。
    これは「ここで話が終わり」という**特別な1トークン**として扱います。

    /// note | 設計上の工夫
    - **学習時**: `<|endoftext|>` でテキストを分割し、
      **区切りをまたいでペアをマージしない** ようにします
      （文書Aの末尾と文書Bの先頭がくっつくのを防ぐ）。
    - 語彙の最後に特殊トークン用のIDを1つ予約します（`vocab_size - 256 - 1` がマージ回数）。
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 改良版 `count_pairs`：集計を引き継げるようにする

    複数の文書（リストのリスト）を1つの集計表にまとめたいので、
    `counts` を **引数で受け取って continue できる** 形に改良します。

    ```python
    def count_pairs(ids, counts=None):
        if counts is None:          # 初回は新しい辞書を用意
            counts = defaultdict(int)
        ...
        return counts               # 同じ辞書を返して次の呼び出しへ渡す
    ```

    marimo では同じ名前を再定義できないので、ここでは `count_pairs_acc`（accumulate版）とします。
    `merge` 関数は 1-3 のものをそのまま再利用します。
    """)
    return


@app.cell
def _():
    from collections import defaultdict as _defaultdict

    def count_pairs_acc(ids, counts=None):
        if counts is None:
            counts = _defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    return (count_pairs_acc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 特殊トークン対応の学習 `train_bpe_special`

    ポイントは最初の2行です。
    `input_text.split(end_token)` で文書ごとに分け、
    **それぞれを別々のID列として** 集計・マージします。
    """)
    return


@app.cell
def _(count_pairs_acc):
    def train_bpe_special(input_text, vocab_size, end_token="<|endoftext|>"):
        # 特殊トークンでテキストを分割（区切りをまたがせない）
        texts = input_text.split(end_token)
        ids_list = [list(text.encode("utf-8")) for text in texts]

        # 基本語彙(0-255) + 特殊トークン(1個) を除いた分がマージ回数
        num_merges = vocab_size - 256 - 1
        merge_rules = {}

        for step in range(num_merges):
            # 全文書ぶんの隣接ペアを1つの集計表にまとめる
            counts = None
            for ids in ids_list:
                counts = count_pairs_acc(ids, counts)

            if not counts:
                break

            best_pair = max(counts, key=counts.get)
            new_id = 256 + step
            merge_rules[best_pair] = new_id

            # 各文書にマージを適用
            for i in range(len(ids_list)):
                ids_list[i] = merge(ids_list[i], best_pair, new_id)

        return merge_rules

    return (train_bpe_special,)


@app.cell
def _(train_bpe_special):
    special_text = "Nago is a town.<|endoftext|>Yanbaru is a forest."
    special_rules = train_bpe_special(special_text, vocab_size=260)
    special_rules  # {(32, 105): 256, (256, 115): 257, (257, 32): 258}
    return (special_rules,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 特殊トークン対応のトークナイザー

    `encode` では `re.split` で `<|endoftext|>` を**保持したまま**分割し、
    特殊トークンに出会ったら専用IDを割り当てます。

    /// note | `re.escape` の小ネタ
    `<|endoftext|>` には正規表現の特殊文字 `|` が含まれます。
    `re.escape(...)` でエスケープしてから使うと、文字通りの文字列として安全に扱えます。
    ///
    """)
    return


@app.cell
def _():
    import re as _re

    class SpecialBPETokenizer:
        def __init__(self, merge_rules, end_token="<|endoftext|>"):
            self.merge_rules = merge_rules
            self.end_token = end_token
            # 特殊トークンのIDは語彙のいちばん最後に予約
            self.end_token_id = 256 + len(merge_rules)

            self.id_to_bytes = {i: bytes([i]) for i in range(256)}
            for (id1, id2), new_id in merge_rules.items():
                self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
            self.id_to_bytes[self.end_token_id] = self.end_token.encode("utf-8")

            self.vocab_size = len(self.id_to_bytes)

        def _encode_text(self, text):
            ids = list(text.encode("utf-8"))
            for merge_pair, new_id in self.merge_rules.items():
                ids = merge(ids, merge_pair, new_id)
            return ids

        def encode(self, input_text):
            # 特殊トークンを「保持したまま」分割する
            pattern = "(" + _re.escape(self.end_token) + ")"
            texts = _re.split(pattern, input_text)

            all_ids = []
            for text in texts:
                if text == self.end_token:
                    all_ids.append(self.end_token_id)
                else:
                    all_ids.extend(self._encode_text(text))
            return all_ids

        def decode(self, ids):
            byte_list = [self.id_to_bytes[i] for i in ids]
            return b"".join(byte_list).decode("utf-8", errors="replace")

    return (SpecialBPETokenizer,)


@app.cell
def _(SpecialBPETokenizer, special_rules):
    special_tokenizer = SpecialBPETokenizer(special_rules)

    special_ids = special_tokenizer.encode("Nago is a town.<|endoftext|>")
    print("encode :", special_ids)
    print("decode :", special_tokenizer.decode(special_ids))
    print("特殊トークンID :", special_tokenizer.end_token_id)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1-6. 事前トークン化（pre-tokenize）

    バイト単位の BPE には、ある弱点があります。
    放っておくと **`" the"` と `"."` のような無関係な塊** までマージしてしまい、
    `" the."` のような変なトークンが生まれることがあります。

    そこで GPT-2 では、まず**正規表現でテキストを「単語っぽい単位」にざっくり区切って**から
    BPE をかけます。これを **事前トークン化（pre-tokenize）** と呼びます。

    /// note | この正規表現が拾うもの
    - `'s` `'re` などの短縮形
    - `▁word`（先頭スペース付きの単語のかたまり）
    - 数字のかたまり、記号のかたまり、空白のかたまり

    `\p{L}`（任意の言語の文字）や `\p{N}`（数字）は、標準の `re` では使えないため
    高機能な **`regex`** ライブラリを使います。
    ///
    """)
    return


@app.cell
def _():
    import regex as regexlib

    def pretokenize(text):
        # GPT-2 で使われている正規表現パターン
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        return regexlib.findall(pattern, text)

    return (pretokenize,)


@app.cell
def _(pretokenize):
    pretokenize("Visit Yanbaru! Love Yanbaru? It's 2024.")
    # ['Visit', ' Yanbaru', '!', ' Love', ' Yanbaru', '?', ' It', "'s", ' 2024', '.']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 事前トークン化つきの学習とトークナイザー

    学習・エンコードのどちらでも、まず `pretokenize` で小片に割ってから BPE を適用します。
    こうすると **単語の境界を尊重した** 自然なトークンが学習されます。

    `tqdm` は進捗バーを表示してくれるライブラリです（学習に時間がかかるときに便利）。
    """)
    return


@app.cell
def _(count_pairs_acc, pretokenize):
    from tqdm.auto import tqdm as _tqdm

    def train_bpe_pretok(input_text, vocab_size, end_token="<|endoftext|>"):
        # ① 特殊トークンで分割 → ② 各片を事前トークン化 → ID列に
        texts = input_text.split(end_token)
        ids_list = []
        for text in texts:
            for pretoken in pretokenize(text):
                ids_list.append(list(pretoken.encode("utf-8")))

        num_merges = vocab_size - 256 - 1
        merge_rules = {}

        for step in _tqdm(range(num_merges), desc="Training BPE"):
            counts = None
            for ids in ids_list:
                counts = count_pairs_acc(ids, counts)
            if not counts:
                break

            best_pair = max(counts, key=counts.get)
            new_id = 256 + step
            merge_rules[best_pair] = new_id

            for i in range(len(ids_list)):
                ids_list[i] = merge(ids_list[i], best_pair, new_id)

        return merge_rules

    return (train_bpe_pretok,)


@app.cell
def _(pretokenize):
    import re as _re2
    import pickle as _pickle

    class BPETokenizer:
        """事前トークン化 + 特殊トークンに対応した実用版BPEトークナイザー"""

        def __init__(self, merge_rules, end_token="<|endoftext|>"):
            self.merge_rules = merge_rules
            self.end_token = end_token
            self.end_token_id = 256 + len(merge_rules)

            self.id_to_bytes = {i: bytes([i]) for i in range(256)}
            for (id1, id2), new_id in merge_rules.items():
                self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
            self.id_to_bytes[self.end_token_id] = self.end_token.encode("utf-8")

            self.vocab_size = len(self.id_to_bytes)

        @staticmethod
        def load_from(filepath):
            with open(filepath, "rb") as f:
                merge_rules = _pickle.load(f)
            return BPETokenizer(merge_rules)

        def _encode_text(self, text):
            ids = list(text.encode("utf-8"))
            for merge_pair, new_id in self.merge_rules.items():
                ids = merge(ids, merge_pair, new_id)
            return ids

        def encode(self, input_text):
            pattern = "(" + _re2.escape(self.end_token) + ")"
            texts = _re2.split(pattern, input_text)

            all_ids = []
            for text in texts:
                if text == self.end_token:
                    all_ids.append(self.end_token_id)
                else:
                    for pretoken in pretokenize(text):  # 事前トークン化してから
                        all_ids.extend(self._encode_text(pretoken))
            return all_ids

        def decode(self, ids):
            byte_list = [self.id_to_bytes[i] for i in ids]
            return b"".join(byte_list).decode("utf-8", errors="replace")

    return (BPETokenizer,)


@app.cell
def _(BPETokenizer, train_bpe_pretok):
    pretok_text = "Visit Yanbaru! Love Yanbaru! Enjoy Yanbaru.<|endoftext|>Good morning!"
    pretok_rules = train_bpe_pretok(pretok_text, vocab_size=270)
    pretok_tokenizer = BPETokenizer(pretok_rules)

    pretok_ids = pretok_tokenizer.encode("Visit Yanbaru!")
    print("encode :", pretok_ids)
    print("decode :", pretok_tokenizer.decode(pretok_ids))

    # 各トークンが「どんな文字列か」を確認
    print("\n--- トークンの中身 ---")
    for token_id in pretok_ids:
        print(f"{token_id} -> {pretok_tokenizer.decode([token_id])!r}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1-7. 実データで学習して保存する（pickle）

    ここまでの仕組みを **実際のソースコード** で学習してみます。
    本書では数百万行のデータ（`tiny_codes.txt`）を `vocab_size=1000` で学習しますが、
    ノートブックでは数秒で終わるよう **小さなサンプル** を使います。

    /// note | `pickle` とは
    Python のオブジェクト（ここでは `merge_rules` という辞書）を、
    **そのままファイルに保存・復元** できる仕組みです。
    一度学習すれば、次回からは保存したルールを読み込むだけで使えます。
    `"wb"` は write-binary（バイナリ書き込み）モードです。
    ///
    """)
    return


@app.cell
def _(BPETokenizer, mo, train_bpe_pretok):
    import pickle

    # 軽量サンプル（実データの先頭60KB）を読み込み
    with open("data/sample_codes.txt", "r", encoding="utf-8") as _f:
        code_text = _f.read()

    # 学習（ノートブック用に vocab_size=400／数秒で完了）
    code_merge_rules = train_bpe_pretok(code_text, vocab_size=400)

    # 学習結果をファイルに保存
    with open("data/merge_rules.pkl", "wb") as _f:
        pickle.dump(code_merge_rules, _f)

    code_tokenizer = BPETokenizer(code_merge_rules)
    mo.md(
        f"""
    - 学習テキスト: **{len(code_text):,} 文字**
    - マージ回数: **{len(code_merge_rules)}**
    - 語彙サイズ: **{code_tokenizer.vocab_size}**
    - 保存先: `data/merge_rules.pkl`
    """
    )
    return (code_text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1-8. 評価：学習されたトークンと圧縮率

    保存したルールを `load_from` で読み込み直して、

    1. **どんなトークンが学習されたか**（最初の方／最後の方）
    2. **圧縮率**（バイト数 ÷ トークン数 ＝ 1トークンあたり何バイトか）

    を確認します。圧縮率が大きいほど「文章を短いトークン列で表せている」＝効率的です。
    """)
    return


@app.cell
def _(BPETokenizer, code_text):
    # ファイルから読み込み直す（1-7とは独立して使える）
    loaded_tokenizer = BPETokenizer.load_from("data/merge_rules.pkl")

    print("=== 最初に学習された10個（頻出＝短い塊） ===")
    for tid in range(256, 266):
        print(f"  ID {tid}: {loaded_tokenizer.id_to_bytes[tid].decode('utf-8', errors='replace')!r}")

    last_id = 255 + len(loaded_tokenizer.merge_rules)
    print("\n=== 最後に学習された10個（より長い塊） ===")
    for tid in range(last_id - 9, last_id + 1):
        print(f"  ID {tid}: {loaded_tokenizer.id_to_bytes[tid].decode('utf-8', errors='replace')!r}")

    # 圧縮率
    sample = code_text[:10000]
    byte_count = len(sample.encode("utf-8"))
    ids_count = len(loaded_tokenizer.encode(sample))
    print("\n=== 圧縮効率 ===")
    print(f"バイト数  : {byte_count:,}")
    print(f"トークン数: {ids_count:,}")
    print(f"圧縮率    : {byte_count / ids_count:.2f} バイト/トークン")
    return (loaded_tokenizer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | 本物の GPT トークナイザーと比べると
    `tiktoken` ライブラリを使うと、GPT-2 や GPT-4 のトークナイザーと圧縮率を比較できます。
    語彙サイズが大きい（GPT-2 は約5万、cl100k_base は約10万）ほど圧縮率は上がります。

    ```python
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")        # 語彙 50,257
    enc = tiktoken.get_encoding("cl100k_base") # 語彙 100,277（GPT-4）
    len(enc.encode(text))
    ```
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1-9. 全文を数値ファイルに変換する（numpy）

    LLM の学習では、テキストを毎回エンコードし直すのは無駄です。
    そこで **全文を一度トークンIDに変換し、数値配列としてファイル保存** しておきます。

    /// note | なぜ `uint16`？
    - `uint16` は **0〜65535** を表せる16bit整数。
    - 語彙サイズが 65536 未満なら、すべてのトークンIDがこの型に収まります。
    - `int64` の **1/4 のファイルサイズ** で済むので、大規模データで効きます。

    `.tofile()` で生のバイナリとして保存し、学習時は `np.fromfile()` で高速に読み込めます。
    ///
    """)
    return


@app.cell
def _(code_text, loaded_tokenizer, mo):
    import numpy as np

    # 全文をトークンIDへ
    all_ids = loaded_tokenizer.encode(code_text)

    # numpy配列(uint16)にしてバイナリ保存
    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile("data/sample_codes.bin")

    mo.md(
        f"""
    - トークンID数: **{len(ids_array):,}**
    - 最初の20個: `{ids_array[:20].tolist()}`
    - 保存先: `data/sample_codes.bin`（{ids_array.nbytes:,} バイト）
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 第1章のまとめ

    | 段階 | 何ができるようになったか |
    |------|--------------------------|
    | 1-1 文字 | 文字 ⇄ 数字（`ord`/`chr`） |
    | 1-2 バイト | UTF-8 で何でも256語彙に |
    | 1-3 BPE学習 | 頻出ペアを繰り返しマージ |
    | 1-4 BPE | 学習済みルールで encode/decode |
    | 1-5 特殊トークン | `<|endoftext|>` で文書を区切る |
    | 1-6 事前トークン化 | 単語境界を尊重したトークン |
    | 1-7 保存 | pickle で学習結果を再利用 |
    | 1-8 評価 | 学習トークンと圧縮率の確認 |
    | 1-9 数値化 | numpy で全文を `.bin` に |

    これで **テキスト → トークンID列** への変換器が完成しました。

    /// tip | 次の章へ
    第2章では、このトークンIDを受け取って「次のトークン」を予測する
    **GPT（Transformer）本体** を、Attention から組み立てていきます。
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
