"""circuit.py: M3 的电路描述 —— graph.py 之上的语义层。

graph.py 只说"算什么"（FFN / 状态表 / taps），这个文件说"一条 RV32I 指令在 token 流里
长什么样"。

**token 不是一拍，是一条指令的产出记录。** 一条指令执行完，电路产出一个 record：写回的
值（`F0` / `V F1` / `V V F2`）、或空拍 `NT`、或跳转 `J`、或存储 `ST0/ST1/SS0/SS1`、或
`BYTE`（putchar）、或 `HALT`。记录的长短随指令而变，1 到 4 个 token 不等——`V/F0/F1/F2`
不是"分片"，是**一个 32 位值的 1/2/3 token 编码**（F0 一个 token 就装得下整个值）。

家族表照抄 Qwenputer 的 `chipformer/rv/spec.py`（协议层，不是 LLM 组件）：12 个数据家族
带载荷 + 8 个控制无载荷，共 230324 个 token；本机再加 `TX`（文本 token，载荷 = **这段
文本里的序号**，也就是行表下标）与 `TCO`/`TCC`（`<cpu_call>` 的两个边界），共 246710 个。

**PC 不存储。** `PC = ROM_BASE + 4·n`：`n` 是这一条指令在代码段里的**字索引**，一张
1 行 `log2(nrom)` 列的表。取指把 n 的二进制位用 `and_` 解成一行 one-hot，ROM 表按该行
读——地址解码 = AND（§5 #6）。Qwenputer 存的是 `anchor + n`（n 限 0..63，到 63 先来一个
`R` 刷新）；这里直接存绝对索引，因为 `J`/`R` 的载荷本来就有 16 位、`ROM_WORDS = 1 << 16`
恰好装得下整个代码段，跳转就是「把载荷写进 n」，anchor 与 R 刷新在这个形状下是多余的。

**家族宽度是外部给定的，不是估的。** 家族表、取指、寄存器堆里没有任何一条指令的名字：
写 8 条指令和写 52 条指令，这三段一行都不用改。ALU 是唯一按“运算类别”（加减/逻辑/移位/
比较）组织的地方——新增一个指令落在现成的类别里，就只是 ROM 字里的一个 uop 编号。

**算术全在 0/1 位上进行。** bf16 能精确表示的整数上界是 256，所以寄存器堆是 32 行 × 32 位
的位平面，加法器是 8 个 nibble 的串行进位：每个中间量要么是 0/1，要么是 ≤31 的 nibble 和。
整数 z 的比较用两个 relu 的差（`relu(z-t+1) - relu(z-t)` 对整数恒为 0/1），
所以“地址解码 = AND、比较 = 两个 relu”这套写法本身就把 float64 挡在门外。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from components.rv32i.constant import (CAP_MAX, COLS, CONTROL_NAMES, CONTROLS,
                                      DROM_BANKS, DROM_BASE, DROM_COLS, DROM_ROWS, FAM_ID,
                                      FAMILIES, FAMILY_SIZES, FINAL, LINK_BITS, NAMES, NFAM,
                                      NPAY, PAGE_BITS, PAGE_LO, PAGE_ROWS, RAM_BANKS, RAM_COLS,
                                      RAM_ROWS, RAM_WORDS, REGBITS, REGS, ROM_BASE, ROM_WORDS,
                                      TEXT_BASE, TEXT_ROW, TEXT_ROWS, TEXT_VOCAB, TEXT_WORDS,
                                      VOCAB)
from components.rv32i.graph import Chan, Expr, Graph

# 这些名字的真源在 `constant.py`，本模块把它们再导出一次：探针一直在用 `cc.REGS` /
# `cc.PAGE_ROWS` / `cc.TEXT_ROWS` 这类写法，不要因为“模块内未引用”就删。

# ---------------- token 层 ----------------

#: 家族表在 `constant.py`；这里只做两件特有的事：展平成 `_FAMID_OF` / `_PAY_OF`
#: 两张长度 `VOCAB` 的表（`_tables`），以及按家族块预展开 8 个派生标志位（`_FLAG`）。


def _tables() -> tuple[list[int], list[int]]:
    """token id -> (家族序号, 载荷)，两张长度 VOCAB 的常量表。

    展平而不是按位打包，是为了让"家族"和"载荷"在电路里都是常量表：编译期对整张
    词表枚举一次，运行期只剩矩阵乘。代价是 vocab 246710 而不是 2^w 的紧凑编码。
    """
    famid, pay = [], []
    for i, (_, size) in enumerate(FAMILY_SIZES):
        famid += [i] * size
        pay += list(range(size))
    for i in range(len(CONTROL_NAMES)):
        famid.append(len(FAMILY_SIZES) + i)
        pay.append(0)
    return famid, pay


_FAMID_OF, _PAY_OF = _tables()
assert len(_FAMID_OF) == VOCAB

#: 派生标志位，按 token 成块预展开——都是 0/1，bf16 精确覆盖。
_FLAGS = ("fin", "val", "jr", "st", "byte", "halt", "q", "diag")


def _flag_of(name: str, fam: str) -> float:
    return float({
        "fin": fam in FINAL,
        "val": fam in ("V", "F0", "F1", "F2"),
        "jr": fam in ("J", "R"),
        "st": fam in ("ST0", "ST1", "SS0", "SS1"),
        "byte": fam == "BYTE",
        "halt": fam == "HALT",
        "q": fam in ("QS", "QD"),
        "diag": fam in ("FLT", "HEX", "BD", "SC"),
    }[name])


_FLAG: dict[str, list[float]] = {k: [_flag_of(k, NAMES[f]) for f in _FAMID_OF] for k in _FLAGS}


def tok(fam: str, pay: int = 0) -> int:
    """(家族, 载荷) -> token id。控制 token 的载荷恒 0。"""
    if fam in CONTROLS:
        if pay:
            raise ValueError(f"{fam} 是控制 token，没有载荷")
        return CONTROLS[fam]
    base, size = FAMILIES[fam]
    if not 0 <= pay < size:
        raise ValueError(f"{fam} 的载荷 {pay} 超出 [0, {size})")
    return base + pay


def split(tid: int) -> tuple[str, int]:
    """token id -> (家族, 载荷)。"""
    if not 0 <= tid < VOCAB:
        raise IndexError(f"token {tid} 超出 vocab {VOCAB}")
    return NAMES[_FAMID_OF[tid]], _PAY_OF[tid]


# ---------------- 文本缓冲（`<cpu_call>` 的输入侧） ----------------
#
# 地址与行宽在 `constant.py`；行号 **1-based**（第 k 个捕获的 token 写第 k 行，第 0 行空着）。


# ---------------- graph 特征 ----------------

def token_features(g: Graph, text: list[bytes] | None = None) -> dict[str, Expr]:
    """家族 one-hot（23）+ 载荷位（16）+ 派生标志（8），共 47 个 token dim。

    它们是 token id 的纯函数，编译期枚举成 feat 表。**没有任何一项是给特定指令用的**：
    电路不认识"这条是 ADD"，它只看 47 个标量。加指令 = 改别处的表达，不改这里。

    `text` 给定时再挂 `TEXT_ROW × 8` 个文本字节位（`tb[k][i]` = 第 k 个字节的第 i 位），
    共 175 个。它是**内容无关**的：`text[id]` 是**词表里第 id 个 token** 的 16 个字节
    （`pytext.vocab_rows`），换一段代码一个字都不改，电路一行也不改（Qwenputer 的
    `text.row_bytes` 是同一件事）。为 None 时不建那 128 个 dim——无文本的机器不该为它付宽度。

    `id` 就是 `TX` 的**载荷**（`flow.call` 发的 tokenizer token id），所以一条 `call` 里每个
    TX 各带各的字节，而电路只有这一张表；写第几行不在这里决定，由 `text_buffer` 的 `cnt`
    自己数。**表按 token id 排而不是按“这段文本里第几个”排**，这是“一份权重收所有程序”
    的那一半（另一半是 `flow.call` 发 id）。
    """
    d: dict[str, Expr] = {}
    for i, n in enumerate(NAMES):
        d[f"fam.{n}"] = g.token_feature(f"fam.{n}", lambda t, i=i: float(_FAMID_OF[t] == i))
    for b in range(NPAY):
        d[f"pay[{b}]"] = g.token_feature(f"pay[{b}]", lambda t, b=b: float((_PAY_OF[t] >> b) & 1))
    for k, col in _FLAG.items():
        d[k] = g.token_feature(k, lambda t, col=col: col[t])
    if text is not None:
        tx = FAM_ID["TX"]

        def tbit(t: int, k: int, i: int) -> float:
            if _FAMID_OF[t] != tx:
                return 0.0
            p = _PAY_OF[t]
            row = text[p] if p < len(text) else b""
            return float((row[k] >> i) & 1) if k < len(row) else 0.0

        for k in range(TEXT_ROW):
            for i in range(8):
                d[f"tb[{k}][{i}]"] = g.token_feature(
                    f"tb{k}.b{i}", lambda t, k=k, i=i: tbit(t, k, i))
    return d


# ---------------- 状态层 ----------------

def cnt_width(nrom: int) -> int:
    """装得下 n ∈ [0, nrom) 的位数。"""
    return max(1, (nrom - 1).bit_length())


def pc_anchor(g: Graph, nrom: int, init_n: int = 0,
              feats: dict[str, Expr] | None = None, out: dict | None = None) -> list[Expr]:
    """`cnt`：指令序号 n 的二进制位，一张 1 行 ncnt 列的表，**带推进写口**。

    **PC 不存储**。`PC = ROM_BASE + 4·n`，而 n 是这一张表的内容，所以电路里既没有
    PC 寄存器，也没有 4 倍缩放——取指时直接把 n 的位解码成行选择，ROM 表和 RAM 表
    共用同一行。这样跳转就是"把 n 改成目标"，而不是"改一个 32 位数"。

    位数用二进制而不是 one-hot：one-hot 要 nrom 个 dim（3000 条指令就是 3000 维），
    二进制只要 log2(nrom)（12 维）。代价是每读一行要过一次 and_ 解码链，而那恰好是
    地址解码该做的事。

    **写口是滞后一拍的，写值和写门两处都是**：写值读 `cnt.next` 的 taps，写门由 `k`
    （`fin` 的 taps）担任。**写门不能用 `beta`**：`beta` 只收同拍的 `Expr`，而 `k` 是
    `Chan`，taps 本来就允许（寄存器堆的写行就是这么拿 `rd` 的）。两个理由：

      * `eval_head` 的次序是「先写后读」，同拍写进去的值在本拍就生效。而本拍的取指
        读的就是这张表，不滞后的话推进在本拍生效，本拍取到的是**下一条**指令
      * `cnt.next = jr ? 载荷 : cnt + 1` 要用到本拍的读出（`n + 1` 的进位链），同拍
        写回来就是环，`check_dag` 会直接报

    滞后一拍之后，推进落在记录终结拍的**下一拍**——也就是下一条指令的第一个 token 上，
    记录期间 `cnt` 一直是本条指令的行号，写行（`rd`，同样滞后一拍）也就指的仍是本条
    指令。`J`/`R` 的载荷是目标字索引，其余终结拍是 `n + 1`；`V` 不算终结拍，所以一条
    记录内部无论几段都不推进。

    另一个副作用是好的：t=0 没有历史，写门为 0，`init_n` 不被冲掉。

    `out` 拿到 `nxt` / `inc` / `carry` 三个中间量：`inc` 是 `cnt + 1` 的位（`carry` 是
    它的溢出），链接值（`link_word`）就从这两个算出来——所以链接值和 `cnt` 是同一层
    的邻居，比寄存器堆的写口早十几层，能直接写进去（见 `link_word`）。
    """
    ncnt = cnt_width(nrom)
    feats = token_features(g) if feats is None else feats
    nxt = g.zeros(ncnt, "cnt.next")         # 写值：下一拍的 n（滞后一拍喂回）
    ret = g.head(
        "cnt", "cnt", rows=1, cols=ncnt,
        q={0: g.one},
        k={0: Chan(feats["fin"], (0.0, 1.0, 0.0, 0.0))},     # 上一拍是终结拍才写
        v={c: Chan(nxt[c], (0.0, 1.0, 0.0, 0.0)) for c in range(ncnt)},
        a=0, z={c: g.one for c in range(ncnt)},
        outs=[(c, 1.0) for c in range(ncnt)],
        init=[{0: [float((init_n >> c) & 1) for c in range(ncnt)]}],
        levels="bit",
    )
    # n + 1：进位链，最低位进 1。异或 = 两个「恰好一个为 1」的 AND 之和
    inc, carry = [], g.one
    for c in range(ncnt):
        inc.append(_node(g, _t_and([1 - ret[c], carry]) + _t_and([ret[c], 1 - carry]),
                         f"cnt.inc{c}"))
        carry = _node(g, _t_and([ret[c], carry]), f"cnt.carry{c}")
    for c, terms in enumerate(mux_bits(feats["jr"], [feats[f"pay[{c}]"] for c in range(ncnt)],
                                       inc)):
        _node(g, terms, f"cnt.next{c}", into=nxt[c])
    if out is not None:
        out.update(nxt=nxt, inc=inc, carry=carry)
    return ret


def link_word(g: Graph, inc: list[Expr], carry: Expr, base: int = 0,
              name: str = "link") -> list[Expr]:
    """`jal`/`jalr` 的链接值：`(PC + 4) >> 2` 的 18 位。

    `PC = ROM_BASE + 4·(base + cnt)`，于是
    `(PC + 4) >> 2 = ROM_BASE/4 + base + cnt + 1 = 0x10000 + base + cnt + 1`。低两位恒 0
    （PC 一定是 4 的倍数），所以这里只算 18 位，写回时直接落在寄存器的 bit2..bit19
    （`regs` 的 `wcols` 第一项就是目标位号，不必和列号一致）。

    **这一路真的在电路里算**，不是从 token 载荷来：入参只有 `cnt + 1`（`pc_anchor`
    已经算好的进位链 `inc`/`carry`）和一个常数。两个后果：

      * `J` 拍自己就能把链接值写进 rd，记录里不再需要一段值载荷（所以 `J` 只有 1 个 token）
      * 它比寄存器堆的写口早十几层（`cnt` 的读出在 level 2，写口在 12 以后）

    **不写行波加法，也不写阈值扫**。行波每加一位多两级，20 位会把整张图从 19 层顶到 33 层
    （实测）；阈值扫的翻点数是 `2^len(cnt)`，一页 512 行就是每位 511 次比较、总共一万多个
    unit。这里拆开看：`cnt + 1` 的低 `len(inc)` 位**就是 `inc`**，溢出只有 `carry` 一位，
    高位部分是常数 `(ROM_BASE >> 2) + (base >> len(inc))` 再叠这一位进位——每一位要么是
    常数、要么是常数和 `carry` 的二选一，全部落在同一层、总共不到十个 unit。

    `base` 必须是 `2^len(inc)` 的倍数（页内口径下是 `页号 × 页大小`），否则进位会从页号
    渗进低位，那个位置的进位链就没算了。`ROM_BASE/4 = 0x10000` 也是个 2 的幂，
    只要 `2^len(inc) <= 0x10000`（即 cnt 不超过 16 位，而 `ROM_WORDS = 1 << 16`）就同样对齐。
    """
    ncnt = len(inc)
    if ncnt > LINK_BITS:
        raise ValueError(f"{name}: cnt 有 {ncnt} 位，链接值只留了 {LINK_BITS} 位")
    if base % (1 << ncnt) or (ROM_BASE >> 2) % (1 << ncnt):
        raise ValueError(f"{name}: base {base} / ROM_BASE 不是 2^{ncnt} 的倍数，进位会渗进低位")
    # 高位 = ROM_BASE/4 的高位 + 页号 + (cnt+1 的溢出)：三者都是常数，只有最后一项在变
    top = ((ROM_BASE >> 2) >> ncnt) + (base >> ncnt)
    if top < 0 or top + 1 >= (1 << (LINK_BITS - ncnt)):
        raise ValueError(f"{name}: 常数项 {top:#x} 装不进 {LINK_BITS - ncnt} 位")
    w = list(inc)
    for i in range(LINK_BITS - ncnt):
        x, y = (top >> i) & 1, ((top + 1) >> i) & 1
        if x == y:
            w.append(g.one if x else Expr())
        else:
            w.append(_node(g, _t_and([carry]) if y else _t_const(1) + _t_and([carry], -1),
                           f"{name}.hi{i}"))
    return w[:LINK_BITS]


def row_onehot(g: Graph, bits: list[Expr], r: int, tag: str = "row") -> Expr:
    """第 r 行的 one-hot 选择子：把 r 的每个二进制位按值 AND 起来。

    地址解码全用它（§5 #6）。`and_` 的输出只会是 0/1，不经过标量比较、不需要 float64
    ——Qwenputer 那版用标量二阶差分算 one-hot，就必须 float64 才能精确，这是本机
    能把地址域压进 bf16 的原因。

    **一个 unit 把 n 个字面量全部 AND 掉**，不是逐位链。`_t_and` 的写法
    `relu(4·sum - (4n-2))` 对任意 n 都只能输出 0/1（全 1 时是 2、否则 ≤ -2），所以链式
    没有任何好处：n 位链要 n 个 unit、**n 层深**，一个 unit 只要 1 层。RAM 的行选择
    接在 ALU 后面，这 6 层的差别直接把整张图从 23 层推到 35 层。
    """
    if not bits:
        raise ValueError("row_onehot 至少要一位")
    if not 0 <= r < (1 << len(bits)):
        raise ValueError(f"行号 {r} 超出 {len(bits)} 位的范围")
    return _and(g, [b if (r >> i) & 1 else g.one - b for i, b in enumerate(bits)],
                f"{tag}[{r}]")


def row_sel(g: Graph, bits: list[Expr], rows: int, tag: str = "row",
            group: int | None = None) -> list[Expr]:
    """把 [0, rows) 全部解成 one-hot 行选择子（每个一行的一个 dim）。

    **分两组，不是每行一条长链**：低的 g 位先解成 2^g 个 one-hot，高的 n-g 位再解成
    2^(n-g) 个，最后逐行相乘。4096 行 /12 位时是 64 + 128 + 4096 = 4288 个 unit，
    而每行一条 12 位长链要 45056 个——差一个数量级。这就是 plan §4 M3 第 2 步说的
    “高位/低位各一组 one-hot，用 and_ 相乘”。

    行数不是 2 的幂也照做：多出来的行号解出来了但没人读，它们只是几个恒 0 的 dim。
    """
    nb = len(bits)
    if rows > (1 << nb):
        raise ValueError(f"{rows} 行装不进 {nb} 位")
    gl = max(0, min(group if group is not None else 6, nb))
    lo_n, hi_n = 1 << gl, 1 << (nb - gl)
    lo = [row_onehot(g, bits[:gl], p, f"{tag}.lo") for p in range(lo_n)] if gl else [g.one]
    hi = ([row_onehot(g, bits[gl:], h, f"{tag}.hi") for h in range(hi_n)]
          if nb - gl else [g.one])
    out = []
    for r in range(rows):
        a, b = lo[r & (lo_n - 1)], hi[r >> gl]
        out.append(a if b is g.one else b if a is g.one else g.and_(a, b, f"{tag}[{r}]"))
    return out


def andpre(lits: list[Expr]) -> Expr:
    """全 1 时恰好 1、否则 <= 0 的**线性**形式（Qwenputer 的 `andpre`）。

    与 `_and` 的唯一区别是**不建节点**：不花 dim、不花 unit、不占一层。代价是这个
    线性组合的系数要进下游 head 的输入投影（`q`/`k`/`z` 本来就是线性投影，所以不收钱）。

    成立的唯一前提是“输入 0/1、下游会削负”：本机 head 的 `q`/`k` 走 `conv(gate=True)`
    的 relu、`z` 走 `np.maximum(zz, 0)`，两处都把 <= 0 精确削成 0。而 andpre 在 0/1
    输入下的取值只能是整数 `{1, 0, -1, ...}`，削完正好 {0, 1}——**不需要 bf16 之外的
    任何精度假设**。这正是它和“用标量比较算 one-hot”（必须 float64）的分水岭。

    常数项 `-(n-1)` 落在 dim 0（`one`），与 `_t_const` 同源。
    """
    if not lits:
        raise ValueError("andpre 至少要一个字面量")
    return sum(lits, Expr()) - (len(lits) - 1)


def row_lits(bits: list[Expr], rows: int, tag: str = "row") -> list[Expr]:
    """`row_sel` 的线性版：每行一个 `andpre`，**一个 dim 都不建**。

    形状上的差别是数量级的：201 页 × 256 行的 ROM 用 `row_sel` 是 256 行选择 + 64 低位
    + 4 高位 = 324 个 dim，用本函数是 0。`row_sel` 保留给“选择子要被别人当 dim 复用”的
    地方（RAM 的 512 行要借给 DROM），两者各有各的用途。

    `tag` 只用于报错，不进 dim 名（没有 dim 可进）。
    """
    nb = len(bits)
    if rows > (1 << nb):
        raise ValueError(f"{tag}: {rows} 行装不进 {nb} 位")
    return [andpre([b if (r >> i) & 1 else 1 - b for i, b in enumerate(bits)])
            for r in range(rows)]


# ---------------- 取指 ----------------
#
# `ROM_BASE` / `ROM_WORDS` / `PAGE_*` / `COLS` 与 `PAGE_BITS = 8` 的由来都在 `constant.py`。

# ---------------- 预译码：指令字 -> ROM 字 ----------------

#: uop 编号完全照拄 Qwenputer 的 `ref.UOPS`——电路按这个编号解码，顺序不能动。
#: `misa` 占一个位置但 MiniRain 没有 CSR 指令，所以永远不会被生成。
UOP_ORDER = ("illegal lui auipc misa "
             "addi slti sltiu xori ori andi slli srli srai "
             "add sub sll slt sltu xor srl sra or and "
             "lb lh lw lbu lhu sb sh sw "
             "beq bne blt bge bltu bgeu jal jalr "
             "ecall ebreak fence "
             "mul mulh mulhsu mulhu div divu rem remu "
             "sh1add sh2add sh3add").split()
UOP = {n: i for i, n in enumerate(UOP_ORDER)}

_IMM_OPS = {"addi", "slti", "sltiu", "xori", "ori", "andi", "slli", "srli", "srai"}
_LOAD_OPS = {"lb", "lh", "lw", "lbu", "lhu"}
_STORE_OPS = {"sb", "sh", "sw"}
_BRANCH_OPS = {"beq", "bne", "blt", "bge", "bltu", "bgeu"}


def _pack(uop: str, rd: int = 0, rs1: int = 0, rs2: int = 0, spare: int = 0) -> int:
    """uop + 三个寄存器槽 + 10 个 spare 位 -> ROM 字。

    格式（Qwenputer `ref` 模块 docstring）：`b0 = uop | S0 << 7`、`b1 = RD << 3 | S1..S3`、
    `b2 = RS1 << 3 | S4..S6`、`b3 = RS2 << 3 | S7..S9`。每字节一个字段，电路一拍解一层。
    静态事实折进去：寄存器号已在字节里，立即数拆到 spare，所以电路不需要再分离字段。
    """
    b0 = UOP[uop] | (spare & 1) << 7
    b1 = rd << 3 | (spare >> 1 & 7)
    b2 = rs1 << 3 | (spare >> 4 & 7)
    b3 = rs2 << 3 | (spare >> 7 & 7)
    return b0 | b1 << 8 | b2 << 16 | b3 << 24


def unpack(rw: int) -> tuple[str, int, int, int, int]:
    """ROM 字 -> (uop, rd, rs1, rs2, spare)。`_pack` 的逆，用来验证打包无损。"""
    b0, b1, b2, b3 = rw & 0xFF, rw >> 8 & 0xFF, rw >> 16 & 0xFF, rw >> 24 & 0xFF
    spare = (b0 >> 7 & 1) | (b1 & 7) << 1 | (b2 & 7) << 4 | (b3 & 7) << 7
    return UOP_ORDER[b0 & 0x7F], b1 >> 3, b2 >> 3, b3 >> 3, spare


def _in_rom(t: int) -> bool:
    return t % 4 == 0 and ROM_BASE <= t < ROM_BASE + 4 * ROM_WORDS


def predecode(word: int, pc: int) -> int:
    """一条 RV32I 指令字 -> 电路直接读的 ROM 字。

    区间拆解（照 Qwenputer 的 `ref.rom_word`）：

      R 型（ALU/MUL/DIV/Zba）      RD = rd, RS1 = rs1, RS2 = rs2
      I 型（ALU imm/load/JALR）    RD = rd, RS1 = rs1, imm12 = RS2(0..4) + S0..S6(5..11)
      存储                           RD = 数据寄存器, RS1 = 基址, imm12 同 I 型
      分支                          RS1, RS2；目标是**有符号字偏移**：RD(0..4) + S0..S5(5..10)，
                                    S6 = 目标非法（非对齐 / 不在 ROM）
      JAL                           RD = rd；目标是 ROM 字索引：RS1(0..4) + RS2(5..9) + S0..S5(10..15)，
                                    S6 = 非法
      LUI / AUIPC                   RD = rd；值的高 20 位：RS1(0..4) + RS2(5..9) + S0..S9(10..19)
      ECALL                         RS1 = 17 (a7), RS2 = 10 (a0)

    **分支与 JAL 的偏移在这里就算好了**，电路不需要做浮点加减：分支存字偏移、JAL 存绝对
    字索引，两者都在本函数里用 Python 整数算完。这是`rom_word` 存在的原因——把控制和
    地址算术从电路里拿走，电路只负责"查行 + 写回"。
    """
    from components.rv32i.isa import decode

    try:
        ins = decode(word)
    except ValueError:
        return _pack("illegal")
    op, imm = ins.name, ins.imm
    if op == "fence.i":
        op = "fence"
    if op in _IMM_OPS or op in _LOAD_OPS or op == "jalr":
        i = imm & 0xFFF
        return _pack(op, ins.rd, ins.rs1, i & 31, i >> 5)
    if op in _STORE_OPS:
        i = imm & 0xFFF
        return _pack(op, ins.rs2, ins.rs1, i & 31, i >> 5)
    if op in ("lui", "auipc"):
        # MiniRain 的 decode 给 U 型的是 20 位 raw（Qwenputer 给的是 32 位符号扩展），
        # 所以 LUI 直接取、AUIPC 先把 imm 挪到高 20 位再加上 pc 再取回。
        up = imm if op == "lui" else ((pc + (imm << 12)) & 0xFFFFFFFF) >> 12
        up &= 0xFFFFF
        return _pack(op, ins.rd, up & 31, up >> 5 & 31, up >> 10)
    if op in _BRANCH_OPS:
        t = (pc + imm) & 0xFFFFFFFF
        off = (imm >> 2) & 0x7FF          # 有符号字偏移，11 位
        bad = int(not _in_rom(t))
        return _pack(op, off & 31, ins.rs1, ins.rs2, off >> 5 | bad << 6)
    if op == "jal":
        t = (pc + imm) & 0xFFFFFFFF
        wi = (t - ROM_BASE) >> 2 & 0xFFFF
        bad = int(not _in_rom(t))
        return _pack(op, ins.rd, wi & 31, wi >> 5 & 31, wi >> 10 | bad << 6)
    if op == "ecall":
        return _pack(op, 0, 17, 10)
    if op in ("ebreak", "fence", "illegal"):
        return _pack(op)
    return _pack(op, ins.rd, ins.rs1, ins.rs2)    # R 型



def load_words(path, base: int = ROM_BASE, n: int | None = None, pre: bool = True) -> list[int]:
    """从 ELF 的代码段取 ROM 字，按 `base` 起算的字索引返回。

    `pre=True` 时逐条过 `predecode`（电路读的就是这个格式）；`pre=False` 拿原始指令字。
    索引的基准是**段首**不是 entry：entry 0x40000 就是索引 0，和 Qwenputer 的
    `anchor` 一致。字数由固件自己定（Qwenputer 的 51241、nucpu 的 34294 都在 16 位里）
    ——这就是 `J`/`R` 取 16 位载荷的依据，真机超过了这个宽度就得改载荷位宽。
    """
    import struct
    from pathlib import Path

    from components.rv32i.elf import parse_elf

    data = Path(path).read_bytes()
    elf = parse_elf(data)
    segs = [s for s in elf.segments if s.vaddr == base]
    if not segs:
        raise ValueError(f"{path}: 没有 vaddr == {base:#x} 的段（有 "
                         f"{[(hex(s.vaddr)) for s in elf.segments]}）")
    seg = segs[0]
    nw = seg.filesz // 4
    n = nw if n is None else min(n, nw)
    words = [struct.unpack_from("<I", data, seg.offset + 4 * i)[0] for i in range(n)]
    return [predecode(w, base + 4 * i) for i, w in enumerate(words)] if pre else words


def chunk_words(words: list[int], rows: int = PAGE_ROWS) -> list[list[int]]:
    """把固件按 `rows` 行切页（最后一页可能不满）。"""
    return [words[i:i + rows] for i in range(0, len(words), rows)] or [[]]


def rom(g: Graph, cnt: list[Expr], words: list[int],
        name: str = "rom") -> tuple[list[Expr], list[Expr]]:
    """ROM：`words` 是**整块固件**，按 512 行切页，一页一个 value head。

    **只读**：`k` 接恒 0 的占位，运行期永不写；`q` 是 `row_lits` 给出的行选择。
    读出就是被选中那一行的 32 个位——一次拿整条指令，不需要三拍 token。

    三十二列而不是一个「指令字」列：bf16 能精确表示的整数上界是 256，32 位字装不进
    一列，拆成位之后每个值都是 0/1，与 word 拆两列的思路同源，只是更彻底。

    **分页**：真固件 34294 条 = 134 页（旧 Qwenputer 固件 51241 条 = 201 页），一页只装
    `PAGE_ROWS` 行。页号在 `cnt` 的高位，页内行在低 `PAGE_BITS` 位。

    这里**不建每页一组并行输出列**——那是页 × 列 = 134 × 32 = 4288 个中间 dim，正好越过
    2048 的悬崖（`plan.md` §4 M4）。用的是 RAM 那一招：**一页一个 value head、共用一组 32 个
    输出位**，靠每页的 `z` 门控，只有页号命中的那一页往残差上落值，其余写 0、累加不变。
    整个 ROM 只有**一个 head 节点**，所以层数和 head 槽都不随页数涨。

    **行选择是线性的（`row_lits`/`andpre`），一个 dim 都不花**；页门仍是真 AND 节点，
    但不再经「页号低位 one-hot」中转。原来这里是 `row_sel` 的显式节点：256 行选择 +
    64 低位 + 4 高位 + 16 个页号 one-hot + 201 个页门 = 541 个 dim（旧固件的数）；现在是
    134 个页门节点，剩下那 340 个全消失，`cnt` 的位线直接进 head 的 q 投影。

    两边不对称是故意的：`q` 走 `conv(gate=True)` 的 relu（削负），共 `andpre` 合法；
    而 `z` 在 `eval_head` 里是**直接查 `isin(zz, (0,1))` 的乘数**（`np.maximum` 发生在
    乘法那一刻，不在取门那一刻），所以 `z` 必须精确 0/1，只能是 `_and` 节点。Qwenputer
    的 `z` 是 `K*gate` 的 pre-activation（过 silu），所以它连页门也能线性——本机要等
    宿主换门后再合。
    页号仍然只花 4 位：`lits_of(pg[:4], p & 15)` 给低位的字面量，高 3 位直接取 `cnt`
    已有的位线，两者拼成同一个 AND。

    返回 `(bit_dims, sel)`：`bit_dims[b]` 是第 b 位的 dim，`sel` 是页内行选择子。
    """
    pages = chunk_words(words)
    n_pages = len(pages)
    if n_pages == 1:
        rows = len(pages[0])
        sel = row_lits(cnt[:cnt_width(rows)], rows, f"{name}.sel")
        gates = [g.one]
    else:
        rows = PAGE_ROWS
        sel = row_lits(cnt[:PAGE_BITS], PAGE_ROWS, f"{name}.sel")
        pg = cnt[PAGE_BITS:]
        gl = min(PAGE_LO, len(pg))          # 页数少的时候高位不存在，只用低位
        hi = pg[gl:]
        if n_pages > (1 << len(pg)):
            raise ValueError(f"{name}: {n_pages} 页装不进 {len(pg)} 位页号")
        gates = [_and(g, lits_of(pg[:gl], p & ((1 << gl) - 1))
                      + [h if (p >> (gl + b)) & 1 else g.one - h
                         for b, h in enumerate(hi)], f"{name}.pg[{p}]")
                 for p in range(n_pages)]

    bit = g.zeros(COLS, f"{name}.bit")
    outs = [(c, 1.0) for c in range(COLS)]
    zs = [{c: gates[i] for c in range(COLS)} for i in range(n_pages)]
    init = [{r: [float((w >> b) & 1) for b in range(COLS)] for r, w in enumerate(pg_)}
            for pg_ in pages]
    zero = g.const(0.0, f"{name}.zero")
    g.head(
        name, "rom", rows=rows, cols=COLS,
        q={r: Chan(sel[r]) for r in range(rows)},
        k={0: Chan(zero)},                       # 恒 0：永不写
        v={},                                    # 内容在 init 里，没有写口
        a=0, z=zs[0],
        outs=outs, into=bit,
        vx=[dict(v={}, a=0, z=zs[i], outs=outs, into=bit) for i in range(1, n_pages)],
        init=init, levels="bit",
    )
    # 每页都 `into=bit`，所以 head 会把同一组 32 个 dim 重复返回 n_pages 份，直接给 bit
    return bit, sel


def fetch(g: Graph, cnt: list[Expr], words: list[int],
          name: str = "rom") -> tuple[list[Expr], list[Expr]]:
    """取指：页号选页、页内行选行，读出一整条指令的 32 位。

    返回 `(bit_dims, sel)`，分页的细节在 rom()。
    """
    return rom(g, cnt, words, name)


# ---------------- 位平面原语 ----------------
#
# 这一层的全部作用是把“整数”变成 0/1 的门电路，且每个中间量都是 bf16 精确的值。三个约定：
#
#   bits   0/1 的 Expr 列表，低位在前；整数就是加权和（线性，免费，不占 unit）
#   int    一个装小整数的 Expr（|值| ≤ 256 才算安全，超过 256 就是 bf16 表示的极限）
#   onehot 恰好一个为 1 的 0/1 列表
#
# 照抄 Qwenputer `q35/prims.py` 的形状，但只保留本机能用的那几条：它的 `t_ge`/`t_lt`
# 有 256 以上的分块分支（为了 12 位地址域），这里 z 的范围都是 nibble 级的，用不上。


def from_bits(bits: list[Expr]) -> Expr:
    """位列表 -> 整数 Expr。纯线性组合，不占 unit。"""
    return sum((b * (1 << i) for i, b in enumerate(bits)), Expr())


def lits_of(bits: list[Expr], v: int) -> list[Expr]:
    """恰好当这些位拼成 v 时全为 1 的字面量列表（AND 的入参）。"""
    return [b if (v >> i) & 1 else 1 - b for i, b in enumerate(bits)]


def _t_const(c) -> list[tuple]:
    """常数项：`c * 1 * relu(1)`。a=1、b=1，silu 在 1 上是精确的。"""
    return [(c, None, None)]        # None 在 _node 里换成 g.one


def _t_and(lits: list[Expr], coef=1, a: Expr | None = None) -> list[tuple]:
    """coef * a * AND(lits)，lits 都是 0/1。

    门写成 `4·(sum - n + 0.5)`：全 1 时是 2、否则 ≤ -2。留 ±2 的裕度而不是掐到 0，
    是因为字面量本身是 fp32 累加出来的，可能带 1e-7 量级的噪声，而 false 必须精确为 0。
    """
    return [(Fraction(coef, 2), a, 4 * sum(lits, Expr()) - (4 * len(lits) - 2))]


def _t_ge(z: Expr, th: int, lo: int, hi: int, coef=1, lits: tuple = ()) -> list[tuple]:
    """coef * [z >= th]（AND lits），z 是 [lo, hi] 内的整数。

    `relu(z-th+1) - relu(z-th)` 对整数 z 恒等于 0/1——两个 ramp 的差把“斜坡”消掉了，
    所以不需要标量比较，也不需要 float64（§5 #6 的另一半：比较 = 两个 relu）。
    lits 用 `m` 把 z 往下推一个大于整段的值：任一为假时两个 ramp 都 ≤ 0。
    """
    if th > hi:
        return []
    n = len(lits)
    if th <= lo and not n:
        return _t_const(coef)
    m = 1 << (hi - lo).bit_length()
    zz = z + m * (sum(lits, Expr()) - n) if n else z
    return [(coef, None, zz - (th - 1)), (-coef, None, zz - th)]


def _t_eq(z: Expr, v: int, lo: int, hi: int, coef=1, lits: tuple = ()) -> list[tuple]:
    """coef * [z == v]（AND lits）。"""
    if not lo <= v or v > hi:
        return []
    return _t_ge(z, v, lo, hi, coef, lits) + _t_ge(z, v + 1, lo, hi, -coef, lits)


def _node(g: Graph, terms, name: str, into: Expr | None = None) -> Expr:
    """一组 unit 项 -> 一个 dim（`into` 给定时累加进已有 dim）。空项直接返回常数 0。"""
    ts = [(c, g.one if a is None else a, g.one if b is None else b) for c, a, b in terms]
    if not ts:
        return Expr() if into is None else into
    return g.ffn(ts, name, into=into)


def _and(g: Graph, lits: list[Expr], name: str) -> Expr:
    return _node(g, _t_and(lits), name)


def _or(g: Graph, lits: list[Expr], name: str) -> Expr:
    """OR 用德摩根：NOT (AND of NOT lits)。"""
    return _node(g, _t_const(1) + _t_and([1 - e for e in lits], -1), name)


def bits_of_int(g: Graph, z: Expr, lo: int, hi: int, n: int, name: str,
                f=None, lits: tuple = ()) -> list[Expr]:
    """f(z) 的低 n 位（f 默认恒等），z 是 [lo, hi] 内的整数。

    每一位 = **(f 在每个翻转点上的符号变化) × [z >= 该点]**。望远镜求和：ramp 在相邻
    翻转点之间一加一减互相消掉，结果恰好落在 {0,1}，而每个翻转点只花 2 个 unit。
    这是“nibble 和 -> 4 个结果位”的全部（Qwenputer 的 `fn_bits`）。
    """
    f = (lambda v: v) if f is None else f
    out = []
    for b in range(n):
        terms = _t_const(1) if (f(lo) >> b) & 1 else []
        if lits:
            terms = _t_and(list(lits)) if terms else []
        for th in range(lo + 1, hi + 1):
            dd = ((f(th) >> b) & 1) - ((f(th - 1) >> b) & 1)
            if dd:
                terms += _t_ge(z, th, lo, hi, dd, lits)
        out.append(_node(g, terms, f"{name}.b{b}"))
    return out


def onehot_of_bits(g: Graph, bits: list[Expr], n: int, name: str) -> list[Expr]:
    """整数取 n 个值时的 one-hot：每个值一个 unit（n 个字面量的 AND）。"""
    return [_and(g, lits_of(bits, v), f"{name}={v}") for v in range(n)]


def mux_bits(sel: Expr, x: list[Expr], y: list[Expr]) -> list[list[tuple]]:
    """`sel ? x : y` 的逐位项组，sel 是 0/1。`y + sel·(x - y)`，每位 3 个 unit。

    返回的是 **项组**而不是节点：调用方常把同一个 mux 的位拼进别的节点里。
    """
    if len(x) != len(y):
        raise ValueError(f"mux 的两位宽不同：{len(x)} vs {len(y)}")
    return [_t_and([y[i]]) + _t_and([sel, x[i]]) + _t_and([sel, y[i]], -1)
            for i in range(len(x))]


def row_add(g: Graph, x: list[Expr], y: list[Expr], n: int, name: str) -> list[Expr]:
    """n 位模加（行波进位），丢掉第 n 位。x/y 都是 n 个 0/1。

    每一位的和 `s = x + y + carry` 只能是 0..3，进位就是 `[s >= 2]`（一个 relu 对）。
    和位**不能**写成 `[s>=1] - [s>=2]`：那个式子只在 s<=2 时等于 `s & 1`，s=3 时也给 0
    （实测踩到过：分支目标只在加法不进到 3 的时候对）。走 `bits_of_int` 的阈值扫，
    s 只有 4 个值，两个阈值就够。nibble 级加法（ALU 用的那种）在这里没好处：宽度只有
    log2(一页)，逐位反而少一半 unit。
    """
    out, carry = [], Expr()
    for i in range(n):
        s = x[i] + y[i] + carry
        out += bits_of_int(g, s, 0, 3, 1, f"{name}.b{i}")
        carry = _node(g, _t_ge(s, 2, 0, 3), f"{name}.c{i}")
    return out


def uops_of(g: Graph, w: list[Expr], name: str = "uop") -> dict[str, Expr]:
    """ROM 字低字节的 uop 编号 -> 每条指令一个 0/1（编号表就是 `UOP_ORDER`）。

    独立成函数是因为两处要用它：`alu` 选运算类别，写入口选「这一拍是不是 JAL/JALR」
    （链接值只在它俩上写）。它们必须拿到**同一组**判定，否则会各建一份。
    """
    uopv = from_bits(w[0:7])
    return {u: _node(g, _t_eq(uopv, i, 0, 127), f"{name}.{u}") for u, i in UOP.items()}


# ---------------- 执行：寄存器堆 ----------------


def reg_index(g: Graph, bits: list[Expr], name: str) -> list[Expr]:
    """5 位寄存器号 -> 32 个行 one-hot。5 个字面量的 AND，一个值一个 unit。"""
    if len(bits) != 5:
        raise ValueError(f"{name}: 寄存器号是 5 位，得到 {len(bits)}")
    return onehot_of_bits(g, bits, REGS, name)


def regs(g: Graph, sels: dict[str, list[Expr]], init: dict | None = None,
         name: str = "regs", wcols: list[tuple[int, Chan]] | None = None,
         wrow: list[Expr] | None = None,
         beta: Expr | None = None) -> dict[str, list[Expr]]:
    """32 行 × 32 位的寄存器堆，每个读口一张**同形的表**。

    为什么每个口一张表而不是一张表多个读口：graph 的 Head 一个节点只有一组 `q`，
    两组独立的行选择必须落在两个节点上；而两个节点各有各的状态。所以写口要写到
    **每一张**上（`wcols`/`wrow`/`beta` 是同一个，所有口共用）。

    三个口：`rs1` / `rs2` 是源操作数，`rd` 是存储指令的数据寄存器（预译码把 SW/SB/SH
    的数据放在 RD 槽里，见 `predecode`）。读就是一次 k-hot 求和，行是位平面所以
    读出的是 32 个 0/1——**不经过任何 norm，也不拆高低字节**。

    `wcols` 是写值列 `[(读出位, 通道)]`：**一个位可以有多列**，源不同、taps 不同，
    读出时逐列累加到同一个位 dim 上（`into`）。值记录为什么要拆三段，见 `value_record`。
    写行 `wrow` 走 taps 滞后一拍：记录拍上的 ROM 已经指向下一条。写口不含 row 0，
    所以 x0 天然写不进去。

    注意写值是**通道值**，不经过输出门：写入发生在记录拍，而那个拍上的 `z` 却是给
    同一拍的读口用的。所以每一个写值必须在自己的表达式里就带上家族门控（`_t_and`），
    否则“不属于本拍的那个来源”也会被写进去。

    `wcols`/`wrow`/`beta` 传 None 时不建写通路（只验“读 + 算”）。
    """
    ncols = len(wcols) if wcols else REGBITS
    kw = {} if wrow is None else {r: Chan(wrow[r], (0.0, 1.0, 0.0, 0.0))
                                  for r in range(1, REGS)}
    out = {}
    for pname, q in sels.items():
        if wcols:                      # 多列 -> 同一个读出位，逐列累加
            dims = g.zeros(REGBITS, f"{name}.{pname}.x")
            vw = {c: ch for c, (_, ch) in enumerate(wcols)}
            outs = [(c, 1.0) for c in range(ncols)]
            into = [dims[b] for b, _ in wcols]
        else:
            vw, into = {}, None
            outs = [(c, 1.0) for c in range(REGBITS)]
        ret = g.head(
            f"{name}.{pname}", "regs", rows=REGS, cols=ncols,
            q={r: Chan(sel) for r, sel in enumerate(q)},
            k=kw, v=vw, a=0, z={c: g.one for c in range(ncols)}, outs=outs, into=into,
            beta=beta, init=[init or {}], levels="bit",
        )
        out[pname] = ret if into is None else dims
    return out


def value_record(g: Graph, feats: dict[str, Expr]) -> tuple[list[tuple[int, Chan]], Expr]:
    """token 流里的**值记录** -> 写值列 + 写门（写回的唯一值来源）。

    电路是验证器：一条指令算出的值不在电路里，而是由**后面几个 token 的载荷**带回来。
    记录有三种形状，段的**位置**是固定的，所以电路里不需要“这是第几段”的状态：

      F0          1 个 token   载荷 = 值 bit0..14（bit14 是符号）→ 符号扩展到 32 位
      V, F1       2 个 token   低 15 位在 V（lag1），中 15 位在 F1 自己的载荷
      V, V, F2    3 个 token   中 15 位在**第一个** V（lag2），低 15 位在第二个 V（lag1），
                               bit30/31 在 F2 的载荷里

    两张表（低段、中段）都靠“段在记录尾部的位置”定位，而不是靠段号：低段永远紧邻
    终结拍（lag1），中段永远在它前面。代价是一条**前置条件**：记录之间必须隔至少
    两个非 V token（终结拍 + 执行拍的 `NT`），否则 lag2 会指到上一条记录的 V 段。

    每个终结拍（F0/F1/F2）都写一次完整 32 位值，历史快照就是那个状态，这正是
    `Chan.taps` 存在的理由。写门 `beta` 只认这三个家族：V 是中间段不写，ST/SS/J 的
    RD 槽是数据寄存器或偏移，也不许写。

    两处细节：

      * 本拍来源的列必须自带家族门控（`_t_and([F0, p[i]])`）：写入拿的是通道值，
        不经过输出门，不然别的家族的载荷也会被写进去
      * 符号扩展也走门控（`sign0`/`sign1` 一个节点被十几列复用），不然 F0 拍上的
        中段/高位列会被其他家族带成 1

    **值必须来自载荷，不能来自 ALU**：ALU 的输出在本拍比写口更晚算出来，用它就成环，
    `check_dag` 会直接报。
    """
    F0, F1, F2 = (feats[f"fam.{n}"] for n in ("F0", "F1", "F2"))
    V = feats["fam.V"]
    p = [feats[f"pay[{b}]"] for b in range(NPAY)]      # 16 位载荷；这三段只用低 15 位
    vp = [_node(g, _t_and([V, p[b]]), f"wb.vpay{b}") for b in range(NPAY)]

    def sel(gate: Expr, bit: Expr, name: str) -> Expr:
        return _node(g, _t_and([gate, bit]), name)

    L1, L2 = (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)
    sgn0 = sel(F0, p[14], "wb.sign0")     # F0 的 bit14 就是 32 位值的符号
    sgn1 = sel(F1, p[14], "wb.sign1")     # F1 的 bit14 = 值 bit29，往上全是它
    cols: list[tuple[int, Chan]] = []
    for i in range(32):
        if i < 15:                    # 低 15 位：F0 本拍 / 紧邻终结拍的那个 V
            cols.append((i, Chan(sel(F0, p[i], f"wb.f0.{i}"))))
            cols.append((i, Chan(vp[i], L1)))
        elif i < 30:                  # 中 15 位：F1 本拍 / 三段记录的**第一段** V
            cols.append((i, Chan(sgn0)))
            cols.append((i, Chan(sel(F1, p[i - 15], f"wb.f1.{i}"))))
            cols.append((i, Chan(vp[i - 15], L2)))
        else:                         # 高 2 位：F0/F1 都是符号扩展，F2 自带
            cols.append((i, Chan(sgn0)))
            cols.append((i, Chan(sgn1)))
            cols.append((i, Chan(sel(F2, p[i - 30], f"wb.f2.{i}"))))
    return cols, F0 + F1 + F2


def record_row(v: int, cols: list[tuple[int, Chan]]) -> list[float]:
    """32 位值 -> 一整行的列值（只填每位的**本拍**列，历史列留 0）。

    一行的合法形状不止一种（列本来就可以分开存），这只是“刚被写进去之后”的那种。
    探针和 `init=` 用它把参考机的寄存器值注入表里。
    """
    row, seen = [0.0] * len(cols), set()
    for c, (b, _) in enumerate(cols):
        if b not in seen:
            seen.add(b)
            row[c] = float((v >> b) & 1)
    return row


def record_value(row, cols: list[tuple[int, Chan]]) -> int:
    """一整行的列值 -> 32 位值。与 head 的读出同构：逐列累加回同一位。"""
    acc: dict[int, float] = {}
    for c, (b, _) in enumerate(cols):
        acc[b] = acc.get(b, 0.0) + float(row[c])
    v = 0
    for b, s in acc.items():
        v |= int(round(s)) << b
    return v


#: 执行拍：`V` 家族、载荷 0。它不携带任何值（每列都要过 `V∧载荷位` 的门，而载荷全是
#: 0），只干两件事：给记录之间隔出一个「非 V 载荷」的 token（`value_record` 的 lag2 靠
#: 它不串到上一条记录的 V 段），并让记录的终结拍取指仍停在本条指令那一行（`cnt` 的推进
#: 滞后一拍，见 `pc_anchor`）。
EXEC = tok("V", 0)


def value_tokens(v: int) -> list[int]:
    """32 位值 -> 值记录的 token（1 / 2 / 3 个）。`value_record` 的逆，段序中 -> 低 -> 终结。

    这里只是把参考机算出的值翻译成 token，不涉及任何指令：凡是写寄存器的指令都走它。
    """
    v &= 0xFFFFFFFF

    def sext(x: int, w: int) -> int:
        return ((x ^ (1 << (w - 1))) - (1 << (w - 1))) & 0xFFFFFFFF

    if sext(v & 0x7FFF, 15) == v:
        return [tok("F0", v & 0x7FFF)]
    if sext(v & 0x3FFFFFFF, 30) == v:
        return [tok("V", v & 0x7FFF), tok("F1", (v >> 15) & 0x7FFF)]
    return [tok("V", (v >> 15) & 0x7FFF), tok("V", v & 0x7FFF), tok("F2", (v >> 30) & 0x3)]


def store_tokens(word: int) -> list[int]:
    """store 要写进 RAM 的整字 -> 记录的前三个 `V`（`store_record` 的逆）。

    段序与 `value_tokens` 相反（低 -> 中 -> 高）：终结拍是 `ST0/ST1`/`SS0/SS1`，它们
    是控制 token、没有载荷，所以那 2 位高位也得占一个 `V`。调用方接上终结拍即可。
    """
    word &= 0xFFFFFFFF
    return [tok("V", word & 0x7FFF), tok("V", (word >> 15) & 0x7FFF),
            tok("V", (word >> 30) & 0x3)]


#: `ecall` 的两个约定号：`a7` = 1 是 putchar，`a7` = 93 是 exit。
#: 预译码把 `ecall` 写成 `_pack("ecall", 0, 17, 10)`，所以 rs1 读口恒读 x17（a7）、
#: rs2 读口恒读 x10（a0）——两个操作数本来就在读口上，电路不必再选一次。
SYSCALL_PUT, SYSCALL_EXIT = 1, 93
PUT_BITS = 9            # `a0 & 511`：0..255 是字节，256..511 是特殊码


def byte_token(a0: int) -> int:
    """`ecall a7=1` 的产出 -> `BYTE` token（载荷 = `a0 & 511`）。`syscall()` 的逆。

    载荷是 9 位而不是 8 位，因为参考机把 256..511 留给特殊码（换行/刷新等），
    只有 0..255 才是真的 putchar。截成 8 位会把这层区分悄悄丢掉。
    """
    return tok("BYTE", a0 & 0x1FF)


def r_token(target: int) -> int:
    """`R` token（anchor 刷新）-> 载荷 = 新的**绝对字索引**。

    参考机的 `R` 是「anchor = 载荷、n 归 0」，因为它的 PC 是 `anchor + n` 两段。
    本机 `cnt` 直接存绝对字索引（§9「锚点推进」），所以两者的载荷含义正好重合：
    `anchor` 就是那个绝对索引，`n` 归 0 这件事由“整张表只有一列 cnt”自然满足。

    `R` 拍什么都不执行：不写 rd（`beta` 只收 `F0/F1/F2`）、不写内存（`wr` 只收
    `ST*/SS*`），ROM 读出的那一行仍算一遍 ALU，但结果没有任何消费者。
    """
    return tok("R", target & 0xFFFF)


# ---------------- 执行：ALU ----------------
#
# ALU 按**运算类别**组织，不按指令逐条布线：
#
#   加法器   4 种操作数组合（+rs2 / +imm / -rs2 / -imm），一条 8 nibble 串行进位链
#   逻辑     与 / 或 / 异或，右操作数由“是不是 I 型”一位选出来
#   移位     桶形：先按 shamt[2:5] 移 0/4/…/28，再按 shamt[0:2] 移 0..3；方向三位
#   比较     相等 = 减法结果全 0；有符号小于 = 符号位异或溢出；无符号小于 = 借位取反
#
# 所以 52 条指令和 8 条指令的差别只是 ROM 字里的 uop 编号，以及“哪些 uop 落进哪个类别”
# 这张表——扩 M 扩展是加一个乘法器 + 几个 gate，不是加 40 条通路。

_IMM_B = ("addi", "slti", "sltiu", "lb", "lh", "lw", "lbu", "lhu",
          "sb", "sh", "sw", "jalr")                  # 右操作数是 imm12
_CLASS_ADD = ("add", "sub", "addi", "jalr", "lb", "lh", "lw", "lbu", "lhu",
              "sb", "sh", "sw")                        # 结果是加法链的值本身
_CLASS_SUB = ("sub", "slt", "sltu", "slti", "sltiu",
              "beq", "bne", "blt", "bge", "bltu", "bgeu")   # 加法链算的是差
_CLASS_LOGIC = ("and", "or", "xor", "andi", "ori", "xori")
_CLASS_SHIFT = ("sll", "srl", "sra", "slli", "srli", "srai")
_CLASS_IMM_LOGIC = ("andi", "ori", "xori")
_CLASS_IMM_SHIFT = ("slli", "srli", "srai")
_CLASS_BRANCH = ("beq", "bne", "blt", "bge", "bltu", "bgeu")
_CMP_S = ("slt", "slti")                                 # 结果 = 那一位比较
_CMP_SU = ("sltu", "sltiu")


def alu(g: Graph, w: list[Expr], ports: dict[str, list[Expr]], name: str = "alu",
        uop: dict[str, Expr] | None = None):
    """一条预译码字 + 三个寄存器读口 -> 32 位结果 + 分支判定。

    返回 dict：

      `res`    32 个 0/1，写回的值（load/store 是**地址**，分支是 0）
      `taken`  1 个 0/1，分支是否 taken（只对 6 条分支有意义）
      `uop`    uop -> 0/1 的字典，下一步拼 record 时按它选家族
      `c32`    加法器最高位的进位（无符号比较的中间量）
      `low`    加法器最低两位（地址的 bit0..1）。单独拿出来是因为 `res[0]` 要等
               `slt`/`sltu` 的比较链（level 46），而这两位只要第一次进位的输出
               （level 24）——`load_value` 的字节/半字选择接在它上面会少 5 层

    `uop` 可以从外面传进来（`uops_of`）：写入口选 JAL/JALR 时也要用它，不能各建一份。
    """
    one = g.one
    b0, b1, b2, b3 = w[0:8], w[8:16], w[16:24], w[24:32]
    sp = [b0[7]] + b1[0:3] + b2[0:3] + b3[0:3]        # S0..S9（`_pack` 的 spare 位）
    U = uops_of(g, w, f"{name}.uop") if uop is None else uop

    def any_of(*names) -> Expr:
        return sum((U[n] for n in names), Expr())

    A, B, C = ports["rs1"], ports["rs2"], ports["rd"]
    imm = b3[3:8] + sp[0:7] + [sp[6]] * 20             # I 型 imm12，符号扩展到 32 位
    upper = b2[3:8] + b3[3:8] + sp[0:10]               # LUI 的 20 位（预译码放在 RS1/RS2/S 里）

    # ---- 加法器：4 种操作数组合共用一条链 ----
    addish = any_of(*_CLASS_ADD)
    use_imm = any_of(*_IMM_B)
    inv = any_of(*_CLASS_SUB)                          # 减法：右操作数取反、进位进 1
    bb = [_node(g, t, f"add.bb{i}") for i, t in enumerate(mux_bits(use_imm, imm, B))]
    nb = [_node(g, _t_and([inv, 1 - bb[i]]) + _t_and([1 - inv, bb[i]]), f"add.nb{i}")
          for i in range(32)]
    cout, res_add = inv, []
    for k in range(8):
        t_k = from_bits(A[4 * k:4 * k + 4]) + from_bits(nb[4 * k:4 * k + 4]) + cout
        res_add += bits_of_int(g, t_k, 0, 31, 4, f"add.d{k}")
        cout = _node(g, _t_ge(t_k, 16, 0, 31), f"add.c{k + 1}")

    # ---- 逻辑 ----
    lb = [_node(g, t, f"logic.bb{i}")
          for i, t in enumerate(mux_bits(any_of(*_CLASS_IMM_LOGIC), imm, B))]
    andv, orv, xorv = any_of("and", "andi"), any_of("or", "ori"), any_of("xor", "xori")
    logic = []
    for i in range(32):
        terms = _t_and([andv, A[i], lb[i]])
        terms += [(1, orv, one)] + _t_and([orv, 1 - A[i], 1 - lb[i]], -1)
        terms += _t_and([xorv, A[i], 1 - lb[i]]) + _t_and([xorv, 1 - A[i], lb[i]])
        logic.append(_node(g, terms, f"logic.b{i}"))

    # ---- 移位：两级桶形 ----
    shamt = [_node(g, t, f"shamt.b{i}")
             for i, t in enumerate(mux_bits(any_of(*_CLASS_IMM_SHIFT), imm[0:5], B[0:5]))]
    hi_lits = [_and(g, lits_of(shamt[2:5], k), f"shamt.hi{k}") for k in range(8)]
    lo_lits = [_and(g, lits_of(shamt[0:2], k), f"shamt.lo{k}") for k in range(4)]
    dirs = (("L", any_of("sll", "slli")),
            ("R", any_of("srl", "srli")),
            ("A", any_of("sra", "srai")))
    y: dict[str, list[Expr]] = {}
    for d, lit in dirs:
        y[d] = []
        for i in range(32):
            terms = []
            for k in range(8):
                j = i - 4 * k if d == "L" else i + 4 * k
                src = A[j] if 0 <= j < 32 else (A[31] if d == "A" and j >= 32 else None)
                if src is not None:
                    terms += _t_and([lit, hi_lits[k], src])
            y[d].append(_node(g, terms, f"sh{d}1.b{i}"))
    shift = []
    for i in range(32):
        terms = []
        for d, lit in dirs:
            for k in range(4):
                j = i - k if d == "L" else i + k
                src = y[d][j] if 0 <= j < 32 else (y[d][31] if d == "A" and j >= 32 else None)
                if src is not None:
                    terms += _t_and([lit, lo_lits[k], src])
        shift.append(_node(g, terms, f"shift.b{i}"))

    # ---- 比较 ----
    eq = _and(g, [1 - b for b in res_add], "cmp.eq")          # A - B == 0
    bsign = bb[31]                 # 右操作数的符号位：slti/sltiu 时是 imm 的，不是 x6 的
    xab = _node(g, _t_and([A[31]]) + _t_and([bsign]) + _t_and([A[31], bsign], -2),
                "cmp.xab")                                     # 符号位不同
    slt = _node(g, _t_and([xab, A[31]]) + _t_and([res_add[31]])
                + _t_and([xab, res_add[31]], -1), "cmp.slt")    # 有符号小于
    sltu = _node(g, _t_const(1) + _t_and([cout], -1), "cmp.sltu")  # 借位取反
    taken = _node(g, _t_and([U["beq"], eq]) + _t_and([U["bne"], one - eq])
                  + _t_and([U["blt"], slt]) + _t_and([U["bge"], one - slt])
                  + _t_and([U["bltu"], sltu]) + _t_and([U["bgeu"], one - sltu]),
                  "cmp.taken")

    # ---- 结果选择：每个类别一位 gate ----
    # slt/slti 与 sltu/sltiu 不算“加法的值”，它们的结果就是那一位比较：所以它们不在
    # `addish` 里，高位因此恒 0，bit0 换成一个 0/1。slti/sltiu 的差仍然走同一条加法链
    # （右操作数由 use_imm 选成 imm），只是不进结果 mux。
    cmp_s, cmp_su = any_of(*_CMP_S), any_of(*_CMP_SU)
    srcs = [([addish], res_add),
            ([U["lui"]], [Expr()] * 12 + upper),
            ([any_of(*_CLASS_LOGIC)], logic),
            ([any_of(*_CLASS_SHIFT)], shift)]
    res = []
    for i in range(32):
        terms = []
        for gate, bits in srcs:
            if bits[i].terms:
                terms += _t_and(list(gate) + [bits[i]])
        if i == 0:
            terms += _t_and([cmp_s, slt]) + _t_and([cmp_su, sltu])
        res.append(_node(g, terms, f"res.b{i}"))
    return {"res": res, "taken": taken, "uop": U, "c32": cout,
            "slt": slt, "sltu": sltu, "eq": eq, "low": res_add[0:2]}


# ---------------- 执行：RAM ----------------

# 尺寸在 `constant.py`。**512 行是硬约束**：换成 256 行，DROM 就借不到 RAM 的行选择
# （理由见 `drom_table`）。


def addr_region(g: Graph, res: list[Expr], name: str = "mem") -> tuple[Expr, Expr]:
    """地址落在哪一段：`(is_ram, is_drom)`，两个都是精确 0/1。

    内存图是 `RAM [0, 0x8000) · DROM [0x20000, 0x40000) · ROM [0x40000, 0x80000)`，
    所以只看 bit15 往上：RAM 全 0、DROM 只有 bit17 为 1、ROM 是 bit18。

    但 `is_drom` 判的是**表覆盖的那 32 KB**（`DROM_BANKS` 块 × 512 字），即
    `[0x20000, 0x28000)`：bit15/16 为 1 的地址（0x28000 往上）还在 `cfg.drom_size`
    里却没有 bank，读出 0——与参考机一致（.rodata 之后就是零填充），而内容真跨过去时
    `drom_table` 会先抛错，不会静默截断。

    写法是**一个 ramp**：`relu(1 - Σ res[15:])` 与 `relu(res[17] - Σ 其余高位)`。两个都对
    整个 32 位地址空间精确（高位任何一位越界都把和推过 0），代价各一个 dim、一个 unit。

    为什么必须两个、不能一个取反：`1 - is_drom` 在 ROM 地址上是 1，RAM 就会往共用输出
    列里落值。现在 RAM 的 16 个 bank 门带 `is_ram`、DROM 的带 `is_drom`，同一拍最多一个
    为 1——这正是 §5 #10 要求的那条不变量（“恰好一个写者写 1，其余写 0”）。
    """
    hi = sum(res[15:], Expr())
    is_ram = _node(g, [(1, None, 1 - hi)], f"{name}.is_ram")
    rest = sum((res[b] for b in range(15, len(res)) if b != 17), Expr())
    is_drom = _node(g, [(1, None, res[17] - rest)], f"{name}.is_drom")
    return is_ram, is_drom


def store_record(g: Graph, feats: dict[str, Expr], name: str = "st") -> list[Chan]:
    """`sb`/`sh`/`sw` 要写进 RAM 的**整字**：记录里三个 `V` 的载荷（`V V V ST/SS`）。

    `ST`/`SS` 是控制 token，自己不带载荷，所以整字只能从前面三个 `V` 取回来：低 15 位在
    lag3、中 15 位在 lag2、高 2 位在 lag1（段序是低 -> 中 -> 高，与 `value_tokens` 的中 ->
    低相反：终结拍不是数据 token，带不了那 2 位）。`sb`/`sh` 的合并后整字由参考机算好，
    这里只管把它原样存进去。

    **`sw` 也走这条路，不用 RD 槽的寄存器读出**。不是因为省事：一列的 taps 是固定的，而
    `sw` 的数据在同拍、`sb`/`sh` 的数据在 1~3 拍前，两种 taps 无法共存于一列；分两半存又
    要多一倍列宽（32 列 × 16 块 bank = 528 dim），读出时还要一位 kind 二选一。写回本来
    就是“值来自载荷、地址来自 ROM”，这里保持同一条原则。

    三个 `V` 之所以能直接被引用，是因为列通道本来就可以带 taps：同一列在 `ST`/`SS` 拍上
    读到的是 1~3 拍前那个 `V` 的载荷。
    """
    V = feats["fam.V"]
    p = [feats[f"pay[{b}]"] for b in range(NPAY)]
    out = []
    for i in range(REGBITS):
        b, lag = (i, 3) if i < 15 else (i - 15, 2) if i < 30 else (i - 30, 1)
        taps = tuple(1.0 if k == lag else 0.0 for k in range(4))
        out.append(Chan(_node(g, _t_and([V, p[b]]), f"{name}.p{i}"), taps))
    return out


def ram_table(g: Graph, res: list[Expr], feats: dict[str, Expr],
              init: dict | None = None, name: str = "ram",
              is_ram: Expr | None = None) -> dict:
    """8192 字 × 32 位 RAM：**16 块 512 行的 bank**，行 = 字索引低 9 位。

    输入就是 ALU 的结果（store/load 的地址 = `rs1 + imm`），字索引 = `res[2:15]`：
    `res[2:11]` 选行、`res[11:15]` 选 bank。

    **为什么不是一块 4096 行**：行选择改成 `row_lits` 之后它不花 dim 了，但**层**仍然花：
    `row_sel` 的 4096 行要 64 低 + 64 高 + 4096 个 AND（3 层），而 16 块 × 512 行只要
    64 + 8 + 512（2 层）。而且行数还决定 DROM 能不能借：见 `drom_table`。

    16 块是拐点，而且 512 行恰好与 ROM 一页同形（`PAGE_ROWS`），行选择器是同一个函数。

    **写出**：`beta = 写拍 ∧ bank one-hot`，写值 `store_record` 的 32 个位（整字覆盖）。
    **读出**：16 块 bank **共用一组 32 个输出列**——每块的输出门 `z` 就是自己的 `lit[h]`，
    所以只有被选中那一块会把读出值落到残差上，其余的写 0、累加不变。
    行选择走 `row_lits`（线性），584 个 dim 全不花；**bank 门 `lit[h]` 仍必须是真 AND**，
    因为它同时当 `z`（输出门）用，而 `z` 在 `eval_head` 里直接查 {0,1}。

    `is_ram` 给定（连上 DROM 之后）时，bank 门换成 `AND(行号字面量, is_ram)`：数目不变
    （还是 16 个），但只读数据段的地址不会再选中 RAM 的某一块。这同时也是**store 侧**的
    保护：写 DROM 在参考机上是 trap，这里至少不会静默改进 RAM。

    读出与写入走同一个地址，所以“同一拍写进去、下一拍能读到”在图上是自然成立的
    （`eval_head` 先写后读）。

    **为什么不是每块 bank 一组独立输出列**（旧形状）：那样 16×32 = 512 个 dim 要
    一直并行活到最后一层——`word[i] = Σ_h lit[h]·cols[h][i]` 的选块只能放在最后。
    压槽后实测峰值 1229 个活跃 dim 里有 **1072 个是 `rv.ram.*`**，其中 512 个就是这批
    输出列。改成共用之后这 512 个塌成 32 个，选块从“最后一层 AND”提前成“写的时候就门控”，
    因为 `lit[h]` 在行/块解码那一层就算完了，比 RAM 的 head 早得多。
    """
    if len(res) < 15:
        raise ValueError(f"{name}: 地址要 15 位（字索引 13 位）")
    sel = row_lits(res[2:11], RAM_ROWS, f"{name}.row")
    # 没有 `is_ram` 时与 `onehot_of_bits` 逐字面量等价（只差一个多出来的 AND 位数）
    extra = [] if is_ram is None else [is_ram]
    lit = [_and(g, lits_of(res[11:15], h) + extra, f"{name}.bank={h}")
           for h in range(RAM_BANKS)]
    st = feats["fam.ST0"] + feats["fam.ST1"]
    ss = feats["fam.SS0"] + feats["fam.SS1"]
    wr = st + ss                       # 这一拍在写 RAM 吗（两个家族互斥，所以是 0/1）
    val = store_record(g, feats, f"{name}.st")

    col = g.zeros(RAM_COLS, f"{name}.col")        # 16 块 bank 共用的一组输出列
    for h in range(RAM_BANKS):
        g.head(f"{name}.b{h}", f"ram{h}", rows=RAM_ROWS, cols=RAM_COLS,
               q={r: Chan(sel[r]) for r in range(RAM_ROWS)},
               k={r: Chan(sel[r]) for r in range(RAM_ROWS)},
               v={c: val[c] for c in range(RAM_COLS)},
               a=0, z={c: lit[h] for c in range(RAM_COLS)},   # 只有本块被选中才落残差
               outs=[(c, 1.0) for c in range(RAM_COLS)],
               into=[col[c] for c in range(RAM_COLS)],
               beta=_node(g, _t_and([wr, lit[h]]), f"{name}.wr{h}"),
               init=[init or {}], levels="bit")

    return {"word": col, "sel": sel, "bank": lit, "wr": wr, "cols": col}


def drom_table(g: Graph, res: list[Expr], sel: list[Expr], words: list[int],
               col: list[Expr], is_drom: Expr, name: str = "drom") -> dict:
    """只读数据段（.rodata）：`words` 是它打包成的 32 位字，**512 行一块**，一块一个 value head。

    形状和 RAM 一模一样（“块 × 512 行 × 32 位”），差别只有三处：**没有写口**（`k` 接恒 0、
    `v` 空，内容全在 `init` 里）、bank 门带 `is_drom`、`sel` **直接借 RAM 已经算好的那 512 个
    行选择**，输出也 `into` 到同一组 `col`。

    能借的三样东西都出自“基址是 2 的幂”：`DROM_BASE = 2^17`，减基址不动 bit 2..16，所以
    字索引低 9 位就是 `res[2:11]`（行）、bank 就是 `res[11:15]`——与 RAM 逐位相同。这也是
    **块大小必须取 512**（而不是 ROM 一页的 256）的理由：换 256 行就借不到行选择，整表
    要多 256 个 dim。

    **尺寸按内容定，不按 `cfg.drom_size`**（与 ROM 拿 `words` 而不是 `cfg.rom_words` 同源）：
    真固件 .rodata 只有 0x6428 B = 6410 字，13 块就够。`res[11:15]` 超过 12 的地址
    （0x26800 往上）没 head，读到 0——参考机那边同样是 0（.rodata 之后就是零）。

    为什么不用 ROM 那套“页号低位 one-hot + 高位位线”：ROM 的页门要 13 个 dim，这里也
    13 个，差不多；但 ROM 的表没有别人能借它的行选择，而这张表有 RAM 可借。
    """
    if len(res) < 15:
        raise ValueError(f"{name}: 地址要 15 位（字索引 13 位）")
    banks = chunk_words(words, DROM_ROWS)
    if len(banks) > DROM_BANKS:
        raise ValueError(f"{name}: {len(banks)} 块超过 {DROM_BANKS}（再多一块就要 `res[15]`，"
                         f"行选择与 bank 位就借不到 RAM 的）")
    lit = [_and(g, lits_of(res[11:15], h) + [is_drom], f"{name}.bank={h}")
           for h in range(len(banks))]
    outs = [(c, 1.0) for c in range(DROM_COLS)]
    zero = g.const(0.0, f"{name}.zero")
    g.head(
        name, "drom", rows=DROM_ROWS, cols=DROM_COLS,
        q={r: Chan(sel[r]) for r in range(DROM_ROWS)},
        k={0: Chan(zero)},                        # 恒 0：永不写
        v={},                                     # 内容在 init 里
        a=0, z={c: lit[0] for c in range(DROM_COLS)},
        outs=outs, into=col,
        vx=[dict(v={}, a=0, z={c: lit[i] for c in range(DROM_COLS)},
                 outs=outs, into=col) for i in range(1, len(banks))],
        init=[{r: [float((w >> b) & 1) for b in range(DROM_COLS)] for r, w in enumerate(bk)}
              for bk in banks],
        levels="bit",
    )
    # 每块都 `into=col`，head 会把同一组 dims 重复返回 len(banks) 份，所以直接给 col
    return {"bank": lit, "banks": len(banks), "cols": col}


def text_buffer(g: Graph, res: list[Expr], feats: dict[str, Expr], word: list[Expr],
                name: str = "txt") -> dict:
    """`<cpu_call>` 的文本缓冲：捕获到的 token 一个一行，读它就是 `lw 0x8000 + 16 i`。

    四个件（由外向内）：

      `armed`  TCO（`<cpu_call>`）置 1、TCC（`</cpu_call>`）清 0 的 1×1 表。它只担一件事：
               call 之外的野文本不许写表。`run`（`GO`/`HALT`）**不建**——那个是管“文本流
               期间 CPU 不推进”的，而本机推进看 `fin`，文本 burst 里根本没有 `EXEC`
      `cnt`    捕获序号，8 位二进制计数器（`cnt_width(160)`）。**写口滞后一拍**，和
               `pc_anchor` 同理：写值 `cnt+1` 要用 `cnt` 自己的读出，同拍写回来就是环。
               于是它存的是“本拍要写的行号”：TCO 那一拍 `V = 1`，下一个 token 就是第 1 行
      `tbuf`   160 行 × 64 列的表，两套 value 头各管 8 个字节（`vx`），`into` 到 RAM 已经
               算好的 32 个输出列。**写行看 `cnt`、读行看地址**：`q` 是 `EA[4:12]`（`lw`
               的地址就是这一行的位置），`k` 是捕获序号。两者用同一张表但不是同一条选择
      `in_text` 地址门：`EA[15] ∧ ¬EA[12:15] ∧ ¬EA[16:32]`，即 `[0x8000, 0x9000)`

    **为什么不用衰减计数器 + 温度计**（Qwenputer 的 `xcap`/`thermo`）：那是 1 + 128 个 dim，
    而且 `2^-k` 到 `k >= 134` 会归零、温度计全判 1、`caprow` 同时点亮一片行——**静默错**。
    二进制计数器是 40 个 dim，而且都是 0/1，没有精度暗礁。

    **行号为什么要 1-based 且滞后一拍**：`cnt` 存的是本拍要写的行，所以 `t=0` 的初值 0
    正好落在“空着”的第 0 行；野文本落在那里也写不坏 1..N。`cnt >= 160` 时 `row_lits`
    没有那一行，`q`/`k` 全 0、静默丢弃——这就是截断（电路容量 159，`CAP_MAX = 128`）。

    **字节怎么进来**：`tb[k][i]` 是 token 特征（第 k 个字节的第 i 位），由 `token_features(g,
    text=...)` 按**整个词表**枚举成 `feat` 那 128 列。所以内容不烤进图：一个 `TX` 只要带自己
    在词表里的 id，字节就已在权重里，与“当前跑的是哪段代码”无关。

    **读出与 RAM/DROM 共列**：每一列的输出门 `z` 是 `(EA[3], EA[2], in_text)` 的组合——
    `EA[3]` 选 8 字节半行、`EA[2]` 选 4 字节字、`in_text` 挡住非文本地址。于是 `load_value`
    原样吃 `mem["word"]`，`lw/lb/lbu/lh/lhu` 全部自动可用，不需要一个专门的 `tmem`。
    写侧不需要额外保护：store 的门是 `wr ∧ lit[h]`，而 `lit` 带 `is_ram`。
    """
    is_text, tco, tcc = feats["fam.TX"], feats["fam.TCO"], feats["fam.TCC"]

    # 1) armed：TCO 置 1、TCC 清 0（同拍写同拍读，所以不需要 taps）
    (armed,) = g.head(f"{name}.arm", "armed", rows=1, cols=1,
                      q={0: g.one}, k={0: Chan(tco + tcc)}, v={0: Chan(tco)},
                      a=0, z={0: g.one}, outs=[(0, 1.0)], levels="bit")
    capw = _and(g, [is_text, armed], f"{name}.capw")

    # 2) cnt：捕获序号。写门 = 上一拍的 TCO 或捕获，写值 = 上一拍的 TCO ? 1 : cnt+1
    nb = cnt_width(TEXT_ROWS)
    nxt = g.zeros(nb, f"{name}.next")
    cnt = g.head(f"{name}.cap", "cap", rows=1, cols=nb,
                 q={0: g.one},
                 k={0: Chan(capw + tco, (0.0, 1.0, 0.0, 0.0))},
                 v={c: Chan(nxt[c], (0.0, 1.0, 0.0, 0.0)) for c in range(nb)},
                 a=0, z={c: g.one for c in range(nb)},
                 outs=[(c, 1.0) for c in range(nb)], levels="bit")
    one = [g.one] + [Expr() for _ in range(nb - 1)]
    # `nxt = TCO ? 1 : cnt+1`：整条向量一起 mux，不要逐位挑 `x`——每一位的 x 不一样
    # （只有最低位是 1），复用 `[g.one]` 会把高位全写成 1（实测踩到：cnt 变 255）。
    for c, terms in enumerate(mux_bits(tco, one, row_add(g, cnt, one, nb, f"{name}.inc"))):
        _node(g, terms, f"{name}.n{c}", into=nxt[c])
    crows = row_lits(cnt, TEXT_ROWS, f"{name}.crow")        # 写行：第几次捕获

    # 3) 地址门 + 读行 + 四个输出门
    # 读行必须是**地址**而不是 `cnt`：`lw 0x8010` 读第 1 行，与捕获到第几个无关。两条
    # 选择子都是 `andpre`（不花 dim），所以分成两份不付代价。
    arows = row_lits(res[4:12], TEXT_ROWS, f"{name}.arow")  # 读行：EA[4:12]
    in_text = _and(g, [res[15]] + [1 - res[b] for b in range(12, 15)]
                   + [1 - res[b] for b in range(16, len(res))], f"{name}.in_text")
    zg = {(x, y): _and(g, [res[3] if x else 1 - res[3],
                           res[2] if y else 1 - res[2], in_text], f"{name}.w{x}{y}")
          for x in range(2) for y in range(2)}

    # 4) tbuf：一套 value 头管 8 个字节（64 列），两套合起来就是 16 字节
    tb = [[feats[f"tb[{k}][{i}]"] for i in range(8)] for k in range(TEXT_ROW)]

    def vhead(x: int) -> dict:
        v, z, outs, into = {}, {}, [], []
        for k in range(8):                       # 半行里的第 k 个字节
            for i in range(8):
                col = 8 * k + i
                v[col] = Chan(tb[8 * x + k][i])
                z[col] = zg[x, k // 4]           # EA[3] 选半行、EA[2] 选字
                outs.append((col, 1.0))
                into.append(word[8 * (k % 4) + i])   # 字节内偏移就是字内的位号
        return {"v": v, "z": z, "outs": outs, "into": into}

    first, second = vhead(0), vhead(1)
    g.head(f"{name}.tb", "tbuf", rows=TEXT_ROWS, cols=8 * (TEXT_ROW // 2),
           q={i: Chan(arows[i]) for i in range(TEXT_ROWS)},
           k={i: Chan(crows[i]) for i in range(TEXT_ROWS)},
           v=first["v"], a=2 * tco,             # 每次 call 开头全表清零，不留上一轮的尾巴
           z=first["z"], outs=first["outs"], into=first["into"],
           beta=capw, vx=[second], levels="bit")
    return {"cnt": cnt, "crow": crows, "arow": arows,
            "in_text": in_text, "capw": capw, "armed": armed}


def load_drom(path, base: int = DROM_BASE) -> list[int]:
    """从 ELF 的只读数据段取 DROM 字表（4 字节小端，末尾不足补 0）。

    与 `load_words` 同形但**不过 `predecode`**：那个格式是指令字才有的。

    没有 `vaddr == base` 的段时返回空表而不是报错：nucpu 固件把 `.rodata` 并在 `.text`
    里（`nucpu.ld`），根本没有 DROM，于是电路不该建那 13 块表——而 Qwenputer 固件
    有 `.rodata`，DROM 的验收归 `probe_drom`（它专门绑着存档的 `firmware-qwenputer.elf`）。
    """
    import struct
    from pathlib import Path

    from components.rv32i.elf import parse_elf

    data = Path(path).read_bytes()
    elf = parse_elf(data)
    segs = [s for s in elf.segments if s.vaddr == base]
    if not segs:
        return []
    seg = segs[0]
    n = -(-seg.filesz // 4)
    buf = data[seg.offset:seg.offset + seg.filesz] + bytes(4 * n - seg.filesz)
    return list(struct.unpack_from(f"<{n}I", buf, 0))


def load_value(g: Graph, word: list[Expr], low: list[Expr], use: dict[str, Expr],
               name: str = "ld") -> list[Expr]:
    """`lw`/`lh`/`lhu`/`lb`/`lbu` 的值：从读出的字里取字节/半字，再按需符号扩展。

    字节由地址 bit0..1 四选一（两级 mux），半字由 bit1 两选一；符号扩展就是「高位移入
    第 7 / 15 位」——`lb`/`lh` 与 `lbu`/`lhu` 的差别只在这几位（`sign` 为 0 时高半部就是 0）。

    `low` 是**加法器最低两位**而不是 `res[0:2]`：load/store 的地址恒来自加法链（`_CLASS_ADD`
    含全部 load/store，`_CLASS_SUB` 一个都不含），所以两者在 load 拍上逐位相等；但 `res[0]`
    要等 `slt`/`sltu` 的比较链，比加法器第一级进位晚 20 个 sublayer，接在上面就是白白多出
    5 层、把整张图从 23 层顶到 27 层（实测）。

    **读出不算入写回**：寄存器堆的写口在 RAM 前面，写值仍来自 token 载荷。这一路是
    “电路能不能自己算出 load 的值”的可验证形式，与 `jump_target` 同性质。
    """
    by = [word[8 * k:8 * k + 8] for k in range(4)]
    lo = [_node(g, t, f"{name}.by{i}") for i, t in enumerate(mux_bits(low[0], by[1], by[0]))]
    hi = [_node(g, t, f"{name}.by{8 + i}")
          for i, t in enumerate(mux_bits(low[0], by[3], by[2]))]
    byte = [_node(g, t, f"{name}.byte{i}") for i, t in enumerate(mux_bits(low[1], hi, lo))]
    half = [_node(g, t, f"{name}.half{i}")
            for i, t in enumerate(mux_bits(low[1], word[16:], word[:16]))]

    sign = use["lb"] + use["lh"]                      # 有符号的两种
    sb_ = _node(g, _t_and([sign, byte[7]]), f"{name}.sgn8")
    sh_ = _node(g, _t_and([sign, half[15]]), f"{name}.sgn16")
    ext_b = byte + [sb_] * 24
    ext_h = half + [sh_] * 16
    lh_all = use["lh"] + use["lhu"]
    lb_all = use["lb"] + use["lbu"]
    out = []
    for i in range(REGBITS):
        terms = _t_and([use["lw"], word[i]]) + _t_and([lh_all, ext_h[i]])
        terms += _t_and([lb_all, ext_b[i]])
        out.append(_node(g, terms, f"{name}.b{i}"))
    return out


def syscall(g: Graph, ports: dict[str, list[Expr]], uop: dict[str, Expr],
            name: str = "sys") -> dict:
    """`ecall` 的两个出口：putchar 的 9 位字符、1 位停机。

    **两个操作数都是现成的**：`predecode` 把 `ecall` 写成 `_pack("ecall", 0, 17, 10)`，
    于是 rs1 读口恒读 x17（`a7`）、rs2 读口恒读 x10（`a0`）。所以这里不需要“选哪个寄存器”，
    只要把两个读口与 uop 门拼起来——`a7 == 1` 出 putchar、`a7 == 93` 出停机。

    **为什么是 32 位全等而不是只比低 7 位**：`a7` 是任意写进 x17 的 32 位数，
    `1 + 2^31` 这种值只比低 7 位会误判成 putchar。全等只多花几个 unit（`_t_and` 是
    一个 unit 项，位数只影响它的线性组合），形状上完全不多一个 dim。

    返回：

      `put`   这一拍是 putchar（0/1）
      `stop`  这一拍是停机（0/1）
      `chars` 9 位 `a0 & 511`，已过 `put` 门——非 putchar 拍全 0

    **它不参与写回**，与 `tgt`/`link`/`quo` 同性质：参考机的 `BYTE`/`HALT` 由 Python
    侧决定，这里把同一件事在电路里再算一遍。这样 “`BYTE` 载荷对不对”才是个能被
    电路否定的命题，而不是把 Python 的结论搬进权重。
    """
    a7, a0 = ports["rs1"], ports["rs2"]
    put = _and(g, [uop["ecall"]] + lits_of(a7, SYSCALL_PUT), f"{name}.put")
    stop = _and(g, [uop["ecall"]] + lits_of(a7, SYSCALL_EXIT), f"{name}.stop")
    chars = [_node(g, _t_and([put, a0[b]]), f"{name}.c{b}") for b in range(PUT_BITS)]
    return {"put": put, "stop": stop, "chars": chars, "a7": a7, "a0": a0}


def jump_target(g: Graph, w: list[Expr], cnt: list[Expr], alu_out: dict,
                ncnt: int, name: str = "tgt") -> dict[str, list[Expr]]:
    """`J` 拍的目标（页内 ncnt 位），三种来源一份选择：

      分支    `cnt + 字偏移`（行波模加；偏移是 11 位有符号，高位按符号位扩到 ncnt）
      JAL     ROM 里的绝对字索引（预译码时算好的 16 位，见 `predecode`）
      JALR    `(rs1 + imm) >> 2`，就是加法器结果的 bit2 往上（右移两位 = 换个起点）

    返回 `{"br", "jal", "jalr", "sel"}`：前三项是三种各自的目标，`sel` 是选完的。

    **偏移比 cnt 窄**：`_pack` 只给分支留了 11 位（±4 KB，RISC-V 的 B 型范围），而分页后
    `cnt` 有 16 位。高位直接复制符号位——它已经是 `boff[10]` 那根 line，重复引用不建新节点。

    **它不参与 `cnt` 的推进**，这是图结构决定的：`cnt` 表的写值必须早于表的读出，
    而这三个目标里有 ALU 结果（level 16 上下）——把它们喂回写值会让表整体后移，
    取指那一段就变成环。所以推进仍走 token 载荷（与 Qwenputer 一致：它的 `btgt`
    也只用来预测 J 载荷，`anchor` 表照样由载荷写）。这一步的作用是把「电路能不能
    自己推出目标」变成可验证的：探针会断言这三项与载荷逐条相等。
    """
    sp = [w[7]] + w[8:11] + w[16:19] + w[24:27]        # S0..S9
    boff = w[11:16] + sp[0:6]                          # 分支：11 位有符号字偏移
    jtgt = w[19:24] + w[27:32] + sp[0:6]               # JAL：16 位绝对字索引
    if ncnt > len(boff):
        boff = boff + [boff[-1]] * (ncnt - len(boff))
    br = row_add(g, cnt, boff[:ncnt], ncnt, f"{name}.br")
    jal = jtgt[:ncnt]
    ea = alu_out["res"][2:2 + ncnt]                    # (rs1 + imm) >> 2
    use = alu_out["uop"]
    brs = sum((use[n] for n in _BRANCH_OPS), Expr())
    sel = [_node(g, t, f"{name}.m0.{i}") for i, t in enumerate(mux_bits(brs, br, ea))]
    sel = [_node(g, t, f"{name}.m1.{i}") for i, t in enumerate(mux_bits(use["jal"], jal, sel))]
    return {"br": br, "jal": jal, "jalr": ea, "sel": sel}


# ---------------- 执行：除法（商位 -> nibble 表） ----------------

QUO_DIGITS = 8          # 32 位商 = 8 个 nibble


def quo_table(g: Graph, feats: dict[str, Expr], name: str = "quo") -> dict:
    """除法的商：8 个 nibble，**行 = 商位序号 k，值 = 那一位的 nibble（0..15）**。

    `QD` 的载荷是 `k << 4 | q`：低 4 位是商位 nibble，高 4 位是它的序号。所以一条 `QD`
    只写一行——8 行天然由载荷里的 k 选出来，电路里不需要“这是第几个 digit”的计数器。
    `QS` 的载荷是**第一个非零商位**的序号（0..7），它是协议里的压缩信息（前导零不必发），
    电路不用它：没发的那几行本来就是 0。

    行存 0..15 用 `levels="byte"`（bf16 精确到 256），**一列就够**；读出后每个 nibble 再过
    `bits_of_int` 展开成 4 位。**为什么是八个 head 而不是一个 8 行表**：一个 Head 只有一组
    `q`，要读某一行就得用 q 选那一行，所以“同一拍读八行”只能是八个节点（各读自己那行）。

    这条通路**不参与写回**：写回值仍来自载荷（`value_tokens`），商表的价值是“电路自己也把
    商拼一遍”，像 `tgt`/`link` 那样与参考机对账——验的是 `QD` 载荷本身，而不是多一条写路。
    """
    qd = feats["fam.QD"]
    dg = [feats[f"pay[{b}]"] for b in range(4)]                # 商位 nibble q
    lit = onehot_of_bits(g, [feats[f"pay[{b}]"] for b in range(4, 8)], QUO_DIGITS,
                         f"{name}.k")
    digs = []
    for k in range(QUO_DIGITS):
        digs.append(g.head(f"{name}.d{k}", "quo", rows=1, cols=1,
                           q={0: g.one},
                           k={0: Chan(lit[k])},
                           v={0: from_bits(dg)},
                           a=0, z={0: g.one}, outs=[(0, 1.0)],
                           beta=qd, levels="byte")[0])
    bits = []
    for k in range(QUO_DIGITS):
        bits += bits_of_int(g, digs[k], 0, 15, 4, f"{name}.d{k}.b")
    return {"bits": bits, "digits": digs, "k": lit, "wr": qd}


def quo_tokens(q: int, digits: int = QUO_DIGITS) -> list[int]:
    """32 位商 -> `QS` + `QD` 序列（`quo_table` 的逆）。高位在前，与参考机的除法次序一致。

    这串 token 落在**值记录之前**（`EXEC, QS, QD.., V, V, F2`）：值记录的段位靠“距终结拍
    多远”定位，那段前置条件不能被 QD 打乱，所以 digit 只能出现在值记录之前。
    """
    q &= 0xFFFFFFFF
    nib = [(q >> (4 * k)) & 0xF for k in range(digits)]
    k0 = max((k for k in range(digits) if nib[k]), default=0)
    return [tok("QS", k0)] + [tok("QD", (k << 4) | nib[k]) for k in range(k0, -1, -1)]


# ---------------- 一步执行：取指 -> 三个读口 -> ALU ----------------

def step_circuit(g: Graph, words: list[int], init: dict | None = None,
                 name: str = "rv", feats: dict[str, Expr] | None = None,
                 wb: tuple | None = None, base: int = 0,
                 ram_init: dict | None = None,
                 drom: list[int] | None = None,
                 text: list[bytes] | None = None) -> dict[str, Any]:
    """把一步执行拼起来，返回一个 dict：

      `cnt` / `cnt_i`  指令序号表（宽度由固件规模定，这里按页大小）与它的节点号
      `w`              ROM 字的 32 个位
      `sels`           rs1 / rs2 / rd 的行 one-hot
      `ports`          三个读口的 32 个位；`ports_i` 是对应三个 head 的节点号
      `wb`             写回的 `[(读出位, 通道)]`：值记录 + **链接值**（18 列，落在 bit2..19）；
                       `beta` 是它的写门
      `link`           链接值 `(PC + 4) >> 2` 的 18 位（电路自己算的，`J` 拍直接写 rd）
      `tgt`            三种跳转目标（分支 / JAL / JALR）与选完的那一份
      `alu`            alu() 的返回值
      `mem`            ram_table() 的返回值（`word` 是按地址读出的整字）；给了 `drom`
                       就多一项 `mem["drom"]`，两张表共用同一组输出列；给了 `text`
                       就再多一项 `mem["text"]`（第三张表，同列）
      `ld`             load 的值：从 `mem["word"]` 里取字节/半字并符号扩展
      `quo`            quo_table() 的返回值（`bits` 是从 `QD` 载荷拼出来的 32 位商）
      `sys`            syscall() 的返回值（`put`/`stop`/`chars`：`ecall` 的两个出口）
      `mem_i`          两块 bank 的节点号（探针往里塞初始状态）

    写回接上之后，一张寄存器表有 `len(wb)` 列（值记录每位 1~3 个来源，再加一列链接值），
    读出仍是 32 位。`wb` 允许把 `value_record` 的结果传进来复用：`init` 的行宽要靠它，
    不然探针得先建一遍再拆。

    **链接值是这一步唯一真的在电路里算出来的写值**（`link_word`）：它只需要 `cnt` 和
    常数，比写口早十几层。跳转目标（`jump_target`）也能算，但参与不了推进——原因写在
    它的 docstring 里。

    **读 DROM 与读 RAM 同一条路**：`drom` 给定时多建一张只读表，它 `q` 借 RAM 已经算好
    的 512 个行选择、`z` 借 bank 位与 `is_drom`、输出 `into` 同一组 `col`——所以 `load_value`
    只认一组 `word`，`lbu`/`lhu` 的字节/半字选择与符号扩展完全不变（它们是地址低位上的
    mux，与表本身无关）。两条 ram/drom 的 bank 门互斥，同一拍最多一个写者（§5 #10）。

    节点号是给探针用的：`evaluate(carry=...)` 用 `(节点号, "S", 0)` 作键塞状态，
    这样一台机器就能跑任意多条用例，不必每个用例重建一张图。

    `text` 给定时挂上 `<cpu_call>` 的文本缓冲（`text_buffer`）：`text[p]` 是第 p 个文本
    token 的 16 个字节，token id 用 `tok("TX", p)`。它和 DROM 一样是个只读表、共用 RAM
    的输出列，所以 `load_value` / `ld` 一行都不用改。注意三个表共用同列的前提是
    **地址不重叠**且只有一个写者：RAM 的 bank 门带 `is_ram`、DROM 带 `is_drom`、文本带
    `in_text`，三者两两互斥。
    """
    rows = len(words)
    feats = token_features(g, text=text) if feats is None else feats
    pc: dict = {}
    cnt = pc_anchor(g, rows, feats=feats, out=pc)
    w, _ = fetch(g, cnt, words, f"{name}.rom")
    U = uops_of(g, w, f"{name}.uop")
    sels = {"rs1": reg_index(g, w[19:24], f"{name}.rs1"),
            "rs2": reg_index(g, w[27:32], f"{name}.rs2"),
            "rd": reg_index(g, w[11:16], f"{name}.rd")}
    cols, beta = value_record(g, feats) if wb is None else wb
    link = link_word(g, pc["inc"], pc["carry"], base, f"{name}.link")
    # 链接值只在 JAL / JALR 的 J 拍上写。写值不经过 beta，所以每列自带这道门。
    gate = _node(g, _t_and([feats["fam.J"], U["jal"]]) + _t_and([feats["fam.J"], U["jalr"]]),
                 f"{name}.link.gate")
    cols = list(cols) + [(2 + i, Chan(_node(g, _t_and([gate, b]), f"{name}.wb.link{i}")))
                         for i, b in enumerate(link) if b.terms]
    ports = regs(g, sels, init=init, name=f"{name}.regs", wcols=cols,
                 wrow=sels["rd"], beta=beta + gate)
    out = alu(g, w, ports, f"{name}.alu", uop=U)
    tgt = jump_target(g, w, cnt, out, len(cnt), f"{name}.tgt")
    # 不连 DROM 也不连文本表时**不建** `addr_region`：那 2 个 dim 是给几张表互斥用的，
    # 单表用不上。文本表一上来就必须建——`is_ram` 让 16 块 RAM bank 在文本地址上闭嘴。
    is_ram = is_drom = None
    if drom or text is not None:
        is_ram, is_drom = addr_region(g, out["res"], f"{name}.mem")
    mem = ram_table(g, out["res"], feats, init=ram_init, name=f"{name}.ram", is_ram=is_ram)
    if drom:
        mem["drom"] = drom_table(g, out["res"], mem["sel"], drom, mem["word"],
                                 is_drom, f"{name}.drom")
    if text is not None:
        mem["text"] = text_buffer(g, out["res"], feats, mem["word"], f"{name}.txt")
    ld = load_value(g, mem["word"], out["low"], U, f"{name}.ld")
    quo = quo_table(g, feats, f"{name}.quo")
    sys = syscall(g, ports, U, f"{name}.sys")
    idx = {n.name: i for i, n in enumerate(g.nodes)}
    return {"cnt": cnt, "cnt_i": idx["cnt"], "w": w, "sels": sels,
            "ports": ports, "ports_i": {p: idx[f"{name}.regs.{p}"] for p in sels},
            "wb": cols, "beta": beta, "link": link, "tgt": tgt, "alu": out,
            "mem": mem, "ld": ld, "quo": quo, "sys": sys,
            "mem_i": [idx[f"{name}.ram.b{h}"] for h in range(RAM_BANKS)]}


# ---------------- 自检 ----------------

def _demo() -> None:
    import numpy as np

    from components.rv32i.graph import evaluate

    # ---- 1. 表本身 ----
    # `VOCAB == 246710` 这条冻结断言在 `constant.py` 的导入段（真源在那里，只写一份）
    print(f"token 层: {NFAM} 个家族 -> vocab {VOCAB}   载荷位 {NPAY}")

    # ---- 2. 编解码往返（全量，不抽样）----
    for tid in range(VOCAB):
        n, p = split(tid)
        assert tok(n, p) == tid, f"往返失败 {tid} -> ({n}, {p})"
    for n, (_, w) in FAMILIES.items():
        for p in (0, w - 1):
            assert split(tok(n, p)) == (n, p)
    for n in CONTROL_NAMES:
        assert split(tok(n)) == (n, 0)
    print("编解码往返: 全 %d 个 token  逐位一致  OK" % VOCAB)

    # ---- 3. 特征：每个家族取首/中/尾三个载荷，家族 one-hot 与载荷位都要对上 ----
    g = Graph(units=256, heads=16)
    feats = token_features(g)
    assert len(feats) == NFAM + NPAY + len(_FLAG), len(feats)
    samples = [tok(n, p) for n, (_, w) in FAMILIES.items()
               for p in sorted({0, w // 2, w - 1})]
    samples += [tok(n) for n in CONTROL_NAMES]
    m = evaluate(g, samples)
    for row, tid in enumerate(samples):
        n, p = split(tid)
        for i, nm in enumerate(NAMES):                       # 家族 one-hot 恰好一个 1
            v = m[row, next(iter(feats[f"fam.{nm}"].terms))]
            assert v == float(i == FAM_ID[n]), f"{tid} fam.{nm} = {v}"
        back = sum(int(m[row, next(iter(feats[f"pay[{b}]"].terms))]) << b for b in range(NPAY))
        assert back == p, f"{tid} 载荷读回 {back} != {p}"
        for k, col in _FLAG.items():
            assert m[row, next(iter(feats[k].terms))] == col[tid], f"{tid} {k}"
    print(f"特征: {len(feats)} 维（家族 one-hot {NFAM} + {NPAY} 位载荷 + {len(_FLAG)} 个标志）"
          f"  {len(samples)} 个样本逐项一致  OK")

    # ---- 4. cnt 装 n、and_ 解 one-hot：这是"PC 不存储"的具体形状 ----
    nrom, n = 4096, 0x1A5
    g2 = Graph(units=512, heads=32)
    f4 = token_features(g2)
    cnt = pc_anchor(g2, nrom, init_n=n, feats=f4)
    ks = {r: row_onehot(g2, cnt, r, "k") for r in sorted({0, 1, n, nrom - 1})}
    m2 = evaluate(g2, [tok("NT", 0)])
    bits = [int(m2[0, next(iter(e.terms))]) for e in cnt]
    assert bits == [(n >> c) & 1 for c in range(cnt_width(nrom))], f"cnt 读出 {bits}"
    for r, k in ks.items():
        got = m2[0, next(iter(k.terms))]
        assert got == float(r == n), f"row_onehot({r}) = {got}，n = {n}"
    print(f"cnt: 1x{cnt_width(nrom)} 表装 n=0x{n:X}   and_ 解码 4 个行号（含命中行）逐位一致  OK")
    g2.check_dag()
    print("check_dag: 无环（cnt 的写值走 taps，不读本拍）  OK")

    # ---- 5. 分组解码：4096 行必须分两组，否则 unit 数多一个数量级 ----
    nrom2 = 4096
    g4 = Graph(units=8192, heads=64)
    f5 = token_features(g4)
    sel = row_sel(g4, pc_anchor(g4, nrom2, init_n=n, feats=f5), nrom2, "sel")
    n_unit = sum(len(nd.terms) for nd in g4.nodes if type(nd).__name__ == "FFN")
    m4 = evaluate(g4, [tok("NT", 0)])
    hit = [r for r in range(nrom2) if m4[0, next(iter(sel[r].terms))] > 0]
    assert hit == [n], f"命中行 {hit} != [{n}]"
    print(f"分组解码: {nrom2} 行 /{cnt_width(nrom2)} 位 -> {n_unit} 个 unit"
          f"（每行一条长链要 {nrom2 * (cnt_width(nrom2) - 1)}）  命中 n=0x{n:X}  OK")

    # ---- 6. 链接值 + 三种跳转目标：8 行小 ROM，注入状态逐位对 ----
    # 固件 page 1 那一段里 **没有 jalr**（196 次跳转全是分支 + 3 次 jal），所以 jalr 这条
    # 支路只能在这里验：它的目标不是 ROM 字段，而是 ALU 结果右移两位，是三条路里唯一
    # 依赖寄存器读出的一条。ROM 字直接 `_pack` 出来，不过 decode——电路读的本来就是这个格式。
    nrow, x5, imm = 8, 0x4000C, 8
    ubne = (-2) & 0x7FF
    words6 = [_pack("addi", 1, 0, 1)] * nrow
    words6[3] = _pack("jalr", 1, 5, imm & 31, imm >> 5)
    words6[4] = _pack("jal", 1, 6 & 31, 6 >> 5 & 31, 6 >> 10)
    words6[5] = _pack("bne", ubne & 31, 1, 2, ubne >> 5)
    g6 = Graph(units=4096, heads=16)
    st6 = step_circuit(g6, words6, feats=token_features(g6), base=0)

    def read_bits(row, exprs) -> int:
        """一组位 dim -> 整数（常数 0 位是空 Expr，跳过）。"""
        v = 0
        for b, e in enumerate(exprs):
            if e.terms:
                v |= int(round(float(row[next(iter(e.terms))]))) << b
        return v

    def inject(n: int, rv: dict[int, int]):
        ncnt = len(st6["cnt"])
        carry = {(st6["cnt_i"], "S", 0):
                 {0: np.array([float((n >> b) & 1) for b in range(ncnt)])}}
        for ni in st6["ports_i"].values():
            carry[(ni, "S", 0)] = {r: np.array(record_row(rv.get(r, 0), st6["wb"]))
                                    for r in range(REGS)}
        return evaluate(g6, [EXEC], carry=carry)[0]

    # 链接值：cnt 是表的内容，PC = ROM_BASE + 4·cnt，所以链接值 = (PC+4)>>2 = 0x10000+cnt+1
    for n in range(nrow):
        got = read_bits(inject(n, {}), st6["link"])
        assert got == (ROM_BASE >> 2) + n + 1, f"cnt={n} 的链接值是 {got:#x}"
    # 三条跳转支路各一次（期望值都是手算的）：jalr 5 = (0x4000C+8)>>2 & 7；jal 6；bne 5-2
    cases = [(3, {5: x5}, 5, "jalr"), (4, {}, 6, "jal"), (5, {}, 3, "分支")]
    for n, rv, want, tag in cases:
        row = inject(n, rv)
        got = read_bits(row, st6["tgt"]["sel"])
        assert got == want, f"{tag}: 目标 {got} != {want}"
    print(f"链接值 {nrow}/{nrow} 条 == (PC+4)>>2  跳转目标 jalr/jal/分支 各一次全对（cnt 3 位） OK")

    # ---- 7. RAM：写一个字，四种 load 读回来 ----
    # 地址来自 ROM 的 rs1/imm（`res` 当地址的第二类指令），写值来自记录里的三个 `V`——
    # 与真机同一条路。0x200 是字索引 0x80（bank 0、行 128），跨 bank 的例子取 0x8000。
    v7 = 0xA1B2C3D4
    words7 = [_pack("sw", 0, 0, 0x200), _pack("lw", 0, 0, 0x200),
              _pack("lhu", 0, 0, 0x202), _pack("lbu", 0, 0, 0x201),
              _pack("lb", 0, 0, 0x203), _pack("sw", 0, 0, 0x8000)]
    g7 = Graph(units=4096, heads=32)
    st7 = step_circuit(g7, words7, feats=token_features(g7))
    n7 = len(st7["cnt"])
    carry7 = {(st7["cnt_i"], "S", 0): {0: np.zeros(n7)}}
    for ni in st7["ports_i"].values():
        carry7[(ni, "S", 0)] = {r: np.array(record_row(0, st7["wb"])) for r in range(REGS)}

    def goto(n: int) -> None:
        carry7[(st7["cnt_i"], "S", 0)] = {0: np.array([float((n >> b) & 1)
                                                     for b in range(n7)])}

    m7 = evaluate(g7, [EXEC] + store_tokens(v7) + [tok("SS0")], carry=carry7)
    got = read_bits(m7[-1], st7["mem"]["word"])
    assert got == v7, f"sw 之后 RAM 字是 {got:#010x}，应为 {v7:#010x}"
    # 小端：v7 的字节是 D4 C3 B2 A1；lb 是符号扩展的，lbu/lhu 不是
    want7 = {1: v7, 2: v7 >> 16, 3: 0xC3, 4: 0xFFFFFFA1}
    for row7, want in want7.items():
        goto(row7)
        got = read_bits(evaluate(g7, [EXEC], carry=carry7)[0], st7["ld"])
        assert got == want, f"第 {row7} 行 load 读到 {got:#010x} != {want:#010x}"
    goto(5)                                  # 另一块 bank：0x8000 -> 字索引 0x2000 -> bank 4
    m7 = evaluate(g7, [EXEC] + store_tokens(0x5A5A5A5A) + [tok("ST0")], carry=carry7)
    assert read_bits(m7[-1], st7["mem"]["word"]) == 0x5A5A5A5A, "跨 bank 的写没落到 ram4"
    print("RAM: 写一次（V V V SS0）-> lw/lhu/lbu/lb 各读一次全对（含符号扩展）"
          "  跨 bank 写一次  OK")

    # ---- 8. 除法：`QD` 载荷 -> quo 表（8 个 nibble），电路自己拼出商 ----
    # 记录形状是 `EXEC, QS, QD..., V, V, F2`：digit 在值记录**之前**，所以值记录的
    # lag1/lag2 段位规则不受影响（前置条件就是不能有 V 混进段位）。
    g8 = Graph(units=4096, heads=32)
    st8 = step_circuit(g8, [_pack("div", 1, 5, 6)], feats=token_features(g8))
    for q8 in (0xFFFFFFFF, 0x12345678, 0x00000005, 0x0):
        row8 = evaluate(g8, [EXEC] + quo_tokens(q8))[-1]
        got = read_bits(row8, st8["quo"]["bits"])
        assert got == q8, f"QD 拼出来的商是 {got:#010x}，应为 {q8:#010x}"
    print(f"除法: QD 八位载荷 -> quo 表 -> 拼回 32 位商  4 个样本全对（含 0 与满位）  OK")


if __name__ == "__main__":
    _demo()
