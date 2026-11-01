"""RV32I 神经 CPU 的电路描述：纯数据，不含 torch、不含 nn.Module。

每个残差标量是一个 dim（下标即整机 residual 的位置）；线性组合（Expr）免费，不占层。
节点只有两类，正好对应一个 transformer 层的两个子层：

  FFN    sum(coef * a * relu(b))     —— 编译成 MLP 的一组 unit
  Head   一个可写状态表               —— q/k/v/beta/alpha 全是 Expr，编译成一组投影

host 侧代码量 = O(节点数)，不随机器规模增长。这是"不是硬布线"的判据：写 8 条指令
和写 52 条指令，这个文件一行都不用改。

与 Qwenputer 的 q35/ir.py 同形，但有三处不同（因为 CPU 是自己的组件，不复用 LLM 的
attention）：

  1. 头不经过 fla 的 Gated DeltaNet。读就是一次 k-hot 求和，没有逐行 RMSNorm，
     所以状态行不需要 ballast 列（省掉 REF_COLS / A_REF / kernel_check）。
  2. 衰减由整数选择子 a in {0,1,2} 直接给出 {1, 0.5, 0}，不用 Qwen 的 exp(dt_bias) 表。
  3. 状态表按 bank 命名（cnt/rom/rom_raw/regs/quo/ram0/ram1），宽度任意，不由 head_dim 决定。

写规则：key 是 one-hot，于是 gated delta 的 beta*(v - k^T S) 就退化成整行覆写，读是
k-hot 求和。两边都不需要 Qwen/fla 的逐行 RMSNorm，也不需要 exp(dt_bias) 衰减表——
"不复用 LLM 组件"省下的就是这些。衰减在写和读之前施加（先乘 alpha，再写，再读）。

反馈规则：一张表的写值只能来自 taps（上一个 token 的值），不能来自它本 token 的读出。
没这条限制就会成环；check_dag() 会把它报出来，而不是安静地写进 0。所以"读 -> 算 ->
写回"总是至少跨一个 token，真机的 trace 也是这么切的。

dim 0 是"在片上的 token"指示子（chip mask），padding 上为 0。裸常数 1 就是它，所以
token 级门控天然成立，不需要另设有效位。

精确性是数值验的，不是靠断言：evaluate() 把同一条 token 流算两遍——float64 理想值，
以及 bf16 硬件模拟（每个被物化的张量先舍入到 bf16，累加在 fp32）——逐节点要求两者相等。
silu 在 0 / >= 18 / <= -104 三段上与 relu 完全一致，所以门控永远走 relu 的精确区间。

量级：残差里放 m 个 ballast 维、每个填 BALLAST，令 m*BALLAST^2/hidden 是 4 的幂，
RMSNorm 的分母就恒等于一个 2 的幂，本文件里的值就是"已经除掉那个幂"的单位值。
RESIDUAL_MAX 是这条前提的边界。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

Num = int | Fraction
ONE = 0  # 常数 1 所在的 dim

K = 128                  # silu 门控缩放：silu(K*n) == K*relu(n)，n 为整数
ALPHA = {0: 1.0, 1: 0.5, 2: 0.0}   # 整数选择子 -> 每 token 的衰减
RESIDUAL_MAX = 2.0 ** 28           # 超过它，RMSNorm 的分母就不再只由 ballast 决定

BANKS = {                          # bank -> 每行最少列数（0 表示由 Head 自己声明）
    "cnt": 1, "rom": 1, "rom_raw": 1, "regs": 1, "quo": 1, "ram0": 1, "ram1": 1,
}


# ---------------- 线性形式 ----------------

class Expr:
    """sum(coef * dim)；常数项落在 dim ONE 上。只能乘常数——乘积用 ffn 节点表达。"""

    __slots__ = ("terms",)

    def __init__(self, terms: dict[int, Num] | None = None):
        self.terms = {k: v for k, v in (terms or {}).items() if v != 0}

    def __add__(self, other: "Expr | Num") -> "Expr":
        other = as_expr(other)
        t = dict(self.terms)
        for k, v in other.terms.items():
            t[k] = t.get(k, 0) + v
        return Expr(t)

    __radd__ = __add__

    def __neg__(self) -> "Expr":
        return Expr({k: -v for k, v in self.terms.items()})

    def __sub__(self, other: "Expr | Num") -> "Expr":
        return self + (-as_expr(other))

    def __rsub__(self, other: "Expr | Num") -> "Expr":
        return as_expr(other) - self

    def __mul__(self, c: Num) -> "Expr":
        if not isinstance(c, (int, Fraction)):
            raise TypeError("Expr 只能乘常数（乘积请用 ffn/gate 表达）")
        return Expr({k: v * c for k, v in self.terms.items()})

    __rmul__ = __mul__

    def __truediv__(self, c: int) -> "Expr":
        return self * Fraction(1, c)

    def __repr__(self) -> str:
        return " + ".join(f"{v}*d{k}" for k, v in self.terms.items()) or "0"


def as_expr(x: "Expr | Num") -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, (int, Fraction)):
        return Expr({ONE: x})
    raise TypeError(f"无法转换为 Expr: {type(x).__name__}")


# ---------------- 节点 ----------------

@dataclass
class Dim:
    name: str
    kind: str   # "one" | "token" | "ffn" | "head"
    level: int  # 子层号；奇数 = head 子层，偶数 = ffn 子层


@dataclass
class Chan:
    """一路投影 + 4 个 conv tap（x_t, x_{t-1}, x_{t-2}, x_{t-3}），再过 relu。

    taps[0] 读本 token，是行选择的正常形式；taps[1:] 读历史 token，是反馈的唯一写法。
    历史值在 token 末尾快照，所以源 dim 可以比读它的节点更晚——这就是写回。
    """

    expr: Expr
    taps: tuple[float, ...] = (1.0,)


@dataclass
class FFN:
    name: str
    terms: list[tuple[Num, Expr, Expr]]  # coef * a * relu(b)
    out: int
    level: int
    into: bool = False                   # out 是已有 dim，本节点累加进去


@dataclass
class VHead:
    """同一组 query/key 的又一份 value 拷贝：自己的值、衰减、写门、输出门。

    用在“同一次行选择、两套数据”上：ROM 的 64 字节列取低字节和高字节，就是一个 q/k
    配两个 value 头。

    多个读口（寄存器 rs1/rs2）不走这里 —— 它们的 q 不同，必须是同一个 bank 上的
    多个 Head 节点，各自被写一遍。
    """

    v: dict[int, Chan]
    a: Expr
    z: dict[int, Expr]
    out: list[tuple[int, int, float]]     # (列, 残差 dim, 权重)
    beta: Expr | None = None
    into: bool = False


@dataclass
class Head:
    """一个可写状态表。q/k 定行，v 定列，beta 定是否写，a 定每 token 的衰减。"""

    name: str
    bank: str
    rows: int
    cols: int
    q: dict[int, Chan]                   # 行 -> 查询通道
    k: dict[int, Chan]                   # 行 -> key 通道（必须 one-hot 或全 0）
    v: dict[int, Chan]                   # 列 -> 写入值
    a: Expr                              # 衰减选择子 in {0,1,2}
    z: dict[int, Expr]                   # 列 -> 输出门
    out: list[tuple[int, int, float]]
    level: int
    beta: Expr | None = None             # 写门 in {0,1}；None = 恒 1
    into: bool = False
    vx: list[VHead] = field(default_factory=list)
    init: list[dict[int, np.ndarray]] | None = None   # 每个 value head 的初值：行 -> 列值
    levels: str = "bit"                  # "bit" | "byte" | "decay"

    def vheads(self) -> list[VHead]:
        return [VHead(self.v, self.a, self.z, self.out, self.beta, self.into), *self.vx]


# ---------------- 图 ----------------

class Graph:
    """电路描述。构造期做容量摊派（每子层最多多少 unit / 多少 head）。"""

    def __init__(self, units: int = 3264, heads: int = 8) -> None:
        self.dims: list[Dim] = [Dim("one", "one", 0)]
        self.token_fns: dict[int, Callable[[int], Num]] = {}
        self.nodes: list[FFN | Head] = []
        self.banks: dict[str, Head] = {}       # bank -> 首个 head，仅用于查容量
        self.one = Expr({ONE: 1})
        self.unit_cap = units                  # 每个 ffn 子层的 unit 上限
        self.head_cap = heads                  # 每个 head 子层的 key head 上限
        self.units: dict[int, int] = {}
        self.heads_used: dict[int, int] = {}

    # --- token 特征 ---------------------------------------------------------

    def token_feature(self, name: str, fn: Callable[[int], Num]) -> Expr:
        """token id -> 一个标量。fn 在编译期对整张词表枚举，所以可以是任意函数。"""
        d = self._dim(name, "token", 0)
        self.token_fns[d] = fn
        return Expr({d: 1})

    def const(self, v: Num, name: str = "const") -> Expr:
        d = self._dim(name, "token", 0)
        self.token_fns[d] = lambda _t: v
        return Expr({d: 1})

    # --- 前向引用 ---------------------------------------------------------

    def zeros(self, n: int, name: str = "z") -> list[Expr]:
        """占 n 个 dim，稍后用 define() 逐个填。

        反馈边必须要这个：寄存器的写值是本层之后才算出来的 ALU 输出，但写入口的
        参数要先于它建出来。taps 让这个循环在 token 轴上合法，zeros 让它在 Python
        里可写。没填的 dim 永远为 0，所以忘填不会造成随机行为，只会读到 0。
        """
        return [Expr({self._dim(f"{name}[{i}]", "ffn", 0): 1}) for i in range(n)]

    def define(self, target: Expr, terms, name: str = "ffn", min_level: int = 0) -> Expr:
        """填一个 zeros() 出来的 dim。"""
        return self.ffn(terms, name, into=target, min_level=min_level)

    # --- 节点 ---------------------------------------------------------------

    def _place(self, lvl: int, odd: bool, need: int, used: dict[int, int], cap: int) -> int:
        sub = lvl + 1
        while (sub % 2 == 1) != odd or used.get(sub, 0) + need > cap:
            sub += 1
        used[sub] = used.get(sub, 0) + need
        return sub

    def ffn(self, terms, name: str = "ffn", into: Expr | None = None, min_level: int = 0) -> Expr:
        terms = [(c, as_expr(a), as_expr(b)) for c, a, b in terms if c != 0]
        lvl = max(self.level(*(e for _, a, b in terms for e in (a, b))), min_level)
        sub = self._place(lvl, False, len(terms), self.units, self.unit_cap)
        if into is not None:
            (d,) = into.terms
            self.dims[d].level = max(self.dims[d].level, sub)
            out = d
        else:
            out = self._dim(name, "ffn", sub)
        self.nodes.append(FFN(name, terms, out, sub, into is not None))
        return Expr({out: 1})

    def and_(self, a: "Expr | Num", b: "Expr | Num", name: str = "and",
             min_level: int = 0) -> Expr:
        """a AND b（两者都必须是 0/1）：一个 unit，relu(a + b - 1)。

        地址解码全用它：第 p 个低位字面量 AND 第 h 个高位字面量，得到 q[p] 的 one-hot。
        这就是"地址是 one-hot 向量"的具体形式——芯片这一版的全部复杂度都在这里，
        而它恰好是 bf16 安全的（relu 的 0/1 输出，不经过标量比较、不需要 float64）。
        """
        return self.ffn([(1, self.one, as_expr(a) + as_expr(b) - 1)], name,
                        min_level=min_level)

    def head(self, name: str, bank: str, rows: int, cols: int, q: dict, k: dict, v: dict,
             a: "Expr | Num" = 0, z: dict | None = None, outs: list | None = None,
             min_level: int = 0, beta=None, into: list | None = None, vx: list | None = None,
             init=None, levels: str = "bit") -> list[Expr]:
        """加一个状态表。outs 是 [(列, 权重)]，每项占一个残差 dim。

        vx 是额外的 value 拷贝（(v, a, z, outs, beta) 的 dict 列表），共享本 head 的
        query/key；它们的输出 dim 排在第一个 value head 之后返回。
        """
        def chans(ch):
            return {i: c if isinstance(c, Chan) else Chan(as_expr(c)) for i, c in ch.items()}

        q, k, v = chans(q), chans(k), chans(v)
        a = as_expr(a)
        z = {i: as_expr(e) for i, e in (z or {}).items()}
        beta = None if beta is None else as_expr(beta)
        outs = list(outs or [])
        vx = [dict(x) for x in (vx or [])]
        if bank not in BANKS:
            raise ValueError(f"{name}: 未知 bank {bank}（可用 {sorted(BANKS)}）")
        if rows < 1 or cols < 1:
            raise ValueError(f"{name}: rows/cols 必须 >= 1")
        prev = self.banks.get(bank)          # 同一个 bank 的多个端口必须同形
        if prev is not None and (prev.rows, prev.cols) != (rows, cols):
            raise ValueError(f"{name}: bank {bank} 已是 {prev.rows}x{prev.cols}，"
                             f"与 {rows}x{cols} 不一致")
        for tag, keys, lim in (("q", q, rows), ("k", k, rows),
                               ("v", v, cols), ("z", z, cols)):
            bad = sorted(i for i in keys if not 0 <= i < lim)
            if bad:
                raise ValueError(f"{name}: {tag} 下标 {bad} 超出 [0, {lim})")
        if any(not 0 <= col < cols for col, _ in outs):
            raise ValueError(f"{name}: outs 列超出 [0, {cols})")

        exprs = [c.expr for ch in (q, k, v) for c in ch.values()] + [a, *z.values()]
        exprs += [] if beta is None else [beta]
        for x in vx:                          # 额外 value 头默认沿用本 head 的衰减
            x["v"], x["a"] = chans(x["v"]), as_expr(x.get("a", a))
            x["z"] = {i: as_expr(e) for i, e in x.get("z", {}).items()}
            x["beta"] = None if x.get("beta") is None else as_expr(x["beta"])
            bad = sorted(i for i in x["v"] if not 0 <= i < cols)
            bad += sorted(i for i in x["z"] if not 0 <= i < cols)
            if bad:
                raise ValueError(f"{name}: 额外 value 头的下标 {bad} 超出 [0, {cols})")
            exprs += [c.expr for c in x["v"].values()] + [x["a"], *x["z"].values()]
            exprs += [] if x["beta"] is None else [x["beta"]]
        lvl = max(self.level(*exprs), min_level)
        sub = self._place(lvl, True, 1, self.heads_used, self.head_cap)

        def out_dims(tag, outs_, into_):
            if into_ is None:
                return [self._dim(f"{name}{tag}[{col}]", "head", sub) for col, _ in outs_]
            if len(into_) != len(outs_):
                raise ValueError(f"{name}{tag}: into 与 outs 长度不一致")
            dims_ = []
            for e in into_:
                (d_,) = e.terms
                self.dims[d_].level = max(self.dims[d_].level, sub)
                dims_.append(d_)
            return dims_

        dims = out_dims("", outs, into)
        extra, ret = [], [Expr({d: 1}) for d in dims]
        for i, x in enumerate(vx):
            xd = out_dims(f".v{i + 1}", x["outs"], x.get("into"))
            extra.append(VHead(x["v"], x["a"], x["z"],
                               [(col, d, w) for (col, w), d in zip(x["outs"], xd)],
                               x["beta"], x.get("into") is not None))
            ret += [Expr({d: 1}) for d in xd]
        self.nodes.append(Head(name, bank, rows, cols, q, k, v, a, z,
                               [(col, d, w) for (col, w), d in zip(outs, dims)], sub,
                               beta, into is not None, extra, init, levels))
        self.banks.setdefault(bank, self.nodes[-1])
        return ret

    # --- 审视 ---------------------------------------------------------------

    def level(self, *exprs: Expr) -> int:
        return max((self.dims[d].level for e in exprs for d in e.terms), default=0)

    def n_layers(self) -> int:
        top = max((n.level for n in self.nodes), default=0)
        return (top + 1) // 2

    def stats(self) -> dict:
        ffns = [n for n in self.nodes if isinstance(n, FFN)]
        hds = [n for n in self.nodes if isinstance(n, Head)]
        return {
            "layers": self.n_layers(),
            "dims": len(self.dims),
            "heads": len(hds),
            "ffn_units": sum(len(n.terms) for n in ffns),
            "max_units": max(self.units.values(), default=0),
            "max_heads": max(self.heads_used.values(), default=0),
            "banks": {n.bank: (n.rows, n.cols) for n in hds},
        }

    def _dim(self, name: str, kind: str, level: int) -> int:
        self.dims.append(Dim(name, kind, level))
        return len(self.dims) - 1

    def check_dag(self) -> None:
        """数据流无环：同 token（lag 0）的边必须指回更早的 level。

        跨 token 的边（taps[0] == 0）不在此列——它读的是上一个 token 的值，而那个值
        本来就在本 token 的后面才被算出来。这条例外正是"写回"能成立的唯一理由：
        一张表的写值只能来自 taps，不能来自它自己本 token 的读出。试图后者会在这里
        报错，而不是安静地写进 0。
        """
        for node in self.nodes:
            for d, same_token in inputs(node):
                if same_token and self.dims[d].level >= node.level:
                    raise ValueError(
                        f"{node.name}: 消费 d{d}({self.dims[d].name}) 的 level "
                        f"{self.dims[d].level} 不早于节点 level {node.level} —— 成环。"
                        f"写回必须走 taps（跨 token），不能依赖本 token 的读出")


def inputs(node: FFN | Head):
    """列出节点消费的 (dim, 是否读本 token 的值)。"""
    if isinstance(node, FFN):
        for _, a, b in node.terms:
            for e in (a, b):
                for d in e.terms:
                    yield d, True
        return
    for ch in (node.q, node.k, node.v):
        for c in ch.values():
            for d in c.expr.terms:
                yield d, bool(c.taps[0])
    for e in (node.a, *node.z.values()):
        for d in e.terms:
            yield d, True
    if node.beta is not None:
        for d in node.beta.terms:
            yield d, True


# ---------------- 数值求值：float64 理想值 + bf16 硬件模拟 ----------------

class ExactnessError(AssertionError):
    pass


def bf16(x: np.ndarray) -> np.ndarray:
    """舍入到最近的 bf16（ties to even），返回 float64。"""
    f = np.asarray(x, dtype=np.float32).copy()
    u = f.view(np.uint32)
    u += 0x7FFF + ((u >> 16) & 1)
    u &= 0xFFFF0000
    return f.astype(np.float64)


def is_bf16(x: np.ndarray) -> np.ndarray:
    return (bf16(x) == x) & np.isfinite(x)


def silu_ok(x: np.ndarray) -> np.ndarray:
    """silu(x) == relu(x)，fp32 与 bf16 都成立：0、>=18（sigmoid 舍入到 1）、<=-104（下溢到 0）。"""
    return (x == 0) | (x >= 18) | (x <= -104)


def require(ok, what: str, x) -> None:
    ok, x = np.asarray(ok), np.asarray(x)
    if not np.all(ok):
        t = np.unravel_index(int(np.argmin(ok)), ok.shape)
        raise ExactnessError(f"{what}: 坏值 {x[t]!r} 在 {t}")


LEVELS = {"bit": np.array([0.0, 1.0]),
          "byte": np.arange(256.0),
          "decay": np.array([0.0] + [2.0 ** -k for k in range(0, 127)])}


def ev(e: Expr, m: np.ndarray) -> np.ndarray:
    if not e.terms:
        return np.zeros(m.shape[0])
    idx = np.fromiter(e.terms.keys(), dtype=np.int64, count=len(e.terms))
    coef = np.array([float(c) for c in e.terms.values()])
    return m[:, idx] @ coef


def conv(c: Chan, m: np.ndarray, check: bool, hist: tuple = (),
         gate: bool = True) -> np.ndarray:
    """conv 投影 + relu。``hist`` 是前几个 token 的投影值（最近的在最后）。

    历史值必须由 evaluate() 在 token 末尾统一快照，不能在这里顺手存：被 taps 读的
    那个 dim 往往在本 token 里更晚才算出来，读的当下它还是 0。

    ``gate=True`` 用于 q/k：输出只能是 0/1（一次精确的行选择）。
    ``gate=False`` 用于 v：输出是写入的数据，取值范围由 head 的 levels 管。
    这里不用 silu，也不乘 K：CPU 是自己的组件，门控就是 relu，不需要把门推到
    silu 的精确区间里。
    """
    if len(c.taps) > 4:
        raise ValueError(f"taps 最多 4 个（历史只留 3 个），得到 {len(c.taps)}")
    x = ev(c.expr, m)
    if check:
        require(is_bf16(x), "投影", x)
    n = len(x)
    xx = np.concatenate([np.asarray(hist, dtype=np.float64), x]) if len(hist) else x
    p = len(xx) - n
    out = np.zeros(len(xx))
    for lag, w in enumerate(c.taps):
        if w:
            out[lag:] += w * xx[: len(xx) - lag]
    out = np.maximum(out[p:], 0.0)
    if check:
        require(is_bf16(out), "conv 输出", out)
        if gate:
            # 行选择子的取值只能是 0/1，否则一个 q 行就带上了倍数，而乘法不在投影里。
            # 同一行被选中两次（add x1,x1,x1）靠两个读口相加，不靠 q=2。
            require(np.isin(out, (0.0, 1.0)), "门控必须是 0/1", out)
    return out


def tapped(g: Graph):
    """所有带 lag>=1 的通道：(历史键, 被延迟的线性形式)。

    历史键与 eval_head 内部用的键一致，只是前面多一个 "h"。
    """
    for i, node in enumerate(g.nodes):
        if isinstance(node, FFN):
            continue
        for part in ("q", "k"):
            for s, c in getattr(node, part).items():
                if any(c.taps[1:]):
                    yield (i, part, s), c.expr
        for x_, vh in enumerate(node.vheads()):
            for s, c in vh.v.items():
                if any(c.taps[1:]):
                    yield (i, "v", x_, s), c.expr


def evaluate(g: Graph, tokens: list[int], check: bool = True,
             carry: dict | None = None) -> np.ndarray:
    """跑一条 token 流，返回每个 token 之后的残差 (n_tokens, n_dims)。

    ``carry`` 用于长流分块：head 的状态和 conv 窗口从一次调用传到下一次（decode 时
    一次一个 token 也走这里）。不传就建一个——taps 依赖它，默默丢掉 taps 会让错误
    安静地通过。

    逐个 token 走完全图，不能一次性向量化：taps 读的是“上一个 token 的那个 dim”，
    而那个 dim 在本 token 里要更晚才被算出来。整列一起算时它还是 0，反馈边就断了。

    一个 token 内的事件次序：先乘 alpha，再写，再读。所以同一 token 上的读看到的是
    刚写进去的值；要读旧值，就把 token 拆开（真机的 trace 就是这么做的）。
    """
    g.check_dag()
    carry = {} if carry is None else carry
    sweep = list(tapped(g))
    n = len(tokens)
    m = np.zeros((n, len(g.dims)))
    toks = np.asarray(tokens, dtype=np.int64)
    m[:, ONE] = toks >= 0
    uniq, inv = np.unique(toks, return_inverse=True)
    for d, fn in g.token_fns.items():
        col = np.array([float(fn(int(u))) if u >= 0 else 0.0 for u in uniq])[inv]
        if check:
            require(is_bf16(col), f"token 特征 {g.dims[d].name}", col)
        m[:, d] = col
    for t in range(n):
        row = m[t : t + 1]
        for i, node in enumerate(g.nodes):
            try:
                if isinstance(node, FFN):
                    eval_ffn(node, row, check)
                else:
                    eval_head(node, row, check, carry, i)
            except ExactnessError as e:
                raise ExactnessError(f"{node.name} @t={t}: {e}") from None
        for key, e in sweep:                      # 全图跑完才快照，否则拿到的是半成品
            carry[("h",) + key] = (*carry.get(("h",) + key, ()), float(ev(e, row)[0]))[-3:]
    if check and n:
        big = np.abs(m[:, 1:]).max(0)
        if len(big) and big.max() > RESIDUAL_MAX:
            d = int(big.argmax()) + 1
            raise ExactnessError(
                f"{g.dims[d].name}: 残差值 {big[d - 1]!r} 会参与 RMSNorm 的分母"
                f"（上限 {RESIDUAL_MAX}）")
    return m


def eval_ffn(node: FFN, m: np.ndarray, check: bool) -> None:
    ideal = np.zeros(m.shape[0])
    hw = np.zeros(m.shape[0])
    for c, a, b in node.terms:
        bv, av = ev(b, m), ev(a, m)
        ideal += float(c) * av * np.maximum(bv, 0.0)
        if check:
            gate = bf16(K * bv)
            require(silu_ok(gate), "门控超出 silu 的精确区间", gate)
            up = bf16(av / K)
            hw += float(c) * bf16(np.maximum(gate, 0.0) * up)
    if check:
        hw = bf16(hw)
        require(hw == ideal, "bf16 输出与精确值不符", np.stack([hw, ideal], 1)[:, 0])
    if node.into:
        ideal = m[:, node.out] + ideal
        if check:
            require(is_bf16(ideal), "残差累加", ideal)
    m[:, node.out] = ideal


def eval_head(node: Head, m: np.ndarray, check: bool, carry: dict | None = None,
              ni: int = 0) -> None:
    carry = {} if carry is None else carry
    n = m.shape[0]

    def hist(*k):
        return carry.get(("h",) + k, ())

    q = (np.stack([conv(c, m, check, hist(ni, "q", s)) for s, c in node.q.items()], 1)
         if node.q else np.zeros((n, 0)))
    k = (np.stack([conv(c, m, check, hist(ni, "k", s)) for s, c in node.k.items()], 1)
         if node.k else np.zeros((n, 0)))
    qi, ki = list(node.q), list(node.k)
    if check:
        nz = (k > 0).sum(1)
        require(nz <= 1, "key 必须 one-hot", nz)
        require(~((nz > 0) & ~(m[:, ONE] > 0)), "key 出现在非 chip token 上", nz)
        for t in np.nonzero((q > 0).sum(1) > 1)[0]:
            vals = q[t][q[t] > 0]
            require(vals == vals[0], "query 必须等值 multi-hot", vals)
    has_key = k.any(1) if k.shape[1] else np.zeros(n, dtype=bool)
    key_row = [ki[int(np.argmax(k[t]))] if has_key[t] else None for t in range(n)]
    has_q = q.any(1) if q.shape[1] else np.zeros(n, dtype=bool)

    for x_, vh in enumerate(node.vheads()):
        v = np.zeros((n, node.cols))
        for col, c in vh.v.items():
            v[:, col] = conv(c, m, check, hist(ni, "v", x_, col), gate=False)
        a = ev(vh.a, m)
        z = {col: ev(e, m) for col, e in vh.z.items()}
        beta = np.ones(n) if vh.beta is None else ev(vh.beta, m)
        writes = has_key & (beta > 0)
        if check:
            require(is_bf16(a) & np.isin(a, list(ALPHA)), "衰减选择子 a", a)
            require((beta == 1.0) | ((beta <= 0) & (beta == np.round(beta))), "写门 beta", beta)
            for col, zz in z.items():
                require(is_bf16(zz) & np.isin(zz, (0.0, 1.0)), f"输出门 z[{col}]", zz)
            written = v[writes]
            if len(written):
                require(np.isin(written, LEVELS[node.levels]),
                        f"状态行数据（{node.levels}）", written)

        if carry is not None and (ni, "S", x_) in carry:
            state = carry[ni, "S", x_]
        else:
            init = (node.init or [{}] * (x_ + 1))
            state = ({r: np.asarray(row, dtype=np.float64).copy()
                      for r, row in init[x_].items()} if x_ < len(init) else {})
            if check and state:
                rows = np.stack(list(state.values()))
                require(np.isin(rows, LEVELS[node.levels]),
                        f"初值（{node.levels}）", rows)

        out = np.zeros((n, node.cols))
        events = np.nonzero((a != 0) | writes | has_q)[0]
        for t in events:
            al = ALPHA[int(round(float(a[t])))]
            if al == 0.0:
                state.clear()
            elif al == 0.5:
                state = {s: row * 0.5 for s, row in state.items()}
            if writes[t]:
                state[key_row[t]] = v[t].copy()
            if has_q[t]:
                o = sum((state[s] for s, w in zip(qi, q[t]) if w > 0 and s in state),
                        np.zeros(node.cols))
                if check:
                    require(is_bf16(o), "读出", o)
                out[t] = o
        if carry is not None:
            carry[ni, "S", x_] = state

        for col, d, w in vh.out:
            zz = z.get(col, np.ones(n))
            val = out[:, col] * np.maximum(zz, 0.0)
            if check:
                require(is_bf16(val * w), f"head 输出列 {col}", val * w)
            if vh.into:
                m[:, d] += val * w
                if check:
                    require(is_bf16(m[:, d]), "head 输出累加", m[:, d])
            else:
                m[:, d] = val * w


# ---------------- 自检：一个 4 行寄存器堆的写-读-衰减 ----------------

def _demo() -> None:
    """最小可运行的机器：写 regs、读 regs、一次衰减、一个 FFN unit，再验两条负向。"""
    g = Graph(units=64, heads=4)

    # token 约定（这里直接用整数）：0=NT, 1=W(d0), 2=R(d0), 3=W(d1), 4=R(d1), 5=NT+halve
    k_row0 = g.token_feature("k_row0", lambda t: 1.0 if t == 1 else 0.0)
    k_row1 = g.token_feature("k_row1", lambda t: 1.0 if t == 3 else 0.0)
    q_row0 = g.token_feature("q_row0", lambda t: 1.0 if t == 2 else 0.0)
    q_row1 = g.token_feature("q_row1", lambda t: 1.0 if t == 4 else 0.0)
    beta = g.token_feature("is_w", lambda t: 1.0 if t in (1, 3) else 0.0)
    halve = g.token_feature("halve", lambda t: 1.0 if t == 5 else 0.0)
    one = g.const(1.0, "one_data")

    # 一张 2 行 1 列的表，两个端口都在同一个 head 上：写口 = k + beta，读口 = q。
    rd = g.head("regs", "regs", rows=2, cols=1,
                q={0: Chan(q_row0), 1: Chan(q_row1)},
                k={0: Chan(k_row0), 1: Chan(k_row1)},
                v={0: Chan(one)},
                a=halve, z={0: g.one}, outs=[(0, 1.0)],
                beta=beta, levels="bit")[0]
    # 一个 FFN unit：coef * 1 * relu(rd)，等价于直接复制 rd
    cp = g.ffn([(1, g.one, rd)], "copy")

    toks = [1, 5, 2, 3, 4]
    m = evaluate(g, toks)
    d_rd, d_cp = next(iter(rd.terms)), next(iter(cp.terms))
    print("自检 1  stats:", g.stats())
    print("自检 1  t=2 读 regs[0]（t=1 减半后）:", m[2, d_rd])
    print("自检 1  t=4 读 regs[1]:", m[4, d_rd])
    print("自检 1  FFN 复制:", m[4, d_cp])
    assert m[2, d_rd] == 0.5 and m[4, d_rd] == 1.0 and m[4, d_cp] == 1.0

    # 自检 2：token 特征不是 bf16 值 -> 必须抛
    bad = Graph()
    bad.token_feature("third", lambda t: 1.0 / 3.0)
    bad.ffn([(1, bad.one, bad.one)], "ffn0")
    try:
        evaluate(bad, [0])
    except ExactnessError as e:
        print("自检 2  非 bf16 特征被拦:", str(e).splitlines()[0])
    else:
        raise SystemExit("自检 2 失败：1/3 没有被拦")

    # 自检 3：silu 门控离开精确区间 -> 必须抛（0.125 是 bf16 值，但 K*0.125 = 16 < 18）
    bad2 = Graph()
    x = bad2.token_feature("x", lambda t: 0.125)
    bad2.ffn([(1, bad2.one, x)], "ffn0")
    try:
        evaluate(bad2, [0])
    except ExactnessError as e:
        print("自检 3  silu 越界被拦:", str(e).splitlines()[0])
    else:
        raise SystemExit("自检 3 失败：0.125 的门没有被拦")

    # 自检 4：一个 key head 配两个 value 拷贝 —— ROM 的低字节/高字节就必须这么描述。
    # MiniRainVLM 的 GQA（value head 数恒等于 kv head 数）写不出这个形状；这里不需要。
    g2 = Graph(units=8, heads=2)
    kw = g2.token_feature("kw", lambda t: 1.0 if t == 1 else 0.0)
    qr = g2.token_feature("qr", lambda t: 1.0 if t == 2 else 0.0)
    lo, hi = g2.head("rom", "rom", rows=2, cols=1,
                     q={0: Chan(qr)}, k={0: Chan(kw)},
                     v={0: Chan(g2.const(2.0, "two"))},
                     a=0, z={0: g2.one}, outs=[(0, 1.0)], levels="byte",
                     vx=[dict(v={0: Chan(g2.const(3.0, "three"))},
                              z={0: g2.one}, outs=[(0, 1.0)])])
    m2 = evaluate(g2, [1, 2])
    print("自检 4  同一 key 两套 value:", m2[1, next(iter(lo.terms))],
          m2[1, next(iter(hi.terms))])
    assert m2[1, next(iter(lo.terms))] == 2.0 and m2[1, next(iter(hi.terms))] == 3.0

    # 自检 5：地址解码 = AND。两个 one-hot 相乘得到一个 0/1 维，全程 bf16。
    g3 = Graph(units=8, heads=2)
    p0 = g3.token_feature("p0", lambda t: 1.0 if t in (1, 3) else 0.0)
    h0 = g3.token_feature("h0", lambda t: 1.0 if t in (1, 2) else 0.0)
    sel = g3.and_(p0, h0, "sel")
    m3 = evaluate(g3, [1, 2, 3])
    got = list(m3[:, next(iter(sel.terms))])
    print("自检 5  AND 地址解码:", got)
    assert got == [1.0, 0.0, 0.0]


if __name__ == "__main__":
    _demo()


# 待办（下一轮）：把 RV32 的每个算子写成本文件里的节点；编译期再写 emit（Expr -> 权重）。
