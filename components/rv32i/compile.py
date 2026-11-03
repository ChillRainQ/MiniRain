"""
compile.py: 把 Graph 编译成 torch 权重。不含 nn.Module、不含训练、不含 float64 设备语义。

名字与内置 compile() 同名。本模块内不用内置那个，调用方也只在这个名字上拿到它；
真要调内置的就写 `builtins.compile`。换来的好处是名字直白：输入电路描述，输出可装载的权重，
这一步就叫编译。

产物是 torch.Tensor（默认 bf16），可以直接 load_state_dict 进 M2 的执行器或宿主模型。
graph.py 仍然是纯 numpy —— 它只做数值验证（float64 理想值 vs bf16 模拟）；
这里是权重的物化层，所以引入 torch 是分层的合理演进，不是把 graph 拖进深度学习栈。

编译产物的形状（M2 的执行器按此实现，M4 的宿主段也按此接）：

  残差 x = [host(d0) | chip(n_dims) | 零区 | ballast(m)]，宽度 hidden = m * 4^k
  chip 的第 d 个 dim 住在 x[d0 + d]。dim 0 是 chip mask，所以"常数 1"就是 x[d0]，
  不需要 bias 项。Linear 一律 bias=False。

  每个子层是一层加性更新（sub 奇数 = head，偶数 = ffn）：

    FFN   gate = bf16(Wb @ x)             Wb 每行 = K * b      (K = 128)
          up   = bf16(Wa @ x)             Wa 每行 = a / K
          h    = bf16(relu(gate) * up)    = a * relu(b)
          x   += Wo @ h                   Wo[slot(out), unit] = coef

    Head  p     = W @ x                   q/k/v 每个通道一行
          v_t   = relu(sum_lag taps[lag] * p_{t-lag})    lag 0 是本 token
          alpha = Walpha @ x              == 1 - 0.5*a，直接算，不查表
          S    <- S * alpha;  S[key] <- v;  o = sum(q 选中的行)
          x   += Wo @ (o * z)

lag >= 1 的那几路读的是**上一个 token 末尾**的投影快照，不是本 token 投影当时的
值：被 taps 引用的 dim 往往在本 token 里更晚才算出来（寄存器的写值就是 ALU 的
结果，而读口在 ALU 之前），所以投影当下看到的是 0，快照才看得到结果。这就是
"写回"在权重里的全部含义，也是 run() 里那次末尾快照存在的理由。

为什么残差用 += 就对：graph.py 里非 into 的写是覆写，但那个 dim 只有一个写者、
写前是 0，所以 += 等价；into 的 dim 有多个写者，+= 正好是累加。两条语义在同一个
处理方式下都成立，因此不需要掩码，也不需要清零。

为什么 K=128：silu(128n) == 128*relu(n) 对整数 n 成立（sigmoid 在 128*1 处已舍入到 1，
在 128*(-1) 处下溢），所以 a*relu(b) 不需要真乘法，只需要 silu 和逐元素乘。
graph.evaluate 用数值验证这条；compile 用 need_bf16 验证系数侧的那一半。

为什么 alpha 是线性的：ALPHA[a] = 1 - 0.5a 恰好在三个取值上都成立（0→1, 1→0.5,
2→0）。所以"每 token 的衰减"就是一个投影，不需要 Qwen 的 exp(dt_bias) 表，也不需要
小查找表。0.5 是 2 的幂，bf16 精确。

为什么权重存 bf16：need_bf16 已经保证每个落进去的数都是 bf16 可表示的，所以存 bf16
零损失，而且正好是设备侧要读的精度。乘法的累加仍在 float64（run() 的验具语义，与
graph.evaluate 同构）；M2 的执行器换成 fp32 累加，那时要重跑 J1 且仍要求容差 0。

权重 schema（torch.save 的 {"manifest": ..., "state": {...}}）：

  manifest       dict: {d0, n_dims, m, hidden, log2_ballast, log2_den, vocab,
                        feat_dims, layers:[{"kind": "ffn"|"head", ...}],
                        banks:{bank: [rows, cols]}}
  feat           (vocab, len(feat_dims))          bf16
  L{i}_Wa/Wb/Wo  ffn 层：(nu, hidden) / (nu, hidden) / (hidden, nu)
  L{i}_Wq_p{s}   head 层：第 s 个 q 通道，(hidden,)；Wk/Wv/Wz/Wbeta/Walpha 同形
  L{i}_Wo_v{j}   head 层第 j 个 value head：(hidden, cols)
  S_{i}_{j}      第 i 层第 j 个 value head 的状态初值：(rows, cols)
"""

from __future__ import annotations

import heapq
import json
import sys
from dataclasses import dataclass, field, replace
from typing import Iterable
from pathlib import Path

import torch

_DIRECT = __package__ in (None, "")
if _DIRECT:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "components.rv32i"

from .graph import (ALPHA, K, ONE, Chan, ExactnessError, Expr, FFN, Graph, Head,
                    silu_ok)

DTYPE = torch.bfloat16             # 权重的存储精度：设备侧读什么就存什么
ACC = torch.float64                # run() 的累加精度，与 graph.evaluate 同构

LOG2_BALLAST = 40                  # ballast 维填 2^40（量级远大于任何 chip 状态值）
BALLAST = float(2.0 ** LOG2_BALLAST)


# ---------------- 残差布局 ----------------

@dataclass(frozen=True)
class Layout:
    """残差的切分。hidden 必须是 m*4^k（m = ballast 个数），RMSNorm 分母才恒为 2 的幂。

    [0, d0)                     host 的 hidden（standalone 时全 0）
    [d0, d0 + n_dims)           chip 的每一个槽（不是每一维：见 slot_of）
    [d0 + n_dims, hidden - m)   未用的零区（补到 m*4^k 留下的空隙）
    [hidden - m, hidden)        ballast，每维填 2^log2_ballast

    零区必须真的为零，否则它也会进分母。所以 compile 会把零区当着面查一遍。

    `slot_of` 是 dim -> 本地槽的映射（`plan_slots` 的结果）。空元组表示恒等——
    M1 的一维一槽；非空时 `n_dims` 是**槽数**而不是 dim 数，多个生命周期不重叠的
    dim 共用一个槽。
    """

    d0: int
    n_dims: int
    hidden: int
    m: int
    log2_ballast: int = LOG2_BALLAST
    slot_of: tuple[int, ...] = ()

    @property
    def log4(self) -> int:
        """hidden / m == 4 ** log4。"""
        return _log4(self.hidden // self.m) if self.m and self.hidden % self.m == 0 else -1

    @property
    def ballast0(self) -> int:
        return self.hidden - self.m

    @property
    def log2_den(self) -> int:
        """RMSNorm 分母 = 2 ** log2_den。sqrt(m * BALLAST^2 / hidden) 恒等于它。"""
        return self.log2_ballast - self.log4

    def slot(self, d: int) -> int:
        return self.d0 + (self.slot_of[d] if self.slot_of else d)

    def slot_at(self, s: int) -> int:
        """本地槽号 -> 残差下标（不经 dim）。"""
        return self.d0 + s

    def slots(self, dims) -> torch.Tensor:
        return torch.tensor([self.slot(d) for d in dims], dtype=torch.long)

    def check(self) -> None:
        if self.m < 1:
            raise ValueError(f"m={self.m} 至少 1（ballast 是分母的唯一来源）")
        if self.log4 < 1:
            raise ValueError(f"hidden={self.hidden} 不是 m={self.m} 的 4 的幂倍"
                             f"（hidden/m={self.hidden / self.m}）—— RMSNorm 分母不会是 2 的幂")
        if self.hidden % 128:
            raise ValueError(f"hidden={self.hidden} 必须是 128 的整数倍")
        if self.hidden - self.m < self.d0 + self.n_dims:
            raise ValueError(f"放不下：hidden-m={self.hidden - self.m} < "
                             f"d0+n_dims={self.d0 + self.n_dims}")


def _log4(n: int) -> int:
    """n == 4^k 时返回 k，否则 -1。"""
    if n < 1:
        return -1
    k = 0
    while n % 4 == 0:
        n //= 4
        k += 1
    return k if n == 1 else -1


def pick_layout(n_dims: int, d0: int = 0, log2_ballast: int = LOG2_BALLAST,
                m_max: int = 64) -> Layout:
    """最小的 hidden = m*4^k，且 hidden - m >= d0 + n_dims（ballast 自己占最后 m 槽）。"""
    best = None
    for m in range(1, m_max + 1):
        for k in range(1, 24):
            h = m * 4 ** k
            if h - m < d0 + n_dims:
                continue
            if h % 128:
                break                     # h 随 k 单调增，本条 m 再大也不会整除 128
            if best is None or h < best.hidden:
                best = Layout(d0, n_dims, h, m, log2_ballast)
            break
    if best is None:
        raise ValueError(f"{n_dims} dims（d0={d0}）放不进任何 m*4^k <= {m_max}*4^23 的布局")
    best.check()
    return best


# ---------------- 槽位复用（M1 欠的那一笔） ----------------

@dataclass
class SlotPlan:
    """dim -> 残差槽。一维一槽是一份特例（`identity`）。

    位置（pos）不是层号，是**执行位置**：同一 level 的 FFN 合在一层算一个位置，
    每个 head 各算一个位置。它与 `compile()` 里 `layers` 的下标一一对应，但压槽会
    在 head 前面插一层“清零层”，所以真正的层下标要查 `Compiled.pos_layer`。
    """

    slot_of: list[int]                  # dim -> 本地槽
    n_slots: int                        # 一共多少槽（= 峰值并发数）
    pos_of: list[int]                   # 节点号 -> 位置号
    n_pos: int                          # 位置个数（一维一槽时就是层数）
    first: list[int]                    # dim -> 首次写的位置（-1 = token 特征，每 token 开头写）
    clears: dict[int, list[int]]        # 位置号 -> 这一位置要先清零的本地槽
    final: list[int]                    # token 末尾仍然持有自己槽的 dim（只有它们可读）

    @staticmethod
    def identity(nd: int, pos_of: list[int] | None = None, n_pos: int = 0,
                 first: list[int] | None = None) -> "SlotPlan":
        """一维一槽：不共用、不清零，`final` 是全部 dim。"""
        return SlotPlan(list(range(nd)), nd, list(pos_of or ()), n_pos or nd,
                        list(first or ()), {}, list(range(nd)))


def plan_slots(g: Graph, order: list[int] | None = None,
               keep: Iterable[int] = ()) -> SlotPlan:
    """按生命周期给每个 dim 分槽。

    规则只有三条：

      1. 一个槽在某一位置被占，直到它的**最后一个读者**过去；下一个 dim 才能接
      2. 被 lag>=1 的 taps 读的 dim 得活到 token 末尾（快照读的是末尾那个 x）
      3. `keep` 里的 dim 也活到 token 末尾——它们**被 Python 读**，没有图内的读者

    第 3 条是给 `ChipSegment.read()` 用的：`sys` 的 `put`/`stop`/`chars` 写完之后图里
    再没人读它们，按第 1 条它们的槽当场就该回收，于是末尾残差那一列已经是别的 dim
    的了（实测 `put` 读出 0.0、`chars[7]` 读到相邻 dim 的值）。凡是要从残差里取的
    dim 都必须进 `keep`，否则读到的是**复用者**的值，而且不会报错。

    剩下要处理的就是“接槽时里面还有旧值”。残差是 `+=`，所以接槽的那一层必须
    同时减掉旧值——一维 = 一个 unit：`-1 * x_old * relu(1)`。它和写值合并成同一个
    `d = round_bf16(Wo @ h)`：因为所有 dim 都只取 0/1（`_tmp_mag` 实测），
    `-old + new` 落在 {-1,0,1}，一次舍入不会丢位。

    FFN 层的清零混在本层里（不添层）；head 层写不了负项，所以在它前面插一层
    纯清零的 FFN——这是压槽唯一的层数代价。
    """
    nd = len(g.dims)
    if order is None:
        order = sorted(range(len(g.nodes)), key=lambda i: (g.nodes[i].level, i))

    # --- 位置号 ---
    pos_of = [0] * len(g.nodes)
    n_pos, i = 0, 0
    while i < len(order):
        k = order[i]
        pos_of[k] = n_pos
        if isinstance(g.nodes[k], FFN):
            sub = g.nodes[k].level
            i += 1
            while i < len(order) and g.nodes[order[i]].level == sub:
                pos_of[order[i]] = n_pos
                i += 1
        else:
            i += 1
        n_pos += 1

    # --- 生命周期 ---
    first = [n_pos] * nd            # 写在哪一位置（token 特征 = -1：每 token 开头由 feat 写）
    last = [-1] * nd                # 最晚被哪一位置读
    to_end = [False] * nd           # 被 taps 读：要活到 token 末尾
    for k in order:
        p = pos_of[k]
        node = g.nodes[k]
        if isinstance(node, FFN):
            first[node.out] = min(first[node.out], p)
            last[node.out] = max(last[node.out], p)      # into 的累加也算“占着槽”
            for _, a, b in node.terms:
                for e in (a, b):
                    for d in e.terms:
                        last[d] = max(last[d], p)
            continue
        for vh in node.vheads():
            for _, d, _ in vh.out:
                first[d] = min(first[d], p)
                last[d] = max(last[d], p)               # into 的累加也算“占着槽”
        # `vheads()` 把第 0 套和 `vx` 里的兄弟套一起给出来，而且给的是 `VHead` 而不是
        # 当初 `head()` 收到的 dict——所以这里只能过 `.v`/`.z` 的属性，不能下标。
        chans = [c for part in (node.q, node.k) for c in part.values()]
        chans += [c for vh in node.vheads() for c in vh.v.values()]
        for c in chans:
            far = any(c.taps[1:])
            for d in c.expr.terms:
                last[d] = max(last[d], p)
                to_end[d] = to_end[d] or far
        # 输出门与衰减是**每套 value head 各一份**（`node.a`/`node.z`/`node.beta` 只是
        # 第 0 套）。ROM 分页把 101 页挂成 101 个兄弟 value head，它们的 `z` 依赖页号位
        # 和页内 one-hot——漏掉这一圈，那些 dim 的 `last` 会偏早，被当成死值提前回收。
        for vh in node.vheads():
            for e in (vh.a, *vh.z.values()):
                for d in e.terms:
                    last[d] = max(last[d], p)
            if vh.beta is not None:
                for d in vh.beta.terms:
                    last[d] = max(last[d], p)

    for d in keep:
        last[d] = n_pos
    for d in range(nd):
        if g.dims[d].kind == "token":
            first[d] = -1
        if to_end[d]:
            last[d] = n_pos
        if first[d] == n_pos:        # 没人写（zeros 没填）：恒为 0，只要早死就行
            first[d] = -1
        last[d] = max(last[d], first[d])

    # --- 贪心分槽：按首次写的时间扫描，最早死的槽最优先 ---
    slot_of = [0] * nd
    heap: list[tuple[int, int]] = []        # (last, slot)
    free: list[int] = []
    owner: dict[int, int] = {}              # slot -> 最后一个持有它的 dim
    used: set[int] = set()
    clears: dict[int, list[int]] = {}
    n_slots = 0
    for d in sorted(range(nd), key=lambda d: (first[d], -last[d])):
        while heap and heap[0][0] < first[d]:
            free.append(heapq.heappop(heap)[1])
        if free:
            s = free.pop()
        else:
            s = n_slots
            n_slots += 1
        slot_of[d] = s
        if first[d] >= 0 and s in used:      # 接一个还热着的槽：这一层得先减掉旧值
            clears.setdefault(first[d], []).append(s)
        used.add(s)
        owner[s] = d
        heapq.heappush(heap, (last[d], s))

    return SlotPlan(slot_of, n_slots, pos_of, n_pos, first, clears,
                    sorted(owner.values()))


# ---------------- 精确性 ----------------

def round_bf16(t: torch.Tensor) -> torch.Tensor:
    """舍入到 bf16（ties to even），返回到 float64。torch 的 .to 就是 RNE，与
    graph.bf16 的位技巧一致。"""
    return t.to(DTYPE).to(ACC)


def is_bf16(t: torch.Tensor) -> torch.Tensor:
    return (round_bf16(t) == t) & torch.isfinite(t)


def require(ok, what: str, x) -> None:
    ok = torch.as_tensor(ok).reshape(-1).to(torch.bool)
    if not bool(ok.all()):
        blank = torch.zeros((), dtype=ACC)
        x = torch.as_tensor(x).reshape(-1).to(ACC) if x is not None else blank
        i = int(torch.argmin(ok.to(torch.int8)))
        raise ExactnessError(f"{what}: 坏值 {float(x[i])!r} 在 ({i},)")


def need_bf16(x, what: str) -> float:
    """x 必须能被 bf16 精确表示，否则权重一存进去就变了。"""
    v = float(x)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{what}: {v!r} 不是有限值")
    if float(torch.tensor(v, dtype=torch.float32).to(DTYPE)) != v:
        raise ValueError(f"{what}: {v!r} 不是 bf16 可表示的值")
    return v


def bf16_parts(x, n: int = 2) -> list[float]:
    """把 x 拆成 n 个 bf16 值之和（大步长贪心，退化成精确值时提前结束）。"""
    out, r = [], float(x)
    for _ in range(n):
        if r == 0.0:
            break
        v = float(torch.tensor(r, dtype=torch.float32).to(DTYPE))
        out.append(v)
        r -= v
    if r != 0.0:
        raise ValueError(f"{x!r} 拆不成 {n} 个 bf16 之和（剩 {r!r}）")
    return out or [0.0]


# ---------------- 编译产物 ----------------

@dataclass
class CChan:
    """编译后的一路通道：稠密权重 + taps。"""

    w: torch.Tensor                    # (hidden,) bf16
    taps: tuple[float, ...]


@dataclass
class CVHead:
    v: dict[int, CChan]
    a: torch.Tensor                    # (hidden,) -> alpha 直接可算
    z: dict[int, torch.Tensor]
    out: list[tuple[int, int, float]]  # (列, dim, 权重)
    beta: torch.Tensor | None
    into: bool
    S0: torch.Tensor                   # (rows, cols) bf16


@dataclass
class FFNLayer:
    sub: int
    names: list[str]
    Wa: torch.Tensor                   # (nu, hidden)
    Wb: torch.Tensor                   # (nu, hidden)
    Wo: torch.Tensor                   # (hidden, nu)
    rows: torch.Tensor                 # (hidden,) bool：本层写哪些槽


@dataclass
class HeadLayer:
    sub: int
    name: str
    bank: str
    rows_n: int
    cols: int
    q: dict[int, CChan]
    k: dict[int, CChan]
    vheads: list[CVHead]


@dataclass
class Compiled:
    layout: Layout
    vocab: int
    feat_dims: list[int]
    feat: torch.Tensor                 # (vocab, len(feat_dims)) bf16
    layers: list                       # FFNLayer | HeadLayer，按 (sub, 序号) 排好
    plan: SlotPlan | None = None       # 压槽方案；None = 一维一槽
    pos_layer: list[int] = field(default_factory=list)   # 位置号 -> “这一位置算完”的层下标

    @property
    def feat_slots(self) -> torch.Tensor:
        return self.layout.slots(self.feat_dims)

    def state(self) -> dict[str, torch.Tensor]:
        """扁平的权重表，key 与 save() 落盘的一致，可直接 load_state_dict 用。"""
        st = {"feat": self.feat}
        for i, l in enumerate(self.layers):
            if isinstance(l, FFNLayer):
                st[f"L{i}_Wa"], st[f"L{i}_Wb"], st[f"L{i}_Wo"] = l.Wa, l.Wb, l.Wo
                continue
            for s, ch in l.q.items():
                st[f"L{i}_Wq_p{s}"] = ch.w
            for s, ch in l.k.items():
                st[f"L{i}_Wk_{s}"] = ch.w
            for j, vh in enumerate(l.vheads):
                st[f"L{i}_Walpha_{j}"] = vh.a
                if vh.beta is not None:
                    st[f"L{i}_Wbeta_{j}"] = vh.beta
                for col, ch in vh.v.items():
                    st[f"L{i}_Wv_{j}_{col}"] = ch.w
                for col, e in vh.z.items():
                    st[f"L{i}_Wz_{j}_{col}"] = e
                st[f"S_{i}_{j}"] = vh.S0
        return st

    def manifest(self) -> dict:
        """结构 + 坐标的完整描述。head 的 taps 与 out 也在这里。

        执行器（M2）要能只靠 manifest + state 重建整个电路，不能依赖同进程里还留着一个
        Compiled 对象。taps 是结构常数而不是权重，放 manifest 比塞进 state 更诚实；
        out 那三个数（列 / dim / 系数）同理。
        """
        L = self.layout
        man = {
            "d0": L.d0, "n_dims": L.n_dims, "m": L.m, "hidden": L.hidden,
            "log2_ballast": L.log2_ballast, "log2_den": L.log2_den,
            "dtype": str(DTYPE).removeprefix("torch."),
            "vocab": self.vocab, "feat_dims": list(self.feat_dims),
            "layers": [_layer_manifest(l) for l in self.layers],
            "banks": {l.bank: [l.rows_n, l.cols] for l in self.layers
                      if isinstance(l, HeadLayer)},
        }
        if self.plan is not None and self.plan.n_slots != len(self.plan.slot_of):
            # 压槽之后 dim 与槽不是一回事，执行器得靠这两张表把语义接回来：
            # slot_of 是 dim -> 槽，final_dims 是 token 末尾仍然持有自己槽的 dim。
            man["slot_of"] = list(self.plan.slot_of)
            man["final_dims"] = list(self.plan.final)
        return man

    def stats(self) -> dict:
        ff = [l for l in self.layers if isinstance(l, FFNLayer)]
        hd = [l for l in self.layers if isinstance(l, HeadLayer)]
        n_ffn = sum(l.Wa.numel() + l.Wb.numel() + l.Wo.numel() for l in ff)
        n_hd = sum(ch.w.numel() for l in hd for ch in (*l.q.values(), *l.k.values()))
        n_hd += sum(vh.a.numel() + (0 if vh.beta is None else vh.beta.numel())
                    + sum(ch.w.numel() for ch in vh.v.values())
                    + sum(e.numel() for e in vh.z.values())
                    + vh.S0.numel()
                    for l in hd for vh in l.vheads)
        return {
            "hidden": self.layout.hidden,
            "d0": self.layout.d0,
            "dims": self.layout.n_dims,
            "clears": 0 if self.plan is None else sum(len(v) for v in self.plan.clears.values()),
            "m": self.layout.m,
            "log2_den": self.layout.log2_den,
            "layers": len(self.layers),
            "ffn_layers": len(ff),
            "head_layers": len(hd),
            "ffn_units": sum(int(l.Wa.shape[0]) for l in ff),
            "weights": n_ffn + n_hd,
            "feat": self.feat.numel(),
            "bytes": (n_ffn + n_hd + self.feat.numel()) * self.feat.element_size(),
        }


def _layer_manifest(l) -> dict:
    """一层的结构描述。taps 与 out 都在这里——执行器靠它把 state 摆回原来的位置。"""
    if isinstance(l, FFNLayer):
        return {"kind": "ffn", "sub": l.sub, "units": int(l.Wa.shape[0])}
    return {
        "kind": "head", "sub": l.sub, "name": l.name, "bank": l.bank,
        "rows": l.rows_n, "cols": l.cols,
        "q": {str(s): list(ch.taps) for s, ch in l.q.items()},
        "k": {str(s): list(ch.taps) for s, ch in l.k.items()},
        "vh": [{"v": {str(c): list(ch.taps) for c, ch in vh.v.items()},
                "z": sorted(int(c) for c in vh.z),
                "out": [[int(c), int(d), float(w)] for c, d, w in vh.out]}
               for vh in l.vheads],
    }


# ---------------- 逐节点编译 ----------------

def _dense(e: Expr, L: Layout, what: str) -> torch.Tensor:
    """一个线性形式 -> (hidden,) 的权重向量。常数项落在 dim 0（chip mask）。"""
    w = torch.zeros(L.hidden, dtype=DTYPE)
    for d, c in e.terms.items():
        w[L.slot(d)] = need_bf16(float(c), what)
    return w


def _cchan(c: Chan, L: Layout, what: str) -> CChan:
    return CChan(_dense(c.expr, L, what), tuple(float(t) for t in c.taps))


def _split_coef(c: float, what: str) -> list[float]:
    """常数拆 bf16：一个 bf16 表示不了的系数拆成 2 个 unit，两个系数都是 bf16。

    只对 coef 做。a / b 的系数不走这里：它们进的是 Wa/Wb，拆开等于把同一个 unit
    复制一遍，而 gate = K*b 的整性正是 silu 精确区间的前提，那一侧直接要求 bf16。
    """
    v = float(torch.tensor(float(c), dtype=torch.float32).to(DTYPE))
    if v == c:
        return [c]
    try:
        return bf16_parts(c, 2)
    except ValueError as e:
        raise ValueError(f"{what}: {c!r} 既不是 bf16，也拆不成两个 bf16（{e}）") from None


def _ffn_layer(sub: int, nodes: list[FFN], L: Layout,
               clears: tuple[int, ...] | list[int] = ()) -> FFNLayer:
    """一个 FFN 子层。`clears` 是要在同一层减掉的旧槽（压槽接槽时的清零）。

    清零一个 unit：a = -x_old（Wa 行 -1/K）、b = 1（Wb 行 K），于是
    `h = relu(K) * (-x_old/K) = -x_old`，`Wo` 行 +1 把这份增量原样落到槽上：
    `d = -x_old`。它和本层的写值落在同一个 `Wo @ h` 里，一次舍入：`-old + new` 在
    dim 只取 0/1 时必是 {-1,0,1}，不丢位。
    """
    rows = []
    for node in nodes:
        for coef, a, b in node.terms:
            for part in _split_coef(float(coef), f"{node.name}.coef"):
                rows.append((node, part, a, b))
    nu = len(rows) + len(clears)
    Wa = torch.zeros((nu, L.hidden), dtype=DTYPE)
    Wb = torch.zeros((nu, L.hidden), dtype=DTYPE)
    Wo = torch.zeros((L.hidden, nu), dtype=DTYPE)
    for u, (node, coef, a, b) in enumerate(rows):
        for d, c in a.terms.items():                 # a/K：K 是 2 的幂，除是精确的
            Wa[u, L.slot(d)] = need_bf16(float(c) / K, f"{node.name}.a[{u}]")
        for d, c in b.terms.items():                 # K*b：乘是精确的
            Wb[u, L.slot(d)] = need_bf16(float(c) * K, f"{node.name}.b[{u}]")
        Wo[L.slot(node.out), u] += need_bf16(coef, f"{node.name}.coef[{u}]")
    for j, s in enumerate(clears):
        u, col = len(rows) + j, L.slot_at(s)
        Wa[u, col] = need_bf16(-1.0 / K, "clear.a")
        Wb[u, L.slot(ONE)] = need_bf16(float(K), "clear.b")
        Wo[col, u] = need_bf16(1.0, "clear.coef")
    touched = Wo.abs().sum(1) > 0
    names = [n.name for n in nodes] + (["clear"] if clears else [])
    return FFNLayer(sub, names, Wa, Wb, Wo, touched)


def _alpha_w(a: Expr, L: Layout, what: str) -> torch.Tensor:
    """alpha = 1 - 0.5*sel。1 落在 dim 0（chip mask）上，所以衰减也是纯粹的投影。

    sel 只能是 0/1/2，映射 {0:1, 1:0.5, 2:0} 恰好是线性的——这就是"不用 exp(dt_bias)
    衰减表"的全部内容。0.5 是 2 的幂，bf16 精确。
    """
    w = torch.zeros(L.hidden, dtype=DTYPE)
    w[L.slot(ONE)] = 1.0
    for d, c in a.terms.items():
        w[L.slot(d)] -= need_bf16(0.5 * float(c), what)
    return w


def _head_layer(node: Head, L: Layout) -> HeadLayer:
    q = {s: _cchan(c, L, f"{node.name}.q[{s}]") for s, c in node.q.items()}
    k = {s: _cchan(c, L, f"{node.name}.k[{s}]") for s, c in node.k.items()}
    vhs = []
    for x_, vh in enumerate(node.vheads()):
        init = node.init[x_] if node.init and x_ < len(node.init) else {}
        S0 = torch.zeros((node.rows, node.cols), dtype=DTYPE)
        for r, row in init.items():
            vals = [need_bf16(float(v), f"{node.name}.init[{r}]") for v in row]
            S0[r, :len(vals)] = torch.tensor(vals, dtype=ACC)
        vhs.append(CVHead(
            v={col: _cchan(c, L, f"{node.name}.v[{col}]") for col, c in vh.v.items()},
            a=_alpha_w(vh.a, L, f"{node.name}.v{x_}.alpha"),
            z={col: _dense(e, L, f"{node.name}.z[{col}]") for col, e in vh.z.items()},
            out=list(vh.out),
            beta=None if vh.beta is None else _dense(vh.beta, L, f"{node.name}.beta"),
            into=vh.into,
            S0=S0,
        ))
    return HeadLayer(node.level, node.name, node.bank, node.rows, node.cols, q, k, vhs)


# ---------------- 入口 ----------------

def compile(g: Graph, vocab: int, d0: int = 0, layout: Layout | None = None,
         log2_ballast: int = LOG2_BALLAST, reuse: bool = True,
         keep: Iterable[int] = ()) -> Compiled:
    """把一个 Graph 编译成 torch 权重。vocab 是 token 特征的枚举上界。

    `reuse`（默认开）按 `plan_slots` 的生命周期分析让不重叠的 dim 共用残差槽；
    传 `reuse=False` 或一个 `n_dims == len(g.dims)` 的 `layout` 就是一维一槽
    （J1 的对照基准）。

    `keep` 是"要在残差里读回来"的 dim（见 `plan_slots` 第 3 条）：它们不许被压槽
    复用，代价是各占一个槽活到 token 末尾。`reuse=False` 时它是多余的（`final` 已经是
    全部 dim），传了也不影响。
    """
    g.check_dag()
    nd = len(g.dims)
    order = sorted(range(len(g.nodes)), key=lambda i: (g.nodes[i].level, i))
    plan = plan_slots(g, order, keep)
    if not reuse:
        plan = SlotPlan.identity(nd, plan.pos_of, plan.n_pos, plan.first)
    if layout is None:
        L = pick_layout(plan.n_slots, d0, log2_ballast)
    elif layout.n_dims == nd:                     # 显式给了 dim 宽的布局：一维一槽
        plan, L = SlotPlan.identity(nd, plan.pos_of, plan.n_pos, plan.first), layout
    elif layout.n_dims == plan.n_slots:
        L = layout
    else:
        raise ValueError(f"layout.n_dims={layout.n_dims} 既不是 dim 数 {nd}，"
                         f"也不是压槽后的 slots {plan.n_slots}")
    L = replace(L, slot_of=tuple(plan.slot_of))
    L.check()
    used = L.d0 + L.n_dims
    if used > L.ballast0:
        raise ValueError(f"放不下：chip 占到 {used}，ballast 从 {L.ballast0} 开始")

    feat_dims = [ONE] + sorted(g.token_fns)
    feat = torch.zeros((vocab, len(feat_dims)), dtype=DTYPE)
    feat[:, 0] = 1.0                                  # dim 0 = chip mask
    for j, d in enumerate(feat_dims[1:], start=1):
        fn = g.token_fns[d]
        col = torch.tensor([float(fn(t)) for t in range(vocab)], dtype=ACC)
        require(is_bf16(col), f"token 特征 {g.dims[d].name}", col)
        feat[:, j] = col                              # 已确认是 bf16 值，舍入无损

    layers: list = []
    pos_layer = [0] * plan.n_pos                  # 位置号 -> 这一位置算完的层下标
    i = 0
    while i < len(order):
        sub = g.nodes[order[i]].level
        j = i
        while j < len(order) and g.nodes[order[j]].level == sub:
            j += 1
        grp = order[i:j]                          # 同一 level 的节点：一个 FFN 层或一批 head
        if isinstance(g.nodes[grp[0]], FFN):
            p = plan.pos_of[grp[0]]
            layers.append(_ffn_layer(sub, [g.nodes[k] for k in grp], L,
                                     plan.clears.get(p, ())))
            pos_layer[p] = len(layers) - 1
        else:
            for k in grp:
                q = plan.pos_of[k]
                if plan.clears.get(q):            # head 写不了负项：它前面单插一层清零
                    layers.append(_ffn_layer(sub, [], L, plan.clears[q]))
                layers.append(_head_layer(g.nodes[k], L))
                pos_layer[q] = len(layers) - 1
        i = j
    return Compiled(L, vocab, feat_dims, feat, layers, plan, pos_layer)


# ---------------- 权重的参考执行（J1 的验具） ----------------

def run(c: Compiled, tokens, check: bool = True, carry: dict | None = None,
        trace: list | None = None) -> torch.Tensor:
    """把编译后的权重跑一遍，返回每个 token 之后的残差 (n, hidden)，dtype 为 ACC。

    ``trace`` 非 None 时把每一层算完的 x 追加进去（压槽的探针要比中间层）。

    这不是 M2 的 nn.Module —— 它是权重的参考执行，用来证明 compile 编对了。和
    graph.evaluate 的差别只有一处：这里没有 dim 语义，只有矩阵乘和状态表。
    累加用 ACC（float64），与 evaluate 同构，所以 J1 的容差可以卡到 0；M2 的设备侧
    执行器换成 fp32 累加，那时要重跑 J1。
    """
    L = c.layout
    carry = {} if carry is None else carry      # (层号, "S", value head) -> S；("h", ...) -> 历史
    n = len(tokens)
    feat_slots = L.slots(c.feat_dims)
    out = torch.zeros((n, L.hidden), dtype=ACC)
    for t in range(n):
        x = torch.zeros(L.hidden, dtype=ACC)
        tok = int(tokens[t])
        if tok >= 0:
            x[feat_slots] = c.feat[tok].to(ACC)
        x[L.ballast0:] = BALLAST
        taps = []
        for li, layer in enumerate(c.layers):
            if isinstance(layer, FFNLayer):
                _run_ffn(layer, x, check)
            else:
                taps += _run_head(layer, x, carry, li, check, L)
            if trace is not None:
                trace.append(x.clone())
        for key, w in taps:            # 末尾快照：读的是本 token 最终的 x，不是投影当下的 x
            h = carry.setdefault(("h",) + key, [])
            h.append(float(w.to(ACC) @ x))
            del h[:-3]                 # taps 最多 4 个，只留 3 个历史
        out[t] = x
    return out


def _run_ffn(layer: FFNLayer, x: torch.Tensor, check: bool) -> None:
    gate = round_bf16(layer.Wb.to(ACC) @ x)
    up = round_bf16(layer.Wa.to(ACC) @ x)
    if check:
        require(silu_ok(gate.numpy()), f"{layer.names}: 门控超出 silu 的精确区间", gate)
    h = round_bf16(torch.clamp(gate, min=0.0) * up)
    d = round_bf16(layer.Wo.to(ACC) @ h)
    x[layer.rows] += d[layer.rows]
    if check:
        require(is_bf16(x[layer.rows]), f"{layer.names}: 残差累加", x[layer.rows])


def _run_head(layer: HeadLayer, x: torch.Tensor, carry: dict, li: int,
              check: bool, L: Layout) -> list:
    """跑一个 head 层，返回 [(历史键, 权重)] —— 需要末尾快照的那几路。"""

    def conv(key, ch: CChan) -> float:
        p = float(ch.w.to(ACC) @ x)
        if check:
            require(is_bf16(torch.tensor([p], dtype=ACC)), f"{layer.name}: 投影", p)
        h = carry.get(("h",) + key, ())
        val = 0.0
        for lag, wt in enumerate(ch.taps):
            if wt:
                # 历史不足时那一路贡献 0，与 graph.conv 把 xx[:len-lag] 切片成空一致。
                val += wt * (p if lag == 0 else (h[-lag] if lag <= len(h) else 0.0))
        v = max(val, 0.0)
        if check:
            require(is_bf16(torch.tensor([v], dtype=ACC)), f"{layer.name}: conv 输出", v)
        return v

    def tapped(key, ch):
        if any(ch.taps[1:]):
            taps.append((key, ch.w))
        return ch

    taps: list = []
    q = {s: conv(("q", li, s), tapped(("q", li, s), ch)) for s, ch in layer.q.items()}
    k = {s: conv(("k", li, s), tapped(("k", li, s), ch)) for s, ch in layer.k.items()}
    if check:
        both = torch.tensor(list(q.values()) + list(k.values()), dtype=ACC)
        require((both == 0.0) | (both == 1.0), f"{layer.name}: 门控必须是 0/1", both)
        require(sum(v > 0 for v in k.values()) <= 1, f"{layer.name}: key 必须 one-hot",
                torch.tensor([sum(v > 0 for v in k.values())], dtype=ACC))
    sel_k = [s for s, v in k.items() if v > 0]
    sel_q = [s for s, v in q.items() if v > 0]
    if check and sel_k:
        require(torch.tensor([x[L.slot(ONE)]], dtype=ACC) > 0,
                f"{layer.name}: key 出现在非 chip token 上", x[L.slot(ONE)])
    if check and sel_q:
        vals = torch.tensor([q[s] for s in sel_q], dtype=ACC)
        require(vals == vals[0], f"{layer.name}: query 必须等值 multi-hot", vals)

    for vhi, vh in enumerate(layer.vheads):
        S = carry.get((li, "S", vhi))
        if S is None:
            S = vh.S0.to(ACC).clone()
            carry[li, "S", vhi] = S
        vv = torch.zeros(layer.cols, dtype=ACC)
        for col, ch in vh.v.items():
            vv[col] = conv(("v", li, vhi, col), tapped(("v", li, vhi, col), ch))
        alpha = float(vh.a.to(ACC) @ x)
        beta = 1.0 if vh.beta is None else float(vh.beta.to(ACC) @ x)
        z = {col: float(w.to(ACC) @ x) for col, w in vh.z.items()}
        if check:
            require(alpha in ALPHA.values(), f"{layer.name}.v{vhi}: 衰减 alpha", alpha)
            require(beta in (1.0, 0.0), f"{layer.name}.v{vhi}: 写门 beta", beta)
            for col, zz in z.items():
                require(zz in (1.0, 0.0), f"{layer.name}.v{vhi}: 输出门 z[{col}]", zz)

        if alpha != 1.0:                          # 1 时乘法是空操作，省掉
            if alpha == 0.0:
                S.fill_(0.0)
            else:
                S *= alpha
        if sel_k and beta > 0:
            S[sel_k[0]] = vv
        o = torch.zeros(layer.cols, dtype=ACC)
        if sel_q:
            o = sum((S[s] for s in sel_q), torch.zeros(layer.cols, dtype=ACC))
            if check:
                require(is_bf16(o), f"{layer.name}.v{vhi}: 读出", o)
        for col, d, w in vh.out:
            val = o[col] * max(z.get(col, 1.0), 0.0) * w
            if check:
                require(is_bf16(val.reshape(1)), f"{layer.name}: 输出列 {col}", val)
            x[L.slot(d)] += val
            if check:
                require(is_bf16(x[L.slot(d)].reshape(1)), f"{layer.name}: 输出累加",
                        x[L.slot(d)])
    return taps


# ---------------- 落盘 ----------------

def save(c: Compiled, path: str | Path) -> Path:
    """torch.save({"manifest": ..., "state": {...}})。state 可以直接 load_state_dict。"""
    p = Path(path)
    torch.save({"manifest": c.manifest(), "state": c.state()}, p)
    return p


def load(path: str | Path) -> tuple[dict, dict[str, torch.Tensor]]:
    """save() 的反向。返回 (manifest, state)。"""
    z = torch.load(Path(path), map_location="cpu", weights_only=True)
    return z["manifest"], z["state"]


def dump_manifest(c: Compiled) -> str:
    return json.dumps(c.manifest(), ensure_ascii=False, indent=1)
