"""
chipformer.py: M2 执行器 —— 把 compile.py 编出来的权重跑成残差序列。

plan.md §3 的 cpu 层。入参是 compile 落盘的 {"manifest", "state"}，出参是每个 token
之后的残差。**bf16 权重 + fp32 累加，张量里不出现 float64**。

它不认识 dim，也不认识指令：只有矩阵乘、状态表和残差坐标。哪些槽是寄存器、哪些位是指令，
全在 compile 编出来的坐标里，这个文件一个名字都不出现。所以 M3 换成 52 条指令的图，
这里一行都不用改 —— 这是"没有硬布线"的判据。

残差形状（manifest 给）：
    x = [host(d0) | chip(n_dims) | 零区 | ballast(m)]
ballast 每维填 2^log2_ballast，只为让 RMSNorm 的分母恒等于 2^log2_den。它不参与任何权重
读取 —— compile 那边用 touched 掩码验过了。

一个 token 的次序（与 compile.run 逐句同构，只差精度）：

    FFN   gate = bf16(Wb @ x);   up = bf16(Wa @ x)
          h    = bf16(relu(gate) * up);   x[rows] += bf16(Wo @ h)

    Head  p     = W @ x                      q/k/v 每个通道一行
          v     = relu(Σ_lag taps[lag] · p_{t-lag})      lag 0 是本 token
          alpha = Walpha @ x  == 1 - 0.5·sel
          S    <- S·alpha;  S[k] <- v;  o = Σ q 选中的行
          x[d] += o[col]·relu(z[col])·w

    末尾快照  carry[("h", ...)] <- W @ x     用的是本 token 最终的 x

最后那一步是唯一一处容易写错的地方：被 taps 引用的 dim 往往在本 token 更晚才算出来
（寄存器的写值就是 ALU 的结果，而读口在 ALU 之前），投影当下看到的是 0，只有 token
末尾的快照才看得到结果。这就是"写回"在权重里的全部含义。

为什么 fp32 够：每个中间量要么本来就是 bf16 值（fp32 是它的超集，装得下且无损），要么是
"小整数 × 2 的幂"这种能精确表示的数。乘法和累加都不产生需要 25 位以上尾数的中间结果，
所以 fp32 累加与 compile.run 的 float64 累加逐位相同，J1 的容差仍然卡在 0。
taps 的滞后组合走 python float（双精度），它是标量控制流而不是数据通路，且参与的每个值
都能被 fp32 精确表示，所以不破坏上面那条。

和另外两条路的分工：
    graph.evaluate   dim 语义，float64 理想值 + bf16 模拟 —— "电路对不对"的真身
    compile.run      权重语义，float64 累加 —— 验"compile 编对了没有"（J1）
    chipformer.py    权重语义，fp32 累加 —— 验"权重装进 nn.Module 之后行为不变"（M2）

权重全部是 register_buffer，不是 nn.Parameter：它们由 compile 现算，没有一个是训出来的。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_DIRECT = __package__ in (None, "")
if _DIRECT:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "components.rv32i"

from .graph import ALPHA, ExactnessError, silu_ok

DTYPE = torch.bfloat16             # 权重的存储与存取精度：设备侧读什么就存什么
ACC = torch.float32                # 累加精度。设备侧没有 float64
HIST = 3                           # taps 最多 4 段，所以每路只留 3 个历史


# ---------------- 精确性（fp32 版，与 compile.py 那套同义） ----------------

def rnd(t: torch.Tensor) -> torch.Tensor:
    """按 bf16 存取一次再回到 fp32。设备侧每个中间量都要过这道门。"""
    return t.to(DTYPE).to(ACC)


def is_bf16(t) -> torch.Tensor:
    t = torch.as_tensor(t, dtype=ACC)
    return (rnd(t) == t) & torch.isfinite(t)


def require(ok, what: str, x=None) -> None:
    ok = torch.as_tensor(ok).reshape(-1).to(torch.bool)
    if not bool(ok.all()):
        v = torch.as_tensor(0.0 if x is None else x).reshape(-1).to(ACC)
        i = int(torch.argmin(ok.to(torch.int8)))
        raise ExactnessError(f"{what}: 坏值 {float(v[i])!r}")


def conv(taps_w, p: float, carry: dict, key) -> float:
    """conv 投影 + relu。p 是本 token 的投影值，lag >= 1 取 carry 里的历史。"""
    hist = carry.get(("h",) + key, ())
    val = 0.0
    for lag, wt in enumerate(taps_w):
        if wt:
            # 历史不足时那一路贡献 0，与 graph.conv 把 xx[:len-lag] 切片成空一致
            val += wt * (p if lag == 0 else (hist[-lag] if lag <= len(hist) else 0.0))
    return max(val, 0.0)


# ---------------- 层 ----------------

class FFN(nn.Module):
    """一个 ffn 子层：nu 个 unit 共享三个矩阵，写回自己那几行。

        gate = bf16(Wb @ x)          Wb 每行 = K·b
        up   = bf16(Wa @ x)          Wa 每行 = a/K
        h    = bf16(relu(gate)·up)   = a·relu(b)
        x   += Wo @ h                Wo[slot, unit] = coef

    旧 ProgFFN 是"relu 分支 + GLU 分支拼成 2·d_ff"，这里只有后者：graph 的项是
    coef·a·relu(b)，a 本身不该过 relu，两条分支也不共用一个 unit 下标。没有 bias ——
    bias 的角色由 dim 0 的 chip mask（常数 1）承担。

    原地改 x：调用方持有本 token 私有的张量，省掉一次 hidden 宽的拷贝。
    """

    def __init__(self, nu: int, hidden: int) -> None:
        super().__init__()
        self.register_buffer("Wb", torch.zeros((nu, hidden), dtype=DTYPE))
        self.register_buffer("Wa", torch.zeros((nu, hidden), dtype=DTYPE))
        self.register_buffer("Wo", torch.zeros((hidden, nu), dtype=DTYPE))
        self.register_buffer("rows", torch.zeros(hidden, dtype=torch.bool))

    def forward(self, x: torch.Tensor, check: bool = False) -> torch.Tensor:
        gate = rnd(self.Wb @ x)
        up = rnd(self.Wa @ x)
        if check:
            require(silu_ok(gate.numpy()), "ffn 门控超出 silu 的精确区间", gate)
        h = rnd(F.relu(gate) * up)
        d = rnd(self.Wo @ h)
        x[self.rows] = x[self.rows] + d[self.rows]
        if check:
            require(is_bf16(x[self.rows]), "ffn 残差累加", x[self.rows])
        return x


class VHead(nn.Module):
    """一张状态表 + 它的衰减 / 写门 / 输出门。同一组 q/k 可以挂多个 VHead（兄弟头）。

    S 不是参数也不是 buffer：它是每个 token 都在变的运行时状态，放在 carry 里按
    (层号, 头号) 索引。模块只持有 S0（初值）和四个投影。
    """

    def __init__(self, hidden: int, rows: int, cols: int, v_keys, z_keys,
                 out_col: torch.Tensor, out_dim: torch.Tensor, out_w: torch.Tensor) -> None:
        super().__init__()
        self.cols = cols
        self.v_keys = list(v_keys)
        self.z_keys = list(z_keys)
        self.taps_v: list[list[float]] = [[] for _ in self.v_keys]
        self.beta_on = False
        #: `prepare()` 之前一律当“要历史”处理，空 taps 正好都落在这一边
        self.plain_v: list[int] = []
        self.lag_v: list[int] = list(range(len(self.v_keys)))
        self.register_buffer("Wv", torch.zeros((len(self.v_keys), hidden), dtype=DTYPE))
        self.register_buffer("Wz", torch.zeros((len(self.z_keys), hidden), dtype=DTYPE))
        self.register_buffer("Walpha", torch.zeros(hidden, dtype=DTYPE))
        self.register_buffer("Wbeta", torch.zeros(hidden, dtype=DTYPE))
        self.register_buffer("S0", torch.zeros((rows, cols), dtype=DTYPE))
        self.register_buffer("out_col", out_col)
        self.register_buffer("out_dim", out_dim)
        self.register_buffer("out_w", out_w)
        self.register_buffer("v_keys_t", torch.tensor(self.v_keys, dtype=torch.long))
        self.register_buffer("z_keys_t", torch.tensor(self.z_keys, dtype=torch.long))
        self.register_buffer("plain_v_t", torch.zeros(0, dtype=torch.long))

    def prepare(self) -> None:
        """按 taps 把 v 通道分两类：`relu(投影)` 的整列批量算，其余逐拍走 `conv`。

        分类只能在这里做——`taps_v` 是 manifest 装完才有的。这一条是整台机器的速度
        分水岭：单层 745 列逐列 `.item()` + 单元素赋值约 4 ms/token，整列一次算完差
        30 倍以上（实测 158 ms/token 的主要部分）。
        """
        plain = [c for c, t in enumerate(self.taps_v)
                 if t and t[0] == 1.0 and not any(t[1:])]
        self.plain_v = plain
        self.lag_v = [c for c in range(len(self.v_keys)) if c not in set(plain)]
        self.plain_v_t = torch.tensor(plain, dtype=torch.long)

    def forward(self, x: torch.Tensor, carry: dict, li: int, j: int,
                sel_k: list, sel_q: list, snap: list, check: bool) -> torch.Tensor:
        dev = x.device
        S = carry.get((li, "S", j))
        if S is None:
            S = self.S0.to(ACC).clone()
            carry[(li, "S", j)] = S
        elif S.device != dev or S.dtype != ACC:
            # carry 可能是 host 侧注入的：探针在 CPU 上建 S、权重在 GPU 上；参考实现的
            # `S0` 又是 float64（`probe_slots._bridge`）。跟 x 走，并且收窄到 ACC ——
            # 值本身是 bf16 精确的，float64 -> fp32 不丢位，但 `index_add_` 不容忍混型
            S = S.to(device=dev, dtype=ACC)
            carry[(li, "S", j)] = S
        vv = torch.zeros(self.cols, dtype=ACC, device=dev)
        if self.v_keys:
            pv = self.Wv @ x
            if len(self.plain_v):
                # `taps = (1,0,0,0)` 就是 relu(投影)：整列一次算完，不逐列 `.item()`
                vv[self.v_keys_t[self.plain_v_t]] = torch.clamp(pv[self.plain_v_t], min=0.0)
            for c in self.lag_v:
                col = self.v_keys[c]
                vv[col] = conv(self.taps_v[c], float(pv[c]), carry, ("v", li, j, col))
                if any(self.taps_v[c][1:]):
                    snap.append((("v", li, j, col), self.Wv[c]))
        alpha = float(self.Walpha @ x)
        beta = float(self.Wbeta @ x) if self.beta_on else 1.0
        #: 输出门默认 1（没有 z 投影的列），只有 `z_keys` 那几列是算出来的 0/1
        zc = None
        if self.z_keys:
            zc = torch.ones(self.cols, dtype=ACC, device=dev)
            zc[self.z_keys_t] = self.Wz @ x
        if check:
            require(alpha in ALPHA.values(), "衰减 alpha", alpha)
            require(beta in (1.0, 0.0), "写门 beta", beta)
            if zc is not None:
                pz = zc[self.z_keys_t]
                require(torch.logical_or(pz == 0.0, pz == 1.0), "输出门 z", pz)

        if alpha != 1.0:                      # 1 时乘法是空操作，省掉
            if alpha == 0.0:
                S.fill_(0.0)
            else:
                S *= alpha
        if sel_k and beta > 0:
            S[sel_k[0]] = vv
        o = torch.zeros(self.cols, dtype=ACC, device=dev)
        if sel_q:
            o = sum((S[s] for s in sel_q), o)
            if check:
                require(is_bf16(o), "状态表读出", o)
        if self.out_dim.numel():
            # 整列一次算完再 `index_add_`：逐列 `x[d] = x[d] + val` 是单层 745 次标量写，
            # 占了整台机器 70% 的时间。两者逐位相同的原因写在模块开头（每个 val 都是
            # bf16×{0,1}×1.0，fp32 里精确，所以“先舍入再加”与“先加再舍入”没差）
            term = o[self.out_col]
            if zc is not None:
                term = term * torch.clamp(zc[self.out_col], min=0.0)
            term = term * self.out_w
            if check:
                require(is_bf16(term), "输出列", term)
            x.index_add_(0, self.out_dim, term)
            if check:
                require(is_bf16(x[self.out_dim]), "输出累加", x[self.out_dim])
        return x


class Head(nn.Module):
    """一个 head 节点：q / k 两组行选择子 + 若干兄弟 value head。

    q 可以有多个通道同时为 1（两个读口同理选中同一行），k 必须 one-hot —— 一条指令只改
    一行。这两个约束在 check 下会当场报出来。
    """

    def __init__(self, hidden: int, name: str, q_keys, k_keys, vheads) -> None:
        super().__init__()
        self.name = name
        self.q_keys = list(q_keys)
        self.k_keys = list(k_keys)
        self.taps_q: list[list[float]] = [[] for _ in self.q_keys]
        self.taps_k: list[list[float]] = [[] for _ in self.k_keys]
        self.plain_q: list[int] = []
        self.lag_q: list[int] = list(range(len(self.q_keys)))
        self.plain_k: list[int] = []
        self.lag_k: list[int] = list(range(len(self.k_keys)))
        self.register_buffer("Wq", torch.zeros((len(self.q_keys), hidden), dtype=DTYPE))
        self.register_buffer("Wk", torch.zeros((len(self.k_keys), hidden), dtype=DTYPE))
        self.register_buffer("plain_q_t", torch.zeros(0, dtype=torch.long))
        self.register_buffer("plain_k_t", torch.zeros(0, dtype=torch.long))
        self.vheads = nn.ModuleList(vheads)

    def prepare(self) -> None:
        """同 `VHead.prepare`：纯 `relu(投影)` 的通道整列算，带滞后的逐拍走 `conv`。

        q/k 在真固件里大部分是 `(1,0,0,0)`，但写口（`regs` 的 `k`）是 `(0,1,0,0)` ——
        所以两组必须分开，不能一律当纯通道。
        """
        def split(taps: list[list[float]], n: int) -> tuple[list[int], list[int]]:
            plain = [i for i, t in enumerate(taps) if t and t[0] == 1.0 and not any(t[1:])]
            return plain, [i for i in range(n) if i not in set(plain)]

        self.plain_q, self.lag_q = split(self.taps_q, len(self.q_keys))
        self.plain_k, self.lag_k = split(self.taps_k, len(self.k_keys))
        self.plain_q_t = torch.tensor(self.plain_q, dtype=torch.long)
        self.plain_k_t = torch.tensor(self.plain_k, dtype=torch.long)

    def forward(self, x: torch.Tensor, carry: dict, li: int,
                check: bool = False) -> tuple[torch.Tensor, list]:
        snap: list = []
        dev = x.device
        qv = torch.zeros(len(self.q_keys), dtype=ACC, device=dev)
        kv = torch.zeros(len(self.k_keys), dtype=ACC, device=dev)
        if self.q_keys:
            pq = self.Wq @ x
            if len(self.plain_q):
                qv[self.plain_q_t] = torch.clamp(pq[self.plain_q_t], min=0.0)
            for i in self.lag_q:
                qv[i] = conv(self.taps_q[i], float(pq[i]), carry, ("q", li, self.q_keys[i]))
                if any(self.taps_q[i][1:]):
                    snap.append((("q", li, self.q_keys[i]), self.Wq[i]))
        if self.k_keys:
            pk = self.Wk @ x
            if len(self.plain_k):
                kv[self.plain_k_t] = torch.clamp(pk[self.plain_k_t], min=0.0)
            for i in self.lag_k:
                kv[i] = conv(self.taps_k[i], float(pk[i]), carry, ("k", li, self.k_keys[i]))
                if any(self.taps_k[i][1:]):
                    snap.append((("k", li, self.k_keys[i]), self.Wk[i]))
        sel_k = [self.k_keys[i] for i in torch.nonzero(kv > 0).flatten().tolist()]
        sel_q = [self.q_keys[i] for i in torch.nonzero(qv > 0).flatten().tolist()]
        if check:
            both = torch.cat([qv, kv])
            if both.numel():
                require(torch.logical_or(both == 0.0, both == 1.0),
                        f"{self.name}: 门控必须是 0/1", both)
            require(len(sel_k) <= 1, f"{self.name}: key 必须 one-hot",
                    torch.tensor([len(sel_k)], dtype=ACC))
            if sel_q:
                vals = qv[qv > 0]
                require(vals == vals[0], f"{self.name}: query 必须等值 multi-hot", vals)
        for j, vh in enumerate(self.vheads):
            vh(x, carry, li, j, sel_k, sel_q, snap, check)
        return x, snap


# ---------------- 执行器 ----------------

class NeuralCPU(nn.Module):
    """这台神经 CPU 的本体：compile 的权重 + manifest 的结构 → 能跑 token 流的 nn.Module。

    一个 token 一拍，一拍走完所有子层。状态表（VHead 的 S）就是寄存器文件与存储器，
    taps 历史就是跨拍的组合延迟。子层“是什么”不在这里——哪些槽是 pc、哪些位是 opcode，
    全在 compile 编出来的坐标里，这个类一个名字都不认识。

    结构全部由 manifest 恢复：哪一层是 ffn、哪些是 head、每路的 taps、每列写到哪个 dim。
    所以它只认落盘文件，不需要同进程里留着一个 Compiled 对象。
    """

    def __init__(self, manifest: dict, state: dict[str, torch.Tensor] | None = None,
                 strict: bool = False) -> None:
        super().__init__()
        self.spec = dict(manifest)
        self.strict = bool(strict)
        self.d0 = int(manifest["d0"])
        self.hidden = int(manifest["hidden"])
        self.n_dims = int(manifest["n_dims"])
        self.m = int(manifest["m"])
        self.ballast0 = self.hidden - self.m
        self.ballast = float(2.0 ** int(manifest["log2_ballast"]))
        self.log2_den = int(manifest["log2_den"])
        self.vocab = int(manifest["vocab"])
        #: dim 号 -> 残差槽号。**压槽之后这两个不是一回事**：`n_dims` 是槽数，
        #: `slot_of` 是按原 dim 排的映射表（一维一槽时它是恒等映射）。manifest 里凡是
        #: 记着“哪个 dim”的地方（token 特征、head 输出列）都得先过这张表
        self.dim_slot = [int(s) for s in manifest.get("slot_of") or range(self.n_dims)]
        #: token 末尾仍持有自己槽的 dim。压槽之后**只有它们能从残差里读回来**，其余的
        #: 槽已经被后面的 dim 接手（读到的会是别人的值，不会报错）。`slot_of` 缺省时
        #: 一维一槽，全部 dim 都还在。
        self.final_dims = [int(d) for d in
                           manifest.get("final_dims") or range(len(self.dim_slot))]
        self.feat_dims = [int(d) for d in manifest["feat_dims"]]
        self.register_buffer("feat_slots",
                             torch.tensor([self.d0 + self.dim_slot[d] for d in self.feat_dims],
                                          dtype=torch.long))
        self.register_buffer("feat",
                             torch.zeros((self.vocab, len(self.feat_dims)), dtype=DTYPE))
        self.layers = nn.ModuleList([self._build(l) for l in manifest["layers"]])
        if state is not None:
            self.load_compiled_state(state)
        self._widen()

    def _widen(self) -> "NeuralCPU":
        """权重转成 fp32 常驻。

        权重是不变的，而原来每次投影都做一次 `W.to(ACC)`（bf16 -> fp32 拷贝）：36 层 ×
        几百列，每 token 重复三万多份拷贝，占整台机器三分之一的时间（实测我编 100 拍
        有 34946 次 `.to`）。内存翻倍换这个，数值一点不变——bf16 值在 fp32 里精确。
        """
        def rewire(m: nn.Module, k: str) -> None:
            m.register_buffer(k, getattr(m, k).to(ACC))

        for l in self.layers:
            if isinstance(l, FFN):
                for k in ("Wb", "Wa", "Wo"):
                    rewire(l, k)
                continue
            rewire(l, "Wq")
            rewire(l, "Wk")
            for vh in l.vheads:
                for k in ("Wv", "Wz", "Walpha", "Wbeta"):
                    rewire(vh, k)
        return self

    # ---- 结构重建：manifest -> 模块 ----

    def _build(self, l: dict) -> nn.Module:
        if l["kind"] == "ffn":
            return FFN(int(l["units"]), self.hidden)
        vhs = [VHead(self.hidden, int(l["rows"]), int(l["cols"]),
                     [int(c) for c in vh["v"]], [int(c) for c in vh["z"]],
                     torch.tensor([o[0] for o in vh["out"]], dtype=torch.long),
                     # out_dim 在这里就加上 d0：模块里只认残差的绝对槽位，
                     # 否则每次写回都要记着“再偏移一次”（d0=0 时看不出错）
                     torch.tensor([self.d0 + self.dim_slot[o[1]] for o in vh["out"]],
                                  dtype=torch.long),
                     torch.tensor([float(o[2]) for o in vh["out"]], dtype=ACC))
               for vh in l["vh"]]
        head = Head(self.hidden, l["name"],
                    [int(k) for k in l["q"]], [int(k) for k in l["k"]], vhs)
        for i, key in enumerate(head.q_keys):
            head.taps_q[i] = [float(t) for t in l["q"][str(key)]]
        for i, key in enumerate(head.k_keys):
            head.taps_k[i] = [float(t) for t in l["k"][str(key)]]
        head.prepare()
        for j, vh in enumerate(head.vheads):
            for c, col in enumerate(vh.v_keys):
                vh.taps_v[c] = [float(t) for t in l["vh"][j]["v"][str(col)]]
            vh.prepare()
        return head

    def load_compiled_state(self, state: dict[str, torch.Tensor]) -> "NeuralCPU":
        """把 compile.py 的扁平 state（`L0_Wa` 这种 key）按结构摆进来。

        不走 load_state_dict：那边认的是模块路径（`layers.0.Wa`），而 compile 落的是
        `L0_Wa`。两套命名都合理，与其在中间做一次改名，不如在这里显式按 manifest 摆一次
        —— 顺手把"缺哪个 key、形状差多少"直接报出来，而不是等 forward 炸在 einsum 里。
        """

        def put(dst: torch.Tensor, key: str) -> None:
            if key not in state:
                raise KeyError(f"state 缺 {key}（manifest 里有对应结构）")
            src = state[key]
            if tuple(src.shape) != tuple(dst.shape):
                raise ValueError(f"{key}: 形状 {tuple(src.shape)} != 期望 {tuple(dst.shape)}")
            dst.copy_(src)

        put(self.feat, "feat")
        for i, l in enumerate(self.layers):
            if isinstance(l, FFN):
                put(l.Wb, f"L{i}_Wb")
                put(l.Wa, f"L{i}_Wa")
                put(l.Wo, f"L{i}_Wo")
                # 行掩码是算出来的，不落盘：Wo 哪几行非零，本层就写哪几行
                l.rows.copy_(l.Wo.abs().sum(1) > 0)
                continue
            for s, key in enumerate(l.q_keys):
                put(l.Wq[s], f"L{i}_Wq_p{key}")
            for s, key in enumerate(l.k_keys):
                put(l.Wk[s], f"L{i}_Wk_{key}")
            for j, vh in enumerate(l.vheads):
                for c, col in enumerate(vh.v_keys):
                    put(vh.Wv[c], f"L{i}_Wv_{j}_{col}")
                for c, col in enumerate(vh.z_keys):
                    put(vh.Wz[c], f"L{i}_Wz_{j}_{col}")
                put(vh.Walpha, f"L{i}_Walpha_{j}")
                vh.beta_on = f"L{i}_Wbeta_{j}" in state
                if vh.beta_on:
                    put(vh.Wbeta, f"L{i}_Wbeta_{j}")
                put(vh.S0, f"S_{i}_{j}")
        return self

    @classmethod
    def from_compiled(cls, c, strict: bool = False) -> "NeuralCPU":
        """同进程里直接装 compile 的产物，省掉一次落盘。"""
        return cls(c.manifest(), c.state(), strict)

    @classmethod
    def from_file(cls, path, strict: bool = False) -> "NeuralCPU":
        """只认落盘文件 —— 这是设备侧真正的装载路径。"""
        z = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls(z["manifest"], z["state"], strict)

    # ---- 前向 ----

    def forward(self, tokens, carry: dict | None = None, check: bool | None = None):
        """跑一串 token，返回 (每个 token 之后的残差 (n, hidden) fp32, carry)。

        tokens 是整数序列，负数表示"这一拍没有 token"（特征全 0，与 compile.run 同语义）。
        carry 跨调用保留，所以一次喂一个 token 就是流式 decode —— 状态表与 taps 历史都在
        carry 里，模型自己没有任何跨 token 的隐状态。
        """
        check = self.strict if check is None else bool(check)
        carry = {} if carry is None else carry
        dev = self.feat.device          # 权重在 CPU 就在 CPU，搬到 GPU 就跟着上去
        n = len(tokens)
        out = torch.zeros((n, self.hidden), dtype=ACC, device=dev)
        for t in range(n):
            tok = int(tokens[t])
            if tok >= self.vocab:
                raise IndexError(f"token {tok} 超出 vocab {self.vocab}")
            x = torch.zeros(self.hidden, dtype=ACC, device=dev)
            if tok >= 0:
                x[self.feat_slots] = self.feat[tok].to(ACC)
            x[self.ballast0:] = self.ballast
            snap: list = []
            for li, layer in enumerate(self.layers):
                if isinstance(layer, FFN):
                    layer(x, check=check)
                else:
                    _, sn = layer(x, carry, li, check=check)
                    snap += sn
            for key, w in snap:      # 末尾快照：读的是本 token 最终的 x，不是投影当下的 x
                h = carry.setdefault(("h",) + key, [])
                h.append(float(w @ x))
                del h[:-HIST]
            out[t] = x
        return out, carry

    def residual(self, tokens, carry: dict | None = None, check: bool | None = None):
        """只要残差的便捷入口（丢掉 carry，等于重新开始一条流）。"""
        return self.forward(tokens, carry, check)[0]


# ---------------- 自检 ----------------

def _demo() -> None:
    from components.rv32i import compile as cc
    from components.rv32i.tests.probe_graph import build, gen

    g, _, _ = build()
    toks = gen(__import__("random").Random(20261101))
    c = cc.compile(g, vocab=1 << 16)
    ex = NeuralCPU.from_compiled(c)
    got, _ = ex(toks)
    want = cc.run(c, toks)
    L = c.layout
    chip = slice(L.d0, L.d0 + L.n_dims)
    assert torch.equal(got[:, chip], want[:, chip].to(ACC)), "fp32 执行器与 float64 参考不一致"
    print(f"自检  fp32 NeuralCPU vs compile.run: 逐位一致  {tuple(got.shape)}")


if __name__ == "__main__":
    _demo()
