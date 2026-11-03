"""host.py: chip 段接进宿主的唯一入口（M4）。

    python components/rv32i/host.py          # 自检：小机，验形状与不变量
    python components/rv32i/tests/probe_host.py   # 验收：真固件 + J3 + J4

**产物是一份 .pt**。`assemble` 每次从图重算 207 MB 权重（15~40 s），那只是“怎么得到
它”；要一台装得起来的神经 CPU 就走 `save_nucpu` / `load_nucpu`：权重 + 被 Python 读
的坐标 + 这段代码的行表，装回来 1~6 s，**不依赖 graph、不依赖 compile**（设备侧只需
`chipformer` + torch）。

    python components/rv32i/tests/probe_whole.py dump        # -> outputs/nucpu.pt
    python components/rv32i/tests/probe_whole.py --pt=outputs/nucpu.pt smoke

`ChipSegment.save` / `from_file` 是更早的那一对：只有段（没有信号坐标），`probe_host`
拿它验往返；一台**整机**走 `save_nucpu` / `load_nucpu`。

chip 段是一台**独立的加性残差机器**：`graph` 描述、`compile` 编出 bf16 权重、
`chipformer.NeuralCPU` 在 fp32 里跑。它自己的残差向量是

    x = [host(d0) | chip(849 槽) | 零区 | ballast(m)]       hidden = m·4^k

`host` 区是给宿主残差预留的前缀（`d0` 取宿主 `hidden_size`），chip 段的写只落在
chip 区与它自己的清零项上，**一个 bit 都不写** `[0, d0)` 与零区。

**与宿主的关系只有一条：旁路。** 宿主 forward 在层栈之前调 `step()` 一次，芯片段
跑自己的 CPU record token 流，宿主的 `hidden_states` 原样不动。所以 J4（"接上宿主
后 ppl 不变"）是**结构性**成立的，不靠"投影权重恰好为 0"——那种做法换个初始值就
失效，而这里连权重都不参与。

这也是 §7「不训练任何新增权重」在 M4 里的唯一落法：宿主与芯片段之间**没有**
可训练的张量。将来要让宿主"看见" CPU 状态，加一个 849→1152 的投影就是 M5 的事，
那时才第一次出现训练参数。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn

from components.rv32i import circuit as cc
from components.rv32i import compile as ccm
from components.rv32i.chipformer import ACC, NeuralCPU
from components.rv32i.graph import Graph

#: `ready/*_1152_*_ready.pth` 的宽度。默认只当"没传 d0 时的兜底"，宿主那边一律
#: 显式传 `config.hidden_size`——这个数字写死在两处迟早会对不上。
DEFAULT_D0 = 1152


def _dim(e) -> int | None:
    """Expr -> 它唯一的那个 dim 号；常数 0 位是空 Expr，返回 None。"""
    return next(iter(e.terms)) if e.terms else None


class ChipSegment(nn.Module):
    """宿主侧的一段：把 `NeuralCPU` 包成"跑一串 token、给回残差"的模块。

    它不认宿主，宿主也不认它——两者唯一的约定是 `step()` 的签名。所以同一个类
    可以挂在 `MiniRainVLM`（图文）或 `MiniRain`（纯文本）上，不需要各写一份。
    """

    def __init__(self, cpu: NeuralCPU, d0: int, sys_dims: dict,
                 flat: dict | None = None) -> None:
        super().__init__()
        #: **不能叫 `self.cpu`**：`nn.Module.cpu()` 是个方法，而子模块只存在 `_modules`
        #: 里、不写 `__dict__`——于是 `self.cpu` 会解析成那个方法，子模块永远取不到
        #: （`'function' object has no attribute 'n_dims'`）。同理避开 `compile`/`to`/`type`。
        self.exec = cpu
        self.d0 = int(d0)
        #: compile 的**扁平** state（`L0_Wb` 这种 key）。只有 `save()` 要用它——
        #: `NeuralCPU.state_dict()` 是模块路径名（`layers.0.Wb`），两者不是一回事，
        #: 而 `load_compiled_state` 认的是前者。`from_file` 装出来的段不带它。
        self._flat = flat
        #: `{"put": dim, "stop": dim, "chars": [9 个 dim]}` —— `ecall` 的两个出口。
        #: 存的是**原 dim 号**，取值时要过 `cpu.dim_slot`（压槽后 dim 与槽不是一回事）
        self.sys_dims = {"put": sys_dims["put"], "stop": sys_dims["stop"],
                         "chars": list(sys_dims["chars"])}
        # 这 11 个 dim 是**被 Python 读**而不是被图读的，`plan_slots` 靠 `keep` 让它们
        # 活到 token 末尾。没进 `final` 的槽在末尾已被别的 dim 接手，读出来的值属于
        # 别人且不报错——所以这一条在构造时就卡死，不留到 ``read()`` 那一刻。
        fin = set(self.exec.final_dims)
        miss = [k for k, d in self._reads() if d is not None and d not in fin]
        if miss:
            raise ValueError(f"sys 出口 {miss} 不在 plan.final 里（compile 时没传 keep？）")

    # ---- 形状 ----

    def _reads(self) -> list[tuple[str, int]]:
        """`read()` 要取的每一列：名字 + 原 dim 号（`None` = 该位恒 0，跳过）。"""
        return ([("put", self.sys_dims["put"]), ("stop", self.sys_dims["stop"])]
                + [(f"chars[{b}]", d) for b, d in enumerate(self.sys_dims["chars"])])

    @property
    def hidden(self) -> int:
        return int(self.exec.hidden)

    @property
    def n_slots(self) -> int:
        return int(self.exec.n_dims)

    @property
    def m(self) -> int:
        return int(self.exec.m)

    @property
    def ballast0(self) -> int:
        return int(self.exec.ballast0)

    @property
    def used(self) -> int:
        """chip 段占到的最高列（不含 ballast）。再往上是零区。"""
        return self.d0 + self.n_slots

    def spec(self) -> dict:
        return {"d0": self.d0, "slots": self.n_slots, "hidden": self.hidden,
                "m": self.m, "ballast0": self.ballast0,
                "log2_den": int(self.exec.log2_den), "zone": self.ballast0 - self.used}

    # ---- 构造 ----

    @classmethod
    def build(cls, firmware: str | Path, d0: int = DEFAULT_D0,
              units: int = 4096, heads: int = 32,
              text: list[bytes] | None = None) -> "ChipSegment":
        """从固件编一份 chip 段。约 10 s / 165 MB 权重（真固件），编一次就落盘复用。

        `text` 给了才有文本桥（收得下 `<cpu_call>` 的 Python），见 `assemble_words`。
        """
        return assemble(firmware, d0, units, heads, text)[0]

    @classmethod
    def from_file(cls, path: str | Path) -> "ChipSegment":
        z = torch.load(Path(path), map_location="cpu", weights_only=True)
        cpu = NeuralCPU(z["manifest"], z["state"])
        return cls(cpu, int(z["d0"]), z["sys_dims"])

    def save(self, path: str | Path) -> Path:
        """落盘**只要这一段**（没有信号坐标，读不出 `cnt`、跑不了整机）。

        `sys_dims` 跟着走——它不在 compile 的 manifest 里，而缺了它就读不出芯片的
        putchar（这正是“dim 号是电路的身份”这件事）。整机落盘用 `save_nucpu`。
        """
        if self._flat is None:
            raise RuntimeError("这个段是从文件装的，没有扁平 state 可再落盘"
                               "——请用 build() 出来的段")
        p = Path(path)
        torch.save({"manifest": self.exec.spec, "state": self._flat,
                    "d0": self.d0, "sys_dims": self.sys_dims}, p)
        return p

    # ---- 前向 ----

    def forward(self, tokens, carry: dict | None = None):
        """跑一串 CPU token，返回 (每个 token 之后的残差 (n, hidden) fp32, carry)。

        残差里 `[0, d0)` 恒为 0（host 区留白）、`[used, ballast0)` 恒为 0（零区）、
        `[ballast0, hidden)` 恒为 `ballast`——这三条是结构性的，`probe_host` 每次
        都断言它们，所以"宿主残差混进 chip 段"或"某层写错列"当场就能看见。
        """
        return self.exec(tokens, carry)

    def step(self, tokens, carry: dict | None = None):
        """`forward` 的正名。宿主只认这个名字（`MiniRainVLM.chip_step` / `attach_chip`）。"""
        return self.forward(tokens, carry)

    def residual(self, tokens, carry: dict | None = None) -> torch.Tensor:
        return self.forward(tokens, carry)[0]

    def read(self, res: torch.Tensor) -> dict:
        """从残差里读 `ecall` 的两个出口。`res` 是 (hidden,) 或 (n, hidden)。

        只读**该 token 末尾**的值（`NeuralCPU` 返回的就是末尾残差）。门是 0/1，这里
        用 0.5 判——fp32 累加在 0/1 上是精确的，这个阈值不会切到别的东西。
        """
        s = self.exec.dim_slot

        def val(d: int) -> float:
            return float(res[..., self.d0 + s[d]].max())

        chars = 0
        for b, d in enumerate(self.sys_dims["chars"]):
            if d is not None and val(d) > 0.5:      # 常数 0 的位没有 dim
                chars |= 1 << b
        return {"put": val(self.sys_dims["put"]) > 0.5,
                "stop": val(self.sys_dims["stop"]) > 0.5,
                "chars": chars}


def _sys_dims(step: dict) -> dict:
    """`step_circuit` 的 `sys` -> dim 号。常数 0 的位在 `chars` 里可能是空 Expr，跳过。"""
    sys = step["sys"]
    put, stop = _dim(sys["put"]), _dim(sys["stop"])
    if put is None or stop is None:
        raise ValueError("syscall 的 put/stop 读成了常数 0——a7 的比较收窄了？")
    return {"put": put, "stop": stop,
            "chars": [_dim(e) for e in sys["chars"]]}


def assemble(firmware: str | Path, d0: int = DEFAULT_D0, units: int = 4096,
             heads: int = 32, text: list[bytes] | None = None,
             keep: tuple[str, ...] = ()) -> tuple:
    """从固件编一份 chip 段，返回 `(seg, graph, compiled, step)`。

    `build()` 只要第一项；`probe_host` 要把后三项拿来注入初值、跟 `compile.run`
    对拍。两者走同一条路，所以"探针验的那台机器"和"挂到宿主上那台"是同一台。
    """
    words = cc.load_words(firmware, pre=True)
    return assemble_words(words, d0, units, heads,
                          drom=cc.load_drom(firmware), text=text, keep=keep)


def assemble_words(words: list[int], d0: int = DEFAULT_D0, units: int = 4096,
                   heads: int = 32, drom=None,
                   text: list[bytes] | None = None,
                   keep: tuple[str, ...] = ()) -> tuple:
    """`assemble` 的字级版（固件已经解成 `words`）：`(seg, graph, compiled, step)`。

    `text` 是 `pytext.vocab_rows()` 那张表：**整个词表**、下标 = tokenizer 的 token id、
    每个 id 一行 16 B（NUL 填充）。给了才建文本桥：电路多 128 个字节位、`mem["text"]`
    多一张 TCO/TX/TCC 表，槽位 414 -> 560。**不给就没有 `<cpu_call>` 的语义**——414 槽
    那台收得到 TX token，但没有一条通路把载荷变成字节，`gather()` 永远读到 0。

    **表与代码无关**：它枚举的是词表，不是程序。所以装配一次之后那份权重收得下任何一段
    ≤`CAP_MAX` 个 token 的代码（`flow.call` 把 token id 当载荷发过来）。早先这里传的是
    “这段代码的行表”、载荷是“这段文本里第几个”，后果是换代码得重装配——2026-11-03 改掉。

    `keep` 是额外想在残差里读回来的信号，按名字给：`cnt`（ROM 行号）/ `w`（取指字）/
    `link`（`PC+4`）/ `tgt`（跳转目标）。它们图内没人再读，不钉住就会在 token 末尾被
    别的 dim 接槽——`probe_whole` 拿它们跟参考机对账，代价是各占一个槽到 token 末尾。
    """
    g = Graph(units=units, heads=heads)
    feats = cc.token_features(g, text=text)
    step = cc.step_circuit(g, words, feats=feats, drom=drom, text=text)
    # `sys` 的 11 个 dim 要给 Python 读，所以编译时得钉住（见 `plan_slots` 第 3 条）：
    # 不进 `final` 的话，末尾残差那一列已经是别的 dim 了。
    sys_dims = _sys_dims(step)
    want = [d for d in (sys_dims["put"], sys_dims["stop"], *sys_dims["chars"])
            if d is not None]
    for k in keep:
        want += [d for d in map(_dim, step["tgt"]["sel"] if k == "tgt" else step[k])
                 if d is not None]
    c = ccm.compile(g, cc.VOCAB, d0=d0, keep=want)
    seg = ChipSegment(NeuralCPU.from_compiled(c), d0, sys_dims, c.state())
    return seg, g, c, step


# ---------------- 整机 pt ----------------
#
# 这台机器的权重不训练，是 `assemble` **现算**出来的（207 MB / 15~27 s）。于是每次想跑
# 一次就重算一次——除了“省时间”，还有一条更硬的理由要落盘：**权重是产物**。一个 .pt
# 就是一台装得起来的神经 CPU，设备侧不必带 graph、不必带 circuit、不必有 torch 之外的
# 任何东西。
#
# 三样缺一不可，后两样都**推不出来**：
#
#   state    207 MB 权重，`Compiled.state()` 的扁平表（bf16，可直接 load_state_dict）
#   signals  `step` 里被 Python 读的那几样：cnt / 取指 / 链接 / 目标各是哪些位、三个
#            寄存器端口落在哪一层、写回表的 32 位怎么摆。manifest 只记**槽**，不记
#            “哪一位是 pc”——那是图的知识，装不回来也就猜不出来
#   text     "vocab"：这张 `feat` 的 128 个 `tb` 位是按**整个词表**枚举的（`pytext.vocab_rows`），
#            所以**一个 pt 收所有程序**——TX 带自己的 token id，字节已在权重里。换代码
#            既不重装配也不重 dump（早先按“这段文本里第几个”排，才那么绑）
#
# `src` 只是记一行“当初 `dump` 时跑的是哪段代码”，给人和日志看的，**不是约束**。

#: 整机 pt 的格式版本。加字段不用动它；改了字段**含义**才 +1。
#: v2（2026-11-03）：`text` 改成按 token id 排的全词表，`feat` 不再绑代码；`rows` 字段撤。
NUCPU_VERSION = 2


def rows_hash(rows: list[bytes]) -> str:
    """行表摘要：**快照**拿它卡“这段话是不是当初那段”（`probe_whole` 的 `stage="prog"`）。

    pt 不用它：那份权重与代码无关，没有“行表”可比。
    """
    return hashlib.sha256(repr(rows).encode()).hexdigest()[:16]


def _layer_of(c, k: int) -> int:
    """`step_circuit` 的节点号 -> `NeuralCPU` 的层号（压槽会在 head 前插清零层）。"""
    return c.pos_layer[c.plan.pos_of[k]]


def signals(c, step: dict) -> dict:
    """`step` 里被 Python 读的坐标 -> 可落盘的整数表。

    两条转换都是“把对象换成身份”：`Expr` -> 它唯一的 dim 号（`_dim`），节点号 -> 层号
    （`_layer_of`）。`wb` 只留位号——每项原本是 `(位, Chan)`，而 `Chan` 是图里的活对象，
    落盘装不回来；读一行本来也只用位号（见 `circuit.record_row`）。

    这件事**只能在这里做**：`Compiled` 的 manifest 只知道槽位活着，不知道哪个槽是
    `cnt`。所以 pt 里必须有这一份，否则装回来的机器一拍都跑不了（读不出对账要的值）。
    """
    return {"cnt": [_dim(e) for e in step["cnt"]],
            "w": [_dim(e) for e in step["w"]],
            "link": [_dim(e) for e in step["link"]],
            "tgt": [_dim(e) for e in step["tgt"]["sel"]],
            "wb": [int(b) for b, _ in step["wb"]],
            "cnt_layer": int(_layer_of(c, step["cnt_i"])),
            "ports": {p: int(_layer_of(c, k)) for p, k in step["ports_i"].items()}}


def _row_of(v: int, bits: list[int]) -> list[float]:
    """32 位值 -> 一整行的列值（只填每位的本拍列）。`circuit.record_row` 的位号版。

    差别只有一处：那边每项还带一个 `Chan`（图里的活对象，pt 里没有），而它一个值也
    不贡献。所以从 pt 复位时用这一份，行形状与写回时**逐位相同**。
    """
    row, seen = [0.0] * len(bits), set()
    for col, b in enumerate(bits):
        if b not in seen:
            seen.add(b)
            row[col] = float((v >> b) & 1)
    return row


def reset_carry(seg: ChipSegment, sig: dict, cnt_row: int,
                regs: list[int]) -> dict:
    """复位那一格的 carry：`cnt` 行号 + 三个端口各 32 行寄存器值。

    只写这四张表：其余层由 `NeuralCPU.forward` 的 `carry.get` 用 `S0` 兜底，与
    `probe_slots._bridge` 的结果逐位相同（那边也是“S0 副本 + 覆盖若干行”，`states`
    里没有的层它一个也不生成）。所以从 pt 起跑不需要图、不需要 `Compiled`，也就
    不需要装配——这正是“权重是一个文件”的意思。
    """
    def sheet(li: int) -> torch.Tensor:
        return seg.exec.layers[li].vheads[0].S0.to(ACC).clone()

    out: dict = {}
    S = sheet(int(sig["cnt_layer"]))
    nbits = len(sig["cnt"])
    S[0, :nbits] = torch.tensor([float((cnt_row >> b) & 1) for b in range(nbits)],
                                dtype=ACC)
    out[(int(sig["cnt_layer"]), "S", 0)] = S
    for li in sig["ports"].values():
        S = sheet(int(li))
        for r in range(cc.REGS):
            row = _row_of(regs[r], sig["wb"])
            S[r, :len(row)] = torch.tensor(row, dtype=ACC)
        out[(int(li), "S", 0)] = S
    return out


def save_nucpu(path: str | Path, seg: ChipSegment, c, step: dict,
               *, src: str = "", fp: str = "") -> Path:
    """整机落盘：`{权重, 信号坐标}` 一个 .pt。

    `fp` 是装配指纹（`probe_whole._fingerprint`）：它认的是**电路**，不含 `feat`，所以 pt
    与 boot 快照拿同一个数互相认亲。`src` 是当初跑的那段代码，只做记录。

    **不存行表**：`text` 是按整个词表枚举的（`pytext.vocab_rows`），换程序不用重装配，
    也就没有“这份 pt 认哪段代码”这回事。
    """
    if seg._flat is None:
        raise RuntimeError("这个段没有扁平 state（不是 build()/assemble() 出来的）")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": NUCPU_VERSION, "fp": fp, "src": src, "text": "vocab",
                "manifest": seg.exec.spec, "state": seg._flat,
                "d0": seg.d0, "sys_dims": seg.sys_dims,
                "signals": signals(c, step), "spec": seg.spec()}, p)
    return p


def load_nucpu(path: str | Path) -> tuple[ChipSegment, dict]:
    """装回来一台整机，返回 `(段, 整个 payload)`。

    比装配便宜两个数量级（207 MB 读盘 vs 从图重算 15~27 s），而且**不依赖图**：
    `NeuralCPU` 按 manifest 重建结构、按 `state` 摆权重，`signals` 补上 manifest 里
    没有的那层语义。`flat` 跟着传进去，于是装出来的段还能再 `save()`（`from_file`
    那条老路不带它，只能跑到 `NeuralCPU` 为止）。

    装回来就是一台**通用**机器：`--src=` 发哪段代码都行（`meta["text"] == "vocab"`），
    字节表按 token id 排在 `feat` 里，与程序无关。
    """
    z = torch.load(Path(path), map_location="cpu", weights_only=True)
    v = int(z.get("version", -1))
    if v != NUCPU_VERSION:
        raise RuntimeError(f"整机 pt 版本 {v} != {NUCPU_VERSION}，重新 `dump`")
    seg = ChipSegment(NeuralCPU(z["manifest"], z["state"]), int(z["d0"]),
                      z["sys_dims"], z["state"])
    return seg, z


def tiny(d0: int = DEFAULT_D0, words: list[int] | None = None,
         units: int = 256) -> ChipSegment:
    """小机：4 行合成 ROM，秒级编完。形状口径与真固件一致（同一条 `compile`），
    只是规模小到能在自检里随口跑——真固件留给 `probe_host`。"""
    if words is None:
        words = [cc._pack("addi", 1, 0, 1), cc._pack("addi", 2, 0, 2),
                 cc._pack("ecall", 0, 17, 10), cc._pack("addi", 3, 0, 3)]
    return assemble_words(words, d0, units)[0]          # heads=32 要够：regs 3 + rom 1 + ram 16


# ---------------- 自检 ----------------

def _demo() -> None:
    """小机自检：4 行合成 ROM + d0=1152，验形状与三条不变量（真固件留给 probe_host）。"""
    d0 = 1152
    seg = tiny(d0)
    sp = seg.spec()
    print(f"形状: d0 {sp['d0']} + {sp['slots']} 槽 = {seg.used}  hidden {sp['hidden']}"
          f"（m={sp['m']}、log2_den={sp['log2_den']}）  零区 {sp['zone']} 格")

    res, _ = seg([cc.EXEC, cc.tok("F0", 0x1234), cc.EXEC])
    host = res[:, :d0]
    zone = res[:, seg.used:seg.ballast0]
    ball = res[:, seg.ballast0:]
    assert not host.any(), "host 区被写了（宿主残差会被污染）"
    assert not zone.any(), f"零区被写了：{int((zone != 0).sum())} 处"
    assert torch.all(ball == seg.exec.ballast), "ballast 被改了（RMSNorm 分母会漂）"
    print(f"不变量: host 区 {d0} 列全 0 · 零区 {sp['zone']} 列全 0 · ballast "
          f"{sp['m']} 列恒 {seg.exec.ballast:g}  OK")

    # dim -> slot：token 特征确实落在 chip 区，不在 host 区
    slots = [seg.d0 + seg.exec.dim_slot[d] for d in seg.exec.feat_dims]
    assert min(slots) >= d0 and max(slots) < seg.used, "特征列跑到 chip 区外面了"
    print(f"映射: {len(slots)} 个特征列落在 [{min(slots)}, {max(slots)}] ⊆ "
          f"[{d0}, {seg.used})  OK")

    # ecall 出口：a7 没写 → put/stop 都不出
    out = seg.read(res[-1])
    assert not out["put"] and not out["stop"], f"非 ecall 拍误判成 {out}"
    print("syscall: 非 ecall 拍不误判  OK")


if __name__ == "__main__":
    _demo()
