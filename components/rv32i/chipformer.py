"""
【已废弃】M0 按 plan.md §2 退役本文件，M2 会整份重写为 graph.py 的设备侧执行器。
保留至今的唯一原因：下面的 npz schema 是 from_weights() 报错信息的锚点。
它为什么不能用（host 在 float64 里扮演 attention）见 plan.md §2.3。


MiniRainVLM-CPU 神经 CPU（载体轨 v8.1）

本文件只有机制原语；语义全部在 npz 权重与 manifest 里，host 不认识指令/寄存器。

npz schema（emit.py 必须按此产出）：
  manifest_json  dict:
      d_stream     int    流宽
      d_ff         int>=1 ProgFFN 隐层宽（0 会退化成常数层）
      feat         dict   协议特征坐标：fam_oh[13] / pay[16] 必给；
                          fin / is_qs / is_qd / is_jr / pay_hi / qdigit 可选（各 1 个坐标）
      qdigit_scale int    商数字填充倍率（默认 0x11111111）
      lag_blocks   [int]  tap 块起点；块宽 = len(feat_coords)，与坐标取值无关
      code_coords  [20]   exit 预测码坐标（NFAM 4 + NPAY 16）
      layers       [{attn_reads: [[bank, addr_c, out_c, w]], mlp: bool}]
      prewrites    [[bank, addr, val, beta, w?]]   层栈之前提交（本 token 的层可读）
      writes       [[bank, addr, val, beta, w?]]   层栈之后提交（下一 token 读）
      strict       bool   验收轨：地址/门控/值非精确整数即抛 ExactnessError
  W_{rom,rom_raw,cnt,regs,quo,ram0,ram1}      状态快照，形状同 ChipState
  L<i>_{Wa,Wb,V,Wo}_{weight,bias}              第 i 层 ProgFFN 权重（i 从 1 起）

调度表条目取值：int = 流坐标（val 取 x[c:c+w]），float = 结构常数（绝对行 / 常数）；
  ["regs", 5,   3, 1.0] -> regs[x[5]] = x[3]
  ["regs", 5.0, 3, 1.0] -> regs[5]    = x[3]
银行列数：cnt 1 / rom 5 / rom_raw 1 / regs 1 / quo 1 / ram0 1 / ram1 1。
坐标区间左闭右开且两两不交叠（feat / lag_blocks / code_coords / attn 输出）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FAMILIES = {
    "NT": 0, "V": 15, "F0": 15, "F1": 15, "F2": 2,
    "J": 16, "R": 16, "ST": 16, "SS": 0, "QS": 4, "QD": 7, "HALT": 0, "BYTE": 9,
}
FAM_ID = {n: i for i, n in enumerate(FAMILIES)}
NFAM = max(1, (len(FAMILIES) - 1).bit_length())
NPAY = 16
TOKENS = [(f, p) for f, w in FAMILIES.items() for p in range(1 << w)]
VOCAB = len(TOKENS)
TOK_ID = {t: i for i, t in enumerate(TOKENS)}
FINAL = ("NT", "V", "F2", "J", "R", "ST", "SS", "HALT", "BYTE")
IS_FINAL = torch.tensor([1.0 if TOKENS[t][0] in FINAL else 0.0 for t in range(VOCAB)])
MARGIN_T = 2.0 ** 20    # GLU 大门（silu(t·g)/t 对整数门精确）


# ---------------- 机制原语（结构常数，无语义） ----------------

class ExactnessError(RuntimeError):
    """精确性违约：地址/门控/值不是精确整数（验收轨抛出，替代静默降级）。"""


def onehot_val(v: torch.Tensor, n: int, strict: bool = False) -> torch.Tensor:
    v = torch.as_tensor(v, dtype=torch.float64)
    if strict:
        fv = float(v)
        if fv != round(fv):
            raise ExactnessError(f"onehot_val: 非整数地址 {fv}")
        if not 0 <= fv < n:
            raise ExactnessError(f"onehot_val: 地址 {fv} 越界 [0, {n})")
    r = torch.arange(n, dtype=torch.float64)
    return torch.relu(v - r + 1.0) - 2.0 * torch.relu(v - r) + torch.relu(v - r - 1.0)


def stream_val(x: torch.Tensor, e, w: int = 1) -> torch.Tensor:
    """调度表取值：int = 流坐标（x[e] / x[e:e+w]），float = 结构常数（必须 float64）。"""
    if isinstance(e, int):
        return x[e] if w == 1 else x[e:e + w]
    return torch.tensor(float(e), dtype=torch.float64)


class WritePort:
    def __init__(self, strict: bool = False) -> None:
        self.ops: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.strict = strict

    def write(self, bank: str, addr, value, beta) -> None:
        self.ops.append((bank, torch.as_tensor(addr, dtype=torch.float64),
                         torch.as_tensor(value, dtype=torch.float64),
                         torch.as_tensor(beta, dtype=torch.float64)))

    def apply(self, state: "ChipState") -> None:
        for bank, addr, v, beta in self.ops:
            S = getattr(state, bank)
            if self.strict:
                fb, fa = float(beta), float(addr)
                if fb not in (0.0, 1.0):
                    raise ExactnessError(f"WritePort: beta={fb} 非精确 0/1（{bank}）—— 半更新不可逆")
                if fa != round(fa) or not 0 <= fa < S.shape[0]:
                    raise ExactnessError(f"WritePort: 地址 {fa} 非法（{bank} 共 {S.shape[0]} 行）—— 静默丢写")
                vv = torch.as_tensor(v, dtype=torch.float64)
                if not bool(torch.all(vv == torch.round(vv))):
                    raise ExactnessError(f"WritePort: {bank} 写入非整数值 {vv.reshape(-1).tolist()}")
                if S.shape[1] > 1 and vv.numel() != S.shape[1]:
                    raise ExactnessError(
                        f"WritePort: {bank} 宽 {S.shape[1]} 列收到 {vv.numel()} 维值（标量广播被禁）")
            k = onehot_val(addr, S.shape[0], self.strict)
            delta = torch.as_tensor(v, dtype=torch.float64).reshape(-1) - (k @ S)   # (width,)
            S.data += beta * torch.outer(k, delta)      # 任意宽度银行的行替换（beta=0 不动）


class ChipState(nn.Module):
    """
    CPU 状态
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("cnt", torch.zeros(5, 1, dtype=torch.float64), persistent=False)      # anchor/n/j/v/divk0
        self.register_buffer("rom", torch.zeros(cfg.rom_size // 4, 5, dtype=torch.float64), persistent=False)
        self.register_buffer("rom_raw", torch.zeros(cfg.rom_size // 4, 1, dtype=torch.float64), persistent=False)
        self.register_buffer("regs", torch.zeros(32, 1, dtype=torch.float64), persistent=False)
        self.register_buffer("quo", torch.zeros(8, 1, dtype=torch.float64), persistent=False)
        self.register_buffer("ram0", torch.zeros(cfg.ram_words // 2, 1, dtype=torch.float64), persistent=False)
        self.register_buffer("ram1", torch.zeros(cfg.ram_words // 2, 1, dtype=torch.float64), persistent=False)


# ---------------- 算术子层（ProgFFN：形状统一，权重逐层来自 npz） ----------------

class ProgFFN(nn.Module):
    """
    可编程算术子层
    """

    def __init__(self, d_stream: int, d_ff: int) -> None:
        super().__init__()
        self.Wa = nn.Linear(d_stream, d_ff, bias=True)
        self.Wb = nn.Linear(d_stream, d_ff, bias=True)
        self.V = nn.Linear(d_stream, d_ff, bias=True)
        self.Wo = nn.Linear(2 * d_ff, d_stream, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.Wa(x))
        h2 = F.silu(MARGIN_T * self.Wb(x)) * self.V(x) / MARGIN_T
        return self.Wo(torch.cat([h1, h2]))



class ChipLayer(nn.Module):
    """
    神经CPU TransformerBlock
    """

    def __init__(self, idx: int, layer: dict, d_stream: int, d_ff: int) -> None:
        super().__init__()
        self.name = f"L{idx + 1}"
        self.attn_reads = layer.get("attn_reads", [])
        self.has_mlp = bool(layer.get("mlp"))
        if self.has_mlp:
            self.ffn = ProgFFN(d_stream, d_ff)

    def forward(self, x: torch.Tensor, state: ChipState, wp: WritePort) -> torch.Tensor:
        # ---- attn 槽：状态读（Q = onehot_val(addr)，K/V = 状态 Parameter 行） ----
        for bank, addr_c, out_c, width in self.attn_reads:
            S = getattr(state, bank)
            addr = x[int(addr_c)]
            k = onehot_val(addr, S.shape[0], wp.strict)
            v = k @ S                                     # (width,)
            if wp.strict and not bool(torch.all(v == torch.round(v))):
                raise ExactnessError(
                    f"attn_reads: {bank} 读出非整数 {v.reshape(-1).tolist()}"
                    f"（addr={float(addr)}）—— 状态含小数则地址索引必然泄漏到邻行")
            x[int(out_c):int(out_c) + int(width)] += v
        # ---- mlp 槽：编译算术（权重来自 npz；全流宽稀疏路由） ----
        if self.has_mlp:
            x = x + self.ffn(x)
        return x


# ---------------- manifest 契约（schema 与静态校验） ----------------

def _validate_manifest(manifest: dict, cfg) -> None:
    """构造期静态校验：违反契约给出指认字段的错误，而不是内部数组的 IndexError。"""
    def need(cond, msg):
        if not cond:
            raise ValueError(f"manifest: {msg}")
    need("d_stream" in manifest and "d_ff" in manifest, "缺 d_stream / d_ff")
    need(int(manifest["d_ff"]) >= 1, "d_ff 至少 1（0 会静默退化成 Wo.bias 常数层，S4）")
    need("layers" in manifest and len(manifest["layers"]) >= 1, "layers 至少一层")
    feat = manifest.get("feat", {})
    feat_ok = {"fam_oh", "pay", "fin", "is_qs", "is_qd", "is_jr", "pay_hi", "qdigit"}
    for nm in feat:
        need(nm in feat_ok, f"feat.{nm} 不是已知特征（可用 {sorted(feat_ok)}）")
    need(float(manifest.get("qdigit_scale", 0x11111111)).is_integer(),
         f"qdigit_scale={manifest.get('qdigit_scale')} 必须整数（strict 轨要求精确整数）")
    need(len(feat.get("fam_oh", [])) == len(FAMILIES),
         f"feat.fam_oh 需要 {len(FAMILIES)} 个坐标（每家族一个），给了 {len(feat.get('fam_oh', []))}")
    need(len(feat.get("pay", [])) == NPAY, f"feat.pay 需要 {NPAY} 个坐标（按位展开）")
    need(len(manifest.get("code_coords", [])) == NFAM + NPAY,
         f"code_coords 需要 {NFAM + NPAY} 个坐标")
    ds = int(manifest["d_stream"])
    def coord_ok(c, w=1):
        return isinstance(c, int) and 0 <= c and c + w <= ds
    for nm, v in feat.items():
        for c in (v if isinstance(v, list) else [v]):
            need(coord_ok(int(c)), f"feat.{nm} 坐标 {c} 超出 d_stream={ds}")
    n_feat = len({int(c) for v in feat.values() for c in
                  (v if isinstance(v, list) else [v])})
    for off in manifest.get("lag_blocks", []):
        need(coord_ok(int(off), n_feat), f"lag_blocks {off} 放不下特征块（稠密宽 {n_feat}）")
    for c in manifest["code_coords"]:
        need(coord_ok(int(c)), f"code_coords {c} 超出 d_stream={ds}")
    bank_cols = {"cnt": 1, "rom": 5, "rom_raw": 1, "regs": 1, "quo": 1,
                 "ram0": 1, "ram1": 1}
    # ---- 坐标占用区登记：越界 + 重叠即报错（compile 契约：坐标无重叠）----
    spans = []
    def claim(c, w, tag):
        c = int(c); w = int(w)
        need(coord_ok(c, w), f"{tag} 坐标 [{c},{c + w}) 越界 d_stream={ds}")
        for a, b, t0 in spans:
            need(c + w <= a or b <= c, f"{tag} 坐标 [{c},{c + w}) 与 {t0} [{a},{b}) 重叠")
        spans.append((c, c + w, tag))
    for nm, v in feat.items():
        for c in (v if isinstance(v, list) else [v]):
            claim(c, 1, f"feat.{nm}")
    for off in manifest.get("lag_blocks", []):
        claim(off, n_feat, f"lag_blocks {off}")
    for c in manifest["code_coords"]:
        claim(c, 1, "code_coords")
    for i, layer in enumerate(manifest["layers"]):
        for bank, addr, out, w in layer.get("attn_reads", []):
            need(bank in bank_cols, f"layers[{i}].attn_reads 未知银行 {bank}")
            need(bank_cols[bank] == int(w), f"layers[{i}].attn_reads {bank} 宽度须 {bank_cols[bank]}")
            # 读地址是引用不是占用：与 writes 的 addr/beta 一样只查越界（addr==out 允许）
            need(coord_ok(int(addr)), f"layers[{i}].attn_reads 地址坐标 {addr} 越界")
            claim(out, int(w), f"layers[{i}].attn_reads {bank} 输出")
    for tbl in ("prewrites", "writes"):
        for j, entry in enumerate(manifest.get(tbl, [])):
            need(len(entry) >= 4, f"{tbl}[{j}] 至少 4 项 [bank, addr, val, beta]")
            bank, addr, val, beta = entry[:4]
            w = int(entry[4]) if len(entry) > 4 else 1
            need(bank in bank_cols, f"{tbl}[{j}] 未知银行 {bank}")
            need(bank_cols[bank] == w, f"{tbl}[{j}] {bank} 宽度须 {bank_cols[bank]}")
            need(not (isinstance(val, float) and bank_cols[bank] > 1),
                 f"{tbl}[{j}] {bank} 是多列银行：val 必须给坐标+宽度（标量常数会静默广播成全列同值，S5）")
            for tag, e in (("addr", addr), ("beta", beta)):
                if isinstance(e, int):
                    need(coord_ok(int(e)), f"{tbl}[{j}].{tag} 坐标 {e} 越界")
            if isinstance(val, int):
                need(coord_ok(int(val), w), f"{tbl}[{j}].val 坐标 {val}+w 越界")
    need(all(("attn_reads" in l) or ("mlp" in l) for l in manifest["layers"]),
         "存在 attn/mlp 全空的层（要么删层，要么显式声明 {\"attn_reads\": [], \"mlp\": False}）")


# ---------------- 整机 ----------------

class NeuralCPU(nn.Module):
    """
    神经 CPU
    """

    def __init__(self, cfg, manifest: dict, strict: bool | None = None) -> None:
        super().__init__()
        _validate_manifest(manifest, cfg)             # 契约先行：报错指认字段
        self.cfg = cfg
        self.manifest = manifest
        # 验收轨：strict 模式全程断言地址/门控/值的精确整数性（ExactnessError）
        self.strict = bool(manifest.get("strict", False)) if strict is None else bool(strict)
        self.state = ChipState(cfg)
        n_layers = len(manifest["layers"])              # 层数由 manifest（compile 的深度摊派）决定
        self.layers = nn.ModuleList(
            [ChipLayer(i, manifest["layers"][i], manifest["d_stream"], manifest["d_ff"])
             for i in range(n_layers)])
        # ---- token 嵌入表：协议特征坐标行，init 物化（结构常数缓冲）----
        # 稠密索引：表宽 = len(feat_coords)（与坐标取值无关），fidx 做坐标->列映射；
        # feat 坐标放高位不再撑爆内存（旧实现表宽 = max(coord)+1）。
        feat = manifest["feat"]                         # {"fam_oh": [coords], "pay": [coords], "fin": c, ...}
        self.feat_coords = sorted(c for v in feat.values() for c in (v if isinstance(v, list) else [v]))
        fidx = {c: i for i, c in enumerate(self.feat_coords)}
        table = torch.zeros(VOCAB, len(self.feat_coords), dtype=torch.float64)
        ar = torch.arange(VOCAB)
        fam_ids = torch.tensor([FAM_ID[TOKENS[t][0]] for t in range(VOCAB)])
        pays = torch.tensor([TOKENS[t][1] for t in range(VOCAB)], dtype=torch.int64)
        if "fam_oh" in feat:
            fam_c = torch.tensor([fidx[int(c)] for c in feat["fam_oh"]])
            table[ar, fam_c[fam_ids]] = 1.0
        if "pay" in feat:
            for i, c in enumerate(feat["pay"]):
                table[ar, fidx[int(c)]] = ((pays >> i) & 1).double()
        for nm, getter in (("fin", lambda t: IS_FINAL[t].item()),
                             ("is_qs", lambda t: float(TOKENS[t][0] == "QS")),
                             ("is_qd", lambda t: float(TOKENS[t][0] == "QD")),
                             ("is_jr", lambda t: float(TOKENS[t][0] in ("J", "R")))):
            if nm in feat:
                table[:, fidx[int(feat[nm])]] = torch.tensor([getter(t) for t in range(VOCAB)])
        # 商数字协议字段（供 prewrites 读取；对所有 token 无差别填充，host 不区分家族）
        qscale = float(manifest.get("qdigit_scale", 0x11111111))
        if "pay_hi" in feat:
            table[:, fidx[int(feat["pay_hi"])]] = (pays >> 4).double()
        if "qdigit" in feat:
            table[:, fidx[int(feat["qdigit"])]] = (pays & 15).double() * qscale
        self.register_buffer("embed", table, persistent=False)
        # ---- 固定码本（±1，向量化构造；与 embed 内容/宽度不同）----
        fb = ((fam_ids.unsqueeze(1) >> torch.arange(NFAM)) & 1).double()
        pb = ((pays.unsqueeze(1) >> torch.arange(NPAY)) & 1).double()
        self.register_buffer("codebook", torch.cat([fb, pb], 1) * 2.0 - 1.0,
                             persistent=False)                              # (VOCAB, 20)
        self.code_coords = [int(c) for c in manifest["code_coords"]]
        # ---- tap：上几个 token 的特征块（默认无 tap；t−1, t−2, ...）----
        self.lag_blocks = [int(o) for o in manifest.get("lag_blocks", [])]
        # ---- 预写表：层栈之前提交的协议写（旧版硬编码的商数字预填已迁入 manifest）----
        self.prewrites = list(manifest.get("prewrites", []))
        self.double()                                     # 全机 float64（位精确语义要求）

    # ---- 权重装载（npz schema 见模块 docstring；缺失 key 一律报错，不静默跳过） ----
    @classmethod
    def from_weights(cls, cfg, path: str | Path, strict: bool | None = None) -> "NeuralCPU":
        z = np.load(path, allow_pickle=False)
        if "manifest_json" not in z:
            raise KeyError("npz 缺 manifest_json（emit.py 必须产出，见模块 docstring 的 npz schema）")
        manifest = json.loads(str(z["manifest_json"]))
        cpu = cls(cfg, manifest, strict)
        missing = []
        for key in ("rom", "rom_raw", "regs", "quo", "ram0", "ram1", "cnt"):
            k = f"W_{key}"
            if k not in z:
                missing.append(k)
            else:
                t = torch.as_tensor(z[k], dtype=torch.float64)
                tgt = getattr(cpu.state, key)
                if t.shape != tgt.shape:
                    raise ValueError(f"{k} 形状 {tuple(t.shape)} != 期望 {tuple(tgt.shape)}")
                if cpu.strict and not bool(torch.all(t == torch.round(t))):
                    raise ExactnessError(
                        f"{k} 含 {int((t != torch.round(t)).sum())} 个非整数 —— 状态快照必须精确整数，"
                        "否则每次地址索引都会泄漏到邻行并被行值量级放大")
                tgt.data.copy_(t)
        for i in range(len(manifest["layers"])):
            ffn = getattr(cpu.layers[i], "ffn", None)
            if ffn is None:
                continue
            for pname, p in ffn.named_parameters():
                k = f"L{i + 1}_{pname.replace('.', '_')}"
                if k not in z:
                    missing.append(k)
                else:
                    t = torch.as_tensor(z[k], dtype=torch.float64)
                    if t.shape != p.shape:
                        raise ValueError(f"{k} 形状 {tuple(t.shape)} != 期望 {tuple(p.shape)}")
                    p.data.copy_(t)
        if missing:
            raise KeyError(f"npz 缺 {len(missing)} 个权重 key: {missing[:5]}{' …' if len(missing) > 5 else ''}"
                           "（key 名见 docstring；emit.py 必须与之一致）")
        return cpu

    def _schedule(self, wp: WritePort, table: list, x: torch.Tensor) -> None:
        """静态调度表 -> WritePort：int 取流坐标、float 取结构常数（host 不认识语义）。"""
        for entry in table:
            bank, addr, val, beta = entry[:4]
            w = int(entry[4]) if len(entry) > 4 else 1
            wp.write(bank, stream_val(x, addr), stream_val(x, val, w), stream_val(x, beta))

    @torch.no_grad()
    def forward(self, t: int, hist: list[int]) -> torch.Tensor:
        wp = WritePort(self.strict)
        # ---- 流初始化：本 token 特征坐标 + tap（t−1, t−2, ... 的特征块）----
        x = torch.zeros(self.manifest["d_stream"], dtype=torch.float64)
        x[self.feat_coords] = self.embed[t]              # 稠密特征行 -> feat 坐标
        fd = torch.arange(len(self.feat_coords))       # tap 块内稠密偏移（块宽 = len，不再吃 feat_max）
        for k, off in enumerate(self.lag_blocks):
            if len(hist) >= k + 2:
                x[off + fd] = self.embed[hist[-k - 2]]  # tap：上 k+1 个 token 的特征行
        # ---- 预写：manifest 声明、层栈之前立即提交（本 token 的层可读）----
        if self.prewrites:
            pre = WritePort(self.strict)
            self._schedule(pre, self.prewrites, x)
            pre.apply(self.state)
        # ---- N 个统一层 ----
        for L in self.layers:
            x = L(x, self.state, wp)
        # ---- 静态写口提交（commit；下一 token 读 post-commit 状态）----
        self._schedule(wp, self.manifest.get("writes", []), x)
        wp.apply(self.state)
        # ---- exit：code 坐标上的 ±1 预测码，tied-embedding 一致度打分 ----
        code = x[self.code_coords]
        if self.strict:
            mag = code.abs()
            if not bool(torch.all(mag == mag[0])):
                raise ExactnessError(
                    f"exit: code 坐标幅值不一致 |v|∈[{float(mag.min())}, {float(mag.max())}]"
                    " —— 码本 margin 仅 2，幅值抖动会让 argmax 偏向小编号 token")
        return self.codebook @ code

    @torch.no_grad()
    def step(self, t: int, hist: list[int]) -> tuple[int, torch.Tensor]:
        """forward + argmax：返回 (下一 token id, logits)。"""
        logits = self.forward(t, hist)
        return int(torch.argmax(logits).item()), logits

    def run(self, max_tokens: int = 1_000_000) -> tuple[list[int], list[int]]:
        """监视器（非 CPU 语义）：驱动 token 序列，按 HALT / BYTE 家族做 I/O。"""
        t = TOK_ID[("NT", 0)]
        hist = [t]
        stdout: list[int] = []
        for _ in range(max_tokens):
            nxt, _ = self.step(t, hist)
            hist.append(nxt)
            if TOKENS[nxt][0] == "HALT":
                return hist, stdout
            if TOKENS[nxt][0] == "BYTE":
                stdout.append(TOKENS[nxt][1] & 0xFF)
            t = nxt
        raise RuntimeError("token limit")


def main() -> None:
    raise SystemExit("chipformer v8.1 已废弃（plan.md M0）：本文件将在 M2 整份重写为 graph.py 的"
                     "设备侧执行器。当前不内置语义/自测，也不接受 npz。")