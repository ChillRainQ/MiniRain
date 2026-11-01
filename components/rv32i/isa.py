"""
RV32I 指令集
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ISA 字宽掩码。本层是纯编码/译码，不依赖 RV32Config（可以直接 python isa.py 跑），
# 所以在这里自持一份；配置层的真源是 core.RV32Config.word_mask。
WORD_MASK = (1 << 32) - 1

OP_OP     = 0b0110011   # R 型算术逻辑（add/sub/sll/...）
OP_IMM    = 0b0010011   # I 型算术逻辑（addi/andi/slli/...）
OP_LUI    = 0b0110111   # U 型：加载高位立即数
OP_AUIPC  = 0b0010111   # U 型：PC 相对高位加
OP_JAL    = 0b1101111   # J 型：无条件跳转
OP_JALR   = 0b1100111   # I 型：寄存器间接跳转
OP_BRANCH = 0b1100011   # B 型：条件分支
OP_LOAD   = 0b0000011   # I 型：读内存
OP_STORE  = 0b0100011   # S 型：写内存
OP_SYSTEM = 0b1110011   # SYSTEM 型：ecall/ebreak
OP_MISC_MEM = 0b0001111 # MISC-MEM 型：fence/fence.i

# 指令主表：助记符 -> (格式, opcode, funct3, funct7)
TABLE: dict[str, tuple[str, int, int, int]] = {
    "add":  ("R", OP_OP,     0b000, 0x00),   # f7=0x00 加
    "sub":  ("R", OP_OP,     0b000, 0x20),   # f7=0x20 减（与 add 同 f3，靠 f7 区分）
    "sll":  ("R", OP_OP,     0b001, 0x00),   # 逻辑左移
    "srl":  ("R", OP_OP,     0b101, 0x00),   # 逻辑右移
    "sra":  ("R", OP_OP,     0b101, 0x20),   # 算术右移（与 srl 同 f3，靠 f7 区分）
    "and":  ("R", OP_OP,     0b111, 0x00),
    "or":   ("R", OP_OP,     0b110, 0x00),
    "xor":  ("R", OP_OP,     0b100, 0x00),
    "addi": ("I", OP_IMM,    0b000, 0),      # 加立即数
    "andi": ("I", OP_IMM,    0b111, 0),
    "ori":  ("I", OP_IMM,    0b110, 0),
    "xori": ("I", OP_IMM,    0b100, 0),
    "slli": ("I", OP_IMM,    0b001, 0x00),   # 移位 I 型的"funct7"藏在 imm[11:5]
    "srli": ("I", OP_IMM,    0b101, 0x00),
    "srai": ("I", OP_IMM,    0b101, 0x20),
    "lui":  ("U", OP_LUI,    0, 0),          # U/J 型无 funct3/funct7
    "auipc":("U", OP_AUIPC,  0, 0),
    "jal":  ("J", OP_JAL,    0, 0),
    "jalr": ("I", OP_JALR,   0b000, 0),
    "beq":  ("B", OP_BRANCH, 0b000, 0),      # 相等
    "bne":  ("B", OP_BRANCH, 0b001, 0),      # 不等
    "blt":  ("B", OP_BRANCH, 0b100, 0),      # 有符号小于
    "bge":  ("B", OP_BRANCH, 0b101, 0),      # 有符号大于等于
    "lw":   ("I", OP_LOAD,   0b010, 0),      # 读 32-bit 字
    "sw":   ("S", OP_STORE,  0b010, 0),      # 写 32-bit 字
    # --- v2 第一批：访存补全（字节/半字） ---
    "lb":   ("I", OP_LOAD,   0b000, 0),      # 读字节，符号扩展
    "lbu":  ("I", OP_LOAD,   0b100, 0),      # 读字节，零扩展
    "lh":   ("I", OP_LOAD,   0b001, 0),      # 读半字，符号扩展（2 对齐）
    "lhu":  ("I", OP_LOAD,   0b101, 0),      # 读半字，零扩展（2 对齐）
    "sb":   ("S", OP_STORE,  0b000, 0),      # 写字节
    "sh":   ("S", OP_STORE,  0b001, 0),      # 写半字（2 对齐）
    # --- v2 第一批：比较置位 ---
    "slt":  ("R", OP_OP,     0b010, 0x00),   # 有符号小于置位
    "sltu": ("R", OP_OP,     0b011, 0x00),   # 无符号小于置位
    "slti": ("I", OP_IMM,    0b010, 0),      # 小于立即数置位（有符号）
    "sltiu":("I", OP_IMM,    0b011, 0),      # 无符号小于立即数置位
    # --- v2 第一批：无符号分支 ---
    "bltu": ("B", OP_BRANCH, 0b110, 0),      # 无符号小于
    "bgeu": ("B", OP_BRANCH, 0b111, 0),      # 无符号大于等于
    # --- v2 第一批：环境调用（占位：trap） ---
    "ecall":("I", OP_SYSTEM, 0b000, 0),       # imm12=0；f7 列复用为 imm12
    "ebreak":("I", OP_SYSTEM, 0b000, 1),     # imm12=1
    # --- v2 第二批：fence 占位（单核顺序模型下是 no-op） ---
    "fence":  ("I", OP_MISC_MEM, 0b000, 0),  # 内存屏障（占位）
    "fence.i":("I", OP_MISC_MEM, 0b001, 0),  # 指令屏障（占位）
    # --- v2 第二批：M 扩展（f7=0x01） ---
    "mul":    ("R", OP_OP, 0b000, 0x01),     # 乘，低 32 位
    "mulh":   ("R", OP_OP, 0b001, 0x01),     # 乘高（有符号×有符号）
    "mulhsu": ("R", OP_OP, 0b010, 0x01),     # 乘高（有符号×无符号）
    "mulhu":  ("R", OP_OP, 0b011, 0x01),     # 乘高（无符号×无符号）
    "div":    ("R", OP_OP, 0b100, 0x01),     # 除（有符号，截断向零）
    "divu":   ("R", OP_OP, 0b101, 0x01),     # 除（无符号）
    "rem":    ("R", OP_OP, 0b110, 0x01),     # 余（有符号）
    "remu":   ("R", OP_OP, 0b111, 0x01),     # 余（无符号）
    # --- v2 第三批：Zba（地址计算加速） ---
    "sh1add": ("R", OP_OP, 0b100, 0x04),     # rd = (rs1 << 1) + rs2
    "sh2add": ("R", OP_OP, 0b100, 0x14),     # rd = (rs1 << 2) + rs2
    "sh3add": ("R", OP_OP, 0b100, 0x34),     # rd = (rs1 << 3) + rs2
}

_BASE_OFF = {"lw", "lb", "lbu", "lh", "lhu", "jalr"}   # 基址寻址写法的 I 型

DEC_TABLE: dict[tuple[int, int, int], str] = {}
for _n, (fmt, op, f3, f7) in TABLE.items():
    # shift-immediates (slli/srli/srai) are told apart by funct7 like R-type
    if fmt == "R" or _n in ("slli", "srli", "srai"):
        DEC_TABLE[(op, f3, f7)] = _n
    elif _n in ("ecall", "ebreak"):
        # SYSTEM 型没有 funct7，imm12（此处存 f7 列）区分两者
        DEC_TABLE[(op, f3, f7)] = _n
    else:
        DEC_TABLE.setdefault((op, f3, 0), _n)


def sx(v: int, bits: int) -> int:
    """符号扩展：把 bits 宽的 v 解释为二进制补码，返回 Python 有符号整数。"""
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


def _check_range(v: int, bits: int, what: str) -> int:
    """立即数范围校验：越界必须报错，不能静默截断（`addi x1, x0, 2048` 会悄悄变成 -2048）。"""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if not lo <= v <= hi:
        raise ValueError(f"{what} 越界：{v} 不在 [{lo}, {hi}]")
    return v


def _check_even(v: int, what: str) -> int:
    """B/J 型偏移以 2 字节为粒度；本 ISA 指令定长 4 字节，所以偏移必须能被 2 整除。"""
    if v & 1:
        raise ValueError(f"{what} 偏移必须是偶数：{v}")
    return v


_FENCE_BITS = (("i", 8), ("o", 4), ("r", 2), ("w", 1))


def _fence_text(imm: int) -> str:
    """fence 的 imm 布局是 imm[11:8]=fm / [7:4]=pred / [3:0]=succ，
    反汇编成 `iorw,iorw` 形式（可无损回喂；fm 非 0 时退回原始十六进制）。"""
    if (imm >> 8) & 0xF:               # fm 非 0（TSO 等预留模式），字母形式表达不了
        return f"0x{imm:03x}"
    if imm == 0:                       # 空屏障：反转回裸 `fence`
        return ""
    def letters(v: int) -> str:
        return "".join(ch for ch, bit in _FENCE_BITS if v & bit) or "0"
    return f"{letters((imm >> 4) & 0xF)},{letters(imm & 0xF)}"


def _fence_set(tok: str) -> int:
    """`iorw` 这类字母集 -> 4 bit；`0` / 空表示空集。"""
    if tok in ("", "0"):
        return 0
    v = 0
    for ch in tok:
        if ch not in "iorw":
            raise ValueError(f"非法 fence 集合 {tok!r}（只允许 i/o/r/w）")
        v |= {"i": 8, "o": 4, "r": 2, "w": 1}[ch]
    return v


def _encode_fence(rest: str) -> int:
    """四种写法都接受：`fence` / `fence iorw,iorw` / `fence 0,0` / `fence 0x0ff`。
    字母形式只表达 fm=0 的情形（pred/succ），fm 靠 `0x` 十六进制写法保留。"""
    _, op, f3, _ = TABLE["fence"]
    rest = rest.strip()
    if not rest:
        imm = 0
    elif rest.lower().startswith("0x"):
        imm = int(rest, 16)
    else:
        parts = [p.strip() for p in rest.split(",")]
        if len(parts) != 2:
            raise ValueError(f"fence 语法错误 {rest!r}（要 `fence iorw,iorw`）")
        imm = (_fence_set(parts[0]) << 4) | _fence_set(parts[1])
    if not 0 <= imm <= 0xFFF:
        raise ValueError(f"fence imm 越界：{imm:#x}")
    return (imm << 20) | (f3 << 12) | op


def _reg(tok: str) -> int:
    """寄存器名解析：'x0'..'x31'，'zero' 是 x0 的别名。"""
    tok = tok.strip()
    if tok == "x0" or tok == "zero":
        return 0
    m = re.fullmatch(r"x(\d+)", tok)
    if not m or not 0 <= int(m.group(1)) <= 31:
        raise ValueError(f"bad register {tok!r}")
    return int(m.group(1))


# ---------------- 编码器：字段 -> 32-bit 指令字 ----------------

def encode_r(name: str, rd: int, rs1: int, rs2: int) -> int:
    """R 型直拼：|f7|rs2|rs1|f3|rd|op|"""
    _, op, f3, f7 = TABLE[name]
    return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op


def encode_i(name: str, rd: int, rs1: int, imm: int) -> int:
    """I 型：|imm12|rs1|f3|rd|op|。
    移位立即数特殊：imm 只有 5 位 shamt（0..31），高位由 funct7 充当（srai=0x20）。"""
    _, op, f3, _ = TABLE[name]
    if name in ("slli", "srli", "srai"):
        if not 0 <= imm <= 31:
            raise ValueError(f"shamt 越界：{imm} 不在 [0, 31]")
        f7 = TABLE[name][3]
        return (f7 << 25) | (imm << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op
    _check_range(imm, 12, "I 型立即数")
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op


def encode_s(name: str, rs1: int, rs2: int, imm: int) -> int:
    """S 型：12 位立即数劈成高 7 位（[31:25]）和低 5 位（[11:7]）两段。"""
    _, op, f3, _ = TABLE[name]
    _check_range(imm, 12, "S 型立即数")
    imm &= 0xFFF
    return ((imm >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | ((imm & 0x1F) << 7) | op


def encode_b(name: str, rs1: int, rs2: int, imm: int) -> int:
    """B 型：字节偏移打散存放 —— imm[12]->31, imm[10:5]->30:25, imm[4:1]->11:8, imm[11]->7。
    imm 为相对本条指令的字节偏移，最低位恒 0（指令 4 字节对齐）。"""
    _, op, f3, _ = TABLE[name]
    _check_even(imm, "B 型")
    _check_range(imm, 13, "B 型")
    imm &= 0x1FFF
    hi = ((imm >> 12) & 1) << 31 | ((imm >> 5) & 0x3F) << 25
    lo = ((imm >> 1) & 0xF) << 8 | ((imm >> 11) & 1) << 7
    return hi | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | lo | op


def encode_u(name: str, rd: int, imm: int) -> int:
    """U 型：20 位立即数直接占据高 20 位，低 12 位是 rd+opcode。
    无符号（0xFFFFF）与有符号（-1）两种写法都接受。"""
    op = TABLE[name][1]
    if not -(1 << 19) <= imm <= (1 << 20) - 1:
        raise ValueError(f"U 型立即数越界：{imm}（合法范围 -524288..1048575）")
    return (imm & 0xFFFFF) << 12 | (rd << 7) | op


def encode_j(name: str, rd: int, imm: int) -> int:
    """J 型：字节偏移打散 —— imm[20]->31, imm[10:1]->30:21, imm[11]->20, imm[19:12]->19:12。"""
    op = TABLE[name][1]
    _check_even(imm, "J 型")
    _check_range(imm, 21, "J 型")
    imm &= 0x1FFFFF
    w = ((imm >> 20) & 1) << 31 | ((imm >> 1) & 0x3FF) << 21 \
        | ((imm >> 11) & 1) << 20 | ((imm >> 12) & 0xFF) << 12
    return w | (rd << 7) | op


@dataclass(frozen=True)
class Instr:
    """
    译码结果
    """
    name: str
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0          # already sign-extended for I/B; raw for U/J before shift
    raw: int = 0

    def __str__(self) -> str:
        """反汇编单行文本（可直接回喂给 assemble，做字节级往返）。"""
        name, fmt = self.name, TABLE[self.name][0]
        if name in ("ecall", "ebreak", "fence.i"):
            return name                                  # 无操作数
        if name == "fence":
            operands = _fence_text(self.imm)
            return f"fence {operands}" if operands else "fence"
        if name in ("lui", "auipc"):
            return f"{name} x{self.rd}, 0x{self.imm & 0xFFFFF:x}"
        if name == "jal":
            return f"{name} x{self.rd}, {self.imm}"
        if name in _BASE_OFF:                            # load 与 jalr：rd, imm(rs1)
            return f"{name} x{self.rd}, {self.imm}(x{self.rs1})"
        if fmt == "S":                                  # 存储：rs2, imm(rs1)
            return f"{name} x{self.rs2}, {self.imm}(x{self.rs1})"
        if fmt == "B":                                  # 分支：rs1, rs2, off
            return f"{name} x{self.rs1}, x{self.rs2}, {self.imm}"
        if fmt == "R":
            return f"{name} x{self.rd}, x{self.rs1}, x{self.rs2}"
        return f"{name} x{self.rd}, x{self.rs1}, {self.imm}"


def decode(word: int) -> Instr:
    """解码 提取指令字"""
    word &= WORD_MASK
    op = word & 0x7F                       # [6:0]  主操作码
    f3 = (word >> 12) & 0x7                # [14:12] funct3
    f7 = (word >> 25) & 0x7F               # [31:25] funct7（移位 I 型里是 imm 高 7 位）
    rd = (word >> 7) & 0x1F                # [11:7]  目的寄存器
    rs1 = (word >> 15) & 0x1F              # [19:15] 源寄存器 1
    rs2 = (word >> 20) & 0x1F              # [24:20] 源寄存器 2
    # 查表键三分支：U/J 无 funct3 字段；R 与移位 I 需 funct7；其余 (op, f3) 定唯一
    imm12_sys = (word >> 20) & 0xFFF
    if op in (OP_LUI, OP_AUIPC, OP_JAL):   # U/J have no funct3 field
        key = (op, 0, 0)
    elif op == OP_OP or (op == OP_IMM and f3 in (0b001, 0b101)):
        key = (op, f3, f7)
    elif op == OP_SYSTEM:                  # ecall/ebreak 由 imm12 区分
        key = (op, f3, imm12_sys)
    else:
        key = (op, f3, 0)
    name = DEC_TABLE.get(key)
    if name is None:
        raise ValueError(f"unknown instruction word 0x{word:08x}")
    fmt = TABLE[name][0]
    # 保留字段必须为 0，否则反汇编无法无损回喂（这些字本身就不是合法指令）
    if name in ("ecall", "ebreak") and (rd or rs1):
        raise ValueError(f"illegal {name}: rd/rs1 必须为 0（word 0x{word:08x}）")
    if name == "fence" and (rd or rs1):
        raise ValueError(f"illegal fence: rd/rs1 必须为 0（word 0x{word:08x}）")
    if name == "fence.i" and (rd or rs1 or imm12_sys):
        raise ValueError(f"illegal fence.i: imm/rd/rs1 必须为 0（word 0x{word:08x}）")
    if fmt == "R":
        return Instr(name, rd, rs1, rs2, 0, word)
    if fmt == "I":
        if name == "fence":
            # imm[11:8]=fm / [7:4]=pred / [3:0]=succ，必须原样保留（不能符号扩展）
            return Instr(name, rd, rs1, 0, imm12_sys, word)
        imm = (word >> 20) & 0xFFF
        if name in ("slli", "srli", "srai"):
            return Instr(name, rd, rs1, 0, imm & 31, word)  # 只留 shamt，丢掉 funct7 位
        return Instr(name, rd, rs1, 0, sx(imm, 12), word)  # 12 位符号扩展
    if fmt == "S":
        imm = ((word >> 25) << 5) | ((word >> 7) & 0x1F)   # 两段拼回
        return Instr(name, 0, rs1, rs2, sx(imm, 12), word)
    if fmt == "B":
        # 按 B 型位序还原 13 位字节偏移（最低位恒 0）
        v = (word >> 31) << 12 | ((word >> 25) & 0x3F) << 5 | ((word >> 8) & 0xF) << 1 | ((word >> 7) & 1) << 11
        return Instr(name, 0, rs1, rs2, sx(v, 13), word)
    if fmt == "U":
        return Instr(name, rd, 0, 0, (word >> 12) & 0xFFFFF, word)
    if fmt == "J":
        # 按 J 型位序还原 21 位字节偏移（最低位恒 0）
        v = ((word >> 31) << 20 | ((word >> 21) & 0x3FF) << 1
             | ((word >> 20) & 1) << 11 | ((word >> 12) & 0xFF) << 12)
        return Instr(name, rd, 0, 0, sx(v, 21), word)
    raise AssertionError(fmt)


def words_to_bytes(words: list[int]) -> bytes:
    """32-bit 字列表 -> 小端字节流（内存/磁盘的存储形态）。"""
    out = bytearray()
    for w in words:
        out += int(w & WORD_MASK).to_bytes(4, "little")
    return bytes(out)


# ---------------- 汇编器：文本 -> 字节流（两遍） ----------------

def assemble(source: str) -> bytes:
    """两遍汇编。
    第一遍：扫描建 label 地址表（指令定长 4 字节，地址 = 序号 * 4）；
    第二遍：生成指令字，B/J 的 label 回填为 目标地址 - 当前地址 的字节偏移。
    行注释支持 '#' 与 '//'。寄存器支持 xN 与 zero 别名。"""
    items: list[tuple] = []      # ("label", name) | ("word", w) | ("ref", name, fixup)
    for lineno, raw in enumerate(source.splitlines(), 1):
        line = raw.split("#", 1)[0].split("//", 1)[0].strip()   # 去注释
        if not line:
            continue
        if line.endswith(":"):                                  # label 行
            items.append(("label", line[:-1].strip()))
            continue
        # 助记符允许带点（fence.i）；\w+ 会把 "fence.i" 切成 "fence" + 剩余 ".i"
        m = re.fullmatch(r"([\w.]+)\s*(.*)", line)
        if not m:
            raise ValueError(f"line {lineno}: cannot parse {line!r}")
        name, rest = m.group(1), m.group(2).strip()
        if name not in TABLE:
            raise ValueError(f"line {lineno}: unknown mnemonic {name!r}")
        fmt = TABLE[name][0]
        parts = [p.strip() for p in rest.split(",") if p.strip()]

        def emit(w: int) -> None:
            items.append(("word", w & WORD_MASK))

        if name == "fence":
            emit(_encode_fence(rest))
        elif name in ("ecall", "ebreak", "fence.i"):
            # imm12 | funct3 | opcode：f3 必须带上，否则 fence.i（f3=1）会退化成 fence
            emit(int(TABLE[name][3]) << 20 | TABLE[name][2] << 12 | TABLE[name][1])
        elif fmt == "R":
            rd, rs1, rs2 = (_reg(p) for p in parts)
            emit(encode_r(name, rd, rs1, rs2))
        elif fmt == "I" and name not in _BASE_OFF:
            rd, rs1, imm = parts
            emit(encode_i(name, _reg(rd), _reg(rs1), int(imm, 0)))
        elif fmt == "I":  # 基址寻址 rd, imm(rs1)：所有 load + jalr
            mm = re.fullmatch(r"(x\d+|zero),\s*(-?\w+)\((x\d+|zero)\)", rest)
            if not mm:
                raise ValueError(f"line {lineno}: bad {name} syntax {rest!r}")
            rd, imm, rs1 = mm.groups()
            emit(encode_i(name, _reg(rd), _reg(rs1), int(imm, 0)))
        elif fmt == "S":
            mm = re.fullmatch(r"(x\d+|zero),\s*(-?\w+)\((x\d+|zero)\)", rest)
            if not mm:
                raise ValueError(f"line {lineno}: bad sw syntax {rest!r}")
            rs2, imm, rs1 = mm.groups()
            emit(encode_s(name, _reg(rs1), _reg(rs2), int(imm, 0)))
        elif fmt == "B":
            rs1, rs2, label = parts
            items.append(("bref", name, _reg(rs1), _reg(rs2), label))   # 偏移第二遍回填
        elif fmt == "U":
            rd, imm = parts
            emit(encode_u(name, _reg(rd), int(imm, 0)))
        elif fmt == "J":
            rd, label = parts
            items.append(("jref", name, _reg(rd), label))

    # ---- 第一遍：label -> 地址 ----
    labels: dict[str, int] = {}
    addr = 0
    for it in items:
        if it[0] == "label":
            labels[it[1]] = addr
        else:
            addr += 4
    # ---- 第二遍：生成与回填 ----
    out: list[int] = []
    for it in items:
        if it[0] == "label":
            continue
        if it[0] == "word":
            out.append(it[1])
        elif it[0] == "bref":
            _, name, rs1, rs2, label = it
            if label in labels:
                off = labels[label] - (len(out) * 4)     # 相对本条指令的字节偏移
            else:
                off = int(label, 0)   # numeric offset (disassembly round-trip)
            out.append(encode_b(name, rs1, rs2, off))
        elif it[0] == "jref":
            _, name, rd, label = it
            if label in labels:
                off = labels[label] - (len(out) * 4)
            else:
                off = int(label, 0)
            out.append(encode_j(name, rd, off))
    return words_to_bytes(out)


def disassemble(data: bytes, base: int = 0) -> list[str]:
    """字节流 -> 反汇编行列表，格式 '地址:  机器字   汇编文本'。"""
    lines = []
    for i in range(0, len(data) - len(data) % 4, 4):
        w = int.from_bytes(data[i:i + 4], "little")
        lines.append(f"{base + i:08x}:  {w:08x}   {decode(w)}")
    return lines


# ---------------- 测试 ----------------

def main() -> None:
    # 1) 每条指令编码后译码，名字必须还原
    cases = [
        ("add", encode_r("add", 1, 2, 3)),
        ("sub", encode_r("sub", 31, 30, 29)),
        ("sll", encode_r("sll", 5, 6, 7)),
        ("srl", encode_r("srl", 5, 6, 7)),
        ("sra", encode_r("sra", 5, 6, 7)),
        ("and", encode_r("and", 1, 1, 1)),
        ("or",  encode_r("or", 1, 2, 3)),
        ("xor", encode_r("xor", 1, 2, 3)),
        ("addi", encode_i("addi", 1, 0, -5)),
        ("andi", encode_i("andi", 1, 2, 0x7FF)),
        ("ori",  encode_i("ori", 1, 2, -1)),
        ("xori", encode_i("xori", 1, 2, 0x55)),
        ("slli", encode_i("slli", 1, 2, 31)),
        ("srli", encode_i("srli", 1, 2, 31)),
        ("srai", encode_i("srai", 1, 2, 31)),
        ("lui",  encode_u("lui", 1, 0xABCDE)),
        ("auipc", encode_u("auipc", 2, 0x12345)),
        ("jal",  encode_j("jal", 1, 2048)),
        ("jalr", encode_i("jalr", 3, 4, 16)),
        ("beq",  encode_b("beq", 1, 2, 4094)),   # B 型最大值（13 位有符号且最低位恒 0）
        ("bne",  encode_b("bne", 1, 2, -4)),
        ("blt",  encode_b("blt", 5, 6, 128)),
        ("bge",  encode_b("bge", 5, 6, -128)),
        ("lw",   encode_i("lw", 1, 2, -8)),
        ("sw",   encode_s("sw", 2, 1, 12)),
        ("lb",   encode_i("lb", 1, 2, -1)),
        ("lbu",  encode_i("lbu", 1, 2, 255)),
        ("lh",   encode_i("lh", 1, 2, -2)),
        ("lhu",  encode_i("lhu", 1, 2, 2)),
        ("sb",   encode_s("sb", 2, 1, 3)),
        ("sh",   encode_s("sh", 2, 1, 6)),
        ("slt",  encode_r("slt", 1, 2, 3)),
        ("sltu", encode_r("sltu", 1, 2, 3)),
        ("slti", encode_i("slti", 1, 2, -3)),
        ("sltiu",encode_i("sltiu", 1, 2, 7)),
        ("bltu", encode_b("bltu", 1, 2, 64)),
        ("bgeu", encode_b("bgeu", 1, 2, -64)),
        ("ecall", 0x00000073),
        ("ebreak", 0x00100073),
        ("fence", 0x0000000F),
        ("fence.i", 0x0000100F),
        ("mul",   encode_r("mul", 1, 2, 3)),
        ("mulh",  encode_r("mulh", 1, 2, 3)),
        ("mulhsu",encode_r("mulhsu", 1, 2, 3)),
        ("mulhu", encode_r("mulhu", 1, 2, 3)),
        ("div",   encode_r("div", 1, 2, 3)),
        ("divu",  encode_r("divu", 1, 2, 3)),
        ("rem",   encode_r("rem", 1, 2, 3)),
        ("remu",  encode_r("remu", 1, 2, 3)),
        ("sh1add", encode_r("sh1add", 1, 2, 3)),
        ("sh2add", encode_r("sh2add", 1, 2, 3)),
        ("sh3add", encode_r("sh3add", 1, 2, 3)),
    ]
    for name, w in cases:
        ins = decode(w)
        assert ins.name == name, (name, ins)

    # 2) 立即数符号扩展抽查（负数必须原样回来）
    assert decode(encode_i("addi", 1, 0, -5)).imm == -5
    assert decode(encode_s("sw", 2, 1, -4)).imm == -4
    assert decode(encode_b("bne", 1, 2, -4)).imm == -4  # B-imm 最低位恒 0，-4 是合法偏移
    assert decode(encode_i("lw", 1, 2, -8)).imm == -8

    # 3) 汇编 -> 反汇编 -> 再汇编，字节级往返必须相等
    src = """
        addi x1, x0, 5        # x1 = 5
        addi x2, x0, -1
        add  x3, x1, x2
        loop:
        addi x1, x1, -1
        bne  x1, x0, loop
        jal  x5, done
        done:
        sw   x3, 8(x0)
        lw   x4, 8(x0)
        jalr x0, 0(x5)
    """
    code = assemble(src)
    lines = disassemble(code)
    for l in lines:
        print(l)
    assert len(code) % 4 == 0
    src2 = "\n".join(l.split(None, 2)[2] for l in lines)   # 取汇编文本列
    assert assemble(src2) == code, "assemble/disassemble round-trip mismatch"
    print(f"isa OK: {len(cases)} encodings, assembler round-trip, "
          f"{len(code)} bytes")


if __name__ == "__main__":
    main()