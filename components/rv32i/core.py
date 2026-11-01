"""
core.py: machine components for ISA-RV32.
"""

from __future__ import annotations

from dataclasses import dataclass, field

WORD_MASK = 0xFFFFFFFF


class RV32Trap(Exception):
    pass

@dataclass(frozen=True)
class RV32Config:
    """
    RV32 machine config.
    """
    word_bits: int = 32
    n_regs: int = 32               # 寄存器个数（x0 恒 0，不占可写口）

    ram_size: int = 0x08000        # 32 KB RAM @ 0x00000：数据段 / 堆 / 栈
    rom_base: int = 0x10000        # ROM 起点（0x8000–0x10000 为未映射空洞，访问即 trap）
    rom_size: int = 0x20000        # 128 KB ROM：程序段，取指专用，无写口

    heap_base: int = 0x00100       # 堆从 0x100 起（低端保留：中断向量等扩展位）
    stack_top: int = 0x08000       # 栈顶（RAM 顶端，向下增长）

    sector_size: int = 512         # 磁盘扇区字节数
    disk_sectors: int = 4096       # 2 MB 磁盘

    @property
    def word_mask(self) -> int:
        """字掩码：由 word_bits 推导，组件统一用它，不再硬编码 0xFFFFFFFF。"""
        return (1 << self.word_bits) - 1

    @property
    def ram_end(self) -> int:
        return self.ram_size                     # 0x08000

    @property
    def rom_end(self) -> int:
        return self.rom_base + self.rom_size     # 0x30000

    @property
    def disk_size(self) -> int:
        return self.sector_size * self.disk_sectors

    @property
    def rom_words(self) -> int:
        """ROM 指令容量 = rom_size / 4；神经版取指权重表的行数。"""
        return self.rom_size // 4

    @property
    def ram_words(self) -> int:
        """RAM 字数；神经版 MemState 的行数（M ∈ R^{ram_words × word_bits}）。"""
        return self.ram_size // 4


class Memory:
    """
    RAM + ROM
    """
    def __init__(self, cfg: RV32Config) -> None:
        self.cfg = cfg
        self._ram = bytearray(cfg.ram_size)
        self._rom = bytearray(cfg.rom_size)

    # ---- 段路由 ----
    def _route(self, addr: int, n: int) -> tuple[bytearray, int]:
        cfg = self.cfg
        if 0 <= addr and addr + n <= cfg.ram_end:
            return self._ram, addr
        if cfg.rom_base <= addr and addr + n <= cfg.rom_end:
            return self._rom, addr - cfg.rom_base
        raise RV32Trap(f"memory access out of range: {addr:#x}+{n}")

    # ---- 引导特权 ----
    def load(self, addr: int, data: bytes) -> None:
        seg, off = self._route(addr, len(data))
        seg[off:off + len(data)] = data

    # ---- 字节/半字访问（lb/lbu/lh/lhu/sb/sh 用；半字 2 对齐） ----
    def read_byte(self, addr: int) -> int:
        seg, off = self._route(addr, 1)
        return seg[off]

    def write_byte(self, addr: int, val: int) -> None:
        seg, off = self._route(addr, 1)
        if seg is self._rom:
            raise RV32Trap(f"write to ROM at {addr:#x}")
        seg[off] = val & 0xFF

    def read_half(self, addr: int) -> int:
        if addr % 2 != 0:
            raise RV32Trap(f"misaligned half access at {addr:#x}")
        seg, off = self._route(addr, 2)
        return int.from_bytes(seg[off:off + 2], "little")

    def write_half(self, addr: int, val: int) -> None:
        if addr % 2 != 0:
            raise RV32Trap(f"misaligned half access at {addr:#x}")
        seg, off = self._route(addr, 2)
        if seg is self._rom:
            raise RV32Trap(f"write to ROM at {addr:#x}")
        seg[off:off + 2] = int(val & 0xFFFF).to_bytes(2, "little")

    # ---- 取指（容忍非对齐，跨段只读 ROM/RAM） ----
    def fetch_word(self, addr: int) -> int:
        seg, off = self._route(addr, 4)
        return int.from_bytes(seg[off:off + 4], "little")

    # ---- 数据访问（4 对齐） ----
    def read_word(self, addr: int) -> int:
        if addr % 4 != 0:
            raise RV32Trap(f"misaligned word access at {addr:#x}")
        seg, off = self._route(addr, 4)
        return int.from_bytes(seg[off:off + 4], "little")

    def write_word(self, addr: int, val: int) -> None:
        if addr % 4 != 0:
            raise RV32Trap(f"misaligned word access at {addr:#x}")
        seg, off = self._route(addr, 4)
        if seg is self._rom:
            raise RV32Trap(f"write to ROM at {addr:#x}")
        seg[off:off + 4] = int(val & self.cfg.word_mask).to_bytes(4, "little")

    # ---- 裸字节窗口 ----
    def read_bytes(self, addr: int, n: int) -> bytes:
        seg, off = self._route(addr, n)
        return bytes(seg[off:off + n])

    def write_bytes(self, addr: int, data: bytes) -> None:
        seg, off = self._route(addr, len(data))
        if seg is self._rom:
            raise RV32Trap(f"write to ROM at {addr:#x}")
        seg[off:off + len(data)] = data

    def reset(self) -> None:
        for i in range(len(self._ram)):
            self._ram[i] = 0
        # ROM 为程序段，reset 不动（由加载器重新 load）

# ---------------- registers ----------------

class RegisterFile:
    """x0..x31，word_bits 宽；x0 恒 0（write 丢弃，组件层保证）。

    神经映射（ISA.md §6.2）：32 槽的小 MemState——one-hot 读、门控写，
    x0 槽写门控恒 0。容量定死，覆盖写，无淘汰问题。
    """
    def __init__(self, cfg: RV32Config) -> None:
        self.cfg = cfg
        self._r = [0] * cfg.n_regs

    def read(self, idx: int) -> int:
        if idx == 0:
            return 0
        return self._r[idx]

    def write(self, idx: int, val: int) -> None:
        if idx != 0:
            self._r[idx] = val & self.cfg.word_mask

    def __getitem__(self, idx: int) -> int:
        return self.read(idx)

    def __setitem__(self, idx: int, val: int) -> None:
        self.write(idx, val)

    def snapshot(self) -> list[int]:
        return list(self._r)

    def reset(self) -> None:
        for i in range(len(self._r)):
            self._r[i] = 0


# ---------------- aggregate state ----------------

@dataclass
class State:
    """CPU 可见状态全集 + factory 胶水接口。

    factory handler 以本接口为准：.regs / .pc / .write() / .lw() / .sw()。
    pc 不入寄存器堆：每指令必读写、行为特殊（+4 或跳转），单独管理。
    """
    cfg: RV32Config = field(default_factory=RV32Config)
    regs: RegisterFile = field(init=False)
    mem: Memory = field(init=False)
    pc: int = 0
    steps: int = 0

    def __post_init__(self) -> None:
        self.regs = RegisterFile(self.cfg)
        self.mem = Memory(self.cfg)
        self.pc = self.cfg.rom_base if self.pc == 0 else self.pc   # 同 reset()：未装载时不落在 RAM 的 0x0

    def reset(self) -> None:
        """复位整机可见状态：清寄存器堆与 RAM（ROM 为程序段，由加载器重新 load）。

        pc 回落到 `cfg.rom_base`（代码所在段的起点），**不是 0**：0x0 属于 RAM，
        复位到那里会取到全零字 -> `decode` 抛 unknown instruction。ELF 的
        `e_entry` 与 rom_base 不相等时，由 `Interpreter.load()` / `load_elf()`
        负责把 pc 改写到真正入口。
        """
        self.regs.reset()
        self.mem.reset()
        self.pc = self.cfg.rom_base
        self.steps = 0

    def write(self, rd: int, val: int) -> None:
        self.regs.write(rd, val)

    def lw(self, rs1: int, imm: int) -> int:
        """地址算术 mod 2^word_bits，访存阶段由 Memory 做边界检查。"""
        addr = (self.regs[rs1] + imm) & self.cfg.word_mask
        return self.mem.read_word(addr)

    def sw(self, rs1: int, imm: int, val: int) -> None:
        addr = (self.regs[rs1] + imm) & self.cfg.word_mask
        self.mem.write_word(addr, val)

    # 字节/半字胶水（lb/lbu/lh/lhu/sb/sh 的 handler 用）
    def lb(self, rs1: int, imm: int) -> int:
        v = self.mem.read_byte((self.regs[rs1] + imm) & self.cfg.word_mask)
        return v - 256 if v & 0x80 else v          # 符号扩展

    def lbu(self, rs1: int, imm: int) -> int:
        return self.mem.read_byte((self.regs[rs1] + imm) & self.cfg.word_mask)

    def lh(self, rs1: int, imm: int) -> int:
        v = self.mem.read_half((self.regs[rs1] + imm) & self.cfg.word_mask)
        return v - 0x10000 if v & 0x8000 else v  # 符号扩展

    def lhu(self, rs1: int, imm: int) -> int:
        return self.mem.read_half((self.regs[rs1] + imm) & self.cfg.word_mask)

    def sb(self, rs1: int, imm: int, val: int) -> None:
        self.mem.write_byte((self.regs[rs1] + imm) & self.cfg.word_mask, val)

    def sh(self, rs1: int, imm: int, val: int) -> None:
        self.mem.write_half((self.regs[rs1] + imm) & self.cfg.word_mask, val)
