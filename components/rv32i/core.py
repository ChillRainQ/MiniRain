"""
core.py: machine components for ISA-RV32.

段布局与字宽的真正来源是 `constant.py`；本文件的字段默认值只是把它接进来，所以
`RV32Config()` 与 `constant.RAM_SIZE` 不可能对不上。单个实例仍可覆盖（测试要改形状时）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_DIRECT = __package__ in (None, "")
if _DIRECT:                       # `python core.py` 直跑时补包路径（与 interpreter.py 同款）
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "components.rv32i"

from .constant import (CAP_MAX, DISK_SECTORS, DROM_BASE, DROM_SIZE, HEAP_BASE, RAM_SIZE,
                       ROM_BASE, ROM_SIZE, SECTOR_SIZE, STACK_TOP, TEXT_BASE, TEXT_ROWS,
                       TEXT_ROW, WORD_BITS, WORD_MASK)


class RV32Trap(Exception):
    pass

@dataclass(frozen=True)
class RV32Config:
    """
    RV32 machine config.
    """
    word_bits: int = WORD_BITS
    n_regs: int = 32               # 寄存器个数（x0 恒 0，不占可写口）

    ram_size: int = RAM_SIZE       # 32 KB RAM @ 0x00000：数据段 / 堆 / 栈
    drom_base: int = DROM_BASE     # 128 KB 只读数据（.rodata）的起点
    drom_size: int = DROM_SIZE
    rom_base: int = ROM_BASE       # 256 KB ROM：程序段，取指专用，无写口
    rom_size: int = ROM_SIZE

    #: 取指是否允许落在 RAM。默认否 = 哈佛：电路的 `fetch()` 就是 `rom()`，RAM 表没有
    #: 取指通路。开着只是让参考机能跑冯诺依曼变体（自改代码 / `.data` 里的跳板）。
    ram_executable: bool = False

    heap_base: int = HEAP_BASE     # 堆从 0x100 起（低端保留：中断向量等扩展位）
    stack_top: int = STACK_TOP     # 栈顶（RAM 顶端，向下增长）

    # ---- `<cpu_call>` 文本缓冲（M5 前置，电路侧的内存，不在 `Memory` 里）----
    # Qwen 的文本 token 落在 `text_base` 起、每行 `text_row` 字节。地址上是 RAM 上界之上的
    # 一块空洞：`text_base` 恰好等于 `stack_top`，但栈是向下长的，不撞车；再往上是
    # `drom_base = 0x20000`，中间还空 105 KB。
    # `cap_max` 是“同时捕获的 token 数”，行号 **1-based**（第 k 个 token 写第 k 行，第 0 行
    # 空着，照 Qwenputer 的约定）；`text_rows = 160` 给 128 个 token 留 32 行余量，因为 7 位
    # 行号只到 127。账在 plan.md §9「载荷上限 80 -> 128」。
    text_base: int = TEXT_BASE
    text_rows: int = TEXT_ROWS
    text_row: int = TEXT_ROW       # 每行字节数 = 单个 token 的最坏 UTF-8 长度
    cap_max: int = CAP_MAX

    sector_size: int = SECTOR_SIZE  # 磁盘扇区字节数
    disk_sectors: int = DISK_SECTORS  # 2 MB 磁盘

    @property
    def word_mask(self) -> int:
        """字掩码：由 word_bits 推导，组件统一用它，不再硬编码 0xFFFFFFFF。"""
        return (1 << self.word_bits) - 1

    @property
    def ram_end(self) -> int:
        return self.ram_size                     # 0x08000

    @property
    def text_size(self) -> int:
        return self.text_rows * self.text_row    # 160 × 16 = 2560 B

    @property
    def text_end(self) -> int:
        return self.text_base + self.text_size   # 0x08A00，仍在 drom_base = 0x20000 之下

    def __post_init__(self) -> None:
        # 第 cap_max 个 token 要写第 cap_max 行，行号从 1 起，所以行数必须严格大于 cap_max
        if self.cap_max >= self.text_rows:
            raise ValueError(f"cap_max={self.cap_max} 装不进 {self.text_rows} 行的文本缓冲")
        if self.text_base < self.ram_end or self.text_end > self.drom_base:
            raise ValueError(f"文本缓冲 {self.text_base:#x}..{self.text_end:#x} 压到了别的段")

    @property
    def drom_end(self) -> int:
        return self.drom_base + self.drom_size   # 0x40000，正好接上 ROM

    @property
    def rom_end(self) -> int:
        return self.rom_base + self.rom_size     # 0x80000

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
    RAM + ROM + 文本缓冲
    """
    def __init__(self, cfg: RV32Config) -> None:
        self.cfg = cfg
        self._ram = bytearray(cfg.ram_size)
        self._drom = bytearray(cfg.drom_size)
        self._rom = bytearray(cfg.rom_size)
        self._text = bytearray(cfg.text_size)

    # ---- 段路由 ----
    def _route(self, addr: int, n: int) -> tuple[bytearray, int]:
        cfg = self.cfg
        if 0 <= addr and addr + n <= cfg.ram_end:
            return self._ram, addr
        if cfg.text_base <= addr and addr + n <= cfg.text_end:
            return self._text, addr - cfg.text_base
        if cfg.drom_base <= addr and addr + n <= cfg.drom_end:
            return self._drom, addr - cfg.drom_base
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
        if seg is not self._ram:
            raise RV32Trap(f"write to read-only memory at {addr:#x}")
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
        if seg is not self._ram:
            raise RV32Trap(f"write to read-only memory at {addr:#x}")
        seg[off:off + 2] = int(val & 0xFFFF).to_bytes(2, "little")

    # ---- 取指（容忍非对齐；只有 ROM 可取指，RAM 受 `ram_executable` 控制） ----
    def fetch_word(self, addr: int) -> int:
        seg, off = self._route(addr, 4)
        if seg is self._text:
            raise RV32Trap(f"fetch from text buffer at {addr:#x}")   # 文本是数据，不可取指
        if seg is self._rom or (seg is self._ram and self.cfg.ram_executable):
            return int.from_bytes(seg[off:off + 4], "little")
        raise RV32Trap(f"fetch outside ROM at {addr:#x}")

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
        if seg is not self._ram:
            raise RV32Trap(f"write to read-only memory at {addr:#x}")
        seg[off:off + 4] = int(val & self.cfg.word_mask).to_bytes(4, "little")

    # ---- 裸字节窗口 ----
    def read_bytes(self, addr: int, n: int) -> bytes:
        seg, off = self._route(addr, n)
        return bytes(seg[off:off + n])

    def write_bytes(self, addr: int, data: bytes) -> None:
        seg, off = self._route(addr, len(data))
        if seg is not self._ram:
            raise RV32Trap(f"write to read-only memory at {addr:#x}")
        seg[off:off + len(data)] = data

    def reset(self) -> None:
        for i in range(len(self._ram)):
            self._ram[i] = 0
        self._text[:] = bytes(len(self._text))    # 文本缓冲每轮由 <cpu_call> 清空
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
        本机只有 ROM 可取指（`ram_executable = False`），复位到那里会抛
        `fetch outside ROM`。ELF 的 `e_entry` 与 rom_base 不相等时，由
        `Interpreter.load()` / `load_elf()` 负责把 pc 改写到真正入口。
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
