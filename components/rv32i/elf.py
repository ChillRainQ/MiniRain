"""
最小 ELF32 解析与装载
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path

# 包导入与脚本直接运行两种方式都支持（同 core/isa/factory 的写法）
_DIRECT = __package__ in (None, "")
if _DIRECT:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "components.rv32i"

from .constant import ROM_BASE
from .core import RV32Trap

EM_RISCV = 243
PT_LOAD = 1


@dataclass(frozen=True)
class Segment:
    vaddr: int
    paddr: int
    offset: int
    filesz: int
    memsz: int
    flags: int


@dataclass(frozen=True)
class ElfFile:
    entry: int
    segments: tuple[Segment, ...]
    machine: int


def parse_elf(data: bytes) -> ElfFile:
    """解析 ELF32 头部 + 程序头表（RV32 小端）。"""
    if len(data) < 52:
        raise RV32Trap("not an ELF (too short)")
    if data[:4] != b"\x7fELF":
        raise RV32Trap("not an ELF (bad magic)")
    if data[4] != 1:                       # EI_CLASS：1 = 32 位
        raise RV32Trap(f"not a 32-bit ELF (EI_CLASS={data[4]})")
    if data[5] != 1:                       # EI_DATA：1 = 小端
        raise RV32Trap(f"not a little-endian ELF (EI_DATA={data[5]})")
    (e_type, e_machine, _ver, e_entry, e_phoff, _shoff, _flags,
     _ehsize, e_phentsize, e_phnum, *_rest) = struct.unpack_from("<16sHHIIIIIHHHHHH", data, 0)[1:]
    if e_machine != EM_RISCV:
        raise RV32Trap(f"not a RISC-V ELF (e_machine={e_machine})")
    if e_phentsize < 32:
        raise RV32Trap(f"bad e_phentsize={e_phentsize} (need >= 32)")
    if e_phoff + e_phnum * e_phentsize > len(data):
        raise RV32Trap(f"program header table out of file: "
                       f"{e_phoff}+{e_phnum}*{e_phentsize} > {len(data)}")
    segments = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        (p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_flags, _align) = \
            struct.unpack_from("<IIIIIIII", data, off)
        if p_type == PT_LOAD:
            if p_offset + p_filesz > len(data):
                raise RV32Trap(f"segment {i} 文件范围越界：{p_offset}+{p_filesz} > {len(data)}")
            segments.append(Segment(p_vaddr, p_paddr, p_offset, p_filesz, p_memsz, p_flags))
    if not segments:
        raise RV32Trap("no PT_LOAD segments")
    return ElfFile(e_entry, tuple(segments), e_machine)


def load_elf(mem, elf: ElfFile, data: bytes) -> int:
    """把各 PT_LOAD 段的文件内容写入 Memory，返回入口地址。"""
    for seg in elf.segments:
        if seg.filesz > seg.memsz:
            raise RV32Trap(f"segment filesz({seg.filesz}) > memsz({seg.memsz}) @ {seg.vaddr:#x}")
        addr = seg.paddr if seg.paddr else seg.vaddr
        if seg.filesz:
            mem.load(addr, data[seg.offset:seg.offset + seg.filesz])
    return elf.entry


# ---------------- 合成分段自测（不依赖外部 gcc） ----------------

def _make_test_elf(base: int = ROM_BASE) -> bytes:
    """构造一个最小合法 ELF32：1 个 PT_LOAD（8 字节 @base），入口同址。"""
    text = b"\x13\x05\x00\x00\x73\x00\x10\x00"   # addi a0,zero,0 ; ebreak（占位字节）
    phoff, phentsize = 52, 32
    e = bytearray(52)
    struct.pack_into("<16sHHIIIIIHHHHHH", e, 0,
                     b"\x7fELF" + bytes([1, 1, 1]) + bytes(9),   # 魔数/32位/小端/版本
                     2, EM_RISCV, 1,              # EXEC, RISC-V, version
                     base,                        # entry
                     phoff, 0, 0, 52, phentsize, 1, 0, 0, 0)
    ph = bytearray(32)
    struct.pack_into("<IIIIIIII", ph, 0,
                     PT_LOAD, phoff + phentsize, base, base,
                     len(text), len(text), 5, 0x1000)
    return bytes(e) + bytes(ph) + text


def main() -> None:
    from .core import Memory, RV32Config
    cfg = RV32Config()
    data = _make_test_elf(cfg.rom_base)
    elf = parse_elf(data)
    assert elf.entry == cfg.rom_base and len(elf.segments) == 1
    mem = Memory(cfg)
    entry = load_elf(mem, elf, data)
    assert entry == cfg.rom_base
    assert mem.fetch_word(cfg.rom_base) == 0x00000513    # addi a0,zero,0
    assert mem.fetch_word(cfg.rom_base + 4) == 0x00100073    # ebreak
    try:
        parse_elf(b"not an elf at all.................")
        raise AssertionError("expected trap for bad magic")
    except Exception:
        pass
    print("elf OK: parse + load synthetic ELF32 (1 PT_LOAD, entry, ebreak bytes)")


if __name__ == "__main__":
    main()