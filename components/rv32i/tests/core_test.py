"""
core_test.py: core.py 组件测试（寄存器堆 / State / factory 接口闭环 / 取指段属性）。

运行：python components/rv32i/tests/core_test.py
"""

import sys
from pathlib import Path

# tests/ 不是包：仓库根入 sys.path，再用包绝对导入（直接运行与 -m 都能跑）
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from components.rv32i.core import RV32Config, RV32Trap, RegisterFile, State
from components.rv32i.factory import factory
from components.rv32i.isa import decode, encode_i


def main() -> None:
    cfg = RV32Config()

    # RegisterFile: x0 / mask / snapshot / reset
    rf = RegisterFile(cfg)
    rf.write(0, 123)
    assert rf[0] == 0
    rf.write(5, (1 << 40) + 0xFF)
    assert rf[5] == 0xFF
    rf[7] = 7
    assert rf.snapshot()[7] == 7
    rf.reset()
    assert rf.snapshot() == [0] * 32

    # State: 胶水接口 / 地址回绕 / reset
    st = State()
    st.mem.load(0x100, (42).to_bytes(4, "little"))
    st.regs[1] = 0x100
    assert st.lw(1, 0) == 42
    st.regs[2] = 0xFFFFFFFF                  # +9 回绕到 0x8（RAM 内）
    assert st.lw(2, 9) == 0
    st.sw(1, 4, 0xDEADBEEF)                   # 写 mem[0x104]
    assert st.mem.read_word(0x104) == 0xDEADBEEF
    # RAM 之上不再是“非法”：紧接着的 0x8000.. 是 `<cpu_call>` 文本缓冲（可读、不可写、拒取指），
    # 真正越界要越过 text_end 才算。
    st.regs[3] = cfg.ram_end - 4              # +8 = 0x8004，落进文本缓冲
    assert st.lw(3, 8) == 0
    st.regs[3] = cfg.text_end - 4             # +8 越过文本缓冲顶端，下面是到 drom_base 的空洞
    try:
        st.lw(3, 8)
        raise AssertionError("expected range trap")
    except RV32Trap:
        pass
    st.reset()
    # pc 回落到 ROM 起点（不是 0：0x0 在 RAM 里，本机只有 ROM 可取指，复位到那儿会抛
    # `fetch outside ROM`）
    assert st.pc == cfg.rom_base and st.steps == 0 and st.regs.snapshot() == [0] * 32
    assert st.mem.read_word(0x104) == 0       # mem 已清

    # factory 通过 State 闭环：两条 addi
    st.mem.load(cfg.rom_base,
                encode_i("addi", 1, 0, 41).to_bytes(4, "little")
                + encode_i("addi", 2, 1, 1).to_bytes(4, "little"))
    for addr in (cfg.rom_base, cfg.rom_base + 4):
        ins = decode(st.mem.fetch_word(addr))
        factory.get_code(ins.name)(st, ins)
    assert st.regs[1] == 41 and st.regs[2] == 42

    # 取指段属性：哈佛——只有 ROM 可取指，RAM 里就算放的是一条合法指令也不许取。
    # `ram_executable` 是给冯诺依曼变体留的显式开关。
    word = encode_i("addi", 1, 0, 41).to_bytes(4, "little")
    cfg_x = RV32Config(ram_executable=True)
    st_x = State(cfg_x)
    st_x.mem.load(cfg_x.heap_base, word)
    assert st_x.mem.fetch_word(cfg_x.heap_base) == int.from_bytes(word, "little")
    st.mem.load(cfg.heap_base, word)
    for addr in (cfg.heap_base, cfg.drom_base):       # RAM / DROM 都不能取指
        try:
            st.mem.fetch_word(addr)
            raise AssertionError(f"expected fetch trap at {addr:#x}")
        except RV32Trap as e:
            assert "fetch outside ROM" in str(e), e

    print("test_core OK: register file / state / factory interface closed / fetch=ROM only")


if __name__ == "__main__":
    main()