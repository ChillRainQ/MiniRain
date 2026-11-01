"""
core_test.py: core.py 组件测试（寄存器堆 / State / factory 接口闭环）。

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
    st.regs[3] = cfg.ram_end - 4              # +8 越过 RAM 顶端 -> trap
    try:
        st.lw(3, 8)
        raise AssertionError("expected range trap")
    except RV32Trap:
        pass
    st.reset()
    # pc 回落到 ROM 起点（不是 0：0x0 在 RAM 里，复位到那儿会取到全零字 -> 非法指令）
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

    print("test_core OK: register file / state / factory interface closed")


if __name__ == "__main__":
    main()