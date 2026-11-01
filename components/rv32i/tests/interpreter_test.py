"""
interpreter_test.py: 参考解释器端到端测试。
程序进 ROM 段（rom_base），数据进 RAM 段；pc 越界停机。

运行：python components/rv32i/tests/interpreter_test.py
"""

import sys
from pathlib import Path

# tests/ 不是包：仓库根入 sys.path，再用包绝对导入（直接运行与 -m 都能跑）
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from components.rv32i.core import RV32Trap
from components.rv32i.factory import RV32Trap_placeholder
from components.rv32i.interpreter import Interpreter, run_asm
from components.rv32i.isa import assemble


def test_fib() -> None:
    it = run_asm("""
        addi x1, x0, 10        # n = 10
        addi x2, x0, 0         # a = 0
        addi x3, x0, 1         # b = 1
    loop:
        beq  x1, x0, done
        add  x4, x2, x3        # t = a + b
        add  x2, x3, x0        # a = b
        add  x3, x4, x0        # b = t
        addi x1, x1, -1
        beq  x0, x0, loop
    done:
    """)
    assert it.state.regs[2] == 55, it.state.regs[2]


def test_gcd() -> None:
    # 减法欧几里得（无 M 扩展）
    it = run_asm("""
        addi x1, x0, 84
        addi x2, x0, 18
    outer:
        beq  x2, x0, done
    inner:
        blt  x1, x2, rot
        sub  x1, x1, x2
        beq  x0, x0, inner
    rot:
        add  x3, x1, x0
        add  x1, x2, x0
        add  x2, x3, x0
        beq  x0, x0, outer
    done:
    """)
    assert it.state.regs[1] == 6, it.state.regs[1]


def test_mem_shifts_logic() -> None:
    it = run_asm("""
        addi x1, x0, 0x100
        addi x2, x0, 42
        sw   x2, 8(x1)          # mem[0x108] = 42
        lw   x3, 8(x1)
        add  x0, x3, x3         # 写 x0 丢弃
        sll  x4, x3, x2         # shamt = 42 & 31 = 10
        srl  x5, x4, x2
        addi x15, x0, -1024
        sra  x6, x15, x2        # -1024 >> 10 = -1
        srl  x16, x15, x2       # 逻辑右移 = 0x3FFFFF
        and  x7, x3, x2
        or   x8, x3, x2
        xor  x9, x3, x2
        lui  x10, 0x12345
    """)
    r = it.state.regs
    assert r[0] == 0 and r[3] == 42
    assert r[4] == 42 << 10 and r[5] == 42
    assert r[6] == 0xFFFFFFFF and r[16] == 0x3FFFFF
    assert r[7] == (42 & 42) and r[8] == (42 | 42) and r[9] == 0
    assert r[10] == 0x12345000
    assert it.state.mem.read_word(0x108) == 42    # RAM 段数据


def test_jal_jalr() -> None:
    it = run_asm("""
        jal  x5, target
        addi x1, x1, 100      # 被跳过
    target:
        addi x1, x1, 1        # x1 = 1
        auipc x12, 0          # x12 = 本条地址
        jalr x13, 8(x12)      # 跳到 x12+8；x13 = 返回地址
        addi x14, x0, 111     # 被跳过？不：jalr 目标就是它
        addi x15, x0, 222
    """)
    r = it.state.regs
    assert r[1] == 1, r[1]
    assert r[13] == r[12] + 8, (r[12], r[13])   # 链接地址
    assert r[14] == 111                          # jalr 目标执行


def test_traps() -> None:
    # 非对齐访问 trap
    try:
        run_asm("""
            addi x1, x0, 0x102
            lw   x2, 0(x1)
        """)
        raise AssertionError("expected misaligned trap")
    except RV32Trap:
        pass
    # 死循环撞 step 上限
    try:
        run_asm("""\n    loop:\n        beq x0, x0, loop\n    """)
        raise AssertionError("expected step limit")
    except RV32Trap as e:
        assert "step limit" in str(e)


def test_v2_batch1() -> None:
    # lb/lbu 符号与零扩展
    it = run_asm("""
        addi x1, x0, 0x100
        addi x2, x0, 0x180
        sb   x2, 0(x1)          # mem[0x100] = 0x80
        lb   x3, 0(x1)          # -128
        lbu  x4, 0(x1)          # 128
        sh   x2, 4(x1)          # mem16[0x104] = 0x0180
        lh   x5, 4(x1)          # 0x0180 = 384
        lhu  x6, 4(x1)
    """)
    r = it.state.regs
    assert r[3] == -128 & 0xFFFFFFFF and r[4] == 128, (r[3], r[4])
    assert r[5] == 384 and r[6] == 384

    # slt/sltu/slti/sltiu：-1 vs 1 的有/无符号对照
    it = run_asm("""
        addi x1, x0, -1         # 0xFFFFFFFF
        addi x2, x0, 1
        slt  x3, x1, x2         # 有符号: -1 < 1 -> 1
        sltu x4, x1, x2         # 无符号: 大 < 小 -> 0
        slti x5, x1, 0          # -1 < 0 -> 1
        sltiu x6, x1, 1         # 无符号大数 < 1 -> 0
    """)
    r = it.state.regs
    assert (r[3], r[4], r[5], r[6]) == (1, 0, 1, 0), (r[3], r[4], r[5], r[6])

    # bltu/bgeu：无符号比较退出循环
    it = run_asm("""
        addi x1, x0, 5
        addi x2, x0, 0
    loop:
        addi x2, x2, 1
        addi x1, x1, -1
        bltu x1, x2, done       # 无符号 x1 < x2 -> 退出
        bgeu x0, x0, loop       # 恒真（0 >=u 0），无条件回环
    done:
    """)
    assert it.state.regs[2] == 3, it.state.regs[2]   # (4,1)(3,2)(2,3) 三轮退出

    # ebreak = 停机点（不是 trap）：run() 吞下信号、halted 置位、pc 停在 ebreak 本身
    it = run_asm("""
        addi x1, x0, 7
        ebreak
        addi x2, x0, 9          # 不应被执行
    """)
    assert it.halted is True, "ebreak 必须置 halted"
    assert it.state.regs[1] == 7 and it.state.regs[2] == 0, "ebreak 之后的指令不得执行"
    assert it.state.pc == it.cfg.rom_base + 4, hex(it.state.pc)

    # 手动 step 驱动时 ebreak 仍会抛出，调用方必须自己接住
    it = Interpreter()
    it.load(assemble("ebreak"))
    try:
        it.step()
    except RV32Trap_placeholder:
        pass
    else:
        raise AssertionError("step() 遇 ebreak 应抛 RV32Trap_placeholder")


def test_v2_batch2() -> None:
    M = 0xFFFFFFFF
    # fence 是 no-op；M 扩展基础向量
    it = run_asm("""
        fence
        fence.i
        addi x1, x0, -1        # 0xFFFFFFFF
        addi x2, x0, 2
        mul  x3, x1, x2        # 低 32：0xFFFFFFFE
        mulh x4, x1, x2        # -2 的高 32 = 全 1
        mulhsu x5, x1, x2      # -1(有符号)×2(无符号) = -2 -> 高 32 全 1
        mulhu x6, x1, x2       # 无符号大数×2 = 0x1FFFFFFFE -> 高 32 = 1
        div  x7, x1, x2        # -1 / 2 截断 = 0（不是 -1！）
        rem  x8, x1, x2        # -1 - 0×2 = -1
    """)
    r = it.state.regs
    assert r[3] == 0xFFFFFFFE and r[4] == M and r[5] == M and r[6] == 1, (
        r[3], r[4], r[5], r[6])
    assert r[7] == 0 and r[8] == M, (r[7], r[8])

    # 截断向零 + 无符号
    it = run_asm("""
        addi x1, x0, -7
        addi x2, x0, 2
        div  x3, x1, x2        # -7/2 = -3（截断），不是 -4
        rem  x4, x1, x2        # -7 - (-3×2) = -1
        divu x5, x1, x2        # 无符号 0xFFFFFFF9 / 2 = 0x7FFFFFFC
        remu x6, x1, x2        # 余 1
    """)
    r = it.state.regs
    assert r[3] == (M - 2) and r[4] == M and r[5] == 0x7FFFFFFC and r[6] == 1, (
        r[3], r[4], r[5], r[6])

    # 除零不 trap：div/divu -> 全 1，rem/remu -> 被除数
    it = run_asm("""
        addi x1, x0, 7
        addi x2, x0, 0
        div  x3, x1, x2
        divu x4, x1, x2
        rem  x5, x1, x2
        remu x6, x1, x2
    """)
    r = it.state.regs
    assert r[3] == M and r[4] == M and r[5] == 7 and r[6] == 7

    # INT_MIN ÷ -1 溢出回绕（不 trap）
    it = run_asm("""
        addi x1, x0, -1
        slli x1, x1, 31        # 0x80000000 = INT_MIN
        addi x2, x0, -1
        div  x3, x1, x2        # 回绕为 INT_MIN
        rem  x4, x1, x2        # 0
    """)
    r = it.state.regs
    assert r[3] == 0x80000000 and r[4] == 0


def test_v2_batch3() -> None:
    it = run_asm("""
        addi x1, x0, 3
        addi x2, x0, 5
        sh1add x3, x1, x2     # 3<<1 + 5 = 11
        sh2add x4, x1, x2     # 3<<2 + 5 = 17
        sh3add x5, x1, x2     # 3<<3 + 5 = 29
        addi x6, x0, 1
        slli x6, x6, 29       # 0x20000000
        sh3add x7, x6, x0     # <<3 溢出回绕 -> 0
    """)
    r = it.state.regs
    assert (r[3], r[4], r[5]) == (11, 17, 29), (r[3], r[4], r[5])
    assert r[7] == 0, r[7]                # 移位溢出截断后加法


def main() -> None:
    test_v2_batch3()
    test_v2_batch2()
    test_fib()
    test_gcd()
    test_mem_shifts_logic()
    test_jal_jalr()
    test_traps()
    test_v2_batch1()
    print("test_interpreter OK: fib/gcd/mem/shifts/jal/jalr/traps/v2-batch1/v2-batch2(M)/v2-batch3(Zba))")


if __name__ == "__main__":
    main()