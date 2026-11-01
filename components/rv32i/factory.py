"""
ISA-RV32 v1 指令语义
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

# 包导入与脚本直接运行两种方式都支持（同 components/vm/nueral_cpu.py 的写法）
_DIRECT = __package__ in (None, "")
if _DIRECT:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "components.rv32i"

from .core import WORD_MASK, RV32Trap
from .isa import Instr, sx


class RV32Trap_placeholder(RV32Trap):
    """ecall/ebreak 停机点。它是 RV32Trap 的子类，所以：

    - `except RV32Trap_placeholder` 仍能精确区分"固件请求停机"；
    - `except RV32Trap` 也能一把接住（写 soc.py 时不必枚举两套类型）。

    注意捕获顺序：先用子类、再用父类，反过来 halt 会被当成异常。
    """
    pass


class OpFactory:
    def __init__(self) -> None:
        self.codes: dict[str, Callable] = {}

    def code(self, fn: Callable) -> Callable:
        # 助记符里的 '.' 在函数名里写成 '_'（fence.i -> fence_i），注册时还原
        self.codes[fn.__name__.removesuffix('_').replace('_', '.')] = fn
        return fn

    def get_code(self, name: str):
        return self.codes.get(name)


factory = OpFactory()


# ---------------- R 型算术逻辑 ----------------

@factory.code
def add(vm, ins: Instr) -> None:
    """加法：rd = rs1 + rs2，模 2^32 回绕（无溢出捕获，RV32I 语义）。"""
    vm.write(ins.rd, vm.regs[ins.rs1] + vm.regs[ins.rs2])


@factory.code
def sub(vm, ins: Instr) -> None:
    """减法：rd = rs1 - rs2，模 2^32 回绕（借位同样不捕获）。"""
    vm.write(ins.rd, vm.regs[ins.rs1] - vm.regs[ins.rs2])


@factory.code
def and_(vm, ins: Instr) -> None:
    """按位与：rd = rs1 & rs2（寄存器-寄存器型）。"""
    vm.write(ins.rd, vm.regs[ins.rs1] & vm.regs[ins.rs2])


@factory.code
def or_(vm, ins: Instr) -> None:
    """按位或：rd = rs1 | rs2（寄存器-寄存器型）。"""
    vm.write(ins.rd, vm.regs[ins.rs1] | vm.regs[ins.rs2])


@factory.code
def xor(vm, ins: Instr) -> None:
    """按位异或：rd = rs1 ^ rs2。"""
    vm.write(ins.rd, vm.regs[ins.rs1] ^ vm.regs[ins.rs2])


@factory.code
def sll(vm, ins: Instr) -> None:
    """逻辑左移：rd = rs1 << (rs2 低 5 位)，低位补 0；移位量只取低 5 位。"""
    vm.write(ins.rd, vm.regs[ins.rs1] << (vm.regs[ins.rs2] & 31))


@factory.code
def srl(vm, ins: Instr) -> None:
    """逻辑右移：rd = rs1 >>> (rs2 低 5 位)，高位补 0（按无符号数处理）。"""
    vm.write(ins.rd, vm.regs[ins.rs1] >> (vm.regs[ins.rs2] & 31))


@factory.code
def sra(vm, ins: Instr) -> None:
    """算术右移：rd = rs1 >> (rs2 低 5 位)，高位补符号位（按有符号数处理）。"""
    vm.write(ins.rd, sx(vm.regs[ins.rs1], 32) >> (vm.regs[ins.rs2] & 31))


# ---------------- I 型算术逻辑（带立即数） ----------------

@factory.code
def addi(vm, ins: Instr) -> None:
    """加立即数：rd = rs1 + imm，imm 为 12-bit 符号扩展值，模 2^32 回绕。"""
    vm.write(ins.rd, vm.regs[ins.rs1] + ins.imm)


@factory.code
def andi(vm, ins: Instr) -> None:
    """按位与立即数：rd = rs1 & imm。imm 已符号扩展，高位 1 实质参与运算
    （教训：gc_init 的 andi a4,a1,-16 被 &0xFFF 截断导致堆指针崩坏）。"""
    vm.write(ins.rd, vm.regs[ins.rs1] & ins.imm)


@factory.code
def ori(vm, ins: Instr) -> None:
    """按位或立即数：rd = rs1 | imm（imm 已符号扩展）。"""
    vm.write(ins.rd, vm.regs[ins.rs1] | ins.imm)


@factory.code
def xori(vm, ins: Instr) -> None:
    """按位异或立即数：rd = rs1 ^ imm（imm 已符号扩展）。"""
    vm.write(ins.rd, vm.regs[ins.rs1] ^ ins.imm)


@factory.code
def slli(vm, ins: Instr) -> None:
    """左移立即数：rd = rs1 << shamt，shamt = imm 低 5 位，低位补 0。"""
    vm.write(ins.rd, vm.regs[ins.rs1] << (ins.imm & 31))


@factory.code
def srli(vm, ins: Instr) -> None:
    """逻辑右移立即数：rd = rs1 >>> shamt，高位补 0。"""
    vm.write(ins.rd, vm.regs[ins.rs1] >> (ins.imm & 31))


@factory.code
def srai(vm, ins: Instr) -> None:
    """算术右移立即数：rd = rs1 >> shamt，高位补符号位。"""
    vm.write(ins.rd, sx(vm.regs[ins.rs1], 32) >> (ins.imm & 31))


# ---------------- U 型（高位立即数） ----------------

@factory.code
def lui(vm, ins: Instr) -> None:
    """加载高位立即数：rd = imm20 << 12，低 12 位清零（imm 为指令字中的 20-bit 原始值）。"""
    vm.write(ins.rd, ins.imm << 12)


@factory.code
def auipc(vm, ins: Instr) -> None:
    """PC 相对高位加：rd = 本条指令地址 + (imm20 << 12)，用于生成 PC 无关的常量地址。"""
    vm.write(ins.rd, vm.pc + (ins.imm << 12))   # vm.pc = 本条指令地址（vm 未推进）


# ---------------- 跳转与分支（控制流：显式设置 vm.pc） ----------------

@factory.code
def jal(vm, ins: Instr) -> int:
    """无条件跳转：x[rd] = pc+4（返回地址），返回 pc+imm。"""
    vm.write(ins.rd, vm.pc + 4)
    return (vm.pc + ins.imm) & WORD_MASK


@factory.code
def jalr(vm, ins: Instr) -> int:
    """寄存器间接跳转：目标 = (rs1 + imm) 且最低位清 0；x[rd] = pc+4。"""
    target = (vm.regs[ins.rs1] + ins.imm) & WORD_MASK & ~1
    vm.write(ins.rd, vm.pc + 4)
    return target


@factory.code
def beq(vm, ins: Instr) -> int | None:
    """相等分支：rs1 == rs2 返回 pc+imm，否则返回 None（默认 +4）。"""
    if vm.regs[ins.rs1] == vm.regs[ins.rs2]:
        return (vm.pc + ins.imm) & WORD_MASK
    return None


@factory.code
def bne(vm, ins: Instr) -> int | None:
    """不等分支：rs1 != rs2 返回 pc+imm，否则 None。"""
    if vm.regs[ins.rs1] != vm.regs[ins.rs2]:
        return (vm.pc + ins.imm) & WORD_MASK
    return None


@factory.code
def blt(vm, ins: Instr) -> int | None:
    """小于分支（有符号）：rs1 < rs2 返回 pc+imm，否则 None。"""
    if sx(vm.regs[ins.rs1], 32) < sx(vm.regs[ins.rs2], 32):
        return (vm.pc + ins.imm) & WORD_MASK
    return None


@factory.code
def bge(vm, ins: Instr) -> int | None:
    """大于等于分支（有符号）：rs1 >= rs2 返回 pc+imm，否则 None。"""
    if sx(vm.regs[ins.rs1], 32) >= sx(vm.regs[ins.rs2], 32):
        return (vm.pc + ins.imm) & WORD_MASK
    return None


# ---------------- 访存 ----------------

@factory.code
def lw(vm, ins: Instr) -> None:
    """读 32-bit 字：rd = mem[rs1 + imm]，小端；地址必须 4 字节对齐，否则 trap。"""
    vm.write(ins.rd, vm.lw(ins.rs1, ins.imm))


@factory.code
def sw(vm, ins: Instr) -> None:
    """写 32-bit 字：mem[rs1 + imm] = rs2，小端；地址必须 4 字节对齐，否则 trap。"""
    vm.sw(ins.rs1, ins.imm, vm.regs[ins.rs2])

# ---------------- v2 第一批：比较置位 ----------------

@factory.code
def slt(vm, ins: Instr) -> None:
    """有符号小于置位：rd = (rs1 <s rs2) ? 1 : 0，两操作数按 32-bit 有符号解释。"""
    vm.write(ins.rd, 1 if sx(vm.regs[ins.rs1], 32) < sx(vm.regs[ins.rs2], 32) else 0)


@factory.code
def sltu(vm, ins: Instr) -> None:
    """无符号小于置位：rd = (rs1 <u rs2) ? 1 : 0。"""
    vm.write(ins.rd, 1 if vm.regs[ins.rs1] < vm.regs[ins.rs2] else 0)


@factory.code
def slti(vm, ins: Instr) -> None:
    """有符号小于立即数置位：rd = (rs1 <s imm) ? 1 : 0。"""
    vm.write(ins.rd, 1 if sx(vm.regs[ins.rs1], 32) < ins.imm else 0)


@factory.code
def sltiu(vm, ins: Instr) -> None:
    """无符号小于立即数置位：imm 先符号扩展再按无符号比较（RV32I 规定）。"""
    vm.write(ins.rd, 1 if vm.regs[ins.rs1] < (ins.imm & WORD_MASK) else 0)


# ---------------- v2 第一批：无符号分支 ----------------

@factory.code
def bltu(vm, ins: Instr) -> int | None:
    """无符号小于分支：rs1 <u rs2 返回 pc+imm，否则 None。"""
    if vm.regs[ins.rs1] < vm.regs[ins.rs2]:
        return (vm.pc + ins.imm) & WORD_MASK
    return None


@factory.code
def bgeu(vm, ins: Instr) -> int | None:
    """无符号大于等于分支：rs1 >=u rs2 返回 pc+imm，否则 None。"""
    if vm.regs[ins.rs1] >= vm.regs[ins.rs2]:
        return (vm.pc + ins.imm) & WORD_MASK
    return None


# ---------------- v2 第一批：字节/半字访存 ----------------

@factory.code
def lb(vm, ins: Instr) -> None:
    """读字节符号扩展：rd = sext8(mem[rs1+imm])。"""
    vm.write(ins.rd, vm.lb(ins.rs1, ins.imm))


@factory.code
def lbu(vm, ins: Instr) -> None:
    """读字节零扩展：rd = zext8(mem[rs1+imm])。"""
    vm.write(ins.rd, vm.lbu(ins.rs1, ins.imm))


@factory.code
def lh(vm, ins: Instr) -> None:
    """读半字符号扩展（2 对齐）：rd = sext16(mem16[rs1+imm])。"""
    vm.write(ins.rd, vm.lh(ins.rs1, ins.imm))


@factory.code
def lhu(vm, ins: Instr) -> None:
    """读半字零扩展（2 对齐）：rd = zext16(mem16[rs1+imm])。"""
    vm.write(ins.rd, vm.lhu(ins.rs1, ins.imm))


@factory.code
def sb(vm, ins: Instr) -> None:
    """写字节：mem[rs1+imm] = rs2 低 8 位。"""
    vm.sb(ins.rs1, ins.imm, vm.regs[ins.rs2])


@factory.code
def sh(vm, ins: Instr) -> None:
    """写半字（2 对齐）：mem16[rs1+imm] = rs2 低 16 位。"""
    vm.sh(ins.rs1, ins.imm, vm.regs[ins.rs2])


# ---------------- v2 第一批：环境调用（占位） ----------------

@factory.code
def ecall(vm, ins: Instr) -> None:
    """环境调用（占位）：v1 无系统调用，直接 trap。将来是打印/系统服务出口。"""
    raise RV32Trap_placeholder("ecall: 系统调用未实现")


@factory.code
def ebreak(vm, ins: Instr) -> None:
    """断点（占位）：调试器接口，v1 直接 trap。"""
    raise RV32Trap_placeholder("ebreak: 断点未实现")

# ---------------- v2 第二批：fence 占位 ----------------

@factory.code
def fence(vm, ins: Instr) -> None:
    """内存屏障（占位）：单核顺序执行模型下是 no-op，保留给多核/乱序扩展。"""


@factory.code
def fence_i(vm, ins: Instr) -> None:
    """指令屏障（占位）：取指不自修改（ROM 无写口）时天然满足，no-op。"""


# ---------------- v2 第二批：M 扩展（神经版 32 拍：shift-add / restoring） ----------------

def _trunc_div(a: int, b: int) -> int:
    """截断向零的整数除法（RV32M 语义；Python // 是向下取整，不可直接用）。"""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


@factory.code
def mul(vm, ins: Instr) -> None:
    """乘低 32 位：rd = (rs1 × rs2) 的低 word_bits 位（模 2^32，与符号无关）。"""
    vm.write(ins.rd, vm.regs[ins.rs1] * vm.regs[ins.rs2])


@factory.code
def mulh(vm, ins: Instr) -> None:
    """乘高（有符号×有符号）：rd = 64 位积的高 32 位。"""
    a = sx(vm.regs[ins.rs1], 32)
    b = sx(vm.regs[ins.rs2], 32)
    vm.write(ins.rd, (a * b) >> 32)


@factory.code
def mulhsu(vm, ins: Instr) -> None:
    """乘高（有符号×无符号）：rs1 按有符号、rs2 按无符号，取 64 位积高 32 位。"""
    a = sx(vm.regs[ins.rs1], 32)
    b = vm.regs[ins.rs2]
    vm.write(ins.rd, (a * b) >> 32)


@factory.code
def mulhu(vm, ins: Instr) -> None:
    """乘高（无符号×无符号）：rd = 64 位积的高 32 位。"""
    vm.write(ins.rd, (vm.regs[ins.rs1] * vm.regs[ins.rs2]) >> 32)


@factory.code
def div(vm, ins: Instr) -> None:
    """有符号除（截断向零）。除零返回全 1；INT_MIN ÷ -1 回绕为 INT_MIN（不 trap）。"""
    a = sx(vm.regs[ins.rs1], 32)
    b = sx(vm.regs[ins.rs2], 32)
    if b == 0:
        vm.write(ins.rd, WORD_MASK)
    elif a == -(1 << 31) and b == -1:
        vm.write(ins.rd, 1 << 31)
    else:
        vm.write(ins.rd, _trunc_div(a, b))


@factory.code
def divu(vm, ins: Instr) -> None:
    """无符号除。除零返回全 1（不 trap）。"""
    b = vm.regs[ins.rs2]
    vm.write(ins.rd, WORD_MASK if b == 0 else vm.regs[ins.rs1] // b)


@factory.code
def rem(vm, ins: Instr) -> None:
    """有符号余（符号跟随被除数）。除零返回 rs1；INT_MIN ÷ -1 余 0。"""
    a = sx(vm.regs[ins.rs1], 32)
    b = sx(vm.regs[ins.rs2], 32)
    if b == 0:
        vm.write(ins.rd, vm.regs[ins.rs1])
    elif a == -(1 << 31) and b == -1:
        vm.write(ins.rd, 0)
    else:
        vm.write(ins.rd, a - _trunc_div(a, b) * b)


@factory.code
def remu(vm, ins: Instr) -> None:
    """无符号余。除零返回 rs1（不 trap）。"""
    b = vm.regs[ins.rs2]
    vm.write(ins.rd, vm.regs[ins.rs1] if b == 0 else vm.regs[ins.rs1] % b)

# ---------------- v2 第三批：Zba ----------------

@factory.code
def sh1add(vm, ins: Instr) -> None:
    """地址加速：rd = (rs1 << 1) + rs2（模 2^32）。等效 slli+add 的单条融合。"""
    vm.write(ins.rd, (vm.regs[ins.rs1] << 1) + vm.regs[ins.rs2])


@factory.code
def sh2add(vm, ins: Instr) -> None:
    """地址加速：rd = (rs1 << 2) + rs2。"""
    vm.write(ins.rd, (vm.regs[ins.rs1] << 2) + vm.regs[ins.rs2])


@factory.code
def sh3add(vm, ins: Instr) -> None:
    """地址加速：rd = (rs1 << 3) + rs2。"""
    vm.write(ins.rd, (vm.regs[ins.rs1] << 3) + vm.regs[ins.rs2])