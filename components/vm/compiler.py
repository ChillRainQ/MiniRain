"""
compiler.py: compiler frontend — lower a program's execution to micro-op IR.

This is the anchor for Stage-2 weight compilation: each MicroOp kind maps to
one fixed layer pattern (attention read / MLP lookup table / unembedding argmax)
in the compiled transformer. Before that happens, the IR must be proven
state-exact against the reference VM: for every executed instruction, simulating
its micro-op sequence from the pre-state must reproduce the post-state exactly
(stack, rstack, pc, full memory, halted).

MicroOp kinds (state ops only; decode is static in this frontend):
    const k dst           dst = k
    pop dst               dst = data_stack.pop()
    push src              data_stack.push(regs[src])
    peek k dst            dst = data_stack[-1-k]        (no pop)
    rpush src             return_stack.push(regs[src])
    rpop dst              dst = return_stack.pop()
    alu op a b dst        dst = a <op> b  (8-bit; div/mod trap on 0)
    read_mem a dst        dst = mem[regs[a]]
    write_mem a v         mem[regs[a]] = regs[v]
    fetch16 dst           dst = mem[pc+1] | mem[pc+2]<<8 ; micro-pc -> pc+3
    brz c target taken    assert (regs[c] == 0) == taken   (fidelity check)
    setpc k / setpc_reg r set micro-pc (absolute)
    halt                  halted = True

fetch16 is defined relative to the instruction-start pc: handlers in the
reference VM read operands from pc_start+1, so fetch16 does the same.
"""

from __future__ import annotations
import sys, pathlib
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "components.vm"
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .core import DEFAULT_CONFIG, Op, VMConfig, VMError, mask_word
from .handlers import factory as default_handlers
from .vm import VM


@dataclass(frozen=True)
class MicroOp:
    kind: str
    args: tuple


@dataclass
class _State:
    """Plain mutable snapshot; MicroVM executes MicroOps against it."""

    mem: bytearray
    stack: list[int]
    rstack: list[int]
    pc: int
    halted: bool = False

    def copy(self) -> _State:
        return _State(bytearray(self.mem), list(self.stack), list(self.rstack), self.pc, self.halted)

    def same_as(self, vm: VM) -> bool:
        return (self.stack == vm.stack and self.rstack == vm.rstack
                and self.pc == vm.pc and self.halted == vm.halted
                and bytes(self.mem) == bytes(vm.memory))


def _alu(op: Op, a: int, b: int) -> int:
    if op is Op.ADD: return mask_word(a + b)
    if op is Op.SUB: return mask_word(a - b)
    if op is Op.MUL: return mask_word(a * b)
    if op is Op.DIV:
        if b == 0: raise VMError("division by zero")
        return a // b
    if op is Op.MOD:
        if b == 0: raise VMError("modulo by zero")
        return a % b
    raise VMError(f"not an alu op: {op}")


def micro_execute(mops: Iterable[MicroOp], st: _State) -> _State:
    """Execute micro-ops against a state snapshot. Registers are per-instruction."""
    regs: dict[str, int] = {}
    for m in mops:
        k, a = m.kind, m.args
        if k == "const":
            regs[a[1]] = a[0]
        elif k == "pop":
            if not st.stack: raise VMError("data stack underflow")
            regs[a[0]] = st.stack.pop()
        elif k == "push":
            st.stack.append(mask_word(regs[a[0]]))
        elif k == "peek":
            off, dst = a
            if len(st.stack) <= off: raise VMError("data stack underflow")
            regs[dst] = st.stack[-1 - off]
        elif k == "rpush":
            st.rstack.append(regs[a[0]] & 0xFFFF)
        elif k == "rpop":
            if not st.rstack: raise VMError("return stack underflow")
            regs[a[0]] = st.rstack.pop()
        elif k == "alu":
            op, ra, rb, dst = a
            regs[dst] = _alu(op, regs[ra], regs[rb])
        elif k == "read_mem":
            ra, dst = a
            regs[dst] = st.mem[mask_word(regs[ra])]
        elif k == "write_mem":
            ra, rv = a
            st.mem[mask_word(regs[ra])] = mask_word(regs[rv])
        elif k == "fetch16":
            dst = a[0]
            lo = st.mem[st.pc + 1]
            hi = st.mem[st.pc + 2]
            regs[dst] = lo | (hi << 8)
            st.pc += 3
        elif k == "brz":
            rc, target, taken = a
            if (regs[rc] == 0) != taken:
                raise VMError(f"brz fidelity mismatch at pc={st.pc:#x}")
            # dynamic trace: fallthrough either way; pc fixed by setpc
        elif k == "jmp":
            st.pc = a[0]
        elif k == "setpc":
            st.pc = a[0]
        elif k == "setpc_reg":
            st.pc = regs[a[0]]
        elif k == "halt":
            st.halted = True
        else:
            raise VMError(f"unknown micro-op kind {k!r}")
    return st


def _lower(op: Op, st: _State) -> list[MicroOp]:
    """Micro-program for one instruction, from the pre-state (instruction-start pc)."""
    pc = st.pc
    nxt = pc + 1  # pc after opcode fetch (no operand)
    if op in (Op.JZ, Op.JMP, Op.CALL):
        nxt = pc + 3

    if op is Op.NOP:   return [MicroOp("setpc", (nxt,))]
    if op is Op.HALT:  return [MicroOp("halt", ()), MicroOp("setpc", (nxt,))]
    if op is Op.PUSH:
        imm = st.mem[pc + 1]
        return [MicroOp("const", (imm, "t0")), MicroOp("push", ("t0",)), MicroOp("setpc", (pc + 2,))]
    if op is Op.POP:   return [MicroOp("pop", ("t0",)), MicroOp("setpc", (nxt,))]
    if op is Op.DUP:   return [MicroOp("peek", (0, "t0")), MicroOp("push", ("t0",)), MicroOp("setpc", (nxt,))]
    if op is Op.DROP:  return [MicroOp("pop", ("t0",)), MicroOp("setpc", (nxt,))]
    if op is Op.SWAP:
        return [MicroOp("pop", ("t0",)), MicroOp("pop", ("t1",)),
                MicroOp("push", ("t0",)), MicroOp("push", ("t1",)), MicroOp("setpc", (nxt,))]
    if op is Op.OVER:  return [MicroOp("peek", (1, "t0")), MicroOp("push", ("t0",)), MicroOp("setpc", (nxt,))]
    if op is Op.ROT:
        return [MicroOp("pop", ("t0",)), MicroOp("pop", ("t1",)), MicroOp("pop", ("t2",)),
                MicroOp("push", ("t1",)), MicroOp("push", ("t2",)), MicroOp("push", ("t0",)),
                MicroOp("setpc", (nxt,))]
    if op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD):
        return [MicroOp("pop", ("t0",)), MicroOp("pop", ("t1",)),
                MicroOp("alu", (op, "t1", "t0", "t2")),
                MicroOp("push", ("t2",)), MicroOp("setpc", (nxt,))]
    if op is Op.LOAD:
        return [MicroOp("pop", ("t0",)), MicroOp("read_mem", ("t0", "t1")),
                MicroOp("push", ("t1",)), MicroOp("setpc", (nxt,))]
    if op is Op.STORE:
        return [MicroOp("pop", ("t0",)), MicroOp("pop", ("t1",)),
                MicroOp("write_mem", ("t1", "t0")), MicroOp("setpc", (nxt,))]
    if op is Op.JMP:
        return [MicroOp("fetch16", ("t0",)), MicroOp("setpc_reg", ("t0",))]
    if op is Op.JZ:
        target = st.mem[pc + 1] | (st.mem[pc + 2] << 8)
        taken = st.stack[-1] == 0
        actual = target if taken else pc + 3
        return [MicroOp("fetch16", ("t0",)), MicroOp("pop", ("t1",)),
                MicroOp("brz", ("t1", target, taken)), MicroOp("setpc", (actual,))]
    if op is Op.CALL:
        target = st.mem[pc + 1] | (st.mem[pc + 2] << 8)
        return [MicroOp("fetch16", ("t0",)), MicroOp("const", (pc + 3, "t1")),
                MicroOp("rpush", ("t1",)), MicroOp("setpc_reg", ("t0",))]
    if op is Op.RET:
        return [MicroOp("rpop", ("t0",)), MicroOp("setpc_reg", ("t0",))]
    raise VMError(f"cannot lower opcode {op}")


def compile_program(
    program: bytes | bytearray | Iterable[int],
    base: int = 0,
    config: VMConfig | None = None,
    handlers: object | None = None,
    max_steps: int = 100_000,
) -> tuple[list[MicroOp], VM]:
    """Run the reference VM, lowering every executed instruction to micro-ops.

    Per instruction: simulate micro-program from pre-state, require exact
    post-state equality with the reference step. Returns the full IR and the
    final reference VM (for final-state comparison).
    """
    cfg = config or DEFAULT_CONFIG
    data = bytes(program)
    vm = VM(config=cfg, handlers=handlers if handlers is not None else default_handlers,
            mem=bytearray(cfg.mem_size))
    vm.memory[base:base + len(data)] = data
    vm.pc = base

    micro: list[MicroOp] = []
    initial: _State | None = None

    while not vm.halted and vm.steps < max_steps:
        st = _State(bytearray(vm.memory), list(vm.stack), list(vm.rstack), vm.pc, vm.halted)
        if initial is None:
            initial = st.copy()

        raw = vm.fetch_u8(vm.pc)
        try:
            op = Op(raw)
        except ValueError:
            raise VMError(f"unknown opcode 0x{raw:02x} at pc={vm.pc:#x}") from None

        mops = _lower(op, st)
        expect = micro_execute(mops, st.copy())
        vm.step()  # reference executes the same instruction
        if not expect.same_as(vm):
            raise VMError(
                f"micro-op mismatch after {op.name} at pc={vm.pc:#x}: "
                f"sim(pc={expect.pc},stack={expect.stack},halted={expect.halted}) vs "
                f"vm(pc={vm.pc},stack={vm.stack},halted={vm.halted})"
            )
        micro.extend(mops)

    if not vm.halted:
        raise VMError(f"step limit exceeded: {max_steps}")

    # whole-program replay: cold MicroVM run must reproduce the final state
    assert initial is not None
    final_sim = micro_execute(micro, initial)
    if not (final_sim.stack == vm.stack and final_sim.rstack == vm.rstack
            and bytes(final_sim.mem) == bytes(vm.memory)):
        raise VMError("whole-program micro replay diverged from reference VM")

    return micro, vm


def stats(micro: list[MicroOp]) -> Counter:
    return Counter(m.kind for m in micro)


def main() -> None:
    from .demos import fib_program, gcd_program, bubble_sort_program

    programs = {
        "fib(10)":        (fib_program(10), [55]),
        "gcd(84,18)":     (gcd_program(84, 18), [6]),
        "sort[5,2,4,1,3]": (bubble_sort_program([5, 2, 4, 1, 3]), [1, 2, 3, 4, 5]),
    }
    total = Counter()
    for name, (prog, expect_stack) in programs.items():
        micro, vm = compile_program(prog)
        assert vm.stack == expect_stack, (name, vm.stack)
        s = stats(micro)
        total += s
        print(f"{name:16s} instructions={vm.steps:5d}  micro_ops={len(micro):5d}  {dict(s)}")
    print("TOTAL micro-op kinds:", dict(total))
    print("compiler frontend OK: IR is state-exact vs reference VM")


if __name__ == "__main__":

    main()