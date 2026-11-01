"""
run.py: program loading and running entry points (vm.py owns the VM class).

Run either way:
    python -m components.vm.run          (from project root)
    double-click run.py                    (direct execution, e.g. on Windows)
The bootstrap below makes relative imports work in the direct case too.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

_DIRECT = __package__ in (None, "")
if _DIRECT:  # launched as a plain script (double-click / python run.py)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "components.vm"

from .core import DEFAULT_CONFIG, Op, TraceEvent, VMConfig, VMError, make_config
from .handlers import factory as default_factory
from .asm import make_asm
from .vm import VM


def make_vm(config: VMConfig | None = None, handlers: object | None = None) -> VM:
    cfg = config or DEFAULT_CONFIG
    return VM(config=cfg, handlers=handlers if handlers is not None else default_factory,
              mem=bytearray(cfg.mem_size))


def load_program(
    program: bytes | bytearray | Iterable[int],
    base: int = 0,
    config: VMConfig | None = None,
    handlers: object | None = None,
) -> VM:
    cfg = config or DEFAULT_CONFIG
    data = bytes(program)
    if base < 0 or base + len(data) > cfg.mem_size:
        raise ValueError("program too large for config.mem_size")
    vm = make_vm(cfg, handlers=handlers)
    vm.memory[base:base + len(data)] = data
    vm.pc = base
    return vm


def run_program(
    program: bytes | bytearray | Iterable[int],
    base: int = 0,
    config: VMConfig | None = None,
    handlers: object | None = None,
    max_steps: int = 100_000,
    trace: bool = False,
) -> tuple[VM, list[TraceEvent] | None]:
    return load_program(program, base=base, config=config, handlers=handlers).run(
        max_steps=max_steps, trace=trace
    )


# ---------------- Smoke tests ----------------

def _expect_stack(program: bytes, expected: list[int], **kw) -> VM:
    vm, _ = run_program(program, **kw)
    assert vm.stack == expected, (program.hex(), vm.stack, expected)
    return vm


def smoke_tests() -> None:
    a = make_asm()
    a.push(6).push(7).op(Op.ADD).op(Op.HALT)
    _expect_stack(a.build(), [13])

    a = make_asm()
    a.push(9).push(4).op(Op.DIV).push(9).push(4).op(Op.MOD).op(Op.ADD).op(Op.HALT)
    _expect_stack(a.build(), [3])

    a = make_asm()
    a.push(1).push(2).op(Op.SWAP).op(Op.OVER).op(Op.ROT).op(Op.HALT)
    _expect_stack(a.build(), [1, 2, 2])

    a = make_asm()
    a.push(10).push(42).op(Op.STORE).push(10).op(Op.LOAD).op(Op.HALT)
    vm = _expect_stack(a.build(), [42])
    assert vm.memory[10] == 42

    a = make_asm()
    a.push(5).call("mul3").op(Op.HALT)
    a.label("mul3")
    a.push(3).op(Op.MUL).op(Op.RET)
    _expect_stack(a.build(), [15])

    a = make_asm()
    for i in range(1, 6):
        a.push(i)
    a.op(Op.ADD).op(Op.ADD).op(Op.ADD).op(Op.ADD).op(Op.HALT)
    _expect_stack(a.build(), [15])

    a = make_asm()
    a.push(0).jz("t").push(111).op(Op.HALT).label("t").push(222).op(Op.HALT)
    _expect_stack(a.build(), [222])
    a = make_asm()
    a.push(1).jz("t").push(111).op(Op.HALT).label("t").push(222).op(Op.HALT)
    _expect_stack(a.build(), [111])

    # loop: sum 1..5 via mem[100]=n, mem[101]=acc (data cells live past the program:
    # program bytes occupy 0..~45, and this is a von Neumann machine)
    a = make_asm()
    a.push(100).push(5).op(Op.STORE)     # mem[100] = n
    a.push(101).push(0).op(Op.STORE)     # mem[101] = acc
    a.label("loop")
    a.push(100).op(Op.LOAD)              # n
    a.jz("done")                         # if n == 0 -> done
    a.push(101).op(Op.DUP).op(Op.LOAD)   # [101, acc]
    a.push(100).op(Op.LOAD)              # [101, acc, n]
    a.op(Op.ADD)                         # [101, acc+n]
    a.op(Op.STORE)                       # mem[101] = acc+n
    a.push(100).op(Op.DUP).op(Op.LOAD)  # [100, n]
    a.push(1).op(Op.SUB)                 # [100, n-1]
    a.op(Op.STORE)                       # mem[100] = n-1
    a.jmp("loop")
    a.label("done")
    a.push(101).op(Op.LOAD).op(Op.HALT)
    _expect_stack(a.build(), [15])

    small = make_config(mem_size=64)
    vm, _ = run_program(make_asm().push(1).push(2).op(Op.ADD).op(Op.HALT).build(), config=small)
    assert vm.stack == [3] and len(vm.memory) == 64

    try:
        run_program(bytes([0xFF, 0x01]))
        raise AssertionError("expected unknown-opcode trap")
    except VMError:
        pass

    try:
        run_program(bytes([int(Op.JMP), 0x00, 0x00]), max_steps=1000)
        raise AssertionError("expected step-limit trap")
    except VMError:
        pass

    try:
        run_program(make_asm().push(1).push(0).op(Op.DIV).op(Op.HALT).build())
        raise AssertionError("expected div-zero trap")
    except VMError:
        pass
    try:
        run_program(make_asm().op(Op.POP).op(Op.HALT).build())
        raise AssertionError("expected underflow trap")
    except VMError:
        pass

    vm, tr = run_program(make_asm().push(1).push(2).op(Op.ADD).op(Op.HALT).build(), trace=True)
    assert tr is not None and [e["op"] for e in tr] == ["PUSH", "PUSH", "ADD", "HALT"]
    assert tr[2]["stack"] == [3]


if __name__ == "__main__":
    smoke_tests()
    print("run smoke tests passed")
    if _DIRECT and sys.stdin.isatty():
        input("\n按回车键退出...")