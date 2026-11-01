"""
handlers.py: OpFactory + instruction handlers (the "opcode" module).

Handler convention (vm.py side):
    op = Op(vm.fetch_u8(vm.pc)); vm.pc += 1
    factory.get_code(op.name.lower())(vm)

Handlers with immediates fetch them themselves (see push/jmp/jz/call), so the
factory keys on the plain function name only.
"""

from __future__ import annotations

from collections.abc import Callable

from .core import VMError


class OpFactory:
    def __init__(self) -> None:
        self.codes: dict[str, Callable] = {}

    def code(self, fn: Callable) -> Callable:
        # dispatch looks up op.name.lower(); 'and_'/'or_'/'xor_' are registered
        # under the keyword-free name so Op.AND -> "and" resolves correctly.
        self.codes[fn.__name__.removesuffix('_')] = fn
        return fn  # NOTE: decorator must return fn, else the name binds to None

    def get_code(self, name: str):
        return self.codes.get(name)


factory = OpFactory()


# ---------------- stack ops ----------------

@factory.code
def nop(vm) -> None:
    pass


@factory.code
def halt(vm) -> None:
    vm.halted = True


@factory.code
def push(vm) -> None:
    imm = vm.fetch_u8(vm.pc)
    vm.pc += 1
    vm.push(imm)


@factory.code
def pop(vm) -> None:
    vm.pop()


@factory.code
def dup(vm) -> None:
    vm.push(vm.peek(0))


@factory.code
def drop(vm) -> None:
    vm.pop()


@factory.code
def swap(vm) -> None:
    a = vm.pop(); b = vm.pop()
    vm.push(a); vm.push(b)


@factory.code
def over(vm) -> None:
    vm.push(vm.peek(1))


@factory.code
def rot(vm) -> None:
    c = vm.pop(); b = vm.pop(); a = vm.pop()
    vm.push(b); vm.push(c); vm.push(a)


# ---------------- ALU ops ----------------

@factory.code
def add(vm) -> None:
    b = vm.pop(); a = vm.pop()
    vm.push(a + b)
    vm.carry = 1 if a + b > 0xFF else 0   # carry out


@factory.code
def sub(vm) -> None:
    b = vm.pop(); a = vm.pop()
    vm.push(a - b)
    vm.carry = 1 if a < b else 0          # borrow out


@factory.code
def mul(vm) -> None:
    b = vm.pop(); a = vm.pop()
    vm.push(a * b)


@factory.code
def div(vm) -> None:
    b = vm.pop(); a = vm.pop()
    if b == 0:
        raise VMError("division by zero")
    vm.push(a // b)


@factory.code
def mod(vm) -> None:
    b = vm.pop(); a = vm.pop()
    if b == 0:
        raise VMError("modulo by zero")
    vm.push(a % b)


# ---------------- memory ops ----------------

@factory.code
def load(vm) -> None:
    vm.push(vm.load_mem(vm.pop()))


@factory.code
def store(vm) -> None:
    # Stack convention: ... addr value ; STORE pops value then addr.
    val = vm.pop(); addr = vm.pop()
    vm.store_mem(addr, val)


# ---------------- control flow ops ----------------

@factory.code
def jmp(vm) -> None:
    vm.pc = vm.fetch_u16(vm.pc)


@factory.code
def jz(vm) -> None:
    target = vm.fetch_u16(vm.pc)
    vm.pc += 2
    if vm.pop() == 0:
        vm.pc = target


@factory.code
def call(vm) -> None:
    target = vm.fetch_u16(vm.pc)
    vm.pc += 2
    vm.rpush(vm.pc)
    vm.pc = target


@factory.code
def ret(vm) -> None:
    vm.pc = vm.rpop()

@factory.code
def dup2(vm) -> None:
    a = vm.peek(1); b = vm.peek(0)
    vm.push(a); vm.push(b)


@factory.code
def drop2(vm) -> None:
    vm.pop(); vm.pop()


@factory.code
def adc(vm) -> None:
    b = vm.pop(); a = vm.pop()
    s = a + b + vm.carry
    vm.push(s)
    vm.carry = 1 if s > 0xFF else 0


@factory.code
def sbb(vm) -> None:
    b = vm.pop(); a = vm.pop()
    d = a - b - vm.carry
    vm.push(d)
    vm.carry = 1 if d < 0 else 0          # borrow out


@factory.code
def and_(vm) -> None:
    b = vm.pop(); a = vm.pop()
    vm.push(a & b)


@factory.code
def or_(vm) -> None:
    b = vm.pop(); a = vm.pop()
    vm.push(a | b)


@factory.code
def xor_(vm) -> None:
    b = vm.pop(); a = vm.pop()
    vm.push(a ^ b)


@factory.code
def shl(vm) -> None:
    a = vm.pop()
    vm.carry = (a >> 7) & 1
    vm.push(a << 1)


@factory.code
def shr(vm) -> None:
    a = vm.pop()
    vm.carry = a & 1
    vm.push(a >> 1)


@factory.code
def jnz(vm) -> None:
    target = vm.fetch_u16(vm.pc)
    vm.pc += 2
    if vm.pop() != 0:
        vm.pc = target


@factory.code
def bank(vm) -> None:
    vm.bank = vm.fetch_u8(vm.pc)
    vm.pc += 1


# ---------------- minimal smoke (FakeVM only; real VM lives in vm.py) ----------------

class _FakeVM:
    def __init__(self) -> None:
        self.stack: list[int] = []
        self.rstack: list[int] = []
        self.mem = bytearray(256)
        self.pc = 0
        self.halted = False
    def push(self, x): self.stack.append(x & 0xFF)
    def pop(self): return self.stack.pop()
    def peek(self, k=0): return self.stack[-1-k]
    def rpush(self, x): self.rstack.append(x)
    def rpop(self): return self.rstack.pop()
    def fetch_u8(self, a): return self.mem[a]
    def fetch_u16(self, a): return self.mem[a] | (self.mem[a+1] << 8)
    def load_mem(self, a): return self.mem[a & 0xFF]
    def store_mem(self, a, v): self.mem[a & 0xFF] = v & 0xFF


def smoke() -> None:
    assert set(factory.codes) == {
        "nop","halt","push","pop","dup","drop","swap","over","rot",
        "add","sub","mul","div","mod","load","store",
        "jmp","jz","call","ret",
    }
    # push fetches its own imm8
    vm = _FakeVM(); vm.mem[0] = 42
    factory.get_code("push")(vm)
    assert vm.stack == [42] and vm.pc == 1

    # add
    vm = _FakeVM(); vm.push(6); vm.push(7)
    factory.get_code("add")(vm)
    assert vm.stack == [13]

    # swap / store / load
    vm = _FakeVM(); vm.push(1); vm.push(2)
    factory.get_code("swap")(vm)
    assert vm.stack == [2, 1]
    vm = _FakeVM(); vm.push(10); vm.push(42)
    factory.get_code("store")(vm)
    assert vm.mem[10] == 42 and vm.stack == []
    vm.push(10)
    factory.get_code("load")(vm)
    assert vm.stack == [42]

    # jz taken: target 0x0005 at mem[0..1], cond 0 -> pc = 5
    vm = _FakeVM(); vm.mem[0] = 5; vm.mem[1] = 0; vm.stack = [0]
    factory.get_code("jz")(vm)
    assert vm.pc == 5 and vm.stack == []

    # div by zero traps
    vm = _FakeVM(); vm.push(1); vm.push(0)
    try:
        factory.get_code("div")(vm)
        raise AssertionError("expected div-zero trap")
    except VMError:
        pass


if __name__ == "__main__":
    smoke()
    print("handlers smoke passed")


