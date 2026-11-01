"""
vm.py: the interpreter. Assembler lives in asm.py (Asm/make_asm).

Dispatch contract (matches handlers.py):
    op = Op(vm.fetch_u8(vm.pc)); vm.pc += 1
    factory.get_code(op.name.lower())(vm)
Handlers fetch their own immediates, so step() does no operand decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .core import (
    DEFAULT_CONFIG,
    Op,
    TraceEvent,
    VMConfig,
    VMError,
    make_config,
    mask_word,
)
from .handlers import factory as default_factory, OpFactory


# ---------------- Interpreter ----------------

@dataclass
class VM:
    config: VMConfig = field(default_factory=VMConfig)
    handlers: OpFactory = default_factory          # duck-typed: has .get_code(name)
    mem: bytearray | None = None
    stack: list[int] = field(default_factory=list)
    rstack: list[int] = field(default_factory=list)
    pc: int = 0
    flags: int = 0
    halted: bool = False
    steps: int = 0

    def __post_init__(self) -> None:
        if self.mem is None:
            self.mem = bytearray(self.config.mem_size)
        elif len(self.mem) != self.config.mem_size:
            raise VMError(
                f"mem length {len(self.mem)} != config.mem_size {self.config.mem_size}"
            )

    @property
    def memory(self) -> bytearray:
        assert self.mem is not None
        return self.mem

    # ----- stack helpers -----
    def _u8(self, x: int) -> int:
        return mask_word(x, self.config.word_mask)

    def push(self, x: int) -> None:
        if len(self.stack) >= self.config.stack_depth:
            raise VMError("data stack overflow")
        self.stack.append(self._u8(x))

    def pop(self) -> int:
        if not self.stack:
            raise VMError("data stack underflow")
        return self.stack.pop()

    def peek(self, k: int = 0) -> int:
        if len(self.stack) <= k:
            raise VMError("data stack underflow")
        return self.stack[-1 - k]

    def rpush(self, x: int) -> None:
        if len(self.rstack) >= self.config.return_depth:
            raise VMError("return stack overflow")
        self.rstack.append(x & 0xFFFF)

    def rpop(self) -> int:
        if not self.rstack:
            raise VMError("return stack underflow")
        return self.rstack.pop()

    # ----- memory -----
    def fetch_u8(self, addr: int) -> int:
        if not 0 <= addr < len(self.memory):
            raise VMError(f"fetch out of range: {addr}")
        return self.memory[addr]

    def fetch_u16(self, addr: int) -> int:
        lo = self.fetch_u8(addr)
        hi = self.fetch_u8(addr + 1)
        return lo | (hi << 8)

    def load_mem(self, addr8: int) -> int:
        # Stage-0: 8-bit address -> only low 256 bytes addressable.
        return self.memory[self._u8(addr8)]

    def store_mem(self, addr8: int, val: int) -> None:
        self.memory[self._u8(addr8)] = self._u8(val)

    # ----- run -----
    def step(self, trace: list[TraceEvent] | None = None) -> bool:
        """Execute one instruction. Returns True if still running."""
        if self.halted:
            return False

        start_pc = self.pc
        raw = self.fetch_u8(self.pc)
        try:
            op = Op(raw)
        except ValueError:
            raise VMError(f"unknown opcode 0x{raw:02x} at pc={start_pc:#x}") from None
        self.pc += 1
        self.steps += 1

        handler = self.handlers.get_code(op.name.lower())
        if handler is None:
            raise VMError(f"no handler for opcode {op.name} at pc={start_pc:#x}")
        handler(self)

        if trace is not None:
            trace.append(TraceEvent(
                pc=start_pc, op=op.name, pc_next=self.pc,
                stack=list(self.stack), rstack=list(self.rstack), steps=self.steps,
            ))
        return not self.halted

    def run(self, max_steps: int = 100_000, trace: bool = False) -> tuple[VM, list[TraceEvent] | None]:
        events: list[TraceEvent] | None = [] if trace else None
        while not self.halted and self.steps < max_steps:
            self.step(trace=events)
        if not self.halted:
            raise VMError(f"step limit exceeded: {max_steps}")
        return self, events

    def result(self) -> int:
        return self.peek(0)

