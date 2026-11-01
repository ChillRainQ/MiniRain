"""
core.py: ISA definition + config. No interpreter, no assembler here.
(vm.py owns the interpreter and assembler; handlers.py owns factory + ops.)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final, TypedDict


class Op(IntEnum):
    NOP   = 0x00
    HALT  = 0x01
    POP   = 0x02
    DUP   = 0x03
    DROP  = 0x04
    SWAP  = 0x05
    OVER  = 0x06
    ROT   = 0x07

    ADD   = 0x10
    SUB   = 0x11
    MUL   = 0x12
    DIV   = 0x13
    MOD   = 0x14

    LOAD  = 0x20
    STORE = 0x21

    DUP2  = 0x08  # ( a b -- a b a b )
    DROP2 = 0x09  # ( a b -- )

    ADC   = 0x15  # a b -> a+b+carry (carry out -> carry)
    SBB   = 0x16  # a b -> a-b-borrow (borrow out -> carry)
    AND   = 0x17
    SHL   = 0x18  # carry out := bit7
    SHR   = 0x19  # carry out := bit0

    JMP   = 0x30
    JZ    = 0x31
    CALL  = 0x32
    RET   = 0x33
    JNZ   = 0x34  # pop cond, jump if != 0

    PUSH  = 0x40  # PUSH imm8

    BANK  = 0x50  # BANK imm8: select 256-byte data window for LOAD/STORE


OP_NAMES: Final[dict[Op, str]] = {op: op.name for op in Op}
IMM16_OPS: Final[set[Op]] = {Op.JMP, Op.JZ, Op.CALL, Op.JNZ}
IMM8_OPS: Final[set[Op]] = {Op.PUSH, Op.BANK}
WORD_MASK: Final[int] = 0xFF


class VMError(Exception):
    """Recoverable VM trap: underflow, overflow, bad opcode, div/mod by zero, etc."""


class TraceEvent(TypedDict):
    pc: int
    op: str
    pc_next: int
    stack: list[int]
    rstack: list[int]
    steps: int


@dataclass(frozen=True)
class VMConfig:
    stack_depth: int = 64
    return_depth: int = 32
    mem_size: int = 8192
    word_mask: int = WORD_MASK


DEFAULT_CONFIG: Final[VMConfig] = VMConfig()


def make_config(**overrides: int) -> VMConfig:
    """Factory for VMConfig with keyword overrides; unknown keys raise TypeError."""
    return VMConfig(**overrides)


def mask_word(x: int, mask: int = WORD_MASK) -> int:
    return x & mask


__all__ = [
    "Op", "OP_NAMES", "IMM16_OPS", "IMM8_OPS", "WORD_MASK",
    "VMError", "TraceEvent", "VMConfig", "DEFAULT_CONFIG",
    "make_config", "mask_word",
]


