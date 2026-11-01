"""
neural_cpu.py: compile the stack-machine ISA into transformer blocks and
weight tensors, executed as hard-attention trace replay.

Blocks (all weights computed from the ISA spec; 0 trained):
    Embed        addr/value -> one-hot vectors (E_addr, fixed identity, saved)
    HardAttention trace-token KV cache; read(addr) = argmax over history of
                 score = [addr_t == addr] * BIG + t   (one-hot QK^T, no softmax;
                 recency term selects the LATEST write to that address)
    Decode       ir(one-hot) @ W_decode -> control vector   (Linear / control ROM)
    MLPTable     alu: one-hot (op,a,b) @ W_alu -> 8-bit result (Linear, exact)
    Unembed      bit vector -> integer token value (exact)

Address map:  0..255 data/program memory, 256..319 data stack slots,
              320..351 return stack slots, 352 REG_PC, 353 REG_SP, 354 REG_RP,
              355 REG_HALT, 511 SINK.
Everything (pc, sp, stacks, memory) lives ONLY in the trace KV. The driver
sequences instructions and dispatches microprograms (Qwenputer-style v1);
all state reads/writes go through the blocks above.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DIRECT = __package__ in (None, "")
if _DIRECT:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "components.vm"

from .core import DEFAULT_CONFIG, Op, VMConfig, VMError
from .handlers import factory as default_handlers
from .vm import VM

ALU_OPS = (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD)
C_U8, C_U16, C_POP, C_PEEK, C_PUSH, C_ALU, C_MEM, C_BR, C_HALT, C_MP = range(10)
NCTRL = 10
MP = {name: i for i, name in enumerate([
    "nop", "halt", "push_imm", "pop", "dup", "drop", "swap", "over", "rot",
    "alu", "load", "store", "jmp", "jz", "call", "ret",
])}

MEM0, STK0, RST0 = 0, 256, 320
REG_PC, REG_SP, REG_RP, REG_HALT = 352, 353, 354, 355
SINK = 511
NADDR = 512
BIG = 1.0e6  # fp64 exact for integers < 2^53
WEIGHTS_PATH = Path(__file__).with_name("cpu_weights.npz")


def _bits8(x: int) -> np.ndarray:
    return np.array([(x >> i) & 1 for i in range(8)], dtype=np.float64)


# ---------------- blocks ----------------

class Embed:
    """Fixed one-hot embedding table for the 512-cell address space."""
    def __init__(self) -> None:
        self.E_addr = np.eye(NADDR, dtype=np.float64)

    def addr_vec(self, addr: int) -> np.ndarray:
        return self.E_addr[addr]


class HardAttention:
    """Trace-token KV cache. read(addr) = exact latest-write lookup, expressed
    as one-hot QK^T scoring + argmax (no softmax)."""
    def __init__(self, embed: Embed) -> None:
        self.embed = embed
        self.kv_addr: list[int] = []
        self.kv_val: list[int] = []
        self.pos = 0

    def write(self, addr: int, val: int) -> None:
        self.kv_addr.append(addr)
        self.kv_val.append(val)
        self.pos += 1

    def read(self, addr: int) -> int:
        # scores_t = <q, k_t> + t, where q = k_t = one-hot(addr_t) * BIG
        if not self.kv_addr:
            return 0
        same = np.fromiter((1.0 if a == addr else 0.0 for a in self.kv_addr),
                           dtype=np.float64, count=len(self.kv_addr))
        if not same.any():
            return 0  # never written -> memory cell is 0
        scores = same * BIG + np.arange(len(self.kv_addr), dtype=np.float64)
        return int(self.kv_val[int(np.argmax(scores))])

    def __len__(self) -> int:
        return len(self.kv_addr)


class Decode:
    """Linear control ROM: one-hot opcode -> control vector."""
    def __init__(self, W: np.ndarray) -> None:
        self.W = W

    def __call__(self, ir: int) -> np.ndarray:
        x = np.zeros(256, dtype=np.float64)
        x[ir] = 1.0
        return x @ self.W


class MLPTable:
    """ALU as an exact Linear lookup table: one-hot (op,a,b) -> 8-bit result."""
    def __init__(self, W: np.ndarray) -> None:
        self.W = W

    def __call__(self, sel: int, a: int, b: int) -> int:
        idx = sel * 65536 + (a & 0xFF) * 256 + (b & 0xFF)
        x = np.zeros(len(ALU_OPS) * 65536, dtype=np.float64)
        x[idx] = 1.0
        bits = x @ self.W
        return int(sum(int(bit) << i for i, bit in enumerate(bits)))


def unembed_bits(bits: np.ndarray) -> int:
    """Exact integer decode of an 8-bit vector (unembedding)."""
    return int(sum(int(b) << i for i, b in enumerate(bits)))


# ---------------- compiled CPU ----------------

class NeuralCPU:
    def __init__(self, config: VMConfig | None = None,
                 W_decode: np.ndarray | None = None,
                 W_alu: np.ndarray | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        if W_decode is None or W_alu is None:
            W_decode, W_alu = self.compile_weights()
        self.W_decode, self.W_alu = W_decode, W_alu
        self.embed = Embed()
        self.attn = HardAttention(self.embed)
        self.decode = Decode(W_decode)
        self.alu = MLPTable(W_alu)
        self.halted = False
        self.steps = 0

    # -------- weight compilation (computed from ISA, not trained) --------
    @staticmethod
    def compile_weights() -> tuple[np.ndarray, np.ndarray]:
        dec = np.zeros((256, NCTRL), dtype=np.float64)
        for op in Op:
            row = dec[int(op)]
            if op is Op.NOP:   row[C_MP] = MP["nop"]
            elif op is Op.HALT:
                row[C_HALT] = 1; row[C_MP] = MP["halt"]
            elif op is Op.PUSH:
                row[C_U8] = 1; row[C_PUSH] = 1; row[C_MP] = MP["push_imm"]
            elif op is Op.POP:  row[C_POP] = 1; row[C_MP] = MP["pop"]
            elif op is Op.DUP:
                row[C_PEEK] = 1; row[C_PUSH] = 1; row[C_MP] = MP["dup"]
            elif op is Op.DROP: row[C_POP] = 1; row[C_MP] = MP["drop"]
            elif op is Op.SWAP:
                row[C_POP] = 2; row[C_PUSH] = 2; row[C_MP] = MP["swap"]
            elif op is Op.OVER:
                row[C_PEEK] = 1; row[C_PUSH] = 1; row[C_MP] = MP["over"]
            elif op is Op.ROT:
                row[C_POP] = 3; row[C_PUSH] = 3; row[C_MP] = MP["rot"]
            elif op in ALU_OPS:
                row[C_POP] = 2; row[C_PUSH] = 1
                row[C_ALU] = ALU_OPS.index(op); row[C_MP] = MP["alu"]
            elif op is Op.LOAD:
                row[C_POP] = 1; row[C_PUSH] = 1; row[C_MEM] = 1; row[C_MP] = MP["load"]
            elif op is Op.STORE:
                row[C_POP] = 2; row[C_MEM] = 2; row[C_MP] = MP["store"]
            elif op is Op.JMP:
                row[C_U16] = 1; row[C_BR] = 1; row[C_MP] = MP["jmp"]
            elif op is Op.JZ:
                row[C_U16] = 1; row[C_POP] = 1; row[C_BR] = 2; row[C_MP] = MP["jz"]
            elif op is Op.CALL:
                row[C_U16] = 1; row[C_BR] = 3; row[C_MP] = MP["call"]
            elif op is Op.RET:
                row[C_BR] = 4; row[C_MP] = MP["ret"]
            else:
                row[C_MP] = -1.0  # ISA v2 op: not in neural v1 weights (M4 recompiles)

        alu = np.zeros((len(ALU_OPS) * 65536, 8), dtype=np.float64)
        grid = np.arange(65536)
        a = (grid >> 8) & 0xFF
        b = grid & 0xFF
        safe_b = np.where(b == 0, 1, b)
        for sel, op in enumerate(ALU_OPS):
            if op is Op.ADD: r = (a + b) & 0xFF
            elif op is Op.SUB: r = (a - b) & 0xFF
            elif op is Op.MUL: r = (a * b) & 0xFF
            elif op is Op.DIV: r = np.where(b == 0, 0, a // safe_b)
            elif op is Op.MOD: r = np.where(b == 0, 0, a % safe_b)
            for bit in range(8):
                alu[sel * 65536 + grid, bit] = (r >> bit) & 1
        return dec, alu

    # -------- persistence --------
    def save_weights(self, path: Path | None = None) -> Path:
        p = path or WEIGHTS_PATH
        np.savez_compressed(p, W_decode=self.W_decode, W_alu=self.W_alu,
                            meta=np.array([NCTRL]))
        return p

    @classmethod
    def from_file(cls, path: Path | None = None, config: VMConfig | None = None):
        p = path or WEIGHTS_PATH
        z = np.load(p)
        return cls(config=config, W_decode=z["W_decode"], W_alu=z["W_alu"])

    # -------- run --------
    def load(self, program: bytes, base: int = 0) -> None:
        self.attn = HardAttention(self.embed)
        self.halted = False
        self.steps = 0
        self.attn.write(REG_PC, base)
        self.attn.write(REG_SP, 0)
        self.attn.write(REG_RP, 0)
        self.attn.write(REG_HALT, 0)
        for i, byte in enumerate(program):
            self.attn.write(base + i, byte)

    def _read(self, addr: int) -> int:
        return self.attn.read(addr)

    def _write(self, addr: int, val: int) -> None:
        self.attn.write(addr, val)

    def step(self) -> None:
        pc0 = self._read(REG_PC)
        ir = self._read(pc0)
        ctrl = self.decode(ir)
        mp = int(ctrl[C_MP])
        if mp < 0:
            raise VMError(f"{op.name}: not in neural v1 weights (ISA v2; recompile M4)")
        nxt = pc0 + (3 if ctrl[C_U16] else (2 if ctrl[C_U8] else 1))
        stored_cell: int | None = None

        def push(v: int) -> None:
            sp = self._read(REG_SP)
            if sp >= self.config.stack_depth:
                raise VMError("data stack overflow")
            self._write(STK0 + sp, v)
            self._write(REG_SP, sp + 1)

        def pop() -> int:
            sp = self._read(REG_SP)
            if sp == 0:
                raise VMError("data stack underflow")
            v = self._read(STK0 + sp - 1)
            self._write(REG_SP, sp - 1)
            return v

        def peek(k: int) -> int:
            sp = self._read(REG_SP)
            if sp <= k:
                raise VMError("data stack underflow")
            return self._read(STK0 + sp - 1 - k)

        if ctrl[C_HALT]:
            self._write(REG_PC, pc0 + 1)
            self._write(REG_HALT, 1)
            self.halted = True
            self.steps += 1
            self._last_stored = None
            return
        if mp == MP["push_imm"]:
            push(self._read(pc0 + 1))
        elif mp == MP["pop"]:
            pop()
        elif mp == MP["dup"]:
            push(peek(0))
        elif mp == MP["drop"]:
            pop()
        elif mp == MP["swap"]:
            x = pop(); y = pop()
            push(x); push(y)
        elif mp == MP["over"]:
            push(peek(1))
        elif mp == MP["rot"]:
            c = pop(); b = pop(); a = pop()
            push(b); push(c); push(a)
        elif mp == MP["alu"]:
            b = pop(); a = pop()
            push(self.alu(int(ctrl[C_ALU]), a, b))
        elif mp == MP["load"]:
            push(self._read(pop()))
        elif mp == MP["store"]:
            val = pop(); addr = pop()
            self._write(addr, val)
            stored_cell = addr
        elif mp == MP["jmp"]:
            nxt = self._read(pc0 + 1) | (self._read(pc0 + 2) << 8)
        elif mp == MP["jz"]:
            target = self._read(pc0 + 1) | (self._read(pc0 + 2) << 8)
            if pop() == 0:
                nxt = target
        elif mp == MP["call"]:
            target = self._read(pc0 + 1) | (self._read(pc0 + 2) << 8)
            rp = self._read(REG_RP)
            if rp >= self.config.return_depth:
                raise VMError("return stack overflow")
            self._write(RST0 + rp, pc0 + 3)
            self._write(REG_RP, rp + 1)
            nxt = target
        elif mp == MP["ret"]:
            rp = self._read(REG_RP)
            if rp == 0:
                raise VMError("return stack underflow")
            self._write(REG_RP, rp - 1)
            nxt = self._read(RST0 + rp - 1)
        # nop: nothing
        self._write(REG_PC, nxt)
        self.steps += 1
        self._last_stored = stored_cell

    def run(self, max_steps: int = 100_000) -> None:
        while not self.halted and self.steps < max_steps:
            self.step()
        if not self.halted:
            raise VMError(f"step limit exceeded: {max_steps}")

    # state views reconstructed PURELY through attention (no mirrors)
    def stack_view(self) -> list[int]:
        sp = self._read(REG_SP)
        return [self._read(STK0 + i) for i in range(sp)]

    def rstack_view(self) -> list[int]:
        rp = self._read(REG_RP)
        return [self._read(RST0 + i) for i in range(rp)]

    def mem_bytes(self) -> bytes:
        return bytes(self._read(a) for a in range(256))


# ---------------- differential test ----------------

def diff_against_reference(program: bytes, name: str, config: VMConfig | None = None) -> None:
    cfg = config or DEFAULT_CONFIG
    ref = VM(config=cfg, handlers=default_handlers, mem=bytearray(cfg.mem_size))
    ref.memory[: len(program)] = program
    neu = NeuralCPU.from_file()
    neu.load(program)

    while not ref.halted:
        ref.step()
        neu.step()
        assert neu._read(REG_PC) == ref.pc, f"{name}: pc"
        assert neu.halted == ref.halted, f"{name}: halted"
        assert neu.stack_view() == list(ref.stack), (
            f"{name}: stack {neu.stack_view()} != {ref.stack}")
        assert neu.rstack_view() == list(ref.rstack), f"{name}: rstack"
        if neu._last_stored is not None:
            assert neu._read(neu._last_stored) == ref.memory[neu._last_stored & 0xFF], \
                f"{name}: stored cell"
    assert neu.mem_bytes() == bytes(ref.memory[:256]), f"{name}: final mem"
    print(f"{name:16s} cycle-exact for {ref.steps} instructions "
          f"(trace tokens: {len(neu.attn)})")


def main() -> None:
    from .demos import fib_program, gcd_program, bubble_sort_program

    # 1) compile from ISA and PERSIST weights to disk
    cpu = NeuralCPU()
    path = cpu.save_weights()
    size_mb = path.stat().st_size / 1e6
    print(f"weights saved: {path}  ({size_mb:.1f} MB)")
    print(f"  W_decode {cpu.W_decode.shape} + W_alu {cpu.W_alu.shape} "
          f"= {cpu.W_decode.size + cpu.W_alu.size:,} params, all computed, 0 trained")

    # 2) load from disk and prove cycle-exactness on all demos
    diff_against_reference(fib_program(10), "fib(10)")
    diff_against_reference(gcd_program(84, 18), "gcd(84,18)")
    diff_against_reference(bubble_sort_program([5, 2, 4, 1, 3]), "bubble_sort")

    # 3) same on-disk weights, new program
    neu = NeuralCPU.from_file()
    neu.load(fib_program(11))
    neu.run()
    print("fib(11) with on-disk weights =", neu.stack_view(), "(expect [89])")
    assert neu.stack_view() == [89]

    print("neural_cpu OK: ISA -> transformer blocks -> weights on disk, "
          "cycle-exact vs reference")


if __name__ == "__main__":
    main()
    if _DIRECT and sys.stdin.isatty():
        input("\n按回车键退出...")