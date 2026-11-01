"""
interpreter.py: ISA-RV32 v1 参考解释器
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
_DIRECT = __package__ in (None, "")
if _DIRECT:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "components.rv32i"

from .core import RV32Config, RV32Trap, State
from .factory import factory
from .isa import assemble, decode


@dataclass
class Interpreter:
    cfg: RV32Config = field(default_factory=RV32Config)
    state: State = field(init=False)
    prog_end: int = 0
    halted: bool = False

    def __post_init__(self) -> None:
        self.state = State(self.cfg)

    def load(self, program: bytes, base: int | None = None) -> None:
        """程序载入 ROM 段，pc 指向入口。base 默认 rom_base。"""
        if base is None:
            base = self.cfg.rom_base
        self.state.mem.load(base, program)
        self.state.pc = base
        self.state.steps = 0
        self.prog_end = base + len(program)
        self.halted = False          # 重载即复位，不然上一次的 halt 会残留

    def step(self) -> None:
        st = self.state
        pc0 = st.pc
        word = st.mem.fetch_word(pc0)          # FETCH（容忍非对齐）
        ins = decode(word)                      # DECODE
        handler = factory.get_code(ins.name)
        if handler is None:
            raise RV32Trap(f"unimplemented {ins.name}")
        npc = handler(st, ins)                  # READ..WRITEBACK（factory）
        # 返回值决定下一条：None -> 顺序 +4；int -> 显式目标（含跳转到自身）
        st.pc = (npc if npc is not None else pc0 + 4) & self.cfg.word_mask
        st.steps += 1

    def run(self, max_steps: int = 1_000_000) -> None:
        """执行直到 pc 越界停机，或固件主动 ebreak/ecall 停机。"""
        from .factory import RV32Trap_placeholder
        while self.state.pc < self.prog_end:
            if self.state.steps >= max_steps:
                raise RV32Trap(f"step limit {max_steps}")
            try:
                self.step()
            except RV32Trap_placeholder:
                self.halted = True          # 固件退出约定：ebreak/ecall = HALT
                return


def run_asm(source: str, cfg: RV32Config | None = None) -> Interpreter:
    it = Interpreter(cfg or RV32Config())
    it.load(assemble(source))
    it.run()
    return it