from __future__ import annotations

from components.vm.core import *

AsmItem = bytes | tuple[str, str]


def u16le(x: int) -> bytes:
    return bytes((x & 0xFF, (x >> 8) & 0xFF))


class Asm:
    """Two-pass assembler. Items are bytes, ('label', name), or ('ref', name)."""

    def __init__(self) -> None:
        self.items: list[AsmItem] = []

    def emit(self, *bs: int) -> Asm:
        self.items.append(bytes(mask_word(b) for b in bs))
        return self

    def label(self, name: str) -> Asm:
        self.items.append(("label", name))
        return self

    def ref(self, name: str) -> Asm:
        self.items.append(("ref", name))
        return self

    def op(self, opcode: Op | int) -> Asm:
        return self.emit(int(opcode))

    def push(self, imm: int) -> Asm:
        return self.emit(int(Op.PUSH), imm)

    def jmp(self, label: str) -> Asm:
        return self.emit(int(Op.JMP)).ref(label)

    def jz(self, label: str) -> Asm:
        return self.emit(int(Op.JZ)).ref(label)

    def call(self, label: str) -> Asm:
        return self.emit(int(Op.CALL)).ref(label)

    def build(self) -> bytes:
        labels: dict[str, int] = {}
        addr = 0
        for item in self.items:
            if isinstance(item, bytes):
                addr += len(item)
            elif isinstance(item, tuple) and item[0] == "label":
                labels[item[1]] = addr
            elif isinstance(item, tuple) and item[0] == "ref":
                addr += 2
            else:
                raise ValueError(f"bad item {item!r}")

        out = bytearray()
        for item in self.items:
            if isinstance(item, bytes):
                out.extend(item)
            elif isinstance(item, tuple) and item[0] == "ref":
                if item[1] not in labels:
                    raise ValueError(f"undefined label {item[1]!r}")
                out.extend(u16le(labels[item[1]]))
            elif isinstance(item, tuple) and item[0] == "label":
                continue
            else:
                raise ValueError(f"bad item {item!r}")
        return bytes(out)


def make_asm() -> Asm:
    return Asm()