"""
demos.py: nontrivial stack-machine demo programs.

Run:  python -m components.vm.demos   (from project root)  or double-click.

Conventions used throughout:
- STORE takes  ... addr value ;  (value on top)
- LOAD takes   ... addr ;
- Only conditional is JZ (jump if popped value == 0). "JNZ" is emulated as
  push cond / jz skip / jmp target / skip:
- All data cells live at mem[100+] / mem[200+], past the code (von Neumann).
- 8-bit cells: fib capped so results stay < 256.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DIRECT = __package__ in (None, "")
if _DIRECT:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "components.vm"

from .asm import make_asm
from .run import run_program
from .core import Op


def fib_program(n: int) -> bytes:
    """a,b = 0,1; repeat n times: a,b = b,a+b; return a.  mem[100]=n,101=a,102=b,103=t"""
    a = make_asm()
    a.push(100).push(n).op(Op.STORE)          # mem[100] = n
    a.push(101).push(0).op(Op.STORE)          # a = 0
    a.push(102).push(1).op(Op.STORE)          # b = 1
    a.label("loop")
    a.push(100).op(Op.LOAD)
    a.jz("done")                              # n == 0 -> done
    # t = a + b      (mem[103])
    a.push(103)                               # [103]
    a.push(101).op(Op.LOAD)                   # [103, a]
    a.push(102).op(Op.LOAD)                   # [103, a, b]
    a.op(Op.ADD)                              # [103, a+b]
    a.op(Op.STORE)                            # mem[103] = a+b
    # a = b
    a.push(101)                               # [101]
    a.push(102).op(Op.LOAD)                   # [101, b]
    a.op(Op.STORE)                            # mem[101] = b
    # b = t
    a.push(102)                               # [102]
    a.push(103).op(Op.LOAD)                   # [102, t]
    a.op(Op.STORE)                            # mem[102] = t
    # n = n - 1
    a.push(100).op(Op.DUP).op(Op.LOAD)        # [100, n]
    a.push(1).op(Op.SUB)                      # [100, n-1]
    a.op(Op.STORE)
    a.jmp("loop")
    a.label("done")
    a.push(101).op(Op.LOAD).op(Op.HALT)        # return a
    return a.build()


def fib_iter(n: int) -> int:
    vm, _ = run_program(fib_program(n))
    return vm.result()


def gcd_program(x: int, y: int) -> bytes:
    """while b != 0: a, b = b, a % b; return a.  mem[110]=a,111=b,112=t"""
    a = make_asm()
    a.push(110).push(x).op(Op.STORE)          # mem[110] = x
    a.push(111).push(y).op(Op.STORE)          # mem[111] = y
    a.label("loop")
    a.push(111).op(Op.LOAD)
    a.jz("done")                              # b == 0 -> done
    # t = a % b
    a.push(112)                               # [112]
    a.push(110).op(Op.LOAD)                   # [112, a]
    a.push(111).op(Op.LOAD)                   # [112, a, b]
    a.op(Op.MOD)                              # [112, a%b]
    a.op(Op.STORE)                            # mem[112] = t
    # a = b
    a.push(110)
    a.push(111).op(Op.LOAD)
    a.op(Op.STORE)
    # b = t
    a.push(111)
    a.push(112).op(Op.LOAD)
    a.op(Op.STORE)
    a.jmp("loop")
    a.label("done")
    a.push(110).op(Op.LOAD).op(Op.HALT)       # return a
    return a.build()


def gcd_euclid(x: int, y: int) -> int:
    vm, _ = run_program(gcd_program(x, y))
    return vm.result()


def bubble_sort_program(vals: list[int]) -> bytes:
    """Ascending bubble sort of vals (values must be 0..255).
    mem[200..204]=array, 210=i, 211=j, 212=t1, 213=t2, 214=tmp.
    No less-than instruction: a<=b tested by countdown (t1,t2 both decrement,
    whichever hits 0 first decides)."""
    base = 200
    a = make_asm()
    # init array
    for k, v in enumerate(vals):
        a.push(base + k).push(v).op(Op.STORE)
    # i = len-1  (outer)
    a.push(210).push(len(vals) - 1).op(Op.STORE)
    a.label("loop_i")
    a.push(210).op(Op.LOAD)
    a.jz("done")                              # i == 0 -> done
    # j = 0
    a.push(211).push(0).op(Op.STORE)
    a.label("loop_j")
    # if i - j == 0 -> next i
    a.push(210).op(Op.LOAD)
    a.push(211).op(Op.LOAD)
    a.op(Op.SUB)
    a.jz("next_i")
    # t1 = arr[j]
    a.push(212)                               # [212]
    a.push(base).push(211).op(Op.LOAD).op(Op.ADD)  # [212, &arr[j]]
    a.op(Op.LOAD)                             # [212, arr[j]]
    a.op(Op.STORE)                            # mem[212] = arr[j]
    # t2 = arr[j+1]
    a.push(213)
    a.push(base).push(211).op(Op.LOAD).op(Op.ADD).push(1).op(Op.ADD)
    a.op(Op.LOAD)
    a.op(Op.STORE)                            # mem[213] = arr[j+1]
    # countdown compare: t1 hits 0 -> a<=b -> no swap ; t2 hits 0 -> a>b -> swap
    a.label("cmp")
    a.push(212).op(Op.LOAD)
    a.jz("no_swap")
    a.push(213).op(Op.LOAD)
    a.jz("do_swap")
    a.push(212).op(Op.DUP).op(Op.LOAD).push(1).op(Op.SUB).op(Op.STORE)
    a.push(213).op(Op.DUP).op(Op.LOAD).push(1).op(Op.SUB).op(Op.STORE)
    a.jmp("cmp")
    a.label("do_swap")
    # tmp = arr[j]
    a.push(214)                               # [214]
    a.push(base).push(211).op(Op.LOAD).op(Op.ADD)
    a.op(Op.LOAD)                             # [214, arr[j]]
    a.op(Op.STORE)                            # mem[214] = arr[j]
    # arr[j] = arr[j+1]
    a.push(base).push(211).op(Op.LOAD).op(Op.ADD)              # [&arr[j]]
    a.push(base).push(211).op(Op.LOAD).op(Op.ADD).push(1).op(Op.ADD)
    a.op(Op.LOAD)                                              # [&arr[j], arr[j+1]]
    a.op(Op.STORE)
    # arr[j+1] = tmp
    a.push(base).push(211).op(Op.LOAD).op(Op.ADD).push(1).op(Op.ADD)  # [&arr[j+1]]
    a.push(214).op(Op.LOAD)                                    # [.., tmp]
    a.op(Op.STORE)
    a.label("no_swap")
    # j = j + 1
    a.push(211).op(Op.DUP).op(Op.LOAD).push(1).op(Op.ADD).op(Op.STORE)
    a.jmp("loop_j")
    a.label("next_i")
    # i = i - 1
    a.push(210).op(Op.DUP).op(Op.LOAD).push(1).op(Op.SUB).op(Op.STORE)
    a.jmp("loop_i")
    a.label("done")
    for k in range(len(vals)):
        a.push(base + k).op(Op.LOAD)
    a.op(Op.HALT)
    return a.build()


def bubble_sort(vals: list[int]) -> list[int]:
    vm, _ = run_program(bubble_sort_program(vals))
    return vm.stack


def main() -> None:
    f = fib_iter(10)
    print(f"fib_iter(10)      = {f}")         # 55
    assert f == 55, f

    g = gcd_euclid(84, 18)
    print(f"gcd_euclid(84,18) = {g}")         # 6
    assert g == 6, g

    arr = [5, 2, 4, 1, 3]
    s = bubble_sort(arr)
    print(f"bubble_sort       = {s}")         # [1, 2, 3, 4, 5]
    assert s == [1, 2, 3, 4, 5], s

    print("demos passed")


if __name__ == "__main__":
    main()
    if _DIRECT and sys.stdin.isatty():
        input("\n按回车键退出...")