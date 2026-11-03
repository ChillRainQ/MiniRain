"""rv32i 的常量真源：跨模块重复的、跨层镜像的，都在这里写一次。

地址图（32 位空间切成四段，段间空洞不算段）：

    0x00000  RAM   32 KB    数据 / 堆 / 栈        heap_base 0x100、stack_top 0x8000
    0x08000  文本  2560 B   `<cpu_call>` 的 token，固件只读、内容由电路写
    0x20000  DROM  128 KB   `.rodata`，只读
    0x40000  ROM   256 KB   代码，取指专用（哈佛：只有这一段可取指）

本文件不 import 任何项目模块（`core` / `isa` / `circuit` / `elf` 直接取用，无循环导入）。
`fw/nucpu` 的 `main.c` 与 `nucpu.ld` 没法 import，靠 `tests/probe_constants.py` 交叉校验。
"""

# ---------------- 字宽 ----------------

WORD_BITS = 32
WORD_MASK = (1 << WORD_BITS) - 1        # 0xFFFFFFFF。`core` / `isa` / `factory` 共用这一份

# ---------------- 地址图 ----------------

RAM_SIZE = 0x08000          # 32 KB：数据段 / 堆 / 栈
DROM_BASE = 0x20000         # 只读数据段（.rodata）。= 2^17，这个"2 的幂"是 `circuit.drom_table`
DROM_SIZE = 0x20000         # 能白借 RAM 的行选择与 bank 位的前提，不是随手取的
ROM_BASE = 0x40000          # 代码段。= 2^18，`circuit.link_word` 的 `ROM_BASE/4 = 0x10000` 同理
ROM_SIZE = 0x40000

HEAP_BASE = 0x00100         # 堆起点（低端保留给中断向量之类的扩展位）
STACK_TOP = 0x08000         # 栈顶 = RAM 顶端，向下长

#: `<cpu_call>` 的文本缓冲。地址就是 RAM 上界之上那块，`TEXT_BASE == STACK_TOP` 不冲突
#: （栈向下长）。行号 **1-based**：第 k 个捕获的 token 写第 k 行，第 0 行空着。
TEXT_BASE = 0x08000
TEXT_ROWS = 160             # 行号 = EA[4:12]（8 位），有效行 1..159
TEXT_ROW = 16               # 每行字节数 = 单个 token 的最坏 UTF-8 长度
CAP_MAX = 128               # 协议上限；电路容量 159，留 31 行余量

SECTOR_SIZE = 512           # 磁盘扇区字节数
DISK_SECTORS = 4096         # 2 MB 磁盘

# ---------------- 派生（地址图算出来的，别另写一份） ----------------

RAM_WORDS = RAM_SIZE // 4               # 8192
DROM_WORDS = DROM_SIZE // 4             # 16384
ROM_WORDS = ROM_SIZE // 4               # 65536：`J`/`R` 的 16 位载荷正是按它定的

DROM_END = DROM_BASE + DROM_SIZE        # 0x40000，正好接上 ROM
ROM_END = ROM_BASE + ROM_SIZE           # 0x80000

TEXT_SIZE = TEXT_ROWS * TEXT_ROW        # 2560 B
TEXT_END = TEXT_BASE + TEXT_SIZE        # 0x08A00，离 drom_base 还空 105 KB
TEXT_WORDS = TEXT_ROW // 4              # 每行 4 个字：`EA[2:4]` 在表里选一个

# ---------------- 表几何（电路侧） ----------------

#: ROM 分页：页内行 = 地址低 `PAGE_BITS` 位。选 8 不是数：ROM 阶段同时活着的 dim 是
#: `2^b`（行选择）+ `ceil(总行/2^b)`（页门），两项相等时最小；34294 条（现役 nucpu
#: 固件）的最优在 b≈8.0，旧固件 51241 条是 b≈7.8，两者都落在 8。
PAGE_BITS = 8
PAGE_ROWS = 1 << PAGE_BITS              # 一页 256 行
PAGE_LO = 4                             # 页号低 4 位解成 one-hot，高位直接取 cnt 的位线

COLS = WORD_BITS                        # 一个 ROM 字 32 位，每列一位

REGS = 32                               # 寄存器条数；x0 永远为 0（第 0 行只读不写）
REGBITS = WORD_BITS                     # 一个寄存器 32 位，位平面存

RAM_BANKS = 16                          # ram0..ram15：bank = 字索引 bit9..12（= 地址 bit11..14）
RAM_ROWS = RAM_WORDS // RAM_BANKS       # 每块 512 行 = 字索引低 9 位（行选择器与 ROM 页共
                                        # 用 `row_lits`，但行数不同：512 vs 256）
RAM_COLS = REGBITS                      # 整字按位平面存，一列一位

DROM_ROWS = RAM_ROWS                    # **必须是 512**：换成 ROM 的 256 行，行选择就借不到了
DROM_BANKS = RAM_BANKS                  # 上限 8192 字：再多一块就要 res[15]，同样借不到
DROM_COLS = REGBITS

#: 链接值有效位数。`(PC+4) >> 2 = 0x10000 + base + cnt + 1`，`cnt < 2^17`、`base < 2^17`，
#: 和 < `2^18`，18 位够（Qwenputer 存的是 16 位，因为它存的是 anchor）。
LINK_BITS = 18

# ---------------- token 层 ----------------

#: 数据家族 -> 载荷宽度。逐项照抄 Qwenputer 的 `spec.FAMILY_SIZES`；顺序即 token id 的
#: 排布，改动它会平移整个 vocab，不要动。
FAMILY_SIZES = [("V", 1 << 15), ("F0", 1 << 15), ("F1", 1 << 15), ("F2", 4),
                ("J", 1 << 16), ("R", 1 << 16), ("BYTE", 512), ("FLT", 8), ("HEX", 16),
                ("QS", 16), ("QD", 128), ("BD", 256),
                # 文本 token：载荷 = tokenizer 的 token id（`TEXT_VOCAB`）。MiniRain 的 token
                # 是芯片 trace 词表，而 Qwenputer 的 `is_text = t >= TEXT_TOKEN` 盖住它整个
                # Qwen 词表——这里必须自成一家族（`FAMILIES["TX"]`，见 `circuit.text_buffer`）。
                ("TX", 1 << 14)]

#: 控制 token：无载荷，一个名字就是一个 id（排在数据家族之后）。
#: `TCO`/`TCC` 是 "<cpu_call>" / "</cpu_call>" 的两个边界（tokenizer 里 id 34/35，宿主映过来），
#: 只夹住文本 burst，**不推进 PC**（所以不在 `FINAL` 里）。
CONTROL_NAMES = ["NT", "ST0", "ST1", "SS0", "SS1", "HALT", "GO", "SC", "TCO", "TCC"]

#: 数据家族 -> (起点, 宽度)。
FAMILIES: dict[str, tuple[int, int]] = {}
_base = 0
for _name, _size in FAMILY_SIZES:
    FAMILIES[_name] = (_base, _size)
    _base += _size
CTRL = _base                                        # 246700：控制 token 的起点
CONTROLS = {n: CTRL + i for i, n in enumerate(CONTROL_NAMES)}
VOCAB = CTRL + len(CONTROL_NAMES)                   # 246710
del _name, _size, _base

#: 终结 token：一条 record 到这里结束，状态在这一刻 commit，n 加一。
FINAL = {"F0", "F1", "F2", "J", "R", "BYTE", "NT", "ST0", "ST1", "SS0", "SS1", "HALT"}

#: 全部家族名，数据在前、控制在后；下标即 FAM_ID。
NAMES = [n for n, _ in FAMILY_SIZES] + CONTROL_NAMES
FAM_ID = {n: i for i, n in enumerate(NAMES)}
NFAM = len(NAMES)                                   # 23（13 个数据家族 + 10 个控制）
NPAY = 16                                           # 最宽载荷是 J/R 的 16 位

#: `<cpu_call>` 的文本词表：载荷上限 = TX 家族的宽度，与 `tokenizers/tokenizer.json` 一致。
TEXT_VOCAB = FAMILIES["TX"][1]                      # 16384

# ---------------- 自洽（导入即验，写错会在 import 时就炸） ----------------

#: vocab 是训练侧的契约（tokenizer + 数据集 + 权重一套），改家族宽度会平移整个 vocab。
assert VOCAB == 246710, f"vocab 变成 {VOCAB} 了（家族宽度被改动过？）"
assert TEXT_BASE == STACK_TOP, "文本缓冲压到了栈顶以下"
assert TEXT_END <= DROM_BASE, "文本缓冲压到 DROM 了"
assert CAP_MAX < TEXT_ROWS, f"cap_max={CAP_MAX} 装不进 {TEXT_ROWS} 行的文本缓冲"
assert TEXT_ROW % 4 == 0 and TEXT_WORDS == 4, "一行的字节数必须是 4 的倍数（按字读）"
assert RAM_ROWS & (RAM_ROWS - 1) == 0, "RAM 每块行数必须是 2 的幂（行选择与 DROM 借位都按它）"
assert ROM_WORDS == ROM_SIZE // 4 and ROM_WORDS <= 1 << 16, "J/R 的载荷只装得下 16 位字索引"
assert (DROM_BASE & (DROM_BASE - 1)) == 0, "DROM 基址必须是 2 的幂，否则借不到 RAM 的位线"
assert NPAY >= max(s for _, s in FAMILY_SIZES).bit_length() - 1, "载荷位不够"
