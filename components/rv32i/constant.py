"""rv32i 的常量真源：跨模块重复的、跨层镜像的，都在这里写一次。

地址图（32 位空间切成四段，段间空洞不算段）：

    0x00000  RAM   32 KB    数据 / 堆 / 栈        heap_base 0x100、stack_top 0x8000
    0x08000  文本  2560 B   `<cpu_call>` 的 token，只读数据
    0x20000  DROM  128 KB   `.rodata`，只读
    0x40000  ROM   256 KB   代码，取指专用（哈佛：只有这一段可取指）

本文件不 import 任何项目模块（`core` / `isa` / `elf` 直接取用，无循环导入）。
"""

# ---------------- 字宽 ----------------

WORD_BITS = 32
WORD_MASK = (1 << WORD_BITS) - 1        # 0xFFFFFFFF。`core` / `isa` / `factory` 共用这一份

# ---------------- 地址图 ----------------

RAM_SIZE = 0x08000          # 32 KB：数据段 / 堆 / 栈
DROM_BASE = 0x20000         # 只读数据段（.rodata）起点
DROM_SIZE = 0x20000
ROM_BASE = 0x40000          # 代码段：取指专用，无写口
ROM_SIZE = 0x40000

HEAP_BASE = 0x00100         # 堆起点（低端保留给中断向量之类的扩展位）
STACK_TOP = 0x08000         # 栈顶 = RAM 顶端，向下长

#: `<cpu_call>` 的文本缓冲。地址就是 RAM 上界之上那块，`TEXT_BASE == STACK_TOP` 不冲突
#: （栈向下长）。行号 **1-based**：第 k 个捕获的 token 写第 k 行，第 0 行空着。
TEXT_BASE = 0x08000
TEXT_ROWS = 160             # 行号 = EA[4:12]（8 位），有效行 1..159
TEXT_ROW = 16               # 每行字节数 = 单个 token 的最坏 UTF-8 长度
CAP_MAX = 128               # 协议上限

SECTOR_SIZE = 512           # 磁盘扇区字节数
DISK_SECTORS = 4096         # 2 MB 磁盘

# ---------------- 派生（地址图算出来的，别另写一份） ----------------

RAM_WORDS = RAM_SIZE // 4               # 8192
DROM_WORDS = DROM_SIZE // 4             # 16384
ROM_WORDS = ROM_SIZE // 4               # 65536

DROM_END = DROM_BASE + DROM_SIZE        # 0x40000，正好接上 ROM
ROM_END = ROM_BASE + ROM_SIZE           # 0x80000

TEXT_SIZE = TEXT_ROWS * TEXT_ROW        # 2560 B
TEXT_END = TEXT_BASE + TEXT_SIZE        # 0x08A00，离 drom_base 还空 105 KB
TEXT_WORDS = TEXT_ROW // 4              # 每行 4 个字

# ---------------- 自洽（导入即验，写错会在 import 时就炸） ----------------

assert TEXT_BASE == STACK_TOP, "文本缓冲压到了栈顶以下"
assert TEXT_END <= DROM_BASE, "文本缓冲压到 DROM 了"
assert CAP_MAX < TEXT_ROWS, f"cap_max={CAP_MAX} 装不进 {TEXT_ROWS} 行的文本缓冲"
assert TEXT_ROW % 4 == 0 and TEXT_WORDS == 4, "一行的字节数必须是 4 的倍数（按字读）"
