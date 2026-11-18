# FP8 最小训练原型（仅 train_bench）

## 范围

- 仅 `tests/train_bench.py --dtype fp8` 开启；默认仍为 bf16，模型主干、P1 训练入口及其默认配置没有改动。
- 使用现有 PyTorch 原生 `torch._scaled_mm`，没有安装或引入 torchao、Transformer Engine 等新依赖。
- E4M3FN、动态 per-tensor amax scaling。Linear 的前向、输入梯度 GEMM、权重梯度 GEMM 均实际使用 FP8，输出 BF16；不是仅模拟量化。
- FP32 主权重和原优化器保持不变，支持现有梯度累积。短卷积、attention 状态/打分、norm、embedding 保留原精度路径。LM head 不量化，logits 在 BF16 autocast 下计算，CE 保留原来的 FP32 稳定归约。
- GDN 的小型 a/b 投影（输出小于 16）保留 BF16；其余矩阵不对齐时补齐到 16，输出和梯度切回原尺寸。
- 处理 `FusedSwiGLU` 内部的 functional Linear，仍保留独立 gate/up 参数和 Muon 分组；参数对象、state_dict 名称和 tied embedding 不变。

## 关键代码

- `tests/fp8_linear.py`：`_quantize` 动态缩放；`_mm` 原生 FP8 GEMM；`_FP8Linear` 自定义反向；`enable_fp8` 仅转换指定 benchmark 实例；`probe_backend` 实测硬件/软件组合是否可运行。
- `tests/train_bench.py`：FP8 参数入口、BF16/FP8 初始 loss 对照、训练和 JSON/TXT 指标输出。
- `tests/test_fp8_linear.py`：真实 CUDA GEMM、补齐尺寸及梯度对照、全零输入、NaN/Inf 拒绝、FP32 主权重/tied 参数不变、BF16 logits 检查。

## 数值与执行限制

- 每次 GEMM 动态更新 amax，无延迟缩放历史。量化时夹到 E4M3 有限范围，全零输入的 scale 设置下限，避免除零。
- 量化输入和 GEMM 输出逐次检查 inf/nan；训练后还检查 loss、所有梯度和 FP32 主权重。发现非有限值直接报错。
- 自定义反向使用量化矩阵近似 BF16 梯度，不对 amax/离散量化过程求导；仅支持一阶梯度。
- **第一版 FP8 为 eager-only**：即使请求默认 compile，也会明确打印并记录 effective compile=False。BF16 的默认 compile 行为不变。当前严格检查包含 GPU→CPU 同步，因此不以加速为目标。
- auto 后端在安装 Liger 时照常选择 Liger；未安装时，FP8 原型选 native CE 以保留 BF16 logits，而不是使用显式 FP32 logits 的 chunked fallback。显式 chunked 在 FP8 原型中也切为 native 并打印说明。结果记录实际选择。
- RTX 4060 的 SM89 已经实际执行；5090 的 SM120 由启动时同样的运行探针验证，本机没有 SM120，不能声称已经实测。它还取决于具体 PyTorch/CUDA 构建。

## 本机实测

设备：RTX 4060 Laptop 8GB，SM89；PyTorch 2.7.1+cu128。小型 MiniRain 3,091,512 参数：hidden=128、4 层、heads=4/KV=2、S=128、microbatch=1、grad_accum=2、Muon+AdamW、native loss、eager，warmup=2、iters=5。FP8 和独立 BF16 基准使用相同 seed/config/data，计时包括前后向和优化器，不包括预热、初始数值对照。

| 指标 | BF16 | FP8 |
| --- | ---: | ---: |
| input tok/s | 2,379 | 1,442 |
| ms/update（256 tokens） | 107.60 | 177.59 |
| allocated 峰值 MiB | 95.0 | 93.5 |
| reserved 峰值 MiB | 552.0 | 552.0 |
| 最后训练 loss | 7.5005 | 7.5106 |
| loss、梯度有限 | 是 | 是 |

**当前 FP8 比 BF16 慢**，且小模型下没有显著显存收益；FP32 主权重、梯度和优化器状态没有压缩。不能把这个小模型数字当作完整 371M 的吞吐或容量结论。

FP8 运行在任何参数更新之前，对同一个模型的相同两份微批次分别计算 BF16 与 FP8 初始 loss：

- BF16：9.736114501953125
- FP8：9.734840393066406
- mean loss 绝对偏差：0.00127410888671875
- 相对偏差：0.0001308642，即约 0.01309%
- 单微批次最大绝对偏差：0.001617431640625

这只是初始随机模型数值对照，不是长训收敛或质量保证。4 项 FP8 单元测试、4 项已有 loss/梯度累积测试均通过；另跑默认 auto 的 FP8 冒烟也通过，实际 master 参数仍全为 FP32。

日志：`outputs/evaluation/fp8_train_smoke.log`、`fp8_bf16_baseline.log`、`fp8_auto_smoke.log`。完整 JSON/TXT 位于 `tests/txt/train_bench_20261118_141123_863700.*`、`train_bench_20261118_141252_726700.*`、`train_bench_20261118_141441_508800.*`。

## 使用

本机冒烟（FP8；BF16 对照将 dtype 改成 bf16，其他参数保持不变）：

```bash
uv run tests/train_bench.py --device cuda:0 --dtype fp8 --no-compile \
  --hidden 128 --layers 4 --heads 4 --kv-heads 2 \
  --train-batch 1 --train-seq 128 --grad-accum 2 \
  --warmup 2 --iters 5 --loss-backend native
```

5090 完整 371M 的首次尝试建议先用 microbatch=1，确认显存及数值后再加大：

```bash
uv run tests/train_bench.py --device cuda:1 --dtype fp8 --no-compile \
  --train-batch 1 --grad-accum 1 --train-seq 4096 --warmup 2 --iters 5
```

后续提速重点为融合 amax/量化、减少严格检查的同步和临时转置、量化缓存及 compile 接入；这次没有用省略安全检查或偷偷回退 BF16 GEMM 来制造更好的 FP8 数字。
