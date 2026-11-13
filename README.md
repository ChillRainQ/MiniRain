# MiniRain-Vision

MiniRain-Vision 是一个从零实现的轻量级视觉语言模型（VLM）。其核心 **MiniRainVLM** 采用**单流统一架构**：不使用独立的视觉编码器 + 跨注意力，而是将图像 patch 作为普通 token 直接注入语言模型的同一序列，由一个 Transformer 主干统一处理文本与图像。

## 模型架构

<div align="center">
  <img src="docs/arch.svg" alt="MiniRain 架构图" width="900"/>
</div>

### 核心组件说明

| 组件 | 作用 | 关键设计 |
|------|------|----------|
| **输入融合** | 文本/图像统一序列 | patch 嵌入 in-place 覆盖 `image_pad` 占位，无独立视觉塔 |
| **2D RoPE** | 位置编码 | 图像 (行,列) 双轴，文本 (t,t) 退化 1D，支持 YaRN 插值 |
| **MoT-GQA** | 注意力（可选） | Q/K/V/O 按模态分 expert，注意力全局共享，纯文本零额外路由开销；`use_mot=True` 时注意力与 MLP 均接入 MoT |
| **BlockAttnRes** | 跨块信息聚合 | 可学习加权历史块尾 + 当前输入，替代恒等残差 |
| **Mask** | 注意力约束 | 文本因果 \| 图像段内双向 \| 图与图之间因果可见 |
| **MLP** | 非线性变换 | SwiGLU（dense）或 LosslessMoE（无辅助损失 MoE） |

### 单流多模态序列

图像经 `Conv2d(patch_size)` 切成 patch 后展平（行优先），覆盖序列中 `<|image_pad|>` 占位位置的文本嵌入，与文本 token 一起走同一条主干：

```
BOS  <|vision_start|> [patch × R*C + 行换行符] <|vision_end|>  文本...  EOS
```

loss 只落在文本 + EOS 上，图像段全部为上下文。

### 2D RoPE（支持 YaRN）

- 图像 patch(i,j) 的位置编码为 `(base+i, base+j)`，行/列各占 head_dim 的一半
- 文本位置退化为 `(t, t)`，即标准 1D RoPE
- 换行符位于 `(base+i, base+C)`，宽图按 `max(R,C)` 推进游标，避免列坐标与后续文本碰撞
- 文本 1D 频率表支持 YaRN 插值（`inference_rope_scaling=True` 启用）；图像 2D 表不做插值

### MoT 注意力（Mixture-of-Transformers）

Q/K/V/O 四个投影按模态解耦为文本/图像两套 expert，注意力计算全局共享。纯文本输入时 `modality_mask=None`，全体直接走文本 expert，**零额外开销退化为普通 GQA**。

`BlockAttnResBlock` 会在 `use_mot=True`（`type_id=3`）时选择 `MoTGroupQueryAttention` 和 `MoTFFN`；普通 Dense/MoE 配置仍使用共享的 GQA 投影。

### 图像双向注意力

注意力 mask = `因果 | 同图段`：文本保持因果，同一张图内的 patch 之间双向可见，padding 只屏蔽 key 维。不同图段之间不做隔离——排在后面的图能因果地看到前面的图。

### 块注意力残差（BlockAttnRes）

块内（`block_size//2` 层为一组）保留经典残差；块间用可学习聚合替代恒等捷径——每层的 attn/mlp 输入 = 对所有前序块尾（含嵌入输出）与当前输入的加权混合，权重由 `attn_res_query` 打分产生（零初始化时等价于均匀平均）。

### 其余组件

- GroupQueryAttention（GQA）+ QK-RMSNorm
- SwiGLU（dense）或 LosslessMoE（`use_moe=True`，无辅助损失的偏置均衡路由 + 共享专家）
- RMSNorm、Flash Attention（`scaled_dot_product_attention`；带图像双向 mask 时退化为显式 mask 路径）
- KV cache 增量解码，generate 内置视觉结构 token 采样屏蔽

## 目录结构

```
architectures/   rain.py(模型主体) / config.py(RainConfig)
components/      attention.py(GQA / MoT-GQA) / block.py(残差块) / mlp.py(SwiGLU/MoE) / norm.py
dataset/         collator.py(VLMCollator 序列拼装+位置编号) / datasets.py
tokenizers/      BPE tokenizer(含 <|vision_start|> 等视觉 token)
train/           train_pre.py / train_sft.py / train_grpo.py / train_util.py
cli/             eval.py
ready/           预训练/SFT/RLAIF 检查点
tests/           测试套件(见下)
```

## 快速开始

```python
import torch
from architectures.config import RainConfig
from architectures.rain_v import MiniRainVForCausalLM

cfg = RainConfig(hidden_size=1024, n_hidden_layers=12, block_size=4,
                 n_attn_heads=8, n_key_value_heads=4)
model = MiniRainVForCausalLM(cfg).cuda()

# 纯文本
ids = torch.randint(3, 6000, (1, 32)).cuda()
out = model(input_ids=ids, labels=ids)

# 图文: 用 VLMCollator 拼序列(自动处理位置编号/截断保护)
from dataset.collator import VLMCollator
collator = VLMCollator(tokenizer, cfg)
batch = collator([{"image": img_tensor, "text": "描述这张图"}])

# 生成
gen = model.generate(inputs=ids, max_new_tokens=64, do_sample=False)
```

## P1 预训练

P1 使用预处理后的 `data/pretrain_bins/p1_text.bin` 与 `p1_text.idx`。默认实验目录为
`exp/pretrain_1024_20/p1`（默认实验名由 hidden-size 和 layers 决定），其中包含
`train.jsonl`、可精确续训的 `latest.pt` 和最终部署权重。
模型初始化时会先从 `init/ConvStem.pt` 加载 `student` 蒸馏权重并严格校验，然后冻结视觉
stem 和视觉专家；默认蒸馏权重对应 `hidden_size=1024`。

```powershell
python train/pretrain_p1.py --dry-run
python train/pretrain_p1.py --auto-resume
```

使用独立实验名时，产物会写入对应子目录：

```powershell
python train/pretrain_p1.py --exp-name pretrainxxxx --auto-resume
```

实验启动时调用 `train/ckpt.py` 的 `save_exp` 保存模型、训练、数据处理代码和依赖配置。
快照在实验根目录；阶段产物在 `p1/`。已有完整快照在续训时不会被当前代码覆盖：

```text
exp/pretrainxxxx/
  architectures/  components/  losses/  train/  dataset/  cli/  scripts/
  project.py  pyproject.toml  uv.lock  run_config.json
  p1/
    step_00001000.pt ... latest.pt
    p1_final_weights.pth
    train.jsonl  metrics.csv  training_report.html
    summary.json
```

`--output-dir` 指定实验根目录，脚本自动在其下创建 `p1/`。
旧布局的断点可用 `--resume exp/旧实验/latest.pt` 指定；历史指标需放到新阶段目录才能延续。

检查点默认每 1000 个优化器 step 保存一次，可用 `--save-every 500` 调整。
实验目录保留最近 5 个 `step_XXXXXXXX.pt` 完整断点，超过数量自动删除最旧的编号断点，
另外维护一份 `latest.pt` 供 `--auto-resume` 使用；训练结束时也保存一次。
最终部署权重 `p1_final_weights.pth` 不参与轮换。

训练指标自动保存到同一实验的 `p1/` 目录，不需要外部平台：

- `train.jsonl`：每个优化器 step 的原始指标，逐行刷新。
- `metrics.csv`：可用于自行分析的表格。
- `training_report.html`：直接用浏览器打开的离线曲线，无网络/CDN 依赖。
- `summary.json`：运行状态、最近指标和最低训练 loss。
- 实验根目录 `run_config.json`：本次启动参数、模型配置、设备、PyTorch 版本及参数量。

记录 loss、step/epoch、AdamW/Muon 学习率、裁剪前梯度范数、AMP scale、有效预测 token
数量、吞吐量、单步/累计耗时、预计剩余时间以及 rank 0 当前/单步峰值显存。
CSV、摘要和 HTML 在保存检查点及训练完成时刷新；`--log-every` 仅控制终端输出。
续训时保留断点及之前的记录，原 JSONL 会先备份为 `train.before-resume-*.jsonl`；
在同一目录启动全新训练也会备份原记录。ETA 基于本次启动的平均单步耗时，不含检查点开销。
