# MiniRain 预训练评估

## lm-eval 内置任务

通用入口直接加载 MiniRain `.pt` 或最终权重，也可传实验目录。对比中文故事续写判断和两项基础能力任务：

```bash
uv run --extra chid bench/lm_eval_pretrain.py --device cuda:1 \
  --ckpt exp/pretrain_text_004/pretrain/step_00011000.pt \
         exp/pretrain_text_004/pretrain/step_00017000.pt \
  --tasks xstorycloze_zh cmmlu_elementary_chinese cmmlu_elementary_commonsense \
  --output outputs/evaluation/004_base_bench.json
```

`--tasks` 可换成 `lambada_openai` 或其他 `lm-eval` 任务名。默认零样本，首次运行需要访问任务数据源。可用 `--limit 64` 先试跑每个任务，正式对比时去掉。加 `--log-samples` 可以把逐题结果也写入 JSON。

## CLUE ChID（lm-eval）

从 [CLUE ChID 数据包](https://storage.googleapis.com/cluebenchmark/tasks/chid_public.zip) 解压 `dev.json` 和 `dev_answer.json` 到 `data/chid/`，或用 `--data-dir` 指向已有的解压目录。使用项目中已有的 `lm-eval`，从仓库根目录运行：

```bash
uv run --extra chid bench/chid.py --device cuda:1 \
  --ckpt exp/pretrain_text_004/pretrain/step_00011000.pt \
         exp/pretrain_text_004/pretrain/step_00017000.pt \
  --output outputs/evaluation/chid_004.json
```

可加 `--limit 64` 冒烟测试。默认跑完 dev 的所有空格。每个空格由 `lm-eval` 独立评分，比较候选成语及其后文在给定前文下的总对数概率。前文和候选分别分词，BOS 加在前文前，避免不同候选改变前文的分词结果。其他空格保留为占位符，不使用正确答案，也不做跨空格唯一匹配。报告的 `acc` 适合按此协议对比同系列检查点。

## 中文自由续写诊断

在仓库根目录指定要评估的检查点。训练中途可以使用具体的 `step_*.pt` 做可复现的对照，也可以传实验目录，由脚本按现有 eval 规则选权重。

```bash
uv run bench/pretrain.py --device cuda:1 \
  --ckpt exp/pretrain_text_004/pretrain/step_00005000.pt \
         exp/pretrain_text_004/pretrain/step_00011000.pt \
         exp/pretrain_text_004/pretrain/step_00017000.pt \
  --output outputs/evaluation/pretrain_text_004.json
```

每个检查点使用相同样例和贪心解码设置。默认生成最多 64 token，CUDA 默认 BF16，CPU 默认 FP32。省略 `--output` 时逐条打印续写和分数。

- **Reference PPL** 衡量在给定前缀时预测参考续写的难度，只计参考后缀，跨样例按 token 加权。评估数据和计分范围与训练 Val 不同。
- **Reference preferred** 统计参考续写的每 token 平均损失低于复述候选的样例数。报告另存平局数和分差，正分差表示更偏好参考续写。前缀和候选分别分词，以便两个候选使用相同的前缀 token。
- **Loops** 统计生成部分出现连续三次完全相同、每次 4–32 token 的片段。JSON 记录首次重复片段的起点（从 1 开始）和原始续写。改写式重复、复述一次和跑题需要结合文本判断。

内置 20 条人工编写的中文诊断样例，覆盖科普、叙事、公告和操作说明，适合追踪同一训练的检查点。要评估更广的分布，可以用 `--cases` 传入自己的训练外 JSONL，每行包含 `id`、`prefix`、`reference` 和 `repeat` 四个字符串字段。跨实验对照时保持样例、分词器、精度和生成长度一致。
