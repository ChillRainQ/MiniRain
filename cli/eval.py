# -*- coding: utf-8 -*-
"""MiniRain 预训练与 SFT 权重评估。

支持当前纯文本 MiniRain 的 final_weights.pth 裸权重及 latest.pt/step_*.pt
训练断点。实验目录内自动查找 sft/ 或 pretrain/ 权重，配置优先使用断点内保存的
config，裸权重从阶段目录或实验根目录的 model_config.json 还原。

用法:
    python cli/eval.py --ckpt pretrain_text_004 --device cuda:1 --prompt "机器学习是"
    python cli/eval.py --ckpt exp/pretrain_text_004/pretrain/latest.pt
    python cli/eval.py --ckpt exp/rainpocket_v1_sft --prompt "你是谁？"
    python cli/eval.py --ckpt pretrain_text_004 --interactive 1
    python cli/eval.py --ckpt pretrain_text_004 --use_cache 0
    python cli/eval.py --ckpt pretrain_text_004 --hf_generate 0
    python cli/eval.py --ckpt pretrain_text_004 --prompt "机器学习是" --logits_topk 10
    python cli/eval.py --ckpt pretrain_text_004 --prompt "机器学习是" --logits_save out.pt --logits_all 1
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, TextStreamer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import project
from architectures.config import RainConfig
from architectures.rain import MiniRainForCausalLM
from train.train_util import get_model_params, setup_seed

# 目录形式的 --ckpt 按此顺序找权重：阶段产物 > 断点 > 其它权重
WEIGHT_PATTERNS = ('final_weights.pth', '*_final_weights.pth', 'latest.pt', 'step_*.pt', '*weights*.pth', '*.pth')
DEFAULT_PROMPTS = [
    '你有什么特长？',
    '为什么天空是蓝色的',
    '请用Python写一个计算斐波那契数列的函数',
    '解释一下"光合作用"的基本过程',
    '如果明天下雨，我应该如何出门',
    '比较一下猫和狗作为宠物的优缺点',
    '解释什么是机器学习',
    '推荐一些中国的美食',
]


def find_weight(path: Path, *, prefer_sft: bool = False):
    """Resolve weights from an explicit file or an experiment directory."""
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    stages = [(path / 'pretrain', path), (path / 'sft',)]
    if prefer_sft:
        stages.reverse()
    for directories in stages:
        for pattern in WEIGHT_PATTERNS:
            found = [p for directory in directories for p in directory.glob(pattern) if p.is_file()]
            if found:
                weight = max(found, key=lambda p: p.stat().st_mtime)
                print(f'[Weight] {path} -> {weight.relative_to(path)}')
                return weight
    raise FileNotFoundError(f'{path} 及其 sft/、pretrain/ 下找不到权重，请用 --ckpt 指定 .pt 或 .pth 文件')


def read_model_config(weight_path: Path, config_path: Path | None):
    """查找模型配置；支持实验根目录、阶段目录和训练 JSON 引用。"""
    candidates = ([Path(config_path)] if config_path is not None else
                  [directory / name for directory in (weight_path.parent, weight_path.parent.parent)
                   for name in ('model_config.json', 'config.json', 'run_config.json', 'training_config.json')])
    for candidate in candidates:
        if not candidate.is_file():
            continue
        raw = json.loads(candidate.read_text(encoding='utf8'))
        stage = raw.get('stage')  # run_config.json 记录的训练阶段，如 p1 / p1_long_16k
        raw = raw.get('model_config', raw)
        if isinstance(raw, str):
            referenced = Path(raw)
            if not referenced.is_absolute():
                referenced = candidate.parent / referenced
            raw = json.loads(referenced.read_text(encoding='utf8'))
        if isinstance(raw, dict) and 'hidden_size' in raw:
            print(f'[Config] {candidate}' + (f' (stage={stage})' if stage else ''))
            return raw
    raise FileNotFoundError(
        f'找不到模型配置：请用 --config 指定 {weight_path.name} 对应的 config.json / run_config.json')


def read_checkpoint(path: Path, config_path: Path | None):
    """返回 (config_dict, state_dict)；兼容裸权重、latest.pt 与 state_dict 包装"""
    config, state, _ = read_checkpoint_with_stage(path, config_path)
    return config, state


def read_checkpoint_with_stage(path: Path, config_path: Path | None):
    if not path.is_file():
        raise FileNotFoundError(path)
    # 训练断点含 RNG/优化器状态，不能使用 weights_only=True（与 train/ckpt.py 一致）
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and isinstance(ckpt.get('model'), dict):
        state = ckpt['model']
    elif isinstance(ckpt, dict) and isinstance(ckpt.get('state_dict'), dict):
        state = ckpt['state_dict']
    else:
        state = ckpt  # save_state_dict 落盘的裸权重
    if not isinstance(state, dict) or not state:
        raise TypeError(f'{path} 不是可识别的权重文件: {type(state).__name__}')
    config_dict = None
    stage = 'sft' if path.parent.name == 'sft' else None
    if isinstance(ckpt, dict):
        config_dict = ckpt.get('config') or ckpt.get('model_config')
        stage = ckpt.get('contract', {}).get('stage') or stage
    if config_path is not None or config_dict is None:
        config_dict = read_model_config(path, config_path)
    return config_dict, state, stage


def build_model(config_dict, state, device, dtype):
    config = RainConfig(**config_dict)
    if any('.attn.gdn.' in key for key in state):
        raise ValueError('该权重使用旧 GDN/GQA 混合结构，不能加载到当前全 GQA MiniRain')
    model = MiniRainForCausalLM(config)
    model.load_state_dict(state, strict=True)
    get_model_params(model, config)
    # 保留转换前的表；先转 fp16 再 .float() 无法恢复丢失的精度。
    rope_buffers = {name: buffer for name in
                    ('freqs_cos', 'freqs_sin')
                    if (buffer := getattr(model.model, name)) is not None}
    model.to(device=device, dtype=dtype)
    for name, buffer in rope_buffers.items():
        setattr(model.model, name, buffer.to(device=device, dtype=torch.float32))
    return model.eval()


@torch.inference_mode()
def text_loss(model, tokenizer, text, device, max_len):
    """预训练模型的正常性检查：中文文本的 loss 应显著低于随机文本"""
    if max_len < 2 or not text:
        raise ValueError('PPL 需要非空文本，max_len 至少为 2')
    # 与文本预训练的 [BOS] + 正文 + [EOS] 保持一致。
    ids = tokenizer(tokenizer.bos_token + text + tokenizer.eos_token,
                    return_tensors='pt', add_special_tokens=False,
                    truncation=True, max_length=max_len)
    input_ids = ids['input_ids'].to(device)
    out = model(input_ids=input_ids, attention_mask=ids['attention_mask'].to(device),
                labels=input_ids)
    tokens = max(1, input_ids.shape[1] - 1)
    return float(out.loss) * tokens, tokens


@torch.inference_mode()
def dump_logits(args, model, tokenizer, input_ids, attention_mask):
    """一次前向取 logits：默认只算最后一个位置，--logits_all 时保留全序列"""
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                logits_to_keep=0 if args.logits_all else 1)
    logits = out.logits.float()
    if args.logits_save:
        path = Path(args.logits_save)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'input_ids': input_ids.cpu(), 'logits': logits.cpu(),
                    'logits_to_keep': 0 if args.logits_all else 1}, path)
        print(f'[Logits] saved {tuple(logits.shape)} -> {path}')
    if args.logits_topk > 0:
        last = logits[0, -1]
        probs = last.softmax(dim=-1)
        values, indices = probs.topk(min(args.logits_topk, probs.numel()))
        print(f'[Logits] 末位置 top-{values.numel()}（vocab={probs.numel()}）')
        for prob, idx in zip(values.tolist(), indices.tolist()):
            piece = tokenizer.decode([idx])
            print(f'  p={prob:8.5f} logit={last[idx].item():8.3f} '
                  f'id={idx:6d} {piece!r}')


def generate_ids(args, model, input_ids, attention_mask, streamer, eos_token_id):
    """默认走 HF GenerationMixin（generate_hf），--hf_generate 0 时回退自定义解码循环"""
    kwargs = dict(inputs=input_ids, attention_mask=attention_mask,
                  max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0,
                  use_cache=bool(args.use_cache), streamer=streamer,
                  eos_token_id=eos_token_id)
    if args.temperature > 0:  # HF 在 do_sample=False 时对 temperature/top_k 会告警
        kwargs.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    if args.hf_generate:
        return model.generate_hf(**kwargs)
    return model.generate(**kwargs)


def encode_prompt(args, tokenizer, prompt):
    if args.mode == 'sft':
        prompt_text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], add_generation_prompt=True,
            open_thinking=args.open_thinking, tokenize=False,
        )
        # Trailing whitespace can tokenize differently when it ends the input.
        # Tokenize before a non-whitespace character, then discard that character's tokens.
        inputs = tokenizer(prompt_text + 'a', add_special_tokens=False,
                           return_offsets_mapping=True, return_tensors='pt')
        offsets = inputs.pop('offset_mapping')[0]
        if ((offsets[:, 0] < len(prompt_text)) & (offsets[:, 1] > len(prompt_text))).any():
            raise ValueError('分词器无法在聊天前缀末尾对齐 token')
        keep = offsets[:, 1] <= len(prompt_text)
        for name in inputs:
            inputs[name] = inputs[name][:, keep]
        length = inputs['input_ids'].shape[1]
        if length > args.max_prompt_tokens:
            raise ValueError(f'套用聊天模板后有 {length} 个 token，超过 --max_prompt_tokens={args.max_prompt_tokens}')
        return inputs.to(args.device)
    return tokenizer(tokenizer.bos_token + prompt, return_tensors='pt',
                     truncation=True, max_length=args.max_prompt_tokens).to(args.device)


def run(args, model, tokenizer):
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print(f'[Gen] seed={args.seed} backend={"hf" if args.hf_generate else "custom"} '
          f'use_cache={args.use_cache} temperature={args.temperature} '
          f'top_p={args.top_p} top_k={args.top_k} mode={args.mode}'
          + (f' open_thinking={args.open_thinking}' if args.mode == 'sft' else ''))
    interactive = args.interactive or args.mode == 'sft'
    prompts = ([args.prompt] if args.prompt else
               (iter(lambda: input('[User]: '), '') if interactive else DEFAULT_PROMPTS))
    for prompt in prompts:
        setup_seed(args.seed)
        inputs = encode_prompt(args, tokenizer, prompt)
        if args.logits_topk > 0 or args.logits_save:
            dump_logits(args, model, tokenizer, inputs['input_ids'], inputs['attention_mask'])
        if not args.prompt and not interactive:
            print(f'[User]: {prompt}')
        print('[MiniRain]: ', end='')
        st = time.time()
        generated = generate_ids(args, model, inputs['input_ids'], inputs['attention_mask'],
                                 streamer if args.stream else None, tokenizer.eos_token_id)
        if not args.stream:
            print(tokenizer.decode(generated[0][inputs['input_ids'].shape[1]:],
                                   skip_special_tokens=True))
        new_tokens = generated[0, inputs['input_ids'].shape[1]:]
        gen_tokens = new_tokens.numel()
        if args.mode == 'sft' and gen_tokens == 1 and new_tokens[0].item() == tokenizer.eos_token_id:
            print('[Gen] 首个 token 是 EOS，模型结束了回复。')
        seconds = time.time() - st
        if args.show_speed:
            backend = 'hf' if args.hf_generate else 'custom'
            print(f'\n[Speed]: {gen_tokens / seconds:.2f} tokens/s '
                  f'(backend={backend}, use_cache={args.use_cache})\n')
        else:
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiniRain 预训练与 SFT 权重评估")
    parser.add_argument('--ckpt', required=True, type=str,
                        help="指定纯文本权重、训练断点或实验目录")
    parser.add_argument('--config', default=None, type=str,
                        help="模型或训练 JSON；默认从检查点或实验目录 model_config.json 读取")
    parser.add_argument('--tokenizer_path', default=project.TOKENIZER_PATH, type=str, help="分词器目录")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help="运行设备（CPU 上回退 fp32）")
    parser.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16',
                        help='CUDA 推理精度，默认与预训练一致为 bf16；CPU 使用 fp32')
    parser.add_argument('--prompt', default='', type=str, help="单条 prompt；省略时 SFT 进入交互，预训练跑内置列表")
    parser.add_argument('--mode', choices=['auto', 'base', 'sft'], default='auto',
                        help="自动识别权重阶段，或手动选择续写和聊天模式")
    parser.add_argument('--open-thinking', action='store_true', help="SFT 聊天时让模型生成思考过程")
    parser.add_argument('--interactive', default=0, type=int, choices=[0, 1], help="交互式输入（1=开）")
    parser.add_argument('--max_prompt_tokens', default=4096, type=int, help="prompt 截断长度")
    parser.add_argument('--max_new_tokens', default=256, type=int,
                        help="最大生成长度；prompt 与生成长度之和须在模型上下文范围内")
    parser.add_argument('--temperature', default=0.8, type=float, help="生成温度，0 表示贪心")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus 采样阈值（0-1）")
    parser.add_argument('--top_k', default=50, type=int, help="top-k 采样（0=关闭）")
    parser.add_argument('--hf_generate', default=1, type=int, choices=[0, 1],
                        help="解码后端：1=HF GenerationMixin（generate_hf，默认），0=自定义解码循环")
    parser.add_argument('--use_cache', default=1, type=int, choices=[0, 1],
                        help="解码复用 KV Cache（1=开，默认；0=每步重算整段 prompt，仅用于速度对比）")
    parser.add_argument('--stream', default=1, type=int, choices=[0, 1], help="流式打印（1=开）")
    parser.add_argument('--show_speed', default=1, type=int, choices=[0, 1], help="显示 tokens/s")
    parser.add_argument('--seed', default=42, type=int,
                        help="随机种子；固定默认值保证可复现（每条 prompt 前重置）")
    parser.add_argument('--logits_topk', default=0, type=int,
                        help=">0 时打印 prompt 末位置的 next-token logits/probs（top-k）")
    parser.add_argument('--logits_save', default='', type=str,
                        help="将 prompt 的 logits 存为 .pt（含 input_ids）；路径相对项目根目录")
    parser.add_argument('--logits_all', default=0, type=int, choices=[0, 1],
                        help="存/算全序列 logits（1）还是仅末位置（0，默认）")
    parser.add_argument('--ppl_text', default='', type=str,
                        help="给定期望文本，打印其 loss/perplexity 验证权重是否学到位"
                             "（建议 ≥100 字：短文本被 BOS 后的首个预测摊薄不了，PPL 会偏高）")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[args.dtype] if device.type == 'cuda' else torch.float32
    if device.type == 'cpu':
        print('[Warn] CPU 上使用 fp32 推理，速度会明显低于 GPU')
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        # Accept project-relative paths and short experiment names.
        candidates = [ROOT / ckpt_path, ROOT / 'exp' / ckpt_path, ROOT / 'outputs' / ckpt_path]
        ckpt_path = next((p for p in candidates if p.exists()), candidates[0])
    ckpt_path = find_weight(ckpt_path, prefer_sft=args.mode != 'base')
    config_dict, state, stage = read_checkpoint_with_stage(ckpt_path, args.config)
    if args.mode == 'auto':
        args.mode = 'sft' if stage == 'sft' else 'base'
    model = build_model(config_dict, state, device, dtype)
    del state
    print(f'[Model] type={model.config.type} hidden={model.config.hidden_size} '
          f'layers={model.config.n_hidden_layers} vocab={model.config.vocab_size} '
          f'rope_table={model.config.max_position_embeddings} dtype={dtype} device={device}')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    if args.ppl_text:
        loss, tokens = text_loss(model, tokenizer, args.ppl_text, device, args.max_prompt_tokens)
        mean_loss = loss / tokens
        print(f'[PPL] tokens={tokens} loss={mean_loss:.4f} ppl={math.exp(mean_loss):.2f}')
    run(args, model, tokenizer)
