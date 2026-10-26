# -*- coding: utf-8 -*-
"""MiniRainVLM 全链路测试：collator 位置编码 / 双向 mask / 嵌入注入 / 前后向 / generate"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from architectures.config import RainConfig
from architectures.rain_v import MiniRainVForCausalLM, build_2d_position_embeddings
from components.attention import MoTGroupQueryAttention, GroupQueryAttention
from dataset.collator import VLMCollator, IMG_START, IMG_END, IMG_PAD, IMG_ROW, PAD, BOS, EOS

TOK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tokenizers')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

passed, failed = [], []
def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  | {detail}" if detail and not cond else ""))

print("=" * 60); print("1. Collator 位置编码")
print("=" * 60)
tok = AutoTokenizer.from_pretrained(TOK_PATH)
col = VLMCollator(tok)

# 1.1 宽图: R=3, C=6，文本位置必须避开列坐标区间 [base, base+C]
img = torch.rand(3, 48, 96)
ids, pos, lab, pv = col.encode_sample(img, "测试文本")
n_pad = (torch.tensor(ids) == IMG_PAD).sum().item()
check("宽图 patch 数 = R*C", n_pad == 18, f"got {n_pad}")
i_end = ids.index(IMG_END)
text_pos = pos[i_end + 1:]                       # 文本 + EOS 的位置
img_rows = [p[0] for p in pos[1:i_end + 1]]      # 图像段全部行坐标
img_cols = [p[1] for p in pos[1:i_end + 1]]
collision = any((r in img_rows) or (c in img_cols) and (r, c) in set(zip(img_rows, img_cols)) for r, c in text_pos)
# 精确判定：文本位置(t,t) 与任何图像 (row,col) 重合才算碰撞
img_coord_set = set(zip(img_rows, img_cols))
collision = any((r, c) in img_coord_set for r, c in text_pos)
check("宽图文本位置不与图像坐标碰撞", not collision)
base = pos[1][0] + 1
first_text = pos[i_end + 1]
check("宽图文本起点 = base+max(R,C)+1", first_text == (base + 6 + 1, base + 6 + 1),
      f"got {first_text}, expect ({base+7},{base+7})")
check("文本位置连续递增", all(text_pos[k+1][0] == text_pos[k][0] + 1 for k in range(len(text_pos)-1)))

# 1.2 换行符位置: 行不变、列出界一格
row_tokens = [i for i, t in enumerate(ids) if t == IMG_ROW]
check("换行符数量 = R", len(row_tokens) == 3, f"got {len(row_tokens)}")
ok_row = all(pos[i] == (base + r, base + 6) for r, i in enumerate(row_tokens))
check("换行符位置 (base+i, base+C)", ok_row)

# 1.3 混合 batch：纯文本样本 padding 后位置/mask 一致
batch = col([{"image": torch.rand(3, 64, 64), "text": "一只狗"},
             {"image": None, "text": "今天天气很好"}])
am = batch["attention_mask"]
check("纯文本样本被 padding 到等长", batch["input_ids"].shape[0] == 2 and batch["input_ids"].shape[1] > 0)
b1_len = int(am[1].sum())
check("padding 位 attention_mask=0", (am[1, b1_len:] == 0).all().item())
check("labels 仅文本+EOS 可学", (batch["labels"][0][batch["input_ids"][0] == IMG_PAD] == -100).all().item())

# 1.4 截断保护：图像段完整，仅砍文本
col_short = VLMCollator(tok, max_len=40)
ids_s, pos_s, lab_s, _ = col_short.encode_sample(torch.rand(3, 64, 64), "很长的文本" * 50)
check("截断后长度 = max_len", len(ids_s) == 40, f"got {len(ids_s)}")
check("截断后图像段完整", (torch.tensor(ids_s) == IMG_PAD).sum().item() == 16 and IMG_END in ids_s)
check("截断后以 EOS 结尾", ids_s[-1] == EOS)
eos_pos, prev_pos = pos_s[-1], pos_s[-2]
check("截断后 EOS 位置连续无空洞", eos_pos == (prev_pos[0] + 1, prev_pos[1] + 1),
      f"prev={prev_pos}, eos={eos_pos}")

print("=" * 60); print("2b. 模态路由 (MoT-GQA idxs / GQA 兼容)")
print("=" * 60)
cfg = RainConfig(hidden_size=256, n_hidden_layers=2, n_attn_heads=4, n_key_value_heads=2)
model = MiniRainVForCausalLM(cfg).to(DEVICE)
vlm = model.model
attn0 = vlm.layers[0].attn
is_mot = isinstance(attn0, MoTGroupQueryAttention)
img_ids_t = torch.tensor([IMG_START, IMG_END, IMG_PAD, IMG_ROW], dtype=torch.long, device=DEVICE)
print(f"  注意力类型: {type(attn0).__name__}")

# 混合序列: 1 图(R=2,C=2) + 文本，单样本无 padding
mix = col([{"image": torch.rand(3, 32, 32), "text": "甲乙丙"}])
mix_ids = mix["input_ids"].to(DEVICE)
mm_mix = torch.isin(mix_ids, img_ids_t).long()
print(f"  混合序列 tokens = {mix_ids[0].tolist()}")
print(f"  modality_mask = {mm_mix.reshape(-1).tolist()}")

# 纯文本序列
txt = col([{"image": None, "text": "今天天气很好"}])
txt_ids = txt["input_ids"].to(DEVICE)
mm_txt = torch.isin(txt_ids, img_ids_t).long()
print(f"  纯文本 tokens = {txt_ids[0].tolist()}")
print(f"  纯文本 mask   = {mm_txt.reshape(-1).tolist()} (全 0)")

if is_mot:
    def show_idxs(tag, ids_tensor):
        mm = torch.isin(ids_tensor, img_ids_t).long()
        idxs = attn0._modal_indices(mm)
        for m, ix in enumerate(idxs):
            kind = "文本" if m == 0 else "图像"
            print(f"    [{tag}] idxs[{m}]({kind}) = {ix.tolist()}")
        return idxs
    idxs_mix = show_idxs("混合", mix_ids)
    img_flat = [i for i, t in enumerate(mix_ids[0].tolist()) if t in (IMG_START, IMG_END, IMG_PAD, IMG_ROW)]
    check("MoT 混合: 图像索引 = 全部图像 token 平坦位置", idxs_mix[1].tolist() == img_flat)
    check("MoT: 文本+图像索引互补覆盖全序列",
          sorted(idxs_mix[0].tolist() + idxs_mix[1].tolist()) == list(range(mix_ids.numel())))
    idxs_txt = show_idxs("纯文本", txt_ids)
    check("MoT 纯文本: 图像索引为空", idxs_txt[1].numel() == 0)
    check("MoT 纯文本: 文本索引覆盖全序列", idxs_txt[0].numel() == txt_ids.numel())
else:
    # 普通 GQA: 模态解耦由 MoTFFN 承担。GQA 的 modality_mask 参数是
    # image_bidirectional_mask 的别名（可见性 mask），与 VLM 的模态指示语义冲突，
    # 因此 block 必须按注意力类型分派、绝不把 modality_mask 传进 GQA。
    # 回归守护：spy block 内 attn 调用，确认 modality_mask 未泄漏进 GQA。
    received = []
    orig_fwd = attn0.forward
    def spy(*a, **kw):
        received.append(kw.get("modality_mask", "ABSENT"))
        return orig_fwd(*a, **kw)
    attn0.forward = spy
    _ = model(input_ids=mix_ids,
              positions=mix["positions"].to(DEVICE),
              attention_mask=mix["attention_mask"].to(DEVICE),
              pixel_values=[x.to(DEVICE) for x in mix["pixel_values"]])
    attn0.forward = orig_fwd
    check("GQA: block 未把 modality_mask 传入注意力（防语义劫持）",
          len(received) > 0 and all(m is None or m == "ABSENT" for m in received),
          f"received={received}")
    check("GQA: modality_mask 图像 token 计数正确",
          mm_mix.sum().item() == len([t for t in mix_ids[0].tolist() if t in (IMG_START, IMG_END, IMG_PAD, IMG_ROW)]))

print("=" * 60); print("3. 双向注意力 mask 语义")
print("=" * 60)

batch2 = col([{"image": torch.rand(3, 32, 32), "text": "甲乙丙"},     # R=2,C=2
              {"image": torch.rand(3, 32, 32), "text": "丁"}])
input_ids = batch2["input_ids"].to(DEVICE)
mask = vlm._build_image_bidirectional_mask(input_ids, batch2["attention_mask"].to(DEVICE), torch.float32)
allow = (mask[0, 0] == 0)  # [S, S] True=可见
ids0 = input_ids[0].tolist()
img_idx = [i for i, t in enumerate(ids0) if t in (IMG_START, IMG_END, IMG_PAD, IMG_ROW)]
txt_idx = [i for i, t in enumerate(ids0) if t not in (IMG_START, IMG_END, IMG_PAD, IMG_ROW)]
check("图内 patch 互相可见(双向)", all(allow[i, j].item() for i in img_idx for j in img_idx))
check("文本因果: 先前文本看不到后文本", not allow[txt_idx[1], txt_idx[-1]].item() if len(txt_idx) > 2 else True)
check("后文本可见先前文本", allow[txt_idx[-1], txt_idx[1]].item() if len(txt_idx) > 2 else True)
check("文本可见先前图像", all(allow[txt_idx[-1], j].item() for j in img_idx))
check("图像不可见未来文本", not any(allow[img_idx[0], t].item() for t in txt_idx if t > img_idx[0]))

# 跨图隔离：构造单样本双图
ids_2img, pos_2img, _, _ = col.encode_sample(torch.rand(3, 32, 32), "中间")  # 只占位
seg_ids = [IMG_START, IMG_PAD, IMG_END] + [IMG_START, IMG_PAD, IMG_END]
seg_ids = torch.tensor([[BOS] + seg_ids + [EOS]], device=DEVICE)
m2 = vlm._build_image_bidirectional_mask(seg_ids, None, torch.float32)[0, 0]
g1 = [1, 2, 3]; g2 = [4, 5, 6]
# 语义记录: allow = causal | same_img → 图2 作为"之后的token"可因果地看见图1（与docstring"跨段隔离"不一致，待确认）
check("跨图: 图2 可见 图1(causal 并集语义)", all((m2[i, j] == 0).item() for i in g2 for j in g1))
check("图1 看不到 图2(未来)", not any((m2[i, j] == 0).item() for i in g1 for j in g2))
check("图1 内部双向", all((m2[i, j] == 0).item() for i in g1 for j in g1))

# padding 屏蔽
padded = torch.tensor([[BOS, IMG_START, IMG_PAD, IMG_END, 5, PAD]], device=DEVICE)
pam = torch.tensor([[1, 1, 1, 1, 1, 0]], device=DEVICE)
m3 = vlm._build_image_bidirectional_mask(padded, pam, torch.float32)[0, 0]
check("padding 作为 key 被屏蔽", (m3[:, 5] == float('-inf')).all().item())

print("=" * 60); print("4. 嵌入注入与前向/反向")
print("=" * 60)
emb = vlm.prepare_inputs_embeds(input_ids, [x.to(DEVICE) for x in batch2["pixel_values"]])
pad_pos = (input_ids[0] == IMG_PAD).nonzero().squeeze(-1)
patch_out = vlm.patch_embed(batch2["pixel_values"][0].unsqueeze(0).to(DEVICE))
ref = vlm.img_norm(patch_out.flatten(2).transpose(1, 2).squeeze(0))  # patch_embed 后经 img_norm 归一
check("patch 嵌入按行主序注入占位坑(含img_norm)",
      torch.allclose(emb[0, pad_pos], ref.to(emb.dtype), atol=1e-5))

out = model(input_ids=input_ids, positions=batch2["positions"].to(DEVICE),
            attention_mask=batch2["attention_mask"].to(DEVICE),
            pixel_values=[x.to(DEVICE) for x in batch2["pixel_values"]],
            labels=batch2["labels"].to(DEVICE))
check("前向产出有限 loss", torch.isfinite(out.loss).item(), f"loss={out.loss.item()}")
out.loss.backward()
g_embed = vlm.embed.weight.grad
g_patch = vlm.patch_embed.stem[0].weight.grad  # ConvStem 首层卷积
check("文本嵌入收到梯度", g_embed is not None and g_embed.abs().sum().item() > 0)
check("patch_embed 收到梯度(原位赋值不断梯度)", g_patch is not None and g_patch.abs().sum().item() > 0)
# 注意: 权重捆绑(tie_word_embeddings)下 embed.weight 同时是 lm_head，
# IMG_PAD 行会从输出侧收到梯度，不能断言其为0
model.zero_grad(set_to_none=True)

print("=" * 60); print("5. generate: prefill/decode/多序列")
print("=" * 60)
model.eval()
prompt = col([{"image": torch.rand(3, 32, 32), "text": "描述:"}])
gen = model.generate(input_ids=prompt["input_ids"].to(DEVICE),
                     positions=prompt["positions"].to(DEVICE),
                     attention_mask=prompt["attention_mask"].to(DEVICE),
                     pixel_values=[x.to(DEVICE) for x in prompt["pixel_values"]],
                     max_new_tokens=8, do_sample=False, eos_token_id=None)
gen_ids = gen[0, prompt["input_ids"].shape[1]:].tolist()
check("贪心生成 8 token(禁提前停)", gen.shape[1] == prompt["input_ids"].shape[1] + 8,
      f"gen_len={gen.shape[1] - prompt['input_ids'].shape[1]}")
struct = set(gen_ids) & {IMG_START, IMG_END, IMG_PAD, 11, 13}
check("生成不含视觉占位/边界(9/10/11/12/13)", len(struct) == 0, f"got {struct}")
check("生成不含换行符 27", IMG_ROW not in gen_ids, f"gen={gen_ids}")

gen2 = model.generate(input_ids=prompt["input_ids"].to(DEVICE),
                      positions=prompt["positions"].to(DEVICE),
                      attention_mask=prompt["attention_mask"].to(DEVICE),
                      pixel_values=[x.to(DEVICE) for x in prompt["pixel_values"]],
                      max_new_tokens=4, num_return_sequences=3, do_sample=True, temperature=1.0)
check("num_return_sequences=3 形状", gen2.shape[0] == 3, f"got {gen2.shape}")

# 纯文本 generate(无 positions) 回归
txt_only = col([{"image": None, "text": "你好"}])
gen3 = model.generate(input_ids=txt_only["input_ids"].to(DEVICE),
                      attention_mask=txt_only["attention_mask"].to(DEVICE),
                      max_new_tokens=4, do_sample=False)
check("纯文本无 positions 生成", gen3.shape[1] >= txt_only["input_ids"].shape[1] + 1)

# 确定性验证 27 号换行符被屏蔽：强制其成为 argmax，若未被屏蔽则必然生成它
cfg_u = RainConfig(hidden_size=256, n_hidden_layers=2, n_attn_heads=4,
                   n_key_value_heads=2, tie_word_embeddings=False)
model_u = MiniRainVForCausalLM(cfg_u).to(DEVICE).eval()
out_u = model_u(input_ids=prompt["input_ids"].to(DEVICE),
                positions=prompt["positions"].to(DEVICE),
                attention_mask=prompt["attention_mask"].to(DEVICE),
                pixel_values=[x.to(DEVICE) for x in prompt["pixel_values"]])
h = out_u.hidden_states[0, -1]
v = h / (h.norm() + 1e-6)
w = model_u.lm_head.weight.data
w.copy_(-100 * v.unsqueeze(0).expand_as(w))   # 其余 token logit = -100|h|
w[IMG_ROW] = 100 * v                          # 27 号 logit = +100|h| → 强制 argmax
gen_u = model_u.generate(input_ids=prompt["input_ids"].to(DEVICE),
                         positions=prompt["positions"].to(DEVICE),
                         attention_mask=prompt["attention_mask"].to(DEVICE),
                         pixel_values=[x.to(DEVICE) for x in prompt["pixel_values"]],
                         max_new_tokens=2, do_sample=False, eos_token_id=None)
gen_u_ids = gen_u[0, prompt["input_ids"].shape[1]:].tolist()
check("强制 argmax 下 27 仍不被生成(屏蔽生效)", IMG_ROW not in gen_u_ids,
      f"gen={gen_u_ids}")

print("=" * 60)
print(f"结果: {len(passed)} PASS, {len(failed)} FAIL")
if failed: print("失败项:", failed); sys.exit(1)
