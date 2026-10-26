"""MiniRainVLM 部件级 + 整机结构测试
与 test_vlm.py(链路/生成)、res_test.py(残差语义)互补: 这里逐部件验证契约。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F

from architectures.config import RainConfig
from architectures.rain_v import (MiniRainVForCausalLM, precompute_freqs_cis_2d,
                                build_2d_position_embeddings)
from components.norm import RMSNorm, NormArgs
from components.attention import (apply_rotary_pos_emb, repeat_kv,
                                  GroupQueryAttention, MoTGroupQueryAttention, AttentionArgs)
from components.block import BlockAttnRes, BlockAttnResBlock
from components.mlp import SwiGLU, LosslessMoE, MLPArgs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
passed, failed = [], []
def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  | {detail}" if detail else ""))

D, NH, NKV = 128, 4, 2          # hidden / q头 / kv头
HD = D // NH                    # head_dim = 32
PS = 16                         # patch_size

# ============================================================
print("=" * 60)
print("1. RMSNorm")
print("=" * 60)
norm = RMSNorm(NormArgs(dim=D, eps=1e-6)).to(DEVICE)
x = torch.randn(2, 7, D, device=DEVICE) * 5
y = norm(x)
manual = (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).type_as(x)
check("数值 == 手工公式", torch.allclose(y, manual, atol=1e-5))
check("归一化后 RMS ≈ 1", ((y.pow(2).mean(-1).sqrt()) - 1).abs().max().item() < 1e-3,
      f"max dev {((y.pow(2).mean(-1).sqrt()) - 1).abs().max().item():.2e}")
xb = x.bfloat16() if DEVICE == "cuda" else x.half()
check("dtype 保持(权重同精度时)", RMSNorm(NormArgs(dim=D, eps=1e-6)).to(DEVICE).to(xb.dtype)(xb).dtype == xb.dtype)
norm.weight.grad = None
norm(x).sum().backward()
check("weight 有梯度", norm.weight.grad is not None and norm.weight.grad.abs().sum() > 0)

# ============================================================
print("=" * 60)
print("2. 2D RoPE 表与 apply_rotary_pos_emb")
print("=" * 60)
cos_t, sin_t = precompute_freqs_cis_2d(HD, end=256)
check("表形状 [end, HD/4]", cos_t.shape == (256, HD // 4) and sin_t.shape == (256, HD // 4))
pos = torch.tensor([[[3, 9]]])
cos, sin = build_2d_position_embeddings(pos, cos_t, sin_t)
c = cos[0, 0]
check("布局 [行|列|行|列]: 行段==行表",
      torch.equal(c[:HD // 4], cos_t[3]) and torch.equal(c[HD // 2:HD // 2 + HD // 4], cos_t[3]))
check("布局: 列段==列表",
      torch.equal(c[HD // 4:HD // 2], cos_t[9]) and torch.equal(c[-HD // 4:], cos_t[9]))
check("配对同频率 cos[i]==cos[i+HD/2]", torch.equal(c[:HD // 2], c[HD // 2:]))
pos_t = torch.tensor([[[7, 7]]])
ct, _ = build_2d_position_embeddings(pos_t, cos_t, sin_t)
check("文本(t,t)退化: 行段==列段", torch.equal(ct[0, 0, :HD // 4], ct[0, 0, HD // 4:HD // 2]))
# YaRN: 不触发时与原表一致; 触发后只动低频
c0, s0 = precompute_freqs_cis_2d(HD, end=256, rope_scaling=None)
c1, _ = precompute_freqs_cis_2d(HD, end=256, rope_scaling={"factor": 8, "original_max_position_embeddings": 2048})
check("YaRN 未触发时与原表一致", torch.equal(c0, c1))
c2, _ = precompute_freqs_cis_2d(HD, end=8192, rope_scaling={"factor": 8.0, "original_max_position_embeddings": 2048,
                                                            "beta_fast": 32.0, "beta_slow": 1.0, "attention_factor": 1.0})
c3, _ = precompute_freqs_cis_2d(HD, end=8192, rope_scaling=None)
diff = (c2 - c3).abs().amax(dim=0)
check("YaRN 触发: 高频不动低频缩放", diff[0].item() == 0.0 and diff[-1].item() > 0,
      f"首 {diff[0]:.1e} 末 {diff[-1]:.1e}")
# 旋转保范数 (attn_factor=1 时是纯旋转)
q = torch.randn(1, 4, NH, HD, device=DEVICE)
posq = torch.stack([torch.arange(4), torch.arange(4)], -1).unsqueeze(0).to(DEVICE)
cosq, sinq = build_2d_position_embeddings(posq, cos_t.to(DEVICE), sin_t.to(DEVICE))
qr, _ = apply_rotary_pos_emb(q, q.clone(), cosq, sinq)
check("旋转保持 L2 范数", torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-5))
check("2D cos [T,h] 也能用 (unsqueeze_dim=1)",
      apply_rotary_pos_emb(q, q.clone(), cosq[0], sinq[0])[0].shape == q.shape)

# ============================================================
print("=" * 60)
print("3. repeat_kv 分组顺序")
print("=" * 60)
xk = torch.arange(1 * 1 * 2 * 3, dtype=torch.float32).view(1, 1, 2, 3)
rep = repeat_kv(xk, 2)
check("形状 [.., n_kv*n_rep, hd]", rep.shape == (1, 1, 4, 3))
check("顺序: 每个kv头连续重复 n_rep 次",
      torch.equal(rep[0, 0, 0], xk[0, 0, 0]) and torch.equal(rep[0, 0, 1], xk[0, 0, 0])
      and torch.equal(rep[0, 0, 2], xk[0, 0, 1]) and torch.equal(rep[0, 0, 3], xk[0, 0, 1]))
check("n_rep=1 原样返回", repeat_kv(xk, 1) is xk)

# ============================================================
print("=" * 60)
print("4. patch_embed 行优先顺序 + prepare_inputs_embeds 覆盖")
print("=" * 60)
cfg = RainConfig(hidden_size=D, n_hidden_layers=2, n_attn_heads=NH, n_key_value_heads=NKV)
model = MiniRainVForCausalLM(cfg).to(DEVICE)
vlm = model.model
# ConvStem(4层stride2=16×降采样): 32x48 → 2x3 网格; flatten(2) 顺序即行优先 (k = i*C + j)
img = torch.rand(3, 32, 48, device=DEVICE)
patches = vlm.patch_embed(img.unsqueeze(0)).flatten(2).transpose(1, 2).squeeze(0)
check("ConvStem 16× 降采样: 32x48 → 2x3=6 patch", patches.shape == (6, D),
      f"got {tuple(patches.shape)}")
# 覆盖: 6 个占位 ↔ 6 个 patch, 文本位不动
ids = torch.tensor([[1, 9, 12, 12, 12, 12, 12, 12, 10, 100, 200, 2]], device=DEVICE)
emb = vlm.prepare_inputs_embeds(ids, [img])
check("占位处 == img_norm(patch) 行 (顺序一致)", torch.allclose(emb[0, 2:8], vlm.img_norm(patches).to(emb.dtype)))
check("文本位 == embed(token)",
      torch.allclose(emb[0, 9], vlm.embed.weight[100]) and torch.allclose(emb[0, 10], vlm.embed.weight[200]))
check("pixel_values=None → 纯 embed", torch.equal(vlm.prepare_inputs_embeds(ids), vlm.embed(ids)))
try:
    vlm.prepare_inputs_embeds(ids[:, :7], [img])   # 5个占位 vs 6个patch
    check("占位/patch 数不匹配触发 assert", False)
except AssertionError:
    check("占位/patch 数不匹配触发 assert", True)

# ============================================================
print("=" * 60)
print("5. MoTGroupQueryAttention 模态路由 + KV cache 等价")
print("=" * 60)
attn = MoTGroupQueryAttention(AttentionArgs(cfg)).to(DEVICE).eval()
x5 = torch.randn(2, 5, D, device=DEVICE)
mm = torch.tensor([[0, 1, 1, 0, 0], [0, 0, 1, 1, 1]], device=DEVICE)
idxs = attn._modal_indices(mm)
q_routed = attn._project_modal(attn.wq, x5, idxs)
flat = x5.reshape(-1, D)
q_manual = torch.cat([attn.wq[0](flat[(mm.reshape(-1) == 0).nonzero().flatten()]),
                      attn.wq[1](flat[(mm.reshape(-1) == 1).nonzero().flatten()])])
q_manual_full = torch.empty_like(q_routed.reshape(-1, q_routed.shape[-1]))
q_manual_full[(mm.reshape(-1) == 0).nonzero().flatten()] = attn.wq[0](flat[(mm.reshape(-1) == 0).nonzero().flatten()])
q_manual_full[(mm.reshape(-1) == 1).nonzero().flatten()] = attn.wq[1](flat[(mm.reshape(-1) == 1).nonzero().flatten()])
check("_project_modal: 文本行==expert0, 图像行==expert1", torch.allclose(q_routed.reshape(-1, q_routed.shape[-1]), q_manual_full, atol=1e-6))
check("idxs=None → 全体走 expert0", torch.allclose(attn._project_modal(attn.wq, x5, None), attn.wq[0](x5)))
# KV cache 等价: 整段 vs 4+2 分段
x6 = torch.randn(1, 6, D, device=DEVICE)
pos6 = torch.stack([torch.arange(6), torch.arange(6)], -1).unsqueeze(0).to(DEVICE)
pe6 = build_2d_position_embeddings(pos6, cos_t.to(DEVICE), sin_t.to(DEVICE))
with torch.no_grad():
    out_full, kv_full = attn(x6, pe6, use_cache=True)
    out_a, kv_a = attn(x6[:, :4], (pe6[0][:, :4], pe6[1][:, :4]), use_cache=True)
    out_b, kv_b = attn(x6[:, 4:], (pe6[0][:, 4:], pe6[1][:, 4:]), past_key_value=kv_a, use_cache=True)
check("prefill vs 分段 decode 输出一致", torch.allclose(out_full[:, 4:], out_b, atol=1e-5),
      f"max diff {(out_full[:, 4:] - out_b).abs().max():.2e}")
check("cache 形状 [B, T, n_kv, hd]", kv_b[0].shape == (1, 6, NKV, HD))
check("分段前段输出一致", torch.allclose(out_full[:, :4], out_a, atol=1e-5))

# ============================================================
print("=" * 60)
print("6. BlockAttnRes 聚合单元")
print("=" * 60)
res = BlockAttnRes(cfg).to(DEVICE)
h1, h2, cur = (torch.randn(2, 5, D, device=DEVICE) for _ in range(3))
out = res([h1, h2], cur)
check("零初始化 → 历史+当前的均值", torch.allclose(out, (h1 + h2 + cur) / 3, atol=1e-6))
check("单候选 → 原样透传", torch.allclose(res([], cur), cur, atol=1e-6))
res.attn_res_query.grad = None
res([h1, h2], cur).sum().backward()
check("≥2 候选时 query 有梯度", res.attn_res_query.grad.abs().sum() > 0)

# ============================================================
print("=" * 60)
print("7. MLP: SwiGLU / LosslessMoE")
print("=" * 60)
sg = SwiGLU(MLPArgs(hidden_size=D, intermediate_size=64, bias=False)).to(DEVICE)
xm = torch.randn(2, 5, D, device=DEVICE)
check("SwiGLU == down(silu(gate(x))*up(x))",
      torch.allclose(sg(xm), sg.down(F.silu(sg.gate(xm)) * sg.up(xm)), atol=1e-6))
for w, n in [(sg.gate.weight, "gate"), (sg.up.weight, "up"), (sg.down.weight, "down")]:
    w.grad = None
sg(xm).sum().backward()
check("SwiGLU 三个权重都有梯度",
      all(w.grad is not None and w.grad.abs().sum() > 0 for w in [sg.gate.weight, sg.up.weight, sg.down.weight]))
moe = LosslessMoE(dim=D, intermediate_size=64, num_experts=4, top_k=2, num_shared_experts=1, aux_scale=0.01).to(DEVICE).eval()
with torch.no_grad():
    out_moe, logits_m, aux_m = moe(xm)
    # 手工复算: 同一路由(无噪声) → top_k 加权 + 共享专家
    flatx = xm.view(-1, D)
    _, w, idx = moe.router(flatx)
    manual = moe.shared_experts[0](flatx)
    for e in range(4):
        sel = (idx == e).nonzero()
        if len(sel) == 0: continue
        ti, ki = sel[:, 0], sel[:, 1]
        manual.index_add_(0, ti, moe.experts[e](flatx[ti]) * w[ti, ki].unsqueeze(-1))
check("MoE 输出 == top_k加权路由+共享 手工复算", torch.allclose(out_moe.view(-1, D), manual, atol=1e-5),
      f"max diff {(out_moe.view(-1, D) - manual).abs().max():.2e}")
check("eval 下 aux_loss == 0", aux_m.item() == 0.0)
bias0 = moe.router.expert_bias.clone()
moe.eval(); moe(xm)
check("eval 不更新 expert_bias", torch.equal(bias0, moe.router.expert_bias))
moe.train(); moe(xm)
check("train 更新 expert_bias (lossless 均衡)", not torch.equal(bias0, moe.router.expert_bias))
check("bias 更新方向: 总量守恒(增减抵消)", moe.router.expert_bias.sum().abs() < 1e-6 + bias0.sum().abs() + 1,
      f"sum={moe.router.expert_bias.sum():.4f}")
moe.eval()

# ============================================================
print("=" * 60)
print("8. BlockAttnResBlock 接线")
print("=" * 60)
cfg_bs2 = RainConfig(hidden_size=D, n_hidden_layers=2, n_attn_heads=NH, n_key_value_heads=NKV, block_size=2)
blk = BlockAttnResBlock(0, cfg_bs2).to(DEVICE).eval()
check("type_id=1 → SwiGLU", isinstance(blk.mlp, SwiGLU))
check("注意力为 GQA/MoT 其一", isinstance(blk.attn, (GroupQueryAttention, MoTGroupQueryAttention)))
check("attn_res/mlp_res 为 BlockAttnRes", isinstance(blk.attn_res, BlockAttnRes) and isinstance(blk.mlp_res, BlockAttnRes))
xb = torch.randn(1, 4, D, device=DEVICE)
posb = torch.stack([torch.arange(4), torch.arange(4)], -1).unsqueeze(0).to(DEVICE)
peb = build_2d_position_embeddings(posb, cos_t.to(DEVICE), sin_t.to(DEVICE))
with torch.no_grad():
    out_b, kv_b, hist = blk(xb, peb, use_cache=True, past_hidden_states=[])
check("返回 (out, kv, history) 形状", out_b.shape == xb.shape and kv_b[0].shape[1] == 4 and len(hist) == 1)
cfg_moe = RainConfig(hidden_size=D, n_hidden_layers=2, n_attn_heads=NH, n_key_value_heads=NKV, use_moe=True)
blk2 = BlockAttnResBlock(0, cfg_moe).to(DEVICE).eval()
check("use_moe → type_id=2 → LosslessMoE", cfg_moe.type_id == 2 and isinstance(blk2.mlp, LosslessMoE),
      f"type_id={cfg_moe.type_id}")

# ============================================================
print("=" * 60)
print("9. MiniRainVLM 结构")
print("=" * 60)
check("层数 == n_hidden_layers 且全为 BlockAttnResBlock",
      len(vlm.layers) == cfg.n_hidden_layers and all(isinstance(l, BlockAttnResBlock) for l in vlm.layers))
sd = model.state_dict()
check("rope2d buffer 非持久(不进 state_dict)", not any("rope2d" in k for k in sd.keys()))
check("buffer 随 .to() 迁移", str(vlm.rope2d_cos.device).startswith(DEVICE))
ids9 = torch.randint(3, 6000, (2, 9), device=DEVICE)
with torch.no_grad():
    h_auto, _, _ = vlm(ids9)
    pos9 = torch.stack([torch.arange(9), torch.arange(9)], -1).unsqueeze(0).expand(2, -1, -1).to(DEVICE)
    h_expl, _, _ = vlm(ids9, positions=pos9)
check("positions=None 等价于显式 arange", torch.allclose(h_auto, h_expl, atol=1e-6))
vlm.eval()
with torch.no_grad():
    h1r, _, _ = vlm(ids9); h2r, _, _ = vlm(ids9)
check("eval(dropout=0) 前向确定性", torch.equal(h1r, h2r))
check("末层输出已过 final text_norm: RMS≈1", (h1r.pow(2).mean(-1).sqrt() - 1).abs().max().item() < 5e-2,
      f"max dev {(h1r.pow(2).mean(-1).sqrt() - 1).abs().max().item():.2e}")
from transformers import DynamicCache
with torch.no_grad():
    h_c, _, _ = vlm(ids9, past_key_values=DynamicCache())
check("传入 HF Cache 对象自动重置为 None", torch.allclose(h_c, h1r, atol=1e-6))

# ============================================================
print("=" * 60)
print("10. MiniRainVForCausalLM 整机结构")
print("=" * 60)
check("权重共享: embed.weight is lm_head.weight", vlm.embed.weight is model.lm_head.weight)
cfg_untied = RainConfig(hidden_size=D, n_hidden_layers=2, n_attn_heads=NH, n_key_value_heads=NKV, tie_word_embeddings=False)
m_untied = MiniRainVForCausalLM(cfg_untied).to(DEVICE)
check("tie_word_embeddings=False → 不共享", m_untied.model.embed.weight is not m_untied.lm_head.weight)
model.eval()
with torch.no_grad():
    o = model(input_ids=ids9)
check("logits 形状 [B,T,V]", o.logits.shape == (2, 9, cfg.vocab_size))
with torch.no_grad():
    o1 = model(input_ids=ids9, logits_to_keep=1)
check("logits_to_keep=1 → [B,1,V] 且==末位", o1.logits.shape == (2, 1, cfg.vocab_size)
      and torch.allclose(o1.logits[:, 0], o.logits[:, -1], atol=1e-6))
lab = ids9.clone()
with torch.no_grad():
    o_loss = model(input_ids=ids9, labels=lab)
man = F.cross_entropy(o.logits[:, :-1].reshape(-1, cfg.vocab_size), lab[:, 1:].reshape(-1))
check("loss == 手工移位 CE", torch.allclose(o_loss.loss, man, atol=1e-5), f"{o_loss.loss.item():.4f} vs {man.item():.4f}")
check("aux_loss 占位为 0", o_loss.aux_loss.item() == 0.0)
# state_dict 往返 → 输出一致
m2 = MiniRainVForCausalLM(cfg).to(DEVICE).eval()
m2.load_state_dict(model.state_dict())
with torch.no_grad():
    o2 = m2(input_ids=ids9)
check("state_dict 往返后输出一致", torch.allclose(o.logits, o2.logits, atol=1e-6),
      f"max diff {(o.logits - o2.logits).abs().max():.2e}")
# MoE 整机
m_moe = MiniRainVForCausalLM(cfg_moe).to(DEVICE).eval()
with torch.no_grad():
    o_moe = m_moe(input_ids=ids9, labels=lab)
check("MoE 整机前向/loss 有限", o_moe.loss is not None and torch.isfinite(o_moe.loss).item(),
      f"loss={o_moe.loss.item():.4f}")
# 参数分组 sanity
n_emb = vlm.embed.weight.numel(); n_patch = sum(p.numel() for p in vlm.patch_embed.parameters())
n_layer = sum(p.numel() for l in vlm.layers for p in l.parameters())
print(f"  参数量: embed {n_emb/1e6:.2f}M | patch {n_patch/1e3:.1f}K | layers {n_layer/1e6:.2f}M | 总 {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
check("各参数组非零且无 NaN", all(v > 0 for v in [n_emb, n_patch, n_layer])
      and all(torch.isfinite(p).all() for p in model.parameters()))
# 图像梯度回流到 patch_embed
model.zero_grad()
o_img = model(input_ids=ids, pixel_values=[img], positions=torch.tensor(
    [[(0, 0), (1, 1), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 4), (5, 5), (6, 6), (7, 7)]],
    device=DEVICE), labels=ids)
o_img.loss.backward()
check("patch_embed 收到梯度", vlm.patch_embed.stem[0].weight.grad is not None and vlm.patch_embed.stem[0].weight.grad.abs().sum() > 0)

print("=" * 60)
print(f"结果: {len(passed)} PASS, {len(failed)} FAIL")
if failed: print("失败项:", failed); sys.exit(1)
