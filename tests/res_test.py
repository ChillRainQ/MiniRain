# -*- coding: utf-8 -*-
"""残差行为测试: BlockAttnRes 启用逻辑与聚合语义
模型: 12 层 x 1024 维, block_size=2 → block_size//2=1 → 每层都 append 历史
"""
import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _device import add_device_args, list_gpus, resolve_device
from architectures.config import RainConfig
from architectures.rain_v import MiniRainVForCausalLM
from components.block import BlockAttnRes

_p = argparse.ArgumentParser(description="BlockAttnRes 残差语义测试")
add_device_args(_p)
_args = _p.parse_args()
if _args.list_gpus:
    list_gpus()
    raise SystemExit(0)
DEVICE, GPU_INDEX = resolve_device(_args.device)
torch.manual_seed(0)

passed, failed = [], []
def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  | {detail}" if detail else ""))

L, D = 12, 1024
cfg = RainConfig(hidden_size=D, n_hidden_layers=L, block_size=2,
                 n_attn_heads=8, n_key_value_heads=4)
model = MiniRainVForCausalLM(cfg).to(DEVICE)
vlm = model.model
STEP = cfg.block_size // 2   # append 间隔: layer_id % STEP == 0 时 append
print(f"模型: {L}层 x {D}维, block_size={cfg.block_size} (append间隔STEP={STEP}), 参数量 {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

input_ids = torch.randint(3, 6000, (2, 32), device=DEVICE)  # 纯文本, 2条x32
labels = input_ids.clone()

print("=" * 60); print(f"1. 历史累积: append 于 layer % {STEP} == 0 (即层 {[l for l in range(L) if l % STEP == 0]})")
print("=" * 60)
records, block_inputs = [], []
orig_fwd = BlockAttnRes.forward
def spy(self, hist, pb):
    records.append(list(hist))          # 记录引用，稍后验身份
    return orig_fwd(self, hist, pb)
BlockAttnRes.forward = spy
hooks = [b.register_forward_pre_hook(lambda m, a: block_inputs.append(a[0])) for b in vlm.layers]
with torch.no_grad():
    model(input_ids=input_ids)
for h in hooks: h.remove()
BlockAttnRes.forward = orig_fwd

# 理论值: attn_res@L 收到 #{k<L: k%STEP==0} = ceil(L/STEP); 若 L%STEP==0 则 append → mlp_res 多一条
exp_attn = [(l + STEP - 1) // STEP for l in range(L)]
exp_mlp  = [l // STEP + 1 for l in range(L)]
ok_attn = all(len(records[2*l]) == exp_attn[l] for l in range(L))
ok_mlp  = all(len(records[2*l+1]) == exp_mlp[l] for l in range(L))
check("attn_res 历史条数 = ceil(L/STEP)", ok_attn,
      f"exp {exp_attn}\n       got {[len(records[2*l]) for l in range(L)]}")
check("mlp_res 历史条数 = L//STEP+1", ok_mlp,
      f"exp {exp_mlp}\n       got {[len(records[2*l+1]) for l in range(L)]}")
check("全前向后历史总长 = 1 + (L-1)//STEP", len(records[-1]) == 1 + (L-1)//STEP,
      f"got {len(records[-1])}")

print("=" * 60); print("2. append 语义: 历史第 j 条 == 第 j*STEP 层输入(引用同一tensor)")
print("=" * 60)
full_hist = records[-1]
append_layers = [l for l in range(L) if l % STEP == 0]
check("历史条目与 append 层输入逐一同源",
      len(full_hist) == len(append_layers) and
      all(full_hist[j] is block_inputs[append_layers[j]] for j in range(len(append_layers))),
      f"append_layers={append_layers}")

print("=" * 60); print("3. 零初始化 → 均匀聚合(mean)")
print("=" * 60)
res = BlockAttnRes(cfg).to(DEVICE)   # attn_res_query 零初始化
hs = [torch.randn(2, 5, D, device=DEVICE) for _ in range(3)]
out = res(hs[:-1], hs[-1])
mean_ref = torch.stack(hs).mean(dim=0)
check("零初始化输出 = 历史均值", torch.allclose(out, mean_ref, atol=1e-5),
      f"max diff = {(out - mean_ref).abs().max().item():.2e}")

print("=" * 60); print("4. query 可学性: 多元素聚合才有梯度 (区分真启用/空转)")
print("=" * 60)
def query_grads(block_size):
    c = RainConfig(hidden_size=D, n_hidden_layers=L, block_size=block_size,
                   n_attn_heads=8, n_key_value_heads=4)
    m = MiniRainVForCausalLM(c).to(DEVICE)
    out = m(input_ids=input_ids, labels=labels)
    out.loss.backward()
    gs = []
    for lyr in m.model.layers:
        gs.append(lyr.attn_res.attn_res_query.grad.abs().sum().item())
        gs.append(lyr.mlp_res.attn_res_query.grad.abs().sum().item())
    return gs, out.loss.item()

g2, loss2 = query_grads(cfg.block_size)
g0, loss0 = query_grads(0)
check(f"block_size={cfg.block_size}: 有限 loss", loss2 == loss2 and abs(loss2) < 1e3, f"loss={loss2:.4f}")
check(f"block_size={cfg.block_size}: 除 layer0.attn_res 外全部 query 有梯度",
      g2[0] == 0 and all(v > 0 for v in g2[1:]),
      f"zero_grad_idx={[i for i,v in enumerate(g2) if v==0]}")
check("block_size=0: 全部 query 梯度恒 0 (单元素softmax=恒等, 证实空转)",
      all(v == 0 for v in g0))
print(f"  佐证: layer0.attn_res 只有1个候选 → 梯度必然为0 → g2[0]={g2[0]}")
print(f"  block_size=2 非零梯度示例: layer11.mlp_res query grad = {g2[-1]:.6f}")

print("=" * 60); print("5. 逐层范数稳定性 (12层, 均匀聚合下不应爆炸/消失)")
print("=" * 60)
norms = []
hk = [b.register_forward_hook(lambda m, a, o: norms.append(o[0].float().pow(2).mean().sqrt().item()))
      for b in vlm.layers]
with torch.no_grad():
    model(input_ids=input_ids)
for h in hk: h.remove()
print("  逐层输出 RMS:", [f"{n:.2f}" for n in norms])
check("所有层输出有限", all(n == n and n < 1e4 for n in norms))
check("末层未消失 (RMS > 0.01)", norms[-1] > 0.01, f"last={norms[-1]:.4f}")
growth = max(norms[i+1] / max(norms[i], 1e-8) for i in range(L-1))
check("相邻层增长倍数 < 5", growth < 5, f"max growth={growth:.2f}")

print("=" * 60); print("6. 历史内容值级验证 (块=2层, 存的应是'上一块最终输出')")
print("=" * 60)
in_clones, out_clones, res_calls = [], [], []
def spy2(self, hist, pb):
    actual = orig_fwd(self, hist, pb)
    hs = hist + [pb]   # 手工复算期望聚合值: Σ softmax(query·text_norm(h))·h
    q = self.attn_res_query
    logits = torch.stack([torch.matmul(self.attn_res_norm(h), q) for h in hs])
    alpha = torch.softmax(logits.float(), dim=0).to(pb.dtype)
    exp = sum(a.unsqueeze(-1) * h for a, h in zip(alpha, hs))
    res_calls.append((list(hist), torch.allclose(actual, exp, atol=1e-5)))
    return actual
BlockAttnRes.forward = spy2
h1 = [b.register_forward_pre_hook(lambda m, a: in_clones.append(a[0].detach().clone())) for b in vlm.layers]
h2 = [b.register_forward_hook(lambda m, a, o: out_clones.append(o[0].detach().clone())) for b in vlm.layers]
with torch.no_grad():
    model(input_ids=input_ids)
for h in h1 + h2: h.remove()
BlockAttnRes.forward = orig_fwd

final_hist = res_calls[-1][0]   # 最后一次 mlp_res 收到的完整历史
ap = [l for l in range(L) if l % STEP == 0]   # append 层: 历史第 j 条 = 第 ap[j] 层输入
check("全部聚合调用: 输出 == 手工复算值(内容被正确使用)",
      all(ok for _, ok in res_calls) and len(res_calls) == 2 * L)
check("history[0] == 嵌入输出(第0层输入)",
      torch.allclose(final_hist[0], in_clones[0], atol=1e-6))
check("history[j] == 第j*STEP-1层输出(上一块的块尾), j≥1",
      all(torch.allclose(final_hist[j], out_clones[ap[j] - 1], atol=1e-6)
          for j in range(1, len(final_hist))))
# 块中间输出 = 块内除最后一层之外的层输出; STEP=1 时无中间层, 检查自动跳过
mid = [(j, m) for j in range(1, len(final_hist))
       for m in range(ap[j] - STEP, ap[j] - 1)]
check("history 不含块中间输出",
      all(not torch.allclose(final_hist[j], out_clones[m], atol=1e-4) for j, m in mid))
check("历史无原地污染(前向结束后仍等于捕获时的值)",
      all(torch.allclose(final_hist[j], in_clones[ap[j]], atol=1e-6)
          for j in range(len(final_hist))))

print("=" * 60)
print(f"结果: {len(passed)} PASS, {len(failed)} FAIL")
if failed: print("失败项:", failed); sys.exit(1)
