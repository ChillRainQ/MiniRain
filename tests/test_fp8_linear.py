"""Real native FP8 forward/backward tests (CUDA SM89+)."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from fp8_linear import _FP8Linear, _quantize, probe_backend, enable_fp8


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class FP8Test(unittest.TestCase):
    def test_runtime_backend(self):
        self.assertTrue(probe_backend('cuda:0')['scaled_mm_probe_passed'])

    def test_padded_forward_and_backward_against_bf16(self):
        torch.manual_seed(42)
        x = torch.randn(2, 7, 19, device='cuda', requires_grad=True)
        w = torch.randn(23, 19, device='cuda', requires_grad=True)
        b = torch.randn(23, device='cuda', requires_grad=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            expected = F.linear(x, w, b)
        stats = {}
        actual = _FP8Linear.apply(x, w, b, stats)
        g = torch.randn_like(expected)
        ref = torch.autograd.grad(expected, (x, w, b), g)
        grads = torch.autograd.grad(actual, (x, w, b), g)
        for a, e in ((actual, expected), *zip(grads, ref)):
            self.assertTrue(torch.isfinite(a).all())
            relative_rmse = (a.float()-e.float()).square().mean().sqrt() / e.float().square().mean().sqrt()
            self.assertLess(relative_rmse.item(), 0.15)
        self.assertEqual(stats['scaled_mm_calls'], 3)
        self.assertEqual(w.dtype, torch.float32)

    def test_zero_scale_and_nonfinite(self):
        q, s = _quantize(torch.zeros(16, 16, device='cuda'), {})
        self.assertTrue(torch.isfinite(s).all())
        self.assertTrue(q.float().eq(0).all())
        for value in (float('inf'), float('nan')):
            with self.assertRaises(FloatingPointError):
                _quantize(torch.full((16, 16), value, device='cuda'), {})

    def test_parameter_identity_and_bf16_logits(self):
        from components.mlp import FusedSwiGLU, MLPArgs
        model = torch.nn.Module()
        model.embed = torch.nn.Embedding(64, 32).cuda()
        model.mlp = FusedSwiGLU(MLPArgs(32, 64, False)).cuda()
        model.lm_head = torch.nn.Linear(32, 64, bias=False).cuda()
        model.lm_head.weight = model.embed.weight
        before = {n: id(p) for n, p in model.named_parameters()}
        info = enable_fp8(model)
        self.assertEqual(before, {n: id(p) for n, p in model.named_parameters()})
        self.assertIs(model.lm_head.weight, model.embed.weight)
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        ids = torch.randint(0, 64, (2, 8), device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            embedding = model.embed(ids)
            logits = model.lm_head(model.mlp(embedding))
            loss = F.cross_entropy(logits.flatten(0, 1), ids.flatten())
        self.assertEqual(embedding.dtype, torch.float32)
        self.assertEqual(logits.dtype, torch.bfloat16)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreaterEqual(info['stats']['scaled_mm_calls'], 6)


if __name__ == '__main__':
    unittest.main()
