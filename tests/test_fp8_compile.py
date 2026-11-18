"""Native FP8 fullgraph capture, delayed state and gradient tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import torch
from fp8_linear import _FP8Linear, _quantize, FP8Controller


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class FP8CompileTest(unittest.TestCase):
    def prepared(self):
        torch.manual_seed(42)
        controller=FP8Controller('cuda',scale_interval=2,check_interval=100)
        state=controller.new_state()
        x=torch.randn(32,64,device='cuda',requires_grad=True)
        w=torch.randn(48,64,device='cuda',requires_grad=True)
        _FP8Linear.apply(x,w,None,state).float().square().mean().backward()
        controller.finish_calibration()
        x.grad=w.grad=None
        return x,w,state,controller

    def test_fullgraph_static(self): self.capture(False)
    def test_fullgraph_dynamic(self): self.capture(True)

    def capture(self,dynamic):
        x,w,state,controller=self.prepared()
        graphs=[]
        def backend(gm,inputs):
            graphs.append(gm)
            return torch._inductor.compile(gm,inputs)
        def fn(x,w): return _FP8Linear.apply(x,w,None,state).float().square().mean()
        compiled=torch.compile(fn,backend=backend,fullgraph=True,dynamic=dynamic)
        scales=state.scales.clone()
        for step in range(3):
            x.grad=w.grad=None
            loss=compiled(x,w);loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all())
            torch.testing.assert_close(state.amax[0],x.detach().bfloat16().float().abs().amax())
            torch.testing.assert_close(state.amax[1],w.detach().bfloat16().float().abs().amax())
            controller.after_step(loss,None)
            if step==0: torch.testing.assert_close(state.scales,scales,rtol=0,atol=0)
        self.assertEqual(controller.scale_updates,2) # bootstrap plus step 2
        self.assertEqual(controller.host_checks,0)
        controller.check_now(loss)
        self.assertTrue(graphs)
        self.assertEqual(len(graphs),1)
        if dynamic:
            # Different token counts, same feature/weight dimensions.
            for rows in (48,64):
                xx=torch.randn(rows,64,device='cuda',requires_grad=True)
                result=compiled(xx,w);result.backward()
                self.assertTrue(torch.isfinite(result))
            self.assertEqual(len(graphs),1)

    def test_nonfinite_flag_on_device(self):
        controller=FP8Controller('cuda')
        state=controller.new_state()
        state.calibrating=False
        for value in (float('nan'),float('inf')):
            state.amax.zero_()
            _quantize(torch.full((32,64),value,device='cuda'),state.scales[0],state.amax[0])
            with self.assertRaises(FloatingPointError): controller.check_now()

    def test_periodic_check_one_host_read(self):
        controller=FP8Controller('cuda',scale_interval=2,check_interval=2)
        controller.new_state()
        controller.after_step(torch.tensor(float('nan'),device='cuda'),None)
        self.assertEqual(controller.host_checks,0)
        with self.assertRaises(FloatingPointError):
            controller.after_step(torch.tensor(1.,device='cuda'),None)
        self.assertEqual(controller.host_checks,1)


if __name__=='__main__': unittest.main()
