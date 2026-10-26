"""Exercise both real function bodies without importing ComfyUI or model weights."""
import ast
import contextlib
import itertools
import os
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get('SEEDVR2_TEST_DEVICE', 'cpu')
torch.set_num_threads(2)


def load_function(size):
    path = ROOT / f'src/models/dit_{size}/attention.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'pytorch_varlen_attention')
    namespace = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['pytorch_varlen_attention']


def reference(q, k, v, cq, ck, *, causal=False, dropout_p=0., deterministic=False):
    parts = []
    qs = torch.tensor_split(q, cq[1:-1].long().cpu(), dim=0)
    ks = torch.tensor_split(k, ck[1:-1].long().cpu(), dim=0)
    vs = torch.tensor_split(v, ck[1:-1].long().cpu(), dim=0)
    for qi, ki, vi in zip(qs, ks, vs):
        out = F.scaled_dot_product_attention(qi.permute(1, 0, 2).unsqueeze(0),
            ki.permute(1, 0, 2).unsqueeze(0), vi.permute(1, 0, 2).unsqueeze(0),
            dropout_p=0. if deterministic else dropout_p, is_causal=causal)
        parts.append(out.squeeze(0).permute(1, 0, 2))
    return torch.cat(parts, dim=0)


def bits(tensor):
    if tensor.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)
    return tensor.detach().cpu().contiguous().view(torch.uint8)


class PreallocationTests(unittest.TestCase):
    def compare(self, size, dtype, lengths=([5, 17, 3], [9, 11, 7]), strided=False,
                causal=False, value_dim=32, autocast=None, dropout=0., deterministic=False):
        torch.manual_seed(1337)
        ql, kl = lengths
        def make(n, dim):
            t = torch.randn(n, 4, dim * (2 if strided else 1), dtype=dtype, device=DEVICE)
            return t[..., ::2] if strided else t
        q, k, v = make(sum(ql), 32), make(sum(kl), 32), make(sum(kl), value_dim)
        before = [bits(t).clone() for t in (q, k, v)]
        cq = torch.tensor([0, *itertools.accumulate(ql)], dtype=torch.int32, device=DEVICE)
        ck = torch.tensor([0, *itertools.accumulate(kl)], dtype=torch.int32, device=DEVICE)
        outputs = []
        for fn in (reference, load_function(size)):
            torch.manual_seed(19)
            with torch.no_grad(), (torch.autocast(DEVICE, autocast) if autocast else contextlib.nullcontext()):
                outputs.append(fn(q, k, v, cq, ck, causal=causal, dropout_p=dropout, deterministic=deterministic))
        old, new = outputs
        self.assertEqual(old.dtype, new.dtype)
        self.assertEqual(old.shape, new.shape)
        self.assertEqual(old.stride(), new.stride())
        self.assertTrue(torch.equal(bits(old), bits(new)))
        for t, original in zip((q, k, v), before):
            self.assertTrue(torch.equal(bits(t), original))
        if new.numel():
            self.assertNotIn(new.untyped_storage().data_ptr(), [t.untyped_storage().data_ptr() for t in (q, k, v)])

    def test_forward_exact_bits_and_input_preservation(self):
        for size, dtype, strided, causal, lengths in itertools.product(
                ['3b', '7b'], [torch.float16, torch.bfloat16, torch.float32], [False, True],
                [False, True], [([31], [31]), ([5,17,3], [9,11,7]), ([0,7,0], [0,9,0]), ([0], [0])]):
            with self.subTest(size=size, dtype=dtype, strided=strided, causal=causal, lengths=lengths):
                self.compare(size, dtype, lengths, strided, causal)

    def test_result_dtype_under_autocast_and_value_dimension(self):
        for size in ['3b', '7b']:
            self.compare(size, torch.float32, autocast=torch.bfloat16)
            self.compare(size, torch.float32, value_dim=16)

    def test_dropout_retains_rng_and_deterministic_flag_behavior(self):
        for size, deterministic in itertools.product(['3b', '7b'], [False, True]):
            self.compare(size, torch.float32, dropout=.15, deterministic=deterministic)

    def test_backward_preserved(self):
        for size in ['3b', '7b']:
            torch.manual_seed(17)
            base = [torch.randn(12, 2, 8, device=DEVICE) for _ in range(3)]
            cq = torch.tensor([0, 5, 12], device=DEVICE)
            results = []
            for fn in (reference, load_function(size)):
                inputs = [x.clone().requires_grad_() for x in base]
                out = fn(*inputs, cq, cq)
                results.append(torch.autograd.grad(out.square().sum(), inputs))
            for a, b in zip(*results):
                self.assertTrue(torch.equal(bits(a), bits(b)))


if __name__ == '__main__':
    unittest.main()
