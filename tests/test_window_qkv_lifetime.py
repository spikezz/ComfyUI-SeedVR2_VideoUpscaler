"""Run the real 7B forward body and observe when its original QKV dies on CPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
import unittest
import weakref

from einops import rearrange
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class Cache:
    def __init__(self, values=None):
        self.values = values or {}

    def namespace(self, _name):
        return self

    def __call__(self, key, factory):
        if key not in self.values:
            self.values[key] = factory()
        return self.values[key]


def forward(retain_original=False):
    path = ROOT / 'src/models/dit_7b/nablocks/mmsr_block.py'
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'NaSwinAttention')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    if retain_original:
        fn.body = [n for n in fn.body if not (
            isinstance(n, ast.Assign) and ast.unparse(n) == 'vid_qkv = None')]
    namespace = dict(torch=torch, Tuple=Tuple, Cache=Cache, rearrange=rearrange,
                     gather_seq_scatter_heads_qkv=lambda x, **_: x,
                     gather_heads_scatter_seq=lambda x, **_: x,
                     safe_pad_operation=F.pad)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['forward']


def run(dtype, strided, retain_original):
    torch.manual_seed(19)
    vid = torch.randn(6, 16 if strided else 8, dtype=dtype)
    txt = torch.randn(2, 16 if strided else 8, dtype=dtype)
    if strided:
        vid, txt = vid[:, ::2], txt[:, ::2]
    inputs = [x.clone() for x in (vid, txt)]
    projection = torch.nn.Linear(8, 24).to(dtype)
    output_projection = torch.nn.Linear(8, 8).to(dtype)
    observed = {}

    def project(v, t):
        projected = projection(v)
        observed['original'] = weakref.ref(projected)
        return projected, projection(t)

    order = torch.tensor([2, 3, 0, 1, 5, 4])
    def partition(x):
        result = x[order]
        assert result.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
        return result

    def concat(v, t):
        return torch.cat([v[:3], t, v[3:], t])

    def unconcat(x):
        return torch.cat([x[:3], x[5:8]]), torch.stack([x[3:5], x[8:10]]).mean(0)

    def attention(q, k, v, **kwargs):
        observed['alive_at_attention'] = observed['original']() is not None
        chunks = []
        for i in (0, 5):
            chunks.append(F.scaled_dot_product_attention(
                q[i:i+5].transpose(0, 1), k[i:i+5].transpose(0, 1),
                v[i:i+5].transpose(0, 1)).transpose(0, 1))
        return torch.cat(chunks)

    model = SimpleNamespace(proj_qkv=project, window_method='test', window=(1, 1, 3),
        head_dim=4, norm_q=lambda v,t:(v,t), norm_k=lambda v,t:(v,t), rope=None,
        attn=attention, proj_out=lambda v,t:(output_projection(v),output_projection(t)))
    cache = Cache({'win_transform': (partition, lambda x:x[order.argsort()],
                                    torch.tensor([[1, 1, 3], [1, 1, 3]]), torch.tensor([2])),
                   'mm_pnp': (concat, unconcat)})
    with torch.no_grad():
        result = forward(retain_original)(model, vid, txt, torch.tensor([[1, 2, 3]]),
                                          torch.tensor([[2]]), cache)
    assert all(torch.equal(x, y) for x, y in zip(inputs, (vid, txt)))
    assert observed['original']() is None
    return result, observed['alive_at_attention']


class WindowQKVLifetimeTests(unittest.TestCase):
    def test_original_projection_dies_before_attention_and_outputs_stay_exact(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for strided in (False, True):
                with self.subTest(dtype=dtype, strided=strided):
                    old, old_alive = run(dtype, strided, True)
                    new, new_alive = run(dtype, strided, False)
                    self.assertTrue(old_alive, 'The reference must reproduce the old retention')
                    self.assertFalse(new_alive, 'Original QKV must be gone before attention')
                    for a, b in zip(old, new):
                        self.assertEqual(a.dtype, b.dtype)
                        self.assertEqual(a.shape, b.shape)
                        self.assertEqual(a.stride(), b.stride())
                        self.assertTrue(torch.equal(a.contiguous().view(torch.uint8),
                                                    b.contiguous().view(torch.uint8)))


if __name__ == '__main__':
    unittest.main()
