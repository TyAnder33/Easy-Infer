# CUDA kernel layer

This directory contains model-independent FP32 operations for the first GPU
forward pass. Public declarations live in `include/easyinfer/kernels/ops.hpp`.

The layer currently provides:

- cuBLASLt linear layers and batched attention matrix multiplications
- token and optional learned-position embedding lookup
- LayerNorm and RMSNorm
- GELU, SiLU, SwiGLU, bias, and residual additions
- packed-QKV splitting plus head reshape/merge
- RoPE with half-split and interleaved layouts
- causal softmax and an unfused causal-attention composition
- KV-cache writes and greedy argmax

The CUDA target is intentionally separate from `easyinfer_core`; nothing is
wired into `GPT2::forward()` yet. The existing CPU-only inspector therefore
still builds on machines without CUDA.

Build the kernel library on the GPU machine with:

```sh
cmake -S . -B build-cuda -DEASYINFER_ENABLE_CUDA=ON
cmake --build build-cuda -j
```

This is a correctness-first implementation. It is FP32-only, uses the same
number of query and KV heads, and materializes the attention score matrix.
Mixed precision, grouped-query attention, algorithm caching, fused attention,
and sampling beyond greedy argmax should follow only after numerical tests pass.
