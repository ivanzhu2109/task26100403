# Low-memory Conv2D (NumPy only)

`conv2d.py` implements stride-1 / no-padding / no-grouping 2D cross-correlation
with per-output-channel bias:

```python
from conv2d import Conv2D
conv = Conv2D(w, bias)   # w: (O, C, Kh, Kw), bias: (O,), float32 C-contiguous
conv.run(x, out)         # x: (C, H, W), out: (O, H-Kh+1, W-Kw+1), overwritten
```

`run` never modifies `x`, `w` or `bias`, and one instance may process
different spatial sizes back to back.

## How it stays within 512 KiB

Instead of materializing the full im2col matrix, `run` tiles both the output
positions (row/column blocks) and the kernel dimension (channel tiles). Each
tile gathers a `(K_tile, L_tile)` window block via `sliding_window_view` +
one contiguous copy, then accumulates `w2d[:, tile] @ block` into an
`(O, L_tile)` accumulator pre-initialized with the bias. Buffer sizes are
planned by `_block_plan` (im2col tile ≤ 256 KiB, accumulator + GEMM temp
≤ 64 KiB each); channel tiling guarantees progress even when a single
receptive field exceeds the budget (acceptance case 2: 1 MiB per field).

## Acceptance

```sh
python3 run_acceptance.py
```

This forces single-threaded BLAS (`VECLIB_MAXIMUM_THREADS=1` etc.) and runs
the full suite in `test_conv2d.py`:

- small random shapes (incl. non-square kernels, ragged dims) vs an
  elementwise float64 reference, `rtol=atol=2e-4`;
- both large shapes `(32,258,258,64,3,3)` and `(1024,31,31,32,16,16)`
  checked against the one-shot im2col+GEMM baseline;
- tracemalloc peak from before construction through consecutive calls
  asserted ≤ 512 KiB + 64 KiB Python overhead, plus a control proving an
  extra hidden 1 MiB buffer fails the check;
- performance: 2 warmup + 5 alternating timed rounds with fresh inputs,
  median of each, asserted ≤ 4x the baseline on both shapes.
