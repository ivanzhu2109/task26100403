# Low-memory Conv2D (NumPy only)

`conv2d.py` implements `Conv2D(w, bias)` / `run(x, out)`: stride-1, no-pad,
ungrouped 2D cross-correlation plus per-output-channel bias, for C-contiguous
float32 arrays. All internal buffers stay within a 512 KiB working-memory
budget by tiling over output rows and input channels, regardless of how large
a single receptive field is.

Run the full acceptance suite (correctness vs float64 reference, large-shape
agreement with the fixed baseline, tracemalloc memory checks incl. a 1 MiB
hog control, and median-of-5 performance assertions vs the baseline):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 -m pytest test_conv2d.py -q
```
