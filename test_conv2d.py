"""Acceptance tests for the low-memory Conv2D component.

Run everything (single-threaded BLAS) with:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 python -m pytest test_conv2d.py -q
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import gc
import statistics
import time
import tracemalloc

import numpy as np
import pytest

from conv2d import Conv2D, WORK_BUDGET_BYTES

RTOL = 2e-4
ATOL = 2e-4
PY_OVERHEAD = 64 * 1024  # extra allowance for Python objects under tracemalloc


def reference_conv(x, w, bias):
    """Straightforward float64 loop reference."""
    O, C, Kh, Kw = w.shape
    Ho, Wo = x.shape[1] - Kh + 1, x.shape[2] - Kw + 1
    ref = np.empty((O, Ho, Wo), dtype=np.float64)
    xd = x.astype(np.float64)
    wd = w.astype(np.float64)
    for o in range(O):
        for i in range(Ho):
            for j in range(Wo):
                ref[o, i, j] = np.sum(xd[:, i:i + Kh, j:j + Kw] * wd[o]) + bias[o]
    return ref


@pytest.mark.parametrize("C,H,W,O,Kh,Kw", [
    (1, 5, 5, 1, 3, 3),          # minimal
    (3, 7, 9, 5, 2, 4),          # non-square kernel, ragged sizes
    (5, 11, 6, 4, 5, 1),         # degenerate Kw=1
    (2, 8, 8, 3, 8, 8),          # kernel == input, single output pixel
    (17, 23, 19, 11, 3, 5),      # odd sizes, several tiles
    (64, 14, 37, 9, 4, 7),       # many channels, wide output
])
def test_small_random_vs_float64(C, H, W, O, Kh, Kw):
    rng = np.random.default_rng(1234)
    x = rng.standard_normal((C, H, W)).astype(np.float32)
    w = (rng.standard_normal((O, C, Kh, Kw)) / np.sqrt(C * Kh * Kw)).astype(np.float32)
    bias = rng.standard_normal(O).astype(np.float32)
    out = np.empty((O, H - Kh + 1, W - Kw + 1), dtype=np.float32)
    x0, w0, b0 = x.copy(), w.copy(), bias.copy()

    Conv2D(w, bias).run(x, out)

    ref = reference_conv(x, w, bias)
    np.testing.assert_allclose(out, ref, rtol=RTOL, atol=ATOL)
    # inputs must be untouched
    np.testing.assert_array_equal(x, x0)
    np.testing.assert_array_equal(w, w0)
    np.testing.assert_array_equal(bias, b0)


def test_reuse_across_spatial_sizes():
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((6, 4, 3, 3)) / 3).astype(np.float32)
    bias = rng.standard_normal(6).astype(np.float32)
    conv = Conv2D(w, bias)
    for H, Wd in [(5, 5), (12, 8), (6, 30), (5, 5)]:
        x = rng.standard_normal((4, H, Wd)).astype(np.float32)
        out = np.empty((6, H - 2, Wd - 2), dtype=np.float32)
        conv.run(x, out)
        np.testing.assert_allclose(out, reference_conv(x, w, bias),
                                   rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Fixed performance baseline (do not tune): im2col via one
# sliding_window_view + ascontiguousarray, a single matmul into out's 2D
# view, then an in-place bias add. Everything is timed.
# ---------------------------------------------------------------------------
from numpy.lib.stride_tricks import sliding_window_view


def baseline_run(x, w, bias, out):
    O, C, Kh, Kw = w.shape
    Ho, Wo = x.shape[1] - Kh + 1, x.shape[2] - Kw + 1
    win = sliding_window_view(x, (Kh, Kw), axis=(1, 2))       # (C,Ho,Wo,Kh,Kw)
    col = np.ascontiguousarray(win.transpose(0, 3, 4, 1, 2))  # (C,Kh,Kw,Ho,Wo)
    col2 = col.reshape(C * Kh * Kw, Ho * Wo)
    np.matmul(w.reshape(O, C * Kh * Kw), col2, out=out.reshape(O, Ho * Wo))
    out += bias[:, None, None]
    return out


SHAPES = [
    (32, 258, 258, 64, 3, 3),
    (1024, 31, 31, 32, 16, 16),
]


def make_inputs(C, H, W, O, Kh, Kw, seed=2026):
    rng = np.random.default_rng(seed)
    K = C * Kh * Kw
    x = rng.standard_normal((C, H, W)).astype(np.float32)
    w = (rng.standard_normal((O, C, Kh, Kw)) / np.sqrt(K)).astype(np.float32)
    bias = rng.standard_normal(O).astype(np.float32)
    out = np.empty((O, H - Kh + 1, W - Kw + 1), dtype=np.float32)
    return x, w, bias, out


@pytest.mark.parametrize("C,H,W,O,Kh,Kw", SHAPES)
def test_large_matches_baseline(C, H, W, O, Kh, Kw):
    x, w, bias, out = make_inputs(C, H, W, O, Kh, Kw)
    ref = baseline_run(x, w, bias, np.empty_like(out))
    Conv2D(w, bias).run(x, out)
    np.testing.assert_allclose(out, ref, rtol=RTOL, atol=ATOL)


MEM_LIMIT = WORK_BUDGET_BYTES + PY_OVERHEAD  # 512 KiB buffers + 64 KiB Python


def _numpy_warmup():
    a = np.empty((64, 256), dtype=np.float32)
    b = np.empty((256, 64), dtype=np.float32)
    np.matmul(a, b, out=np.empty((64, 64), dtype=np.float32))
    sliding_window_view(np.empty((2, 8, 8), dtype=np.float32), (3, 3), axis=(1, 2))


def _measured_peak(conv_factory, x, w, bias, out):
    """Trace from before construction through consecutive runs."""
    _numpy_warmup()
    gc.collect()
    tracemalloc.start()
    try:
        conv = conv_factory(w, bias)
        conv.run(x, out)          # first call: allocates persistent buffers
        conv.run(x, out)          # consecutive call: buffers reused
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


@pytest.mark.parametrize("C,H,W,O,Kh,Kw", SHAPES)
def test_memory_within_budget(C, H, W, O, Kh, Kw):
    x, w, bias, out = make_inputs(C, H, W, O, Kh, Kw)
    peak = _measured_peak(Conv2D, x, w, bias, out)
    assert peak <= MEM_LIMIT, f"peak {peak} bytes exceeds {MEM_LIMIT}"


class _WastefulConv(Conv2D):
    """Control: deliberately allocates an extra 1 MiB array per run."""

    def run(self, x, out):
        self._junk = np.empty(1024 * 1024 // 4, dtype=np.float32)  # 1 MiB
        super().run(x, out)


def test_memory_check_catches_1mib_hog():
    C, H, W, O, Kh, Kw = SHAPES[0]
    x, w, bias, out = make_inputs(C, H, W, O, Kh, Kw)
    peak = _measured_peak(_WastefulConv, x, w, bias, out)
    assert peak > MEM_LIMIT, "control must FAIL the memory check"


MAX_SLOWDOWN = 4.0


@pytest.mark.parametrize("C,H,W,O,Kh,Kw", SHAPES)
def test_performance_vs_baseline(C, H, W, O, Kh, Kw):
    conv = Conv2D(*make_inputs(C, H, W, O, Kh, Kw)[1:3])
    out_ours = np.empty((O, H - Kh + 1, W - Kw + 1), dtype=np.float32)

    for r in range(2):  # warmup, untimed
        x, w, bias, out_b = make_inputs(C, H, W, O, Kh, Kw, seed=100 + r)
        baseline_run(x, w, bias, out_b)
        conv.run(x, out_ours)

    t_base, t_ours = [], []
    for r in range(5):  # 5 alternating rounds, fresh inputs each round
        x, w, bias, out_b = make_inputs(C, H, W, O, Kh, Kw, seed=2026 + r)

        def time_base():
            t0 = time.perf_counter()
            baseline_run(x, w, bias, out_b)
            return time.perf_counter() - t0

        def time_ours():
            t0 = time.perf_counter()
            conv.run(x, out_ours)
            return time.perf_counter() - t0

        if r % 2 == 0:  # alternate who runs first
            tb, to = time_base(), time_ours()
        else:
            to, tb = time_ours(), time_base()
        t_base.append(tb)
        t_ours.append(to)

    med_base, med_ours = statistics.median(t_base), statistics.median(t_ours)
    ratio = med_ours / med_base
    print(f"\nshape {(C, H, W, O, Kh, Kw)}: baseline {med_base:.4f}s, "
          f"ours {med_ours:.4f}s, ratio {ratio:.2f}x")
    assert med_ours <= MAX_SLOWDOWN * med_base, (
        f"too slow: {med_ours:.4f}s vs baseline {med_base:.4f}s "
        f"({ratio:.2f}x > {MAX_SLOWDOWN}x)")
