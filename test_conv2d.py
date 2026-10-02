"""Acceptance tests for the low-memory Conv2D component."""

import os

for _var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import unittest

import numpy as np

from conv2d import Conv2D

RTOL = 2e-4
ATOL = 2e-4


def make_data(C, H, W, O, Kh, Kw, seed=2026):
    """Standard-normal float32 inputs; weights scaled by 1/sqrt(K)."""
    rng = np.random.default_rng(seed)
    K = C * Kh * Kw
    x = rng.standard_normal((C, H, W)).astype(np.float32)
    w = (rng.standard_normal((O, C, Kh, Kw)) / np.sqrt(K)).astype(np.float32)
    bias = rng.standard_normal(O).astype(np.float32)
    out = np.empty((O, H - Kh + 1, W - Kw + 1), dtype=np.float32)
    return x, w, bias, out


def ref_conv2d_float64(x, w, bias):
    """Independent elementwise float64 reference (explicit loops)."""
    O, C, Kh, Kw = w.shape
    Ho = x.shape[1] - Kh + 1
    Wo = x.shape[2] - Kw + 1
    x64 = np.asarray(x, dtype=np.float64)
    w64 = np.asarray(w, dtype=np.float64)
    b64 = np.asarray(bias, dtype=np.float64)
    ref = np.empty((O, Ho, Wo), dtype=np.float64)
    for o in range(O):
        for i in range(Ho):
            for j in range(Wo):
                acc = 0.0
                for c in range(C):
                    acc += np.sum(x64[c, i:i + Kh, j:j + Kw] * w64[o, c])
                ref[o, i, j] = acc + b64[o]
    return ref


class TestCorrectness(unittest.TestCase):
    SHAPES = [
        # (C, H, W, O, Kh, Kw)
        (1, 1, 1, 1, 1, 1),      # minimal
        (3, 7, 5, 2, 2, 4),      # non-square kernel, non-square image
        (5, 6, 9, 3, 3, 2),      # non-square kernel, ragged dims
        (2, 11, 13, 5, 3, 3),    # prime-ish dims
        (4, 5, 5, 2, 5, 5),      # kernel == image: single output pixel
        (1, 9, 4, 3, 1, 4),      # Kh=1, Kw==W
        (3, 8, 8, 1, 1, 1),      # 1x1 kernel
        (6, 10, 7, 5, 4, 1),     # Kw=1
        (2, 6, 6, 3, 6, 1),      # Kh==H, Kw=1
    ]

    def test_small_shapes_vs_float64_reference(self):
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                x, w, bias, out = make_data(*shape)
                out[...] = np.nan  # prove run overwrites every element
                Conv2D(w, bias).run(x, out)
                ref = ref_conv2d_float64(x, w, bias)
                self.assertFalse(np.isnan(out).any())
                np.testing.assert_allclose(out, ref, rtol=RTOL, atol=ATOL)

    def test_inputs_not_modified(self):
        x, w, bias, out = make_data(3, 8, 9, 4, 3, 2)
        x0, w0, b0 = x.copy(), w.copy(), bias.copy()
        Conv2D(w, bias).run(x, out)
        np.testing.assert_array_equal(x, x0)
        np.testing.assert_array_equal(w, w0)
        np.testing.assert_array_equal(bias, b0)

    def test_instance_reused_across_spatial_sizes(self):
        C, O, Kh, Kw = 3, 4, 3, 2
        _, w, bias, _ = make_data(C, 5, 5, O, Kh, Kw)
        conv = Conv2D(w, bias)
        for H, W in ((5, 5), (9, 4), (3, 2)):
            with self.subTest(H=H, W=W):
                x, _, _, out = make_data(C, H, W, O, Kh, Kw, seed=H * 100 + W)
                conv.run(x, out)
                ref = ref_conv2d_float64(x, w, bias)
                np.testing.assert_allclose(out, ref, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    unittest.main()


import tracemalloc

from conv2d import MEM_LIMIT, plan_buffer_bytes

PY_OVERHEAD = 64 * 1024  # allowed Python-object overhead on top of MEM_LIMIT

CASE1 = (32, 258, 258, 64, 3, 3)
CASE2 = (1024, 31, 31, 32, 16, 16)


def _traced_peak(fn):
    """Peak bytes traced by tracemalloc while fn runs."""
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


class TestMemory(unittest.TestCase):
    def test_buffer_plan_within_limit(self):
        for shape in (CASE1, CASE2):
            C, H, W, O, Kh, Kw = shape
            with self.subTest(shape=shape):
                self.assertLessEqual(
                    plan_buffer_bytes(C, O, Kh, Kw, H - Kh + 1, W - Kw + 1),
                    MEM_LIMIT)

    def test_traced_peak_within_limit(self):
        limit = MEM_LIMIT + PY_OVERHEAD
        for shape in (CASE1, CASE2):
            C, H, W, O, Kh, Kw = shape
            with self.subTest(shape=shape):
                # Caller arrays and a second spatial size, all before tracing.
                x1, w, bias, out1 = make_data(*shape)
                x2, _, _, out2 = make_data(C, H - 3, W + 5, O, Kh, Kw, seed=7)
                np.matmul(np.eye(4, dtype=np.float32),
                          np.eye(4, dtype=np.float32))  # NumPy/BLAS warmup

                def workload():
                    conv = Conv2D(w, bias)      # construction is traced
                    conv.run(x1, out1)          # consecutive calls ...
                    conv.run(x2, out2)          # ... on different sizes
                    conv.run(x1, out1)

                peak = _traced_peak(workload)
                self.assertLessEqual(
                    peak, limit,
                    f"traced peak {peak} bytes exceeds {limit} bytes")

    def test_memory_check_catches_extra_1mib(self):
        """Control: a component hiding an extra 1 MiB must fail the check."""
        limit = MEM_LIMIT + PY_OVERHEAD
        x, w, bias, out = make_data(*CASE1)
        np.matmul(np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32))

        def leaky_workload():
            conv = Conv2D(w, bias)
            junk = np.zeros(1 << 20, dtype=np.uint8)  # hidden 1 MiB buffer
            conv.run(x, out)
            return junk

        peak = _traced_peak(leaky_workload)
        self.assertGreater(peak, limit,
                           "memory check failed to detect an extra 1 MiB")


import time
from statistics import median

from numpy.lib.stride_tricks import sliding_window_view

MAX_SLOWDOWN = 4.0  # component must stay within 4x of the baseline
WARMUP_ROUNDS = 2
TIMED_ROUNDS = 5


def baseline_conv2d(x, w, bias, out):
    """Reference baseline: one-shot im2col + single GEMM + in-place bias."""
    O, C, Kh, Kw = w.shape
    Ho = x.shape[1] - Kh + 1
    Wo = x.shape[2] - Kw + 1
    K, L = C * Kh * Kw, Ho * Wo
    win = sliding_window_view(x, (Kh, Kw), axis=(1, 2))  # (C,Ho,Wo,Kh,Kw)
    win = win.transpose(0, 3, 4, 1, 2)                   # (C,Kh,Kw,Ho,Wo)
    cols = np.ascontiguousarray(win).reshape(K, L)
    np.matmul(w.reshape(O, K), cols, out=out.reshape(O, L))
    out += bias.reshape(O, 1, 1)
    return out


class TestLargeAndPerformance(unittest.TestCase):
    def test_large_shapes_match_baseline(self):
        for shape in (CASE1, CASE2):
            with self.subTest(shape=shape):
                x, w, bias, out = make_data(*shape)
                ref = baseline_conv2d(x, w, bias, np.empty_like(out))
                Conv2D(w, bias).run(x, out)
                np.testing.assert_allclose(out, ref, rtol=RTOL, atol=ATOL)

    def _bench(self, shape):
        C, H, W, O, Kh, Kw = shape
        rng = np.random.default_rng(2026)

        def fresh():
            x = rng.standard_normal((C, H, W)).astype(np.float32)
            w = (rng.standard_normal((O, C, Kh, Kw))
                 / np.sqrt(C * Kh * Kw)).astype(np.float32)
            b = rng.standard_normal(O).astype(np.float32)
            out = np.empty((O, H - Kh + 1, W - Kw + 1), dtype=np.float32)
            return x, w, b, out

        def timed(fn):
            x, w, b, out = fresh()      # fresh inputs, generation untimed
            t0 = time.perf_counter()
            fn(x, w, b, out)
            return time.perf_counter() - t0

        def ours(x, w, b, out):
            Conv2D(w, b).run(x, out)

        for _ in range(WARMUP_ROUNDS):
            timed(ours)
            timed(baseline_conv2d)
        t_ours, t_base = [], []
        for _ in range(TIMED_ROUNDS):   # alternating measurement
            t_ours.append(timed(ours))
            t_base.append(timed(baseline_conv2d))
        return median(t_ours), median(t_base)

    def test_performance_within_4x_baseline(self):
        for shape in (CASE1, CASE2):
            with self.subTest(shape=shape):
                t_ours, t_base = self._bench(shape)
                ratio = t_ours / t_base
                print(f"\n{shape}: ours {t_ours*1e3:.1f} ms, "
                      f"baseline {t_base*1e3:.1f} ms, ratio {ratio:.2f}x")
                self.assertLessEqual(
                    ratio, MAX_SLOWDOWN,
                    f"{shape}: {ratio:.2f}x slower than baseline "
                    f"(limit {MAX_SLOWDOWN}x)")


if __name__ == "__main__":
    unittest.main()
