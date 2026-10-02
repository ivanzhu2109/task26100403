"""Low-memory Conv2D (cross-correlation) component built on NumPy only.

API:
    conv = Conv2D(w, bias)   # w: (O, C, Kh, Kw) float32, C-contiguous
    conv.run(x, out)         # x: (C, H, W), out: (O, H-Kh+1, W-Kw+1)

Stride 1, no padding, no groups, plus a per-output-channel bias.
`run` overwrites `out` and never modifies x, w or bias. One instance may
process different spatial sizes back to back (no concurrency assumed).

Working memory (all internal buffers) stays within 512 KiB regardless of
input size. A single receptive field may exceed that budget: the kernel
dimension is then tiled channel-wise so execution is never rejected.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided, sliding_window_view

# Working-memory budget for internal buffers (bytes).
WIN_BUDGET = 256 * 1024   # im2col tile buffer
ACC_BUDGET = 64 * 1024    # output accumulator buffer (a same-sized GEMM
                          # result temporary may coexist with it)
MEM_LIMIT = 512 * 1024    # total working-memory limit (excl. caller arrays)

_L_TARGET = 256           # preferred output positions per GEMM tile


def _block_plan(C, O, Kh, Kw, Ho, Wo):
    """Pick (rows_blk, cols_blk, ch_tile) for the blocked im2col.

    The im2col tile holds rows_blk*cols_blk output positions by
    ch_tile*Kh*Kw kernel elements and stays within WIN_BUDGET; the (O, L)
    accumulator stays within ACC_BUDGET. Channel tiling guarantees progress
    even when one full receptive field exceeds the budget.
    """
    win_elems = WIN_BUDGET // 4   # float32 elements
    acc_elems = ACC_BUDGET // 4
    kpc = Kh * Kw

    l_cap = max(1, acc_elems // O)          # accumulator is (O, L_blk)
    l_want = min(_L_TARGET, l_cap)
    if Wo >= l_want:
        rows_blk, cols_blk = 1, l_want
    else:
        rows_blk = max(1, min(Ho, l_want // Wo))
        cols_blk = Wo
    l_blk = rows_blk * cols_blk

    ch_tile = min(C, win_elems // (l_blk * kpc))
    if ch_tile < 1:
        # Even one channel of windows for l_blk positions is too big:
        # fall back to a single row, then to a column chunk.
        rows_blk = 1
        cols_blk = min(Wo, max(1, win_elems // kpc))
        l_blk = cols_blk
        ch_tile = max(1, min(C, win_elems // (l_blk * kpc)))
    return rows_blk, cols_blk, ch_tile


def plan_buffer_bytes(C, O, Kh, Kw, Ho, Wo):
    """Total bytes of internal buffers a run with these shapes allocates."""
    rows_blk, cols_blk, ch_tile = _block_plan(C, O, Kh, Kw, Ho, Wo)
    l_max = rows_blk * cols_blk
    k_max = ch_tile * Kh * Kw
    win_buf = k_max * l_max * 4
    acc_buf = O * l_max * 4
    return win_buf + acc_buf + acc_buf  # accumulator + GEMM temp


class Conv2D:
    """Blocked im2col + GEMM conv2d with bounded working memory."""

    def __init__(self, w, bias):
        w = np.asanyarray(w)
        bias = np.asanyarray(bias)
        if w.ndim != 4 or bias.ndim != 1 or bias.shape[0] != w.shape[0]:
            raise ValueError("w must be (O, C, Kh, Kw), bias must be (O,)")
        if w.dtype != np.float32 or bias.dtype != np.float32:
            raise TypeError("w and bias must be float32")
        if not w.flags.c_contiguous or not bias.flags.c_contiguous:
            raise ValueError("w and bias must be C-contiguous")
        self._w = w
        self._bias = bias

    @property
    def w(self):
        return self._w

    @property
    def bias(self):
        return self._bias

    def run(self, x, out):
        w = self._w
        bias = self._bias
        if x.ndim != 3 or out.ndim != 3:
            raise ValueError("x must be (C, H, W), out must be (O, Ho, Wo)")
        if x.dtype != np.float32 or out.dtype != np.float32:
            raise TypeError("x and out must be float32")
        if not x.flags.c_contiguous or not out.flags.c_contiguous:
            raise ValueError("x and out must be C-contiguous")
        C, H, W = x.shape
        O, Cw, Kh, Kw = w.shape
        Ho, Wo = out.shape[1], out.shape[2]
        if (Cw != C or out.shape[0] != O or bias.shape[0] != O
                or Ho != H - Kh + 1 or Wo != W - Kw + 1 or Ho < 1 or Wo < 1):
            raise ValueError("shape mismatch between x, w, bias and out")

        rows_blk, cols_blk, ch_tile = _block_plan(C, O, Kh, Kw, Ho, Wo)
        kpc = Kh * Kw
        l_max = rows_blk * cols_blk
        k_max = ch_tile * kpc

        w2d = w.reshape(O, C * kpc)              # view, no copy
        out2d = out.reshape(O, Ho * Wo)          # view, no copy
        win = sliding_window_view(x, (Kh, Kw), axis=(1, 2))  # (C,Ho,Wo,Kh,Kw)

        xb = np.empty((k_max, l_max), dtype=np.float32)   # im2col tile
        acc = np.empty((O, l_max), dtype=np.float32)      # output tile
        item = xb.strides[-1]
        s_k, s_l = xb.strides[0] // item, xb.strides[1] // item

        for i0 in range(0, Ho, rows_blk):
            i1 = min(i0 + rows_blk, Ho)
            r = i1 - i0
            for j0 in range(0, Wo, cols_blk):
                j1 = min(j0 + cols_blk, Wo)
                cb = j1 - j0
                lb = r * cb
                acc_blk = acc[:, :lb]
                acc_blk[:] = bias[:, None]
                for c0 in range(0, C, ch_tile):
                    c1 = min(c0 + ch_tile, C)
                    ct = c1 - c0
                    kb = ct * kpc
                    # Gather windows into xb[:kb, :lb], laid out (c,p,q | i,j):
                    # element (ch,p,q,rr,cc) maps to offset
                    # (ch*kpc + p*Kw + q)*s_k + (rr*cb + cc)*s_l.
                    dest = as_strided(
                        xb,
                        shape=(ct, Kh, Kw, r, cb),
                        strides=(kpc * s_k * item, Kw * s_k * item,
                                 s_k * item, cb * s_l * item, s_l * item),
                    )
                    np.copyto(dest, win[c0:c1, i0:i1, j0:j1]
                              .transpose(0, 3, 4, 1, 2))
                    acc_blk += w2d[:, c0 * kpc:c1 * kpc] @ xb[:kb, :lb]
                out2d[:, i0 * Wo + j0:(i1 - 1) * Wo + j1] = acc_blk
        return out
