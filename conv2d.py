"""Low-memory Conv2D (cross-correlation) component built on NumPy only.

Public API:
    conv = Conv2D(w, bias)   # w: (O, C, Kh, Kw) float32, bias: (O,) float32
    conv.run(x, out)         # x: (C, H, W) float32, out: (O, H-Kh+1, W-Kw+1)

Only stride=1, no padding, no groups. `run` overwrites `out` and never
modifies `x`, `w` or `bias`. The same object may be reused across calls
with different spatial sizes (not thread-safe).

Memory strategy: the im2col expansion is computed tile by tile, tiling
over output-row blocks (hb) and input-channel chunks (cc). All internal
buffers (one im2col tile + two O x tile accumulators) are persistent and
their total size never exceeds WORK_BUDGET_BYTES, no matter how large a
single receptive field (C*Kh*Kw) is.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Hard working-memory budget for all internal buffers (persistent or
# temporary), excluding the caller's x / w / bias / out arrays.
WORK_BUDGET_BYTES = 512 * 1024

_ITEMSIZE = 4  # float32


class Conv2D:
    """Tiled, memory-bounded 2D cross-correlation with per-channel bias."""

    def __init__(self, w: np.ndarray, bias: np.ndarray) -> None:
        w = np.asarray(w)
        bias = np.asarray(bias)
        if w.ndim != 4:
            raise ValueError("w must have shape (O, C, Kh, Kw)")
        if w.dtype != np.float32 or bias.dtype != np.float32:
            raise ValueError("w and bias must be float32")
        if bias.shape != (w.shape[0],):
            raise ValueError("bias must have shape (O,)")
        self._w = w
        self._bias = bias
        self._w2d = w.reshape(w.shape[0], -1)  # (O, C*Kh*Kw) view
        self._buf: np.ndarray | None = None  # persistent im2col tile buffer
        self._acc: np.ndarray | None = None  # persistent tile accumulator
        self._scr: np.ndarray | None = None  # persistent matmul scratch

    # -- internal helpers -------------------------------------------------

    @staticmethod
    def _ensure(buf: np.ndarray | None, elems: int) -> np.ndarray:
        if buf is None or buf.size < elems:
            buf = np.empty(elems, dtype=np.float32)
        return buf

    def _tile_shape(self, C: int, O: int, Kh: int, Kw: int,
                    Ho: int, Wo: int) -> tuple[int, int]:
        """Pick (hb, cc): output-row block and channel-chunk sizes.

        Maximizes the smaller matmul dimension (min of K = cc*Kh*Kw and
        L = hb*Wo), which keeps the tiled GEMMs efficient, subject to
        buf + acc + scr <= WORK_BUDGET_BYTES. Never rejects: in the worst
        case it falls back to hb = cc = 1.
        """
        best_hb, best_cc, best_score = 1, 1, -1
        for hb in range(Ho, 0, -1):
            acc_bytes = 2 * O * hb * Wo * _ITEMSIZE
            per_cc = Kh * Kw * hb * Wo * _ITEMSIZE
            room = WORK_BUDGET_BYTES - acc_bytes
            cc = min(C, room // per_cc)
            if cc < 1:
                continue  # this hb cannot fit even one channel chunk
            score = min(cc * Kh * Kw, hb * Wo)
            if score > best_score:
                best_hb, best_cc, best_score = hb, int(cc), score
        return best_hb, best_cc

    # -- public API -------------------------------------------------------

    def run(self, x: np.ndarray, out: np.ndarray) -> None:
        w, bias = self._w, self._bias
        O, C, Kh, Kw = w.shape
        if x.ndim != 3 or x.shape[0] != C:
            raise ValueError("x must have shape (C, H, W)")
        Ho, Wo = x.shape[1] - Kh + 1, x.shape[2] - Kw + 1
        if Ho < 1 or Wo < 1:
            raise ValueError("kernel larger than input")
        if out.shape != (O, Ho, Wo):
            raise ValueError("out must have shape (O, H-Kh+1, W-Kw+1)")
        if x.dtype != np.float32 or out.dtype != np.float32:
            raise ValueError("x and out must be float32")

        hb, cc = self._tile_shape(C, O, Kh, Kw, Ho, Wo)

        buf_elems = cc * Kh * Kw * hb * Wo
        acc_elems = O * hb * Wo
        self._buf = self._ensure(self._buf, buf_elems)
        self._acc = self._ensure(self._acc, acc_elems)
        self._scr = self._ensure(self._scr, acc_elems)
        buf, acc, scr = self._buf, self._acc, self._scr
        w2d, w = self._w2d, self._w  # locals; w never written

        for r0 in range(0, Ho, hb):
            h = min(hb, Ho - r0)
            cols = h * Wo
            acc2 = acc[: O * cols].reshape(O, cols)
            scr2 = scr[: O * cols].reshape(O, cols)
            first = True
            for c0 in range(0, C, cc):
                c = min(cc, C - c0)
                # (c, h, Wo, Kh, Kw) window view of x, no copy
                win = sliding_window_view(
                    x[c0:c0 + c, r0:r0 + h + Kh - 1, :], (Kh, Kw), axis=(1, 2))
                # gather tile as (c, Kh, Kw, h, Wo) into the persistent buffer
                tile = buf[: c * Kh * Kw * cols].reshape(c, Kh, Kw, h, Wo)
                np.copyto(tile, win.transpose(0, 3, 4, 1, 2))
                tile2 = tile.reshape(c * Kh * Kw, cols)
                wblk = w2d[:, c0 * Kh * Kw:(c0 + c) * Kh * Kw]
                if first:
                    np.matmul(wblk, tile2, out=acc2)
                    first = False
                else:
                    np.matmul(wblk, tile2, out=scr2)
                    acc2 += scr2
            acc2 += bias[:, None]
            np.copyto(out[:, r0:r0 + h, :], acc2.reshape(O, h, Wo))
