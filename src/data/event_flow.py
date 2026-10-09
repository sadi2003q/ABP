"""
Background motion estimated from the events alone, for STAGE B1
(pseudo-labels with no motion capture, no depth and no inertial sensor).

For each frame window we fit a smooth "background flow": how far each
pixel of the static scene moves over the window. The flow is a simple
formula with a few numbers (parameters). We pick the parameters that
make the motion-compensated event image as sharp as possible
(contrast maximisation): when the background motion is undone
correctly, the events of each static edge pile up on one line.

Unlike stage B0 (gyroscope rotation only), this flow also absorbs the
image motion caused by the camera moving sideways / forwards
(translation), as long as the background is roughly one surface.

Flow models. (u, v) is the displacement in pixels over the whole
window; x, y are pixel coordinates centred on the image and divided by
half the larger image side (so x, y are within about [-1, 1] and every
parameter is in pixels):

  translation (2 numbers): u = a0,                 v = a1
  affine      (6 numbers): u = a0 + a1 x + a2 y,   v = a3 + a4 x + a5 y
  planar      (8 numbers): affine + a6 (x*x, x*y) + a7 (x*y, y*y)
              The planar model is the exact image motion of a flat
              surface seen by a camera that rotates and translates by
              a small amount.

Sharpness = variance of the blurred image of warped events. Its exact
gradient with respect to the parameters is computed analytically, so
each fit takes a few dozen image evaluations.

Protection against "event collapse" (a known failure: a zoom-capable
flow can squeeze all events towards one point, which looks very sharp
but is meaningless):
  * sharpness is measured with events warped to the MIDDLE of the
    window, so squeezing the early events means stretching the late
    ones and collapse no longer pays off;
  * the zoom-like (linear and quadratic) terms are limited to a small
    fraction of the image size;
  * the whole-image shift is fitted first; the other terms are only
    then allowed to refine it.
The fitted numbers still describe the displacement over the WHOLE
window, so the flow is used exactly like the other geometries.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

MODELS = {"translation": 2, "affine": 6, "planar": 8}

# Limits on the parameters over one window. The shift limit is in pixels;
# the zoom-like limits are fractions of half the larger image side (s):
# a linear term of 0.1*s is a 10 % stretch / rotation per window, far more
# than real background motion between neighbouring frames.
TRANSLATION_LIMIT = 64.0
LINEAR_LIMIT_FRACTION = 0.10
QUADRATIC_LIMIT_FRACTION = 0.05


def _normalised(x, y, H, W):
    s = max(H, W) / 2.0
    return (x - W / 2.0) / s, (y - H / 2.0) / s


def _basis(x, y, H, W, model):
    """(Bu, Bv), each (N, P): u = Bu @ theta, v = Bv @ theta."""
    xn, yn = _normalised(np.asarray(x, np.float64), np.asarray(y, np.float64), H, W)
    one, zero = np.ones_like(xn), np.zeros_like(xn)
    if model == "translation":
        Bu = np.stack([one, zero], 1)
        Bv = np.stack([zero, one], 1)
    elif model == "affine":
        Bu = np.stack([one, xn, yn, zero, zero, zero], 1)
        Bv = np.stack([zero, zero, zero, one, xn, yn], 1)
    elif model == "planar":
        Bu = np.stack([one, xn, yn, zero, zero, zero, xn * xn, xn * yn], 1)
        Bv = np.stack([zero, zero, zero, one, xn, yn, xn * yn, yn * yn], 1)
    else:
        raise ValueError(f"unknown flow model {model!r}; choose from {sorted(MODELS)}")
    return Bu, Bv


def _vertical_shift_index(model):
    return 1 if model == "translation" else 3


def _bounds(model, H, W):
    T = (-TRANSLATION_LIMIT, TRANSLATION_LIMIT)
    if model == "translation":
        return [T, T]
    s = max(H, W) / 2.0
    L = (-LINEAR_LIMIT_FRACTION * s, LINEAR_LIMIT_FRACTION * s)
    b = [T, L, L, T, L, L]
    if model == "planar":
        Q = (-QUADRATIC_LIMIT_FRACTION * s, QUADRATIC_LIMIT_FRACTION * s)
        b += [Q, Q]
    return b


def _clip(theta, bounds):
    return np.clip(np.asarray(theta, np.float64), [b[0] for b in bounds], [b[1] for b in bounds])


def flow_field(theta, H, W, model) -> np.ndarray:
    """Background displacement over the window for every pixel, (H, W, 2) float32."""
    v, u = np.mgrid[0:H, 0:W]
    Bu, Bv = _basis(u.ravel(), v.ravel(), H, W, model)
    theta = np.asarray(theta, np.float64)
    return np.stack([Bu @ theta, Bv @ theta], 1).reshape(H, W, 2).astype(np.float32)


# ----------------------------------------------------------------------
# Sharpness of the warped event image and its gradient
# ----------------------------------------------------------------------

def _splat(x, y, H, W):
    """Bilinear accumulation of one unit per event; events outside are dropped."""
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    Wp = W + 2
    img = np.zeros((H + 2) * Wp, np.float64)       # 1-pixel zero border
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xx, yy = x0 + dx + 1, y0 + dy + 1
        ok = (xx >= 1) & (xx <= W) & (yy >= 1) & (yy <= H)
        img += np.bincount(yy[ok] * Wp + xx[ok], weights=w[ok], minlength=img.size)
    return img.reshape(H + 2, Wp)[1:H + 1, 1:W + 1]


def _sharpness(x, y, H, W, sigma) -> float:
    I = ndimage.gaussian_filter(_splat(x, y, H, W), sigma, mode="constant")
    return float(I.var())


def _sharpness_and_position_gradient(x, y, H, W, sigma):
    """
    Variance V of the blurred event image, and dV/dx, dV/dy for every event.
    V = mean_p (I_p - mean I)^2,  I = Gaussian * bilinear splat.
    dV/d(event position) = gradient of the bilinear interpolation of
    (2 / #pixels) * Gaussian * (I - mean I), evaluated at the event.
    """
    I = ndimage.gaussian_filter(_splat(x, y, H, W), sigma, mode="constant")
    mu = I.mean()
    V = float(((I - mu) ** 2).mean())
    J = ndimage.gaussian_filter(I - mu, sigma, mode="constant") * (2.0 / (H * W))
    Jp = np.zeros((H + 2, W + 2))
    Jp[1:H + 1, 1:W + 1] = J

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    inside = (x0 >= -1) & (x0 <= W - 1) & (y0 >= -1) & (y0 <= H - 1)
    xi = np.clip(x0 + 1, 0, W)
    yi = np.clip(y0 + 1, 0, H)
    J00, J10 = Jp[yi, xi], Jp[yi, xi + 1]
    J01, J11 = Jp[yi + 1, xi], Jp[yi + 1, xi + 1]
    gx = (1 - fy) * (J10 - J00) + fy * (J11 - J01)
    gy = (1 - fx) * (J01 - J00) + fx * (J11 - J10)
    gx[~inside] = 0.0
    gy[~inside] = 0.0
    return V, gx, gy


class _Problem:
    """
    Events of one window, prepared so each parameter guess is cheap to score.
    alpha = how far along the warp each event is moved (fit: event time
    fraction minus 0.5, i.e. warped to the middle of the window).
    """

    def __init__(self, x, y, alpha, H, W, model):
        self.x, self.y, self.alpha = x, y, alpha
        self.H, self.W, self.model = H, W, model
        self.Bu, self.Bv = _basis(x, y, H, W, model)

    def warp(self, theta, n=None):
        s = slice(None) if n is None else slice(0, n)
        a = self.alpha[s]
        return self.x[s] - a * (self.Bu[s] @ theta), self.y[s] - a * (self.Bv[s] @ theta)

    def sharpness(self, theta, sigma, n=None) -> float:
        xw, yw = self.warp(theta, n)
        return _sharpness(xw, yw, self.H, self.W, sigma)

    def value_and_gradient(self, theta, sigma):
        xw, yw = self.warp(theta)
        V, gx, gy = _sharpness_and_position_gradient(xw, yw, self.H, self.W, sigma)
        # x_warped = x - alpha * Bu @ theta  ->  d x_warped / d theta = -alpha * Bu
        grad = -(self.Bu.T @ (self.alpha * gx) + self.Bv.T @ (self.alpha * gy))
        return V, grad


def fit_background_flow(
    xy, t, t0: float, t1: float, H: int, W: int,
    model: str = "planar",
    init=None,
    sigmas=(3.0, 1.0),
    max_events: int = 80_000,
    search_radius: float = 24.0,
    search_step: float = 8.0,
    search_events: int = 30_000,
    max_iter: int = 40,
    seed: int = 0,
):
    """
    Fit the background flow of one window from its events.

    1. Whole-image shift: coarse grid search on a subsample with a wide
       blur, then gradient refinement (wide blur, then narrow blur).
    2. Other models: start from that shift (or from `init`, e.g. the
       previous window's answer, if it is sharper) and refine all
       parameters the same way.
    Sharpness is always measured with events warped to the middle of the
    window (see the module notes on event collapse).

    Returns (theta, info). info["gain"] = sharpness with the fitted flow
    divided by sharpness with no compensation (narrow blur; >1 = sharper).
    """
    P = MODELS[model]
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    t = np.asarray(t, np.float64)
    n = len(t)
    if n < 200:
        return np.zeros(P), {"fitted": False, "events": n, "gain": 1.0, "at_bound": False,
                             "started_from_previous": False}

    order = np.random.default_rng(seed).permutation(n)[:max_events]   # random order: any prefix is a fair subsample
    x, y = xy[order, 0], xy[order, 1]
    to_middle = np.clip((t[order] - t0) / max(t1 - t0, 1e-9), 0.0, 1.0) - 0.5

    def refine(prob, theta, bounds):
        for sigma in sigmas:
            ref = max(prob.sharpness(theta, sigma), 1e-12)   # scale so the objective starts at -1

            def fun(th, sigma=sigma, ref=ref):
                V, g = prob.value_and_gradient(th, sigma)
                return -V / ref, -g / ref

            res = minimize(fun, theta, jac=True, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": max_iter})
            if np.all(np.isfinite(res.x)) and -res.fun >= 1.0:
                theta = res.x
        return theta

    # ---- 1. whole-image shift ----
    prob_t = _Problem(x, y, to_middle, H, W, "translation")
    bounds_t = _bounds("translation", H, W)
    r = np.arange(-search_radius, search_radius + 1e-9, search_step)
    grid = [np.array([dx, dy]) for dy in r for dx in r]
    m = min(search_events, len(x))
    shift = grid[int(np.argmax([prob_t.sharpness(th, sigmas[0], n=m) for th in grid]))].copy()
    shift = refine(prob_t, _clip(shift, bounds_t), bounds_t)

    # ---- 2. full model, starting from the shift ----
    started_from_init = False
    if model == "translation":
        prob, bounds, theta = prob_t, bounds_t, shift
    else:
        prob = _Problem(x, y, to_middle, H, W, model)
        bounds = _bounds(model, H, W)
        theta = np.zeros(P)
        theta[0], theta[_vertical_shift_index(model)] = shift
        if init is not None and len(init) == P and np.all(np.isfinite(init)):
            init_c = _clip(init, bounds)
            if prob.sharpness(init_c, sigmas[0]) > prob.sharpness(theta, sigmas[0]):
                theta, started_from_init = init_c, True
        theta = refine(prob, theta, bounds)

    final_sigma = sigmas[-1]
    base = max(prob.sharpness(np.zeros(P), final_sigma), 1e-12)
    gain = prob.sharpness(theta, final_sigma) / base
    at_bound = any(abs(th - lo) < 1e-6 or abs(th - hi) < 1e-6 for th, (lo, hi) in zip(theta, bounds))
    return theta, {"fitted": True, "events": n, "gain": float(gain), "at_bound": bool(at_bound),
                   "started_from_previous": bool(started_from_init)}
