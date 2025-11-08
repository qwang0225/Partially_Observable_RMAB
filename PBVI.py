import numpy as np
from utils import *
from fast_PBVI import pava_isotonic

# Caches for faster PBVI Whittle computation
_LOCAL_MODEL_CACHE = {}
_V_CACHE = {}
_LAST_LAMBDA = None


def _q(x, eps=1e-6):
    """Quantize a float to a coarse grid for stable cache keys."""
    return float(np.round(float(x) / eps) * eps)


# --- Deterministic local belief construction (sigma-style) ---
def _F0_bar(b, th0, th1, P01, P11):
    """Passive: predict -> observe via theta -> Bayes -> next-belief expectation."""
    b = float(np.clip(b, 0.0, 1.0))
    pt = predict_belief(b, a=0, P01=P01, P11=P11)
    py0, p0_next, py1, p1_next = passive_posterior_terms(pt, th0, th1)
    return float(py0 * p0_next + py1 * p1_next)


def _F1_bar_reveal(b, P01, P11):
    """Active = perfect reveal (collapse). Expected next belief is the active prediction."""
    b = float(np.clip(b, 0.0, 1.0))
    return float(b * P11[1] + (1.0 - b) * P01[1])


def _build_sigma_B(p, th0, th1, P01, P11):
    """Deterministic, small set of informative belief points (no rollout/depth)."""
    p = float(np.clip(p, 0.0, 1.0))
    F0p = _F0_bar(p, th0, th1, P01, P11)
    F1p = _F1_bar_reveal(p, P01, P11)
    # one-step passives to capture local curvature
    F0F0 = _F0_bar(F0p, th0, th1, P01, P11)
    F0F1 = _F0_bar(F1p, th0, th1, P01, P11)

    B = np.array([0.0, 1.0, p, F0p, F1p, F0F0, F0F1], dtype=float)
    B = np.unique(np.round(np.clip(B, 0.0, 1.0), 8))
    return B


def get_or_build_local_model_sigma(p, th0, th1, P01, P11, rng=None):
    """
    Builds a tiny local belief set and its kernels near p.

    Returns
    -------
    B : np.ndarray (sorted ascending)
    P_passive : (S,S) row-stochastic
    P_active  : (S,S) row-stochastic
    """
    key = (
        _q(p, 1e-5), _q(th0, 1e-5), _q(th1, 1e-5),
        tuple(np.round(np.asarray(P01).ravel(), 6)),
        tuple(np.round(np.asarray(P11).ravel(), 6)),
        "reveal",
    )
    if key in _LOCAL_MODEL_CACHE:
        return _LOCAL_MODEL_CACHE[key]

    # 1) tiny deterministic belief set
    B = _build_sigma_B(p, th0, th1, P01, P11)
    S = len(B)

    # 2) Active kernel (reveal -> collapse to {0,1} according to active prediction)
    P_active = np.zeros((S, S))
    for i, bi in enumerate(B):
        pt = predict_belief(bi, a=1, P01=P01, P11=P11)
        for j, wj in split_to_grid(0.0, B):
            P_active[i, j] += (1.0 - pt) * wj
        for j, wj in split_to_grid(1.0, B):
            P_active[i, j] += pt * wj

    # 3) Passive kernel (mixture over posteriors given noisy observation)
    P_passive = np.zeros((S, S))
    for i, bi in enumerate(B):
        pt = predict_belief(bi, a=0, P01=P01, P11=P11)
        py0, p0_next, py1, p1_next = passive_posterior_terms(pt, th0, th1)
        for j, wj in split_to_grid(p0_next, B):
            P_passive[i, j] += py0 * wj
        for j, wj in split_to_grid(p1_next, B):
            P_passive[i, j] += py1 * wj

    _LOCAL_MODEL_CACHE[key] = (B, P_passive, P_active)
    return B, P_passive, P_active


def get_or_build_local_model(p, th0, th1, P01, P11, rng,
                             n_rollouts=8, depth=3, active_reveals_H=True,
                             jitter=1e-6):
    """Build or fetch a small local belief set and its kernels near p.

    Returns B (sorted array), P0 (passive kernel), P1 (active kernel).
    Uses a cache keyed by rounded inputs to avoid re-building.
    """
    key = (
        _q(p, 1e-5), _q(th0, 1e-5), _q(th1, 1e-5),
        tuple(np.round(np.asarray(P01).ravel(), 6)),
        tuple(np.round(np.asarray(P11).ravel(), 6)),
        int(n_rollouts), int(depth), bool(active_reveals_H)
    )
    if key in _LOCAL_MODEL_CACHE:
        return _LOCAL_MODEL_CACHE[key]

    # Build local B similarly to smc_value_iter_local
    B = [0.0, 1.0]
    if p not in B:
        B.append(float(p))
    B = np.array(sorted(B))
    
    for _ in range(n_rollouts):
        b = float(p)
        for _d in range(depth):
            pt = predict_belief(b, a=0, P01=P01, P11=P11)
            py0, p0_next, py1, p1_next = passive_posterior_terms(pt, th0, th1)
            for cand in (p0_next, p1_next):
                cand = float(np.clip(cand, 0.0, 1.0))
                if np.min(np.abs(B - cand)) > 1e-6:
                    B = np.sort(np.append(B, cand + rng.uniform(-jitter, jitter)))
            if active_reveals_H and rng.random() < 0.3:
                # Include active collapse endpoints already captured by {0,1}
                pass
            b = p1_next if rng.random() < py1 else p0_next

    B = np.clip(B, 0.0, 1.0)
    B = np.unique(np.round(B, 8))
    S = len(B)

    # Active kernel
    P_active = np.zeros((S, S))
    for i, bi in enumerate(B):
        pt = predict_belief(bi, a=1, P01=P01, P11=P11)
        for j, wj in split_to_grid(0.0, B):
            P_active[i, j] += (1.0 - pt) * wj
        for j, wj in split_to_grid(1.0, B):
            P_active[i, j] += pt * wj

    # Passive kernel
    P_passive = np.zeros((S, S))
    for i, bi in enumerate(B):
        pt = predict_belief(bi, a=0, P01=P01, P11=P11)
        py0, p0_next, py1, p1_next = passive_posterior_terms(pt, th0, th1)
        for j, wj in split_to_grid(p0_next, B):
            P_passive[i, j] += py0 * wj
        for j, wj in split_to_grid(p1_next, B):
            P_passive[i, j] += py1 * wj

    _LOCAL_MODEL_CACHE[key] = (B, P_passive, P_active)
    return B, P_passive, P_active


def value_iter_with_warm_start(B, P0, P1, lam, gamma, V0=None, vi_tol=1e-6, max_iter=500):
    """Run VI on (B,P0,P1) with reward r0=B, r1=B-lam, optionally warm-starting V."""
    S = len(B)
    r0 = B.copy()
    r1 = B.copy() - float(lam)
    V = V0.copy() if V0 is not None else np.zeros(S)
    for _ in range(max_iter):
        V_old = V.copy()
        Q0 = r0 + gamma * (P0 @ V_old)
        Q1 = r1 + gamma * (P1 @ V_old)
        V = np.maximum(Q0, Q1)
        if np.max(np.abs(V - V_old)) < vi_tol:
            break
    return V, Q0, Q1

def value_iter_with_pava(B, P0, P1, lam, gamma, V0=None,
                         vi_tol=1e-6, max_iter=500, eta=1.0):
    """
    Projected Value Iteration with monotone (non-decreasing) isotonic projection via PAVA.
    - B must be sorted ascending.
    - eta in (0,1]: damping factor toward the projected value.
    Returns (V, Q0, Q1).
    """
    S = len(B)
    r0 = B.copy()
    r1 = B.copy() - float(lam)
    V = V0.copy() if V0 is not None else np.zeros(S)

    def _pava_nondec(y):
        try:
            return pava_isotonic(y, x=B)
        except Exception:
            return pava_isotonic(y)

    for _ in range(max_iter):
        V_old = V.copy()
        Q0 = r0 + gamma * (P0 @ V_old)
        Q1 = r1 + gamma * (P1 @ V_old)
        V_raw = np.maximum(Q0, Q1)

        # Monotone projection and optional damping
        V_proj = _pava_nondec(V_raw)
        V = (1.0 - eta) * V_old + eta * V_proj

        if np.max(np.abs(V - V_old)) < vi_tol:
            break
    return V, Q0, Q1

    
def mean_pbvi_whittle_index(p_star, gamma, th0, th1, P01, P11, rng,
                            lam_lo=-1.0, lam_hi=1.0, tol=1e-5, max_iter=50,
                            use_sigma=False):
    """Fast Whittle index with caching and warm-started VI/bisection."""
    global _LAST_LAMBDA, _V_CACHE

    # 1) local model (cached)
    if use_sigma:
        B, P0, P1 = get_or_build_local_model_sigma(p_star, th0, th1, P01, P11, rng)
    else:
        B, P0, P1 = get_or_build_local_model(
            p_star, th0, th1, P01, P11, rng, n_rollouts=1, depth=1, active_reveals_H=True
        )
    key_V = (tuple(np.round(B, 10)), _q(gamma, 1e-6))
    V0 = _V_CACHE.get(key_V)

    # 2) bracket around last lambda if available
    if _LAST_LAMBDA is not None:
        mid = float(_LAST_LAMBDA)
        lo, hi = max(lam_lo, mid - 0.2), min(lam_hi, mid + 0.2)
        if lo >= hi:
            lo, hi = lam_lo, lam_hi
    else:
        lo, hi = lam_lo, lam_hi

    # 3) bisection with PAVA-projected VI (warm start)
    for _ in range(max_iter):
        lam = 0.5 * (lo + hi)
        V, Q0, Q1 = value_iter_with_warm_start(B, P0, P1, lam, gamma, V0, vi_tol=1e-6, max_iter=50)
        _V_CACHE[key_V] = V
        ip = int(np.argmin(np.abs(np.asarray(B) - float(p_star))))
        if Q1[ip] > Q0[ip]:
            lo = lam
        else:
            hi = lam
        if hi - lo < tol:
            break
        V0 = V

    w = 0.5 * (lo + hi)
    _LAST_LAMBDA = w
    return w
