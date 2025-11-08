import numpy as np


def closed_form_whittle_index(b, gamma, P01_vec, P11_vec):
    """
    Closed-form Whittle index for binary 'reveal/restart' arm (subsidy form).
    Uses expected next-belief under active: F1(b) = b*P11(1) + (1-b)*P01(1).

    Args
    - b: current belief P(H=1)
    - gamma: discount factor in (0,1)
    - P01_vec: length-2 array-like, [P01(passive), P01(active)] where P01(a) = P(s_t=1 | s_{t-1}=0, a)
    - P11_vec: length-2 array-like, [P11(passive), P11(active)] where P11(a) = P(s_t=1 | s_{t-1}=1, a)

    Returns
    - Whittle index value (float)
    """
    P01_pass, P01_act = float(P01_vec[0]), float(P01_vec[1])
    P11_pass, P11_act = float(P11_vec[0]), float(P11_vec[1])
    alpha = P11_pass - P01_pass
    beta = P01_pass
    # Passive next-belief (affine)
    F0 = alpha * float(b) + beta
    # Active next-belief: reveal then apply active transition in expectation
    F1 = float(b) * P11_act + (1.0 - float(b)) * P01_act
    # Marginal value coefficient for passive dynamics
    A = 1.0 / (1.0 - float(gamma) * alpha + 1e-16)
    # If your immediate rewards differ, add ΔR(b) here (often 0).
    # W(b) = ΔR(b) + γ * A * (F1 - F0)
    return float(gamma) * A * (F1 - F0)


def compute_closed_form_wis(p_belief, gamma, P01_all, P11_all):
    """Vectorized helper to compute closed-form Whittle indices for all arms.

    Args
    - p_belief: array shape (N,), beliefs per arm
    - gamma: scalar
    - P01_all: array shape (N,2) with [P01(passive), P01(active)] per arm
    - P11_all: array shape (N,2) with [P11(passive), P11(active)] per arm

    Returns
    - wis: array shape (N,) Whittle indices
    """
    N = len(p_belief)
    wis = np.empty(N, dtype=float)
    for i in range(N):
        wis[i] = closed_form_whittle_index(
            b=float(p_belief[i]),
            gamma=gamma,
            P01_vec=P01_all[i],
            P11_vec=P11_all[i],
        )
    return wis


class ClosedFormPolicyController:
    """
    Maintains 10-step freeze for action selection using closed-form indices,
    while SMC-based theta learning can continue externally.

    Usage pattern per episode:
      - create instance with freeze_period and gamma
      - at each t, call select_actions(p_belief, budget, P01_all, P11_all, t)
      - optionally retrieve last computed indices via .last_wis
      - you may update .frozen_theta0/.frozen_theta1 externally every freeze
    """
    def __init__(self, gamma, freeze_period=10):
        self.gamma = float(gamma)
        self.freeze_period = int(freeze_period)
        self.last_update_t = None
        self.last_wis = None
        self.frozen_theta0 = None
        self.frozen_theta1 = None

    def maybe_update_snapshot(self, t, theta0_mean=None, theta1_mean=None):
        if t == 0 or (self.last_update_t is None) or (t - self.last_update_t) >= self.freeze_period:
            self.last_update_t = int(t)
            if theta0_mean is not None:
                self.frozen_theta0 = np.array(theta0_mean, dtype=float, copy=True)
            if theta1_mean is not None:
                self.frozen_theta1 = np.array(theta1_mean, dtype=float, copy=True)

    def select_actions(self, p_belief, budget, P01_all, P11_all, t,
                       theta0_mean=None, theta1_mean=None):
        # Freeze snapshot every freeze_period
        self.maybe_update_snapshot(t, theta0_mean=theta0_mean, theta1_mean=theta1_mean)

        # Recompute Whittle indices every time step 
        self.last_wis = compute_closed_form_wis(p_belief, self.gamma, P01_all, P11_all)

        N = len(p_belief)
        actions = np.zeros(N, dtype=int)
        k = int(min(budget, N))
        if k > 0:
            idx = np.argpartition(-self.last_wis, k - 1)[:k]
            actions[idx] = 1
        return actions
