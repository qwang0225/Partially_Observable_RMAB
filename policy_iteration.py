import numpy as np
import matplotlib.pyplot as plt

def _build_P_pi(P0, P1, pi):
    """Row-wise selector: for each state i, pick row i from P0 if pi[i]==0 else from P1."""
    P_pi = P0.copy()
    mask = (pi.astype(int) == 1)
    if np.any(mask):
        P_pi[mask, :] = P1[mask, :]
    return P_pi


def _policy_iteration_for_lambda(B, P0, P1, gamma, lam, max_outer=50):
    """Run policy iteration to optimality for a fixed λ on (B,P0,P1).
    Returns V, Q0, Q1, pi.
    """
    S = len(B)
    I = np.eye(S)
    pi = np.zeros(S, dtype=int)
    for _ in range(int(max_outer)):
        P_pi = _build_P_pi(P0, P1, pi)
        A = I - gamma * P_pi
        r_pi = B - (pi * lam)
        try:
            V = np.linalg.solve(A, r_pi)
        except np.linalg.LinAlgError:
            V = np.linalg.solve(A + 1e-12 * I, r_pi)
        Q0 = B + gamma * (P0 @ V)
        Q1 = (B - lam) + gamma * (P1 @ V)
        pi_new = (Q1 >= Q0).astype(int)
        if np.array_equal(pi_new, pi):
            return V, Q0, Q1, pi
        pi = pi_new
    return V, Q0, Q1, pi

def whittle_index_pi_secant(b_star, B, P0, P1, gamma,
                            lam0=-1.0, lam1=1.0, tol=1e-6, max_it=25):
    """Compute Whittle index via root-finding on g(λ)=Q1(b*;λ)-Q0(b*;λ), where
    Q0,Q1 are evaluated under the optimal policy for that λ using policy iteration.
    Uses a secant step with a safe fallback when slopes are tiny.
    """
    idx = int(np.argmin(np.abs(B - float(b_star))))
    lam0_eff, lam1_eff = float(lam0), float(lam1)
    _, Q00, Q10, _ = _policy_iteration_for_lambda(B, P0, P1, gamma, lam0_eff)
    g0 = float(Q10[idx] - Q00[idx])
    _, Q01, Q11, _ = _policy_iteration_for_lambda(B, P0, P1, gamma, lam1_eff)
    g1 = float(Q11[idx] - Q01[idx])

    lam_prev, g_prev = float(lam0_eff), float(g0)
    lam_cur, g_cur = float(lam1_eff), float(g1)
    for _ in range(int(max_it)):
        denom = (g_cur - g_prev)
        lam_new = 0.5 * (lam_cur + lam_prev) if abs(denom) < 1e-14 else lam_cur - g_cur * (lam_cur - lam_prev) / denom
        _, Q0n, Q1n, _ = _policy_iteration_for_lambda(B, P0, P1, gamma, lam_new)
        gn = float(Q1n[idx] - Q0n[idx])
        if abs(gn) < tol:
            return float(lam_new)
        lam_prev, g_prev = lam_cur, g_cur
        lam_cur, g_cur = lam_new, gn
    return float(lam_cur)


def compute_value_and_advantage(B, P0, P1, gamma, lam):
    """Run PI for λ and return (V, Q0, Q1, advantage=Q1-Q0)."""
    V, Q0, Q1, _ = _policy_iteration_for_lambda(B, P0, P1, gamma, lam)
    adv = Q1 - Q0
    return V, Q0, Q1, adv


def plot_value_and_advantage(B, V, Q0, Q1, out_path=None, title=None, n_plot_points=201):
    """Plot V(b) and advantage(b)=Q1-Q0 over belief grid B.
    Saves to out_path if provided; otherwise shows the plot.
    """
    import matplotlib.pyplot as plt
    B = np.asarray(B, dtype=float)
    V = np.asarray(V, dtype=float)
    Q0 = np.asarray(Q0, dtype=float)
    Q1 = np.asarray(Q1, dtype=float)
    adv = Q1 - Q0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    # Densify curves for smoother visualization (linear interpolation)
    x_min, x_max = float(np.min(B)), float(np.max(B))
    n_plot = max(int(n_plot_points), len(B))
    x_plot = np.linspace(x_min, x_max, n_plot)
    V_plot = np.interp(x_plot, B, V)
    A_plot = np.interp(x_plot, B, adv)

    ax1.plot(x_plot, V_plot, linewidth=1.6)
    ax1.scatter(B, V, color='C0', s=22, zorder=3)
    ax1.set_ylabel('V(b)')
    ax1.grid(True, alpha=0.3)
    if title:
        ax1.set_title(title)

    ax2.plot(x_plot, A_plot, color='C1', linewidth=1.6, label='Q1-Q0')
    ax2.scatter(B, adv, color='C1', s=22, zorder=3)
    ax2.axhline(0.0, color='k', linewidth=1, linestyle='--', alpha=0.6)
    ax2.set_xlabel('belief b')
    ax2.set_ylabel('advantage')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_value_and_advantage_direct(B, V, advantage, out_path=None, title=None, sigma_points=None, n_plot_points=201):
    """Plot V(b) and provided advantage(b) over belief grid B.
    Saves to out_path if provided; otherwise shows the plot.
    """
 
    B = np.asarray(B, dtype=float)
    V = np.asarray(V, dtype=float)
    advantage = np.asarray(advantage, dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    # Densify curves for smoother visualization (linear interpolation)
    x_min, x_max = float(np.min(B)), float(np.max(B))
    n_plot = max(int(n_plot_points), len(B))
    x_plot = np.linspace(x_min, x_max, n_plot)
    V_plot = np.interp(x_plot, B, V)
    A_plot = np.interp(x_plot, B, advantage)

    ax1.plot(x_plot, V_plot, linewidth=1.6)
    ax1.scatter(B, V, color='C0', s=22, zorder=3)
    ax1.set_ylabel('V(b)')
    ax1.grid(True, alpha=0.3)
    if title:
        ax1.set_title(title)

    ax2.plot(x_plot, A_plot, color='C1', linewidth=1.6, label='Q1-Q0')
    ax2.scatter(B, advantage, color='C1', s=22, zorder=3)
    ax2.axhline(0.0, color='k', linewidth=1, linestyle='--', alpha=0.6)
    ax2.set_xlabel('belief b')
    ax2.set_ylabel('advantage')
    ax2.grid(True, alpha=0.3)

    # Optionally overlay discrete sigma points as scatter on both subplots
    if sigma_points is not None:
        sigma_points = np.asarray(sigma_points, dtype=float)
        sigma_points = sigma_points[(sigma_points >= 0.0) & (sigma_points <= 1.0)]
        if sigma_points.size > 0:
            # Interpolate the curves onto sigma points for visualization
            V_sigma = np.interp(sigma_points, B, V)
            A_sigma = np.interp(sigma_points, B, advantage)
            ax1.scatter(sigma_points, V_sigma, color='C0', s=28, zorder=3, label='sigma pts')
            ax2.scatter(sigma_points, A_sigma, color='C1', s=28, zorder=3, label='sigma pts')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
