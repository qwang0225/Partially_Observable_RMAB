import numpy as np
import matplotlib.pyplot as plt
from PBVI import mean_pbvi_whittle_index, get_or_build_local_model_sigma
from policy_iteration import (
    whittle_index_pi_secant,
    compute_value_and_advantage,
    plot_value_and_advantage_direct,
)
from thompson_update import ts_soft_update_single, ts_labeled_update_single
from smc_update import smc_update_single
from closed_from_policy import ClosedFormPolicyController
from priority_index import select_priority_actions
from utils import beta_paras, plot_theta_tracking_pvbi, plot_results, plot_runtime, print_results_summary
import time 

# Numeric policy mapping to avoid string checks
POLICY_MAP = {
    0: "smc_pbvi",
    1: "oracle",
    2: "random",
    3: "myopic",
    4: "thompson_pbvi",
    6: "closed_form",
    7: "priority_index",
    10: "pi_whittle",
}
POLICY_NAME_TO_ID = {v: k for k, v in POLICY_MAP.items()}


def simulate_rmab(
    num_arms=100,
    budget=40,
    horizon=10,
    episodes=10,
    gamma=0.95,
    seed=0,
    M_particles=500,
    M_keep=40,
    ess_ratio=0.5,
    lw_a=0.98,
    policy_id=0,  # numeric policy id; see POLICY_MAP in this file
    noise_std=0.05,  # std used for transition noise and theta_true perturbation
    verbose=False,
):
    rng = np.random.default_rng(seed)
    
    A = np.array([[0.95, 0.0],
              [0.0, 0.95]])  # decay or persistence matrix

    # Base transition matrix (shared template)
    base_transition_responsive = np.array([
        [[0.95, 0.05], [0.2, 0.8]],  # from state 0
        [[0.25, 0.75], [0.05, 0.95]],  # from state 1
    ], dtype=float)
    
    base_transition_mod_responsive = np.array([
    [[0.95, 0.05], [0.4, 0.6]],  # from state 0
    [[0.4, 0.6], [0.3, 0.7]],  # from state 1
], dtype=float)
    
    base_transition_less_responsive = np.array([
        [[0.95, 0.05], [0.5, 0.5]],  # from state 0
        [[0.45, 0.55], [0.4, 0.6]],  # from state 1
    ], dtype=float)

    # Build per-patient transition matrices by adding noise to P(next=1)
    def perturb_transition(base_tm, rng=rng, std=noise_std):
        tm = base_tm.copy()
        for s_prev in (0, 1):
            for a in (0, 1):
                p1 = float(tm[s_prev, a, 1])
                p1_noisy = np.clip(p1 + rng.normal(0.0, std), 1e-3, 1.0 - 1e-3)
                tm[s_prev, a, 1] = p1_noisy
                tm[s_prev, a, 0] = 1.0 - p1_noisy
        return tm 
    
    patient_types = ['responsive', 'moderate', 'less_responsive']
    type_proportions = [0.3, 0.4, 0.3]  # 30%, 40%, 30%
    
    patient_type_indices = rng.choice(
        len(patient_types), 
        size=num_arms, 
        p=type_proportions
    )
    transition_matrices = np.zeros((num_arms, 2, 2, 2))
    
    for i in range(num_arms):
        type_idx = patient_type_indices[i]
        if type_idx == 0:
            transition_matrices[i] = perturb_transition(base_transition_responsive)
        elif type_idx == 1:
            transition_matrices[i] = perturb_transition(base_transition_mod_responsive)
        else:
            transition_matrices[i] = perturb_transition(base_transition_less_responsive)
            
    # transition_matrices = np.stack([
    #     perturb_transition(base_transition, rng, trans_noise_std)
    #     for _ in range(num_arms)
    # ], axis=0)  # shape: (num_arms, 2, 2, 2)

    # Extract per-patient P01/P11 for convenience in PVBI/Oracle
    P01_all = transition_matrices[:, 0, :, 1]  # shape (num_arms, 2)
    P11_all = transition_matrices[:, 1, :, 1]  # shape (num_arms, 2)

    # True theta parameters - fixed for all policies (initial)
    theta_true = np.zeros((num_arms, 2))
    a0, b0 = beta_paras(0.7, 5)
    a1, b1 = beta_paras(0.3, 5)
    for i in range(num_arms):
        th0 = np.clip(rng.beta(a0, b0) + rng.normal(0.0, noise_std), 1e-3, 1.0 - 1e-3)
        th1 = np.clip(rng.beta(a1, b1) + rng.normal(0.0, noise_std), 1e-3, 1.0 - 1e-3)
        while th1 >= th0:
            th0 = np.clip(rng.beta(a0, b0) + rng.normal(0.0, noise_std), 1e-3, 1.0 - 1e-3)
            th1 = np.clip(rng.beta(a1, b1) + rng.normal(0.0, noise_std), 1e-3, 1.0 - 1e-3)
        theta_true[i, 0] = th0
        theta_true[i, 1] = th1
    
    theta_prior = np.zeros((num_arms, 2))
    theta_prior[:, 0] = 0.5
    theta_prior[:, 1] = 0.5
    
    theta0_smc_episodes = np.zeros((episodes, num_arms))
    theta1_smc_episodes = np.zeros((episodes, num_arms))
    # Track true theta per episode (snapshot at end of each episode)
    theta0_true_episodes = np.zeros((episodes, num_arms))
    theta1_true_episodes = np.zeros((episodes, num_arms))

    # Initialize particles for SMC-based policies (SMC-PBVI, closed-form, priority, pi_whittle)
    if policy_id == 0 or policy_id == 6 or policy_id == 7 or policy_id == 10:
        def init_particles(size, th0=True):
            if th0:
                th = rng.beta(1.0, 1.0, size=size)
            else:
                th = rng.beta(1.0, 1.0, size=size)
            return np.clip(th, 1e-4, 1 - 1e-4)

        theta0_particles = np.vstack([init_particles(M_particles, th0=True) for _ in range(num_arms)])
        theta1_particles = np.vstack([init_particles(M_particles, th0=False) for _ in range(num_arms)])
        w_particles = np.full((num_arms, M_particles), 1.0 / M_particles)
        if policy_id == 6:
            cf_controller = ClosedFormPolicyController(gamma=gamma, freeze_period=5)
    elif policy_id == 4:
        # Initialize Beta(1,1) priors for theta0/theta1 per arm
        a0 = np.ones(num_arms)
        b0 = np.ones(num_arms)
        a1 = np.ones(num_arms)
        b1 = np.ones(num_arms)

    # For policy 10: track end-of-episode curves for two random arms (no interpolation)
    if policy_id == 10:
        sel_pi10_arms = np.sort(rng.choice(num_arms, size=min(2, num_arms), replace=False))
        pi10_last = {int(a): {"B": None, "V": None, "Adv": None} for a in sel_pi10_arms}
    else:
        sel_pi10_arms = None
        pi10_last = None

    total_returns = []

    for ep in range(episodes):
        # Initialize states and beliefs
        s = rng.integers(0, 2, size=num_arms)
        p_belief = rng.random(num_arms)
        ep_return = 0.0

        if verbose:
            print(f"Episode {ep+1} - Initial states: {s}")

        for t in range(horizon):
            # Select actions based on numeric policy id
            if policy_id == 0:
                actions = select_pvbi_actions(
                    p_belief,
                    theta0_particles,
                    theta1_particles,
                    w_particles,
                    budget,
                    gamma,
                    P01_all,
                    P11_all,
                    verbose,
                    rng
                )
            elif policy_id == 6:
                th0_mean = np.sum(w_particles * theta0_particles, axis=1)
                th1_mean = np.sum(w_particles * theta1_particles, axis=1)
                actions = cf_controller.select_actions(
                    p_belief=p_belief,
                    budget=budget,
                    P01_all=P01_all,
                    P11_all=P11_all,
                    t=t,
                    theta0_mean=th0_mean,
                    theta1_mean=th1_mean,
                )
            
            elif policy_id == 10:
                # Whittle via policy-iteration on tiny local model per arm
                th0_mean = np.sum(w_particles * theta0_particles, axis=1)
                th1_mean = np.sum(w_particles * theta1_particles, axis=1)
                num_arms_local = len(p_belief)
                wis = np.empty(num_arms_local, dtype=float)
                for i in range(num_arms_local):
                    # Build deterministic sigma local model for speed
                    B, P0, P1 = get_or_build_local_model_sigma(
                        float(p_belief[i]), float(th0_mean[i]), float(th1_mean[i]), P01_all[i], P11_all[i], rng
                    )
                    lam_i = whittle_index_pi_secant(
                        b_star=float(p_belief[i]), B=B, P0=P0, P1=P1, gamma=gamma,
                        lam0=-1.0, lam1=1.0, tol=1e-6, max_it=20
                    )
                    wis[i] = lam_i
                actions = np.zeros(num_arms_local, dtype=int)
                active_idx = np.argpartition(-wis, budget-1)[:budget]
                actions[active_idx] = 1
            elif policy_id == 7:
                # Priority-index: posterior theta mean + exploration bonus
                th0_mean = np.sum(w_particles * theta0_particles, axis=1)
                th1_mean = np.sum(w_particles * theta1_particles, axis=1)
                actions = select_priority_actions(
                    p_belief=p_belief,
                    budget=budget,
                    th0_mean=th0_mean,
                    th1_mean=th1_mean,
                    P01_all=P01_all,
                    P11_all=P11_all,
                    gamma=gamma,
                    lam_explore=0.1,
                    rng=rng,
                )
            elif policy_id == 1:
                    actions = select_oracle_actions(s, budget, p_belief, transition_matrices)
            elif policy_id == 2:
                actions = select_random_actions(num_arms, budget, rng)
            elif policy_id == 3:
                actions = select_myopic_actions(p_belief, budget)
            elif policy_id == 4:
                actions = select_thompson_actions(
                    p_belief=p_belief,
                    budget=budget,
                    a0=a0, b0=b0, a1=a1, b1=b1,
                    P01_all=P01_all,
                    P11_all=P11_all,
                    gamma=gamma,
                    verbose=verbose,
                    rng=rng,
                )
            
            
            # Environment step
            rewards_t = 0.0
            for i in range(num_arms):
                a = int(actions[i])
                
                # State transition
                probs_next = transition_matrices[i, s[i], a]
                s[i] = 1 if rng.random() < probs_next[1] else 0
                
                # Observation
                p_y1 = theta_true[i, s[i]]
                y = 1 if rng.random() < p_y1 else 0
                
                # Belief update
                if a == 1:
                    # Active action: perfect health observation. only update theta estimation 
                    if policy_id in [0, 6, 7, 10]:
                            theta0_particles[i], theta1_particles[i], w_particles[i], _ = smc_update_single(
                            p=p_belief[i], action=a, y=y, P01=P01_all[i], P11=P11_all[i],
                            theta0=theta0_particles[i], theta1=theta1_particles[i], w=w_particles[i], 
                            rng=rng, H=s[i])
                    elif policy_id == 4:
                             a0[i], b0[i], a1[i], b1[i] = ts_labeled_update_single(p_belief[i], y, a0=a0[i], b0=b0[i], a1=a1[i], b1=b1[i])
                    p_belief[i] = float(s[i])
                else:
                    # Passive action: update health belief and theta 
                        if policy_id in [0, 6, 7, 10]:
                            theta0_particles[i], theta1_particles[i], w_particles[i], p_belief[i] = smc_update_single(
                            p=p_belief[i], action=a, y=y, P01=P01_all[i], P11=P11_all[i],
                            theta0=theta0_particles[i], theta1=theta1_particles[i], w=w_particles[i], 
                            rng=rng, H=None)
                        elif policy_id == 1:
                            p_belief[i] = update_belief_default(p_belief[i], a, y, theta_true[i], P01_all[i], P11_all[i]
                        )
                        elif policy_id == 4:
                            a0[i], b0[i], a1[i], b1[i], p_belief[i] = ts_soft_update_single(
                            p_cur=p_belief[i], y=y, action=a, P01=P01_all[i], P11=P11_all[i],
                            a0=a0[i], b0=b0[i], a1=a1[i], b1=b1[i], rng=rng
                        )
                        else:
                            p_belief[i] = update_belief_default(
                            p_belief[i], a, y, theta_prior[i], P01_all[i], P11_all[i]
                        )
                        
                rewards_t += float(s[i])
                # Evolve true theta with small Gaussian noise and keep 0 < th1 < th0 < 1
                # epsilon = rng.normal(0.0, noise_std, size=2)
                # theta_true[i] = np.clip(theta_true[i] @ A.T + epsilon, 1e-3, 1 - 1e-3)
                # if theta_true[i, 1] >= theta_true[i, 0]:
                #     theta_true[i, 1] = max(1e-3, theta_true[i, 0] - 1e-3)

            ep_return += (gamma ** t) * rewards_t

            if verbose and t % 5 == 0:  # Print every 5 steps to reduce output
                healthy_count = np.sum(s)
                print(f"  t={t}: {healthy_count}/{num_arms} healthy, reward={rewards_t:.2f}")

        # Record per-episode parameter snapshots
            if policy_id == 0:
                theta0_smc_episodes[ep] = np.sum(w_particles * theta0_particles, axis=1)
                theta1_smc_episodes[ep] = np.sum(w_particles * theta1_particles, axis=1)
            elif policy_id == 6:
                # record the last frozen snapshot if present; otherwise current mean
                th0_mean = np.sum(w_particles * theta0_particles, axis=1)
                th1_mean = np.sum(w_particles * theta1_particles, axis=1)
                if 'cf_controller' in locals() and cf_controller.frozen_theta0 is not None:
                    theta0_smc_episodes[ep] = cf_controller.frozen_theta0
                    theta1_smc_episodes[ep] = cf_controller.frozen_theta1
                else:
                    theta0_smc_episodes[ep] = th0_mean
                    theta1_smc_episodes[ep] = th1_mean
            elif policy_id == 4:
                theta0_smc_episodes[ep] = a0 / (a0 + b0)
                theta1_smc_episodes[ep] = a1 / (a1 + b1)
            elif policy_id == 7:
                theta0_smc_episodes[ep] = np.sum(w_particles * theta0_particles, axis=1)
                theta1_smc_episodes[ep] = np.sum(w_particles * theta1_particles, axis=1)
            
            elif policy_id == 10:
                theta0_smc_episodes[ep] = np.sum(w_particles * theta0_particles, axis=1)
                theta1_smc_episodes[ep] = np.sum(w_particles * theta1_particles, axis=1)
        # Always record true theta (end of episode snapshot)
        theta0_true_episodes[ep] = theta_true[:, 0]
        theta1_true_episodes[ep] = theta_true[:, 1]

        total_returns.append(ep_return)
        if verbose:
            print(f"Episode {ep+1} final return: {ep_return:.3f}")

    # For policy 10: after each episode, snapshot end-of-episode curves for selected arms
    # Compute at episode end using final beliefs and current theta means
    if policy_id == 10 and sel_pi10_arms is not None:
        th0_mean_final = np.sum(w_particles * theta0_particles, axis=1)
        th1_mean_final = np.sum(w_particles * theta1_particles, axis=1)
        for i in sel_pi10_arms:
            i = int(i)
            B_i, P0_i, P1_i = get_or_build_local_model_sigma(
                float(p_belief[i]), float(th0_mean_final[i]), float(th1_mean_final[i]), P01_all[i], P11_all[i], rng
            )
            lam_i = whittle_index_pi_secant(
                b_star=float(p_belief[i]), B=B_i, P0=P0_i, P1=P1_i, gamma=gamma,
                lam0=-1.0, lam1=1.0, tol=1e-6, max_it=20
            )
            V_i, Q0_i, Q1_i, adv_i = compute_value_and_advantage(B_i, P0_i, P1_i, gamma, lam_i)
            pi10_last[i]["B"] = B_i
            pi10_last[i]["V"] = V_i
            pi10_last[i]["Adv"] = adv_i

    # For policy 10: after all episodes, plot the end-of-episode V and advantage for the selected arms
    if policy_id == 10 and sel_pi10_arms is not None:
        for a in sel_pi10_arms:
            Bp = pi10_last[int(a)]["B"]
            Vp = pi10_last[int(a)]["V"]
            Ap = pi10_last[int(a)]["Adv"]
            if Bp is None:
                continue
            plot_value_and_advantage_direct(
                Bp, Vp, Ap,
                out_path=f"pi10_arm{int(a)}_end_value_adv.png",
                title=f"PI10 Arm {int(a)} End of last episode",
                sigma_points=Bp,
                n_plot_points=401,
            )

    # Return per-episode true and estimated theta trajectories
    return total_returns, theta0_true_episodes, theta1_true_episodes, theta0_smc_episodes, theta1_smc_episodes

def select_random_actions(num_arms, budget, rng):
    """Random policy: select budget arms randomly"""
    actions = np.zeros(num_arms, dtype=int)
    idx = rng.choice(num_arms, size=budget, replace=False)
    actions[idx] = 1
    return actions

def select_myopic_actions(p_belief, budget):
    """Myopic policy: select arms with smallest belief of being healthy.

    Chooses the `budget` arms with the smallest `p_belief` values.
    Returns a binary action array with 1 for selected arms.
    """
    num_arms = len(p_belief)
    actions = np.zeros(num_arms, dtype=int)
    # Indices of the smallest beliefs
    selected = np.argpartition(p_belief, budget - 1)[:budget]
    actions[selected] = 1
    return actions

def select_oracle_actions(s, budget, p_belief, trans_mat):
    """Oracle policy: knows true states, treats unhealthy patients first"""
    num_arms = len(s)
    actions = np.zeros(num_arms, dtype=int)
    treatment_benefit = np.zeros(num_arms)
    
    for i in range(num_arms):
        excepted_health_treated = trans_mat[i, s[i], 1, 1]
        excepted_unhealth_treated = trans_mat[i, s[i], 0, 1]
        treatment_benefit[i] = excepted_health_treated - excepted_unhealth_treated
    selected = np.argpartition(-treatment_benefit, budget-1)[:budget]
    
    actions[selected] = 1
    return actions

def select_pvbi_actions(p_belief, theta0_particles, theta1_particles, w_particles,
                       budget, gamma, P01_all, P11_all, verbose, rng):
    """PVBI policy: uses Whittle indices based on current beliefs and particles"""
    num_arms = len(p_belief)
    wis = np.empty(num_arms)

    # Per-arm posterior mean parameters from particles
    th0_mean = np.sum(w_particles * theta0_particles, axis=1)
    th1_mean = np.sum(w_particles * theta1_particles, axis=1)

    for i in range(num_arms):
        wis[i] = mean_pbvi_whittle_index(
            p_star=float(p_belief[i]),
            gamma=gamma,
            th0=float(th0_mean[i]),
            th1=float(th1_mean[i]),
            P01=P01_all[i],
            P11=P11_all[i],
            rng=rng
        )
    
    if verbose:
        print(f"Whittle indices: min={wis.min():.3f}, max={wis.max():.3f}, mean={wis.mean():.3f}")
    
    # Select arms with highest Whittle indices
    actions = np.zeros(num_arms, dtype=int)
    active_idx = np.argpartition(-wis, budget-1)[:budget]
    actions[active_idx] = 1
    
    return actions  

def select_thompson_actions(p_belief, budget, a0, b0, a1, b1, P01_all, P11_all, gamma, verbose=False, rng=None):
    
    num_arms = len(p_belief)
    wis = np.empty(num_arms)
    if rng is None:
        rng = np.random.default_rng()
    # Sample theta parameters from Beta posteriors (Thompson sampling)
    th0 = rng.beta(a0, b0)
    th1 = rng.beta(a1, b1)
    for i in range(num_arms):
        wis[i] = mean_pbvi_whittle_index(
            p_star=float(p_belief[i]),
            gamma=gamma,
            th0=float(th0[i]),
            th1=float(th1[i]),
            P01=P01_all[i],
            P11=P11_all[i],
            rng=rng
        )
    
    if verbose:
        print(f"Whittle indices: min={wis.min():.3f}, max={wis.max():.3f}, mean={wis.mean():.3f}")
    
    # Select arms with highest Whittle indices
    actions = np.zeros(num_arms, dtype=int)
    active_idx = np.argpartition(-wis, budget-1)[:budget]
    actions[active_idx] = 1
    
    return actions  

def update_belief_default(current_belief, action, observation, theta_true_i, P01, P11):
    """Update belief using known theta (for Oracle and Random policies)"""
    # Predict belief after transition
    pt = current_belief * P11[action] + (1.0 - current_belief) * P01[action]
    
    # Update based on observation
    th0, th1 = theta_true_i[0], theta_true_i[1]
    if observation == 1:
        denom = (1.0 - pt) * th0 + pt * th1
        p_post = (pt * th1) / (denom + 1e-16)
    else:
        denom = (1.0 - pt) * (1.0 - th0) + pt * (1.0 - th1)
        p_post = (pt * (1.0 - th1)) / (denom + 1e-16)
    
    return float(np.clip(p_post, 0.0, 1.0))

def run_experiments():
    """Run comparison experiments"""
    num_arms_list = [40, 100]
    # Include policy 10 (policy-iteration Whittle); drop 'none' policy
    policy_ids = [0, 1, 2, 3, 6, 10]
    episodes = 20
    horizon = 40
    gamma = 0.95
    seed = 42  # Fixed seed for reproducibility

    results = {}
    
    for N in num_arms_list:
        budget_list = [int(N*0.2), int(N*0.3)]
        for B in budget_list:
            key = (N, B)
            results[key] = {}
            
            for pol_id in policy_ids:
                pol = POLICY_MAP[pol_id]
                print(f"\n=== Testing {pol} with {N} arms, budget {B}, horizon {horizon} ===")
                t0 = time.perf_counter()
                returns, theta0_true_over_eps, theta1_true_over_eps, theta0_est_over_eps, theta1_est_over_eps = simulate_rmab(
                    num_arms=N,
                    budget=B,
                    horizon=horizon,
                    episodes=episodes,
                    gamma=gamma,
                    seed=seed,
                    policy_id=pol_id,
                    verbose=False,
                )
                # SMC-PBVI/Thompson-PBVI: plot theta tracking over episodes
                elapsed = time.perf_counter() - t0
                if pol_id == 0: 
                    plot_theta_tracking_pvbi(
                        theta0_true_over_eps=theta0_true_over_eps,
                        theta1_true_over_eps=theta1_true_over_eps,
                        theta0_est_over_eps=theta0_est_over_eps,
                        theta1_est_over_eps=theta1_est_over_eps,
                        arms_to_plot=min(5, N),
                        title=f"SMC-PBVI Theta Tracking (N={N}, B={B})",
                        out_path=f"smc_pbvi_theta_tracking_N{N}_B{B}.png",
                        sample_random=True,
                        sample_seed=seed,
                    )
                elif pol_id == 4:
                    plot_theta_tracking_pvbi(
                        theta0_true_over_eps=theta0_true_over_eps,
                        theta1_true_over_eps=theta1_true_over_eps,
                        theta0_est_over_eps=theta0_est_over_eps,
                        theta1_est_over_eps=theta1_est_over_eps,
                        arms_to_plot=min(5, N),
                        title=f"Thompson-PBVI Theta Tracking (N={N}, B={B})",
                        out_path=f"thompson_pbvi_theta_tracking_N{N}_B{B}.png",
                        sample_random=True,
                        sample_seed=seed,
                    )
                
                results[key][pol] = {
                    "mean": float(np.mean(returns)),
                    "runtime_sec": elapsed,
                }
                print(f"{pol} - Average return: {results[key][pol]['mean']:.3f}")

    # Plot and print results
    policy_names = [POLICY_MAP[i] for i in policy_ids]
    plot_results(results, policy_list=policy_names)
    plot_runtime(results, policy_list=policy_names)
    print_results_summary(results, policy_list=policy_names)
    return results

if __name__ == "__main__":
    results = run_experiments()
