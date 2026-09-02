
import numpy as np
import pandas as pd

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

class KratosArchitectureEngine:
    """
    Evaluates core equations of the Kratos Predictive Architecture
    using dynamic state variables C_K and step evaluation C_{K+1}.
    """
    def __init__(self, C_max=200.0, beta=0.002, K_m=1000.0, tau_max=3600.0):
        self.C_max = C_max # Theoretical absolute capacity ceiling (Kr)
        self.beta = beta # Global baseline growth parameter
        self.K_m = K_m # Prefrontal half-saturation constant (s)
        self.tau_max = tau_max # Metabolic depletion ceiling (s)

    def calculate_initial_calibration(self, tau_e_calib, L_calib):
        """Eq. 1: Dynamic Baseline Calibration Index (C_K)"""
        if tau_e_calib < 600.0:
            return None # Invalidated trial (< 600s floor criterion)
        W_calib = tau_e_calib * L_calib
        C_K_initial = (self.C_max * W_calib) / (W_calib + self.K_m)
        return C_K_initial

    def calculate_workload_factor(self, TLX_avg):
        """Eq. 5 & 6: Piecewise Workload Acceleration Factor (L)"""
        if TLX_avg < 30.0:
            return 0.0 # Discarded trial due to lack of friction
        elif 30.0 <= TLX_avg <= 50.0:
            return TLX_avg / 50.0
        else:
            return 1.0 + 2.0 * ((TLX_avg - 50.0) / 50.0)

    def calculate_readiness_modifier(self, S_norm, F_sub, N_status, F_K, w1=0.4, w2=0.3, w3=0.3):
        """
        Eq. 7: Fully Expanded Systemic Readiness Modifier (alpha_K)
        alpha_K = max(0.15, min(1.0, (w1S_norm + w2(1 - F_sub) + w3*N_status) * (1 - F_K)))
        """
        biometric_composite = (w1 * S_norm) + (w2 * (1.0 - F_sub)) + (w3 * N_status)
        alpha_K = max(0.15, min(1.0, biometric_composite * (1.0 - F_K)))
        return alpha_K

    def calculate_recovery_period(self, L, tau_e, W_calib, lambda_rec=1.0, tau_base=1800.0):
        """Eq. 11: Non-linear Homeostatic Recovery Period (tau_p)"""
        W_active = L * tau_e
        excitotoxic_penalty = max(0.0, (tau_e - self.tau_max) / self.tau_max)
        tau_p = lambda_rec * tau_base * (W_active / W_calib) * np.exp(excitotoxic_penalty)
        return tau_p

    def calculate_dynamic_brake(self, C_K):
        """Eq. 3: Linear Proximity Dynamic Brake (beta_eff)"""
        return max(0.0, self.beta * ((self.C_max - C_K) / self.C_max))

    def calculate_capacity_expansion(self, C_K, alpha_K, beta_eff, W_active, tau_p):
        """Eq. 2: Capacity Adaptation Step (C_{K+1})"""
        delta_C = C_K * (alpha_K * beta_eff * (W_active / (W_active + tau_p)))
        C_K_next = C_K + delta_C
        return C_K_next, delta_C

    def calculate_fatigue_accumulation(self, F_K, W_active, W_calib, tau_rest, tau_p, gamma_fatigue=0.85, eta=0.15):
        """Eq. 18: Post-trial Fatigue Accumulation (F_{K+1})"""
        unrecovered_ratio = 1.0 - min(1.0, tau_rest / tau_p) if tau_p > 0 else 0.0
        F_K_next = (gamma_fatigue * F_K) + (eta * (W_active / W_calib) * unrecovered_ratio)
        return max(0.0, min(1.0, F_K_next))

    def calculate_disuse_decay_threshold(self, C_K, L_session, TLX_avg):
        """
        Eq. 13-16: Minimum Maintenance Friction (L_min) & Dynamic Decay Rate (gamma)
        L_min = 1.0 * (C_K / 100.0) anchored to nominal 100 Kr benchmark.
        """
        L_min = 1.0 * (C_K / 100.0)
        if L_session >= L_min and TLX_avg >= 30.0:
            gamma = 0.000 # Growth / maintenance lock active
        else:
            gamma = 0.002 # Passive decay active (0.002 day^-1)
        return L_min, gamma


def run_10k_simulations(num_runs=10000):
    engine = KratosArchitectureEngine()

    # Fully Unconstrained Input Distributions Across 10,000 Cohorts
    tau_calib_dist = np.random.uniform(600.0, 3600.0, num_runs) # Full spectrum calibration times (s)
    L_calib_dist = np.random.uniform(1.0, 2.5, num_runs) # Full spectrum calibration difficulty
    tlx_dist = np.random.uniform(30.0, 100.0, num_runs) # Full operational TLX scores
    tau_e_dist = np.random.uniform(600.0, 7200.0, num_runs) # Full execution duration spectrum (s)

    # Biometrics for Fully Expanded Readiness (alpha_K)
    s_norm_dist = np.random.uniform(0.4, 1.0, num_runs)
    f_sub_dist = np.random.uniform(0.0, 0.60, num_runs)
    n_status_dist = np.random.uniform(0.5, 1.0, num_runs)
    tau_rest_dist = np.random.uniform(1800.0, 43200.0, num_runs) # Inter-session rest (s)

    results = []

    for i in range(num_runs):
        # Step 1: Capacity Evaluation (C_K) — Purely Input-Driven
        tau_c = tau_calib_dist[i]
        L_c = L_calib_dist[i]
        W_calib = tau_c * L_c
        C_K = engine.calculate_initial_calibration(tau_c, L_c)

        # Skip invalidated trials under 600s floor criterion
        if C_K is None:
            continue

        # Step 2: Readiness (Fully Expanded Formula) & Operational Workload
        F_K = np.random.uniform(0.0, 0.2)
        alpha_K = engine.calculate_readiness_modifier(
            S_norm=s_norm_dist[i],
            F_sub=f_sub_dist[i],
            N_status=n_status_dist[i],
            F_K=F_K
        )
        L_active = engine.calculate_workload_factor(tlx_dist[i])
        W_active = L_active * tau_e_dist[i]

        # Step 3: Dynamic Brake & Homeostatic Recovery Window
        beta_eff = engine.calculate_dynamic_brake(C_K)
        tau_p = engine.calculate_recovery_period(L_active, tau_e_dist[i], W_calib)

        # Step 4: Capacity Expansion Evaluation (C_{K+1})
        C_K_next, delta_C = engine.calculate_capacity_expansion(C_K, alpha_K, beta_eff, W_active, tau_p)

        # Step 5: Post-Trial Fatigue (F_{K+1}) & Maintenance Friction (L_min = C_K / 100)
        F_K_next = engine.calculate_fatigue_accumulation(F_K, W_active, W_calib, tau_rest_dist[i], tau_p)
        L_min, gamma_decay = engine.calculate_disuse_decay_threshold(C_K, L_active, tlx_dist[i])

        results.append({
            'Run_ID': i + 1,
            'Calibration_W_calib': round(W_calib, 2),
            'Capacity_C_K_Kr': round(C_K, 4),
            'Workload_L': round(L_active, 4),
            'Readiness_alpha_K': round(alpha_K, 4),
            'Dynamic_Brake_beta_eff': round(beta_eff, 6),
            'Recovery_tau_p_sec': round(tau_p, 2),
            'Delta_C_Kr': round(delta_C, 6),
            'Capacity_C_Kplus1_Kr': round(C_K_next, 4),
            'Fatigue_F_Kplus1': round(F_K_next, 4),
            'Min_Maint_L_min': round(L_min, 4),
            'Decay_Engaged_gamma': gamma_decay
        })

    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    print(f"--- Running 10,000 Unconstrained Kratos Engine Simulations (SEED = {GLOBAL_SEED}) ---")
    df_results = run_10k_simulations(num_runs=10000)

    # Save complete dataset to CSV
    csv_filename = "kratos_10k_unconstrained_runs.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"\nDataset successfully saved to '{csv_filename}'.")

    # Print Statistical Summary Table
    print("\n=====================================================================================")
    print(" 10,000-RUN UNCONSTRAINED KRATOS ENGINE STATISTICAL SUMMARY")
    print("=====================================================================================")
    summary_metrics = {
        'Parameter Metric': [
            'Capacity State C_K (Kr)',
            'Expanded Readiness alpha_K',
            'Workload Acceleration L',
            'Dynamic Brake beta_eff',
            'Recovery Window tau_p (hours)',
            'Capacity Step Delta C (Kr)',
            'Evaluated Capacity C_{K+1} (Kr)',
            'Post-Trial Fatigue F_{K+1}',
            'Maintenance Threshold L_min'
        ],
        'Mean': [
            df_results['Capacity_C_K_Kr'].mean(),
            df_results['Readiness_alpha_K'].mean(),
            df_results['Workload_L'].mean(),
            df_results['Dynamic_Brake_beta_eff'].mean(),
            (df_results['Recovery_tau_p_sec'] / 3600.0).mean(),
            df_results['Delta_C_Kr'].mean(),
            df_results['Capacity_C_Kplus1_Kr'].mean(),
            df_results['Fatigue_F_Kplus1'].mean(),
            df_results['Min_Maint_L_min'].mean()
        ],
        'Min': [
            df_results['Capacity_C_K_Kr'].min(),
            df_results['Readiness_alpha_K'].min(),
            df_results['Workload_L'].min(),
            df_results['Dynamic_Brake_beta_eff'].min(),
            (df_results['Recovery_tau_p_sec'] / 3600.0).min(),
            df_results['Delta_C_Kr'].min(),
            df_results['Capacity_C_Kplus1_Kr'].min(),
            df_results['Fatigue_F_Kplus1'].min(),
            df_results['Min_Maint_L_min'].min()
        ],
        'Max': [
            df_results['Capacity_C_K_Kr'].max(),
            df_results['Readiness_alpha_K'].max(),
            df_results['Workload_L'].max(),
            df_results['Dynamic_Brake_beta_eff'].max(),
            (df_results['Recovery_tau_p_sec'] / 3600.0).max(),
            df_results['Delta_C_Kr'].max(),
            df_results['Capacity_C_Kplus1_Kr'].max(),
            df_results['Fatigue_F_Kplus1'].max(),
            df_results['Min_Maint_L_min'].max()
        ]
    }

    df_summary = pd.DataFrame(summary_metrics)
    print(df_summary.to_string(index=False))
    print("=====================================================================================\n")
