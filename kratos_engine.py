"""
Kratos Predictive Architecture Engine
=====================================
Evaluates core equations of the Kratos Predictive Architecture using dynamic 
state variables C_K and step evaluation C_{K+1}.

This script runs a 10,000-cohort Monte Carlo simulation to evaluate unconstrained 
input distributions across the Kratos equations.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


class KratosArchitectureEngine:
    """
    Kratos Predictive Architecture Core Engine.
    
    Attributes:
        C_max (float): Theoretical absolute capacity ceiling (Kr).
        beta (float): Global baseline growth parameter.
        K_m (float): Prefrontal half-saturation constant (s).
        tau_max (float): Nominal metabolic depletion ceiling (s).
        C_0 (float): Nominal anchor.
        mu (float): Capacity scaling exponent.
        k_steep (float): Softplus activation steepness.
        kappa (float): Asymptotic scaling factor of the neuro-metabolic penalty.
    """

    def __init__(
        self, 
        C_max: float = 200.0, 
        beta: float = 0.002, 
        K_m: float = 1000.0, 
        tau_max: float = 3600.0, 
        C_0: float = 100.0, 
        mu: float = 0.5, 
        k_steep: float = 2.0, 
        kappa: float = 0.5
    ):
        # Baseline Parameters
        self.C_max = C_max
        self.beta = beta
        self.K_m = K_m
        self.tau_max = tau_max

        # V17 Dynamic Parameters
        self.C_0 = C_0
        self.mu = mu
        self.k_steep = k_steep
        self.kappa = kappa

    def calculate_initial_calibration(self, tau_e_calib: float, L_calib: float) -> Optional[float]:
        """
        Eq. 1: Dynamic Baseline Calibration Index (C_K)
        
        Args:
            tau_e_calib: Execution duration of the calibration trial (s).
            L_calib: Workload acceleration factor of the calibration trial.
            
        Returns:
            Initial capacity state C_K (Kr), or None if trial is invalidated.
        """
        if tau_e_calib < 600.0:
            return None  # Invalidated trial (< 600s floor criterion)
        
        W_calib = tau_e_calib * L_calib
        C_K_initial = (self.C_max * W_calib) / (W_calib + self.K_m)
        return C_K_initial

    def calculate_workload_factor(self, TLX_avg: float) -> float:
        """
        Eq. 5 & 6: Piecewise Workload Acceleration Factor (L)
        
        Args:
            TLX_avg: NASA-TLX average score [0-100].
            
        Returns:
            Calculated workload factor L.
        """
        if TLX_avg < 30.0:
            return 0.0  # Discarded trial due to lack of friction
        elif 30.0 <= TLX_avg <= 50.0:
            return TLX_avg / 50.0
        else:
            return 1.0 + 2.0 * ((TLX_avg - 50.0) / 50.0)

    def calculate_readiness_modifier(
        self, 
        S_norm: float, 
        F_sub: float, 
        N_status: float, 
        F_K: float, 
        w1: float = 0.4, 
        w2: float = 0.3, 
        w3: float = 0.3
    ) -> float:
        """
        Eq. 7: Fully Expanded Systemic Readiness Modifier (alpha_K)
        """
        biometric_composite = (w1 * S_norm) + (w2 * (1.0 - F_sub)) + (w3 * N_status)
        alpha_K = max(0.15, min(1.0, biometric_composite * (1.0 - F_K)))
        return alpha_K

    def calculate_recovery_period(
        self, 
        C_K: float, 
        alpha_K: float, 
        L: float, 
        tau_e: float, 
        W_calib: float, 
        lambda_rec: float = 1.0, 
        tau_base: float = 1800.0
    ) -> float:
        """
        Eq. 11 & 12: Dynamic Metabolic Scaling & C-infinity Softplus Penalty
        """
        # Eq. 11: Dynamic effective ceiling
        tau_max_eff = self.tau_max * ((C_K / self.C_0) ** self.mu) * alpha_K

        # Eq. 12: Softplus recovery transition
        W_active = L * tau_e
        penalty_term = self.kappa * np.log(1.0 + np.exp(self.k_steep * ((tau_e - tau_max_eff) / tau_max_eff)))
        tau_p = lambda_rec * tau_base * (W_active / W_calib) * (1.0 + penalty_term)
        return tau_p

    def calculate_dynamic_brake(self, C_K: float) -> float:
        """
        Eq. 3: Linear Proximity Dynamic Brake (beta_eff)
        """
        return max(0.0, self.beta * ((self.C_max - C_K) / self.C_max))

    def calculate_capacity_expansion(
        self, 
        C_K: float, 
        alpha_K: float, 
        beta_eff: float, 
        W_active: float, 
        W_calib: float
    ) -> Tuple[float, float]:
        """
        Eq. 2: Decoupled Capacity Adaptation Step (C_{K+1})
        """
        delta_C = C_K * (alpha_K * beta_eff * (W_active / (W_calib + W_active)))
        C_K_next = C_K + delta_C
        return C_K_next, delta_C

    def calculate_fatigue_accumulation(
        self, 
        F_K: float, 
        W_active: float, 
        W_calib: float, 
        tau_rest: float, 
        tau_p: float, 
        gamma_fatigue: float = 0.85, 
        eta: float = 0.15
    ) -> float:
        """
        Eq. 13: Post-trial Fatigue Accumulation (F_{K+1})
        """
        unrecovered_ratio = 1.0 - min(1.0, tau_rest / tau_p) if tau_p > 0 else 0.0
        F_K_next = (gamma_fatigue * F_K) + (eta * (W_active / W_calib) * unrecovered_ratio)
        return max(0.0, min(1.0, F_K_next))

    def calculate_disuse_decay_threshold(self, C_K: float, L_session: float, TLX_avg: float) -> Tuple[float, float]:
        """
        Eq. 17-18: Minimum Maintenance Friction (L_min) & Dynamic Decay Rate (gamma)
        """
        L_min = 1.0 * (C_K / self.C_0)
        
        if L_session >= L_min and TLX_avg >= 30.0:
            gamma = 0.000  # Growth / maintenance lock active
        else:
            gamma = 0.002  # Passive decay active (0.002 day^-1)
        return L_min, gamma


def run_10k_simulations(num_runs: int = 10000) -> pd.DataFrame:
    """
    Executes a Monte Carlo simulation across unconstrained inputs for the Kratos Engine.
    """
    engine = KratosArchitectureEngine()

    # Fully Unconstrained Input Distributions Across Cohorts
    tau_calib_dist = np.random.uniform(600.0, 3600.0, num_runs) 
    L_calib_dist = np.random.uniform(1.0, 2.5, num_runs) 
    tlx_dist = np.random.uniform(30.0, 100.0, num_runs) 
    tau_e_dist = np.random.uniform(600.0, 7200.0, num_runs) 

    # Biometrics for Fully Expanded Readiness (alpha_K)
    s_norm_dist = np.random.uniform(0.4, 1.0, num_runs)
    f_sub_dist = np.random.uniform(0.0, 0.60, num_runs)
    n_status_dist = np.random.uniform(0.5, 1.0, num_runs)
    tau_rest_dist = np.random.uniform(1800.0, 43200.0, num_runs) 

    results = []

    for i in range(num_runs):
        # Step 1: Capacity Evaluation (C_K)
        tau_c = tau_calib_dist[i]
        L_c = L_calib_dist[i]
        C_K = engine.calculate_initial_calibration(tau_c, L_c)

        # Skip invalidated trials under 600s floor criterion
        if C_K is None:
            continue
            
        W_calib = tau_c * L_c

        # Step 2: Readiness & Operational Workload
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
        tau_p = engine.calculate_recovery_period(C_K, alpha_K, L_active, tau_e_dist[i], W_calib)

        # Step 4: Capacity Expansion Evaluation (C_{K+1})
        C_K_next, delta_C = engine.calculate_capacity_expansion(C_K, alpha_K, beta_eff, W_active, W_calib)

        # Step 5: Post-Trial Fatigue (F_{K+1}) & Maintenance Friction
        F_K_next = engine.calculate_fatigue_accumulation(F_K, W_active, W_calib, tau_rest_dist[i], tau_p)
        L_min, gamma_decay = engine.calculate_disuse_decay_threshold(C_K, L_active, tlx_dist[i])

        results.append({
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
    
    # Assign IDs smoothly after all valid runs are collected.
    df.insert(0, 'Run_ID', range(1, len(df) + 1))
    
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
    
    summary_data = df_results.copy()
    summary_data['Recovery_tau_p_hours'] = summary_data['Recovery_tau_p_sec'] / 3600.0
    
    display_columns = {
        'Capacity_C_K_Kr': 'Capacity State C_K (Kr)',
        'Readiness_alpha_K': 'Expanded Readiness alpha_K',
        'Workload_L': 'Workload Acceleration L',
        'Dynamic_Brake_beta_eff': 'Dynamic Brake beta_eff',
        'Recovery_tau_p_hours': 'Recovery Window tau_p (hours)',
        'Delta_C_Kr': 'Capacity Step Delta C (Kr)',
        'Capacity_C_Kplus1_Kr': 'Evaluated Capacity C_{K+1} (Kr)',
        'Fatigue_F_Kplus1': 'Post-Trial Fatigue F_{K+1}',
        'Min_Maint_L_min': 'Maintenance Threshold L_min'
    }
    
    # Calculate Mean, Min, and Max natively
    df_summary = summary_data[list(display_columns.keys())].agg(['mean', 'min', 'max']).T
    df_summary = df_summary.rename(index=display_columns).reset_index()
    df_summary.columns = ['Parameter Metric', 'Mean', 'Min', 'Max']
    
    print(df_summary.to_string(index=False))
    print("=====================================================================================\n")
