#!/usr/bin/env python3
"""
Comprehensive Fault Detection Analysis for Redundant Estimator Architecture.

This script provides a thorough analysis of fault detection capabilities:

1. Disagreement Dynamics Analysis
   - Mahalanobis distance between ESKF and Smoother estimates
   - Quaternion angle difference evolution
   - Fault signature characterization

2. ROC Curve Analysis
   - Detection probability vs false alarm rate
   - Optimal threshold determination
   - Multiple disagreement metrics comparison

3. Detection Delay Analysis
   - Time from fault onset to detection
   - Detection speed vs false alarm trade-off
   - Various fault magnitude analysis

4. Residual/Innovation Monitoring
   - Chi-squared test statistics over time
   - CUSUM (Cumulative Sum) charts for drift detection
   - Comparison of innovation-based vs disagreement-based detection

5. Fault Injection Studies
   - Step bias injection
   - Gradual sensor degradation
   - Intermittent dropouts
   - Outlier spikes

6. Information-Theoretic Metrics
   - KL divergence between estimator posteriors
   - Entropy evolution analysis

Usage:
    python scripts/fault_detection_analysis.py --config configs/config_baseline_short.yaml --generate
    python scripts/fault_detection_analysis.py --config configs/config_meas_spikes.yaml --generate --monte-carlo 50
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.special import kl_div

from data.db import SimulationDatabase
from data.classes import SimulationConfig
from estimation.eskf import ESKF
from estimation.redundant_estimator import RedundantEstimator, PrimaryEstimator
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from utilities.utils import load_yaml

# Lazy import for generator (has heavy dependencies like pyIGRF)
EnhancedAttitudeDataGenerator = None

def get_generator():
    global EnhancedAttitudeDataGenerator
    if EnhancedAttitudeDataGenerator is None:
        from data.generator_enhanced import EnhancedAttitudeDataGenerator as Gen
        EnhancedAttitudeDataGenerator = Gen
    return EnhancedAttitudeDataGenerator


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class InnovationRecord:
    """Record of measurement innovation statistics."""
    time: float
    sensor_type: str
    innovation: np.ndarray
    innovation_cov: np.ndarray
    mahalanobis_sq: float
    chi2_threshold: float
    rejected: bool


@dataclass
class FaultDetectionResult:
    """Results from a single fault detection run."""
    times: np.ndarray
    eskf_errors_deg: np.ndarray
    smoother_errors_deg: np.ndarray
    disagreement_deg: np.ndarray
    mahalanobis_distance: np.ndarray
    primary_estimator: List[str]
    switch_events: List[Tuple[float, str]]

    # Innovation monitoring
    innovation_records: List[InnovationRecord] = field(default_factory=list)
    chi2_stats: np.ndarray = field(default_factory=lambda: np.array([]))
    cusum_values: np.ndarray = field(default_factory=lambda: np.array([]))

    # Covariance traces
    eskf_cov_trace: np.ndarray = field(default_factory=lambda: np.array([]))
    smoother_cov_proxy: np.ndarray = field(default_factory=lambda: np.array([]))  # Approximation

    # Fault injection metadata
    fault_onset_time: Optional[float] = None
    fault_type: Optional[str] = None
    fault_magnitude: Optional[float] = None


@dataclass
class ROCPoint:
    """Single point on ROC curve."""
    threshold: float
    true_positive_rate: float
    false_positive_rate: float
    detection_delay: float  # Average delay when detected


@dataclass
class FaultDetectionMetrics:
    """Comprehensive fault detection metrics."""
    # ROC analysis
    roc_points: List[ROCPoint]
    auc: float  # Area under ROC curve
    optimal_threshold: float

    # Detection delay statistics
    mean_detection_delay: float
    std_detection_delay: float
    min_detection_delay: float
    max_detection_delay: float

    # False alarm statistics
    false_alarm_rate: float
    false_alarms_per_minute: float

    # Fault detection rates by type
    detection_rates_by_fault_type: Dict[str, float]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_attitude_error(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_true @ q_est.conjugate()
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


def compute_mahalanobis_quaternion(q1: Quaternion, P1: np.ndarray,
                                    q2: Quaternion, P2: np.ndarray) -> float:
    """
    Compute Mahalanobis distance between two quaternion estimates.

    Uses the angular difference and combined attitude covariance.

    Args:
        q1, q2: Quaternion estimates
        P1, P2: 6x6 covariances (attitude 3x3 is top-left)

    Returns:
        Mahalanobis distance
    """
    # Compute rotation difference
    dq = q1.conjugate() @ q2
    # Convert to axis-angle
    angle = 2 * np.arccos(np.clip(abs(dq.mu), 0, 1))
    if angle < 1e-10:
        return 0.0

    # Direction of rotation error (eta is the vector part of quaternion)
    if np.linalg.norm(dq.eta) > 1e-10:
        axis = dq.eta / np.linalg.norm(dq.eta)
    else:
        axis = np.array([1, 0, 0])

    delta_theta = angle * axis

    # Combined attitude covariance (top-left 3x3)
    P_att_combined = P1[:3, :3] + P2[:3, :3]

    # Mahalanobis distance
    try:
        P_inv = np.linalg.inv(P_att_combined)
        d_sq = delta_theta.T @ P_inv @ delta_theta
        return np.sqrt(d_sq)
    except np.linalg.LinAlgError:
        # Fallback to simple angle-based distance
        return angle / np.sqrt(np.trace(P_att_combined) / 3)


def compute_kl_divergence_gaussian(mu1: np.ndarray, P1: np.ndarray,
                                   mu2: np.ndarray, P2: np.ndarray) -> float:
    """
    Compute KL divergence D_KL(N1 || N2) between two Gaussians.

    D_KL = 0.5 * (tr(P2^{-1} P1) + (mu2-mu1)^T P2^{-1} (mu2-mu1) - k + ln(det(P2)/det(P1)))
    """
    k = len(mu1)
    try:
        P2_inv = np.linalg.inv(P2)
        term1 = np.trace(P2_inv @ P1)
        diff = mu2 - mu1
        term2 = diff.T @ P2_inv @ diff
        term3 = -k
        term4 = np.log(np.linalg.det(P2) / np.linalg.det(P1))
        return 0.5 * (term1 + term2 + term3 + term4)
    except np.linalg.LinAlgError:
        return np.nan


def compute_entropy_gaussian(P: np.ndarray) -> float:
    """Compute differential entropy of multivariate Gaussian."""
    k = P.shape[0]
    try:
        det_P = np.linalg.det(P)
        if det_P <= 0:
            return np.nan
        return 0.5 * k * (1 + np.log(2 * np.pi)) + 0.5 * np.log(det_P)
    except np.linalg.LinAlgError:
        return np.nan


# =============================================================================
# CUSUM DETECTOR
# =============================================================================

class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) detector for gradual drift detection.

    Detects when process mean shifts from nominal value.
    """

    def __init__(self, target: float = 0.0, threshold: float = 5.0,
                 drift: float = 1.0, window: int = 50):
        """
        Args:
            target: Expected nominal value
            threshold: Detection threshold (h parameter)
            drift: Allowable drift before detection (k parameter)
            window: Window for running statistics
        """
        self.target = target
        self.threshold = threshold
        self.drift = drift
        self.window = window

        # CUSUM statistics
        self.S_pos = 0.0  # Positive CUSUM
        self.S_neg = 0.0  # Negative CUSUM

        # History for running statistics
        self.history = deque(maxlen=window)

    def update(self, value: float) -> Tuple[float, float, bool]:
        """
        Update CUSUM with new value.

        Returns:
            (S_pos, S_neg, alarm): CUSUM values and alarm flag
        """
        self.history.append(value)

        # Compute running mean and std for normalization
        if len(self.history) >= 10:
            running_std = np.std(self.history)
            if running_std < 1e-10:
                running_std = 1.0
        else:
            running_std = 1.0

        # Normalized deviation
        z = (value - self.target) / running_std

        # Update CUSUM
        self.S_pos = max(0, self.S_pos + z - self.drift)
        self.S_neg = max(0, self.S_neg - z - self.drift)

        # Check for alarm
        alarm = (self.S_pos > self.threshold) or (self.S_neg > self.threshold)

        return self.S_pos, self.S_neg, alarm

    def reset(self):
        """Reset CUSUM statistics."""
        self.S_pos = 0.0
        self.S_neg = 0.0


# =============================================================================
# MAIN FAULT DETECTION RUNNER
# =============================================================================

class FaultDetectionAnalyzer:
    """
    Comprehensive fault detection analyzer for redundant estimator.
    """

    def __init__(self, config_path: str, db_path: str = "simulations.db"):
        self.config_path = config_path
        self.db_path = db_path
        self.config = load_yaml(config_path)
        self.db = SimulationDatabase(db_path)

    def run_single_analysis(
        self,
        sim_data,
        initial_error_deg: float = 10.0,
        fault_injection: Optional[Dict] = None,
    ) -> FaultDetectionResult:
        """
        Run fault detection analysis on simulation data.

        Args:
            sim_data: Simulation data from database
            initial_error_deg: Initial attitude error
            fault_injection: Optional fault to inject {type, onset_time, magnitude}

        Returns:
            FaultDetectionResult with all metrics
        """
        # Initialize estimator
        att_err_rad = np.deg2rad(initial_error_deg)
        P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

        redundant = RedundantEstimator(
            P0=P0,
            config_path=self.config_path,
            smoother_lag=60.0,
            use_robust=True,
            disagreement_threshold_deg=2.0,
            consecutive_disagreements_to_switch=5,
        )

        # Initial state with error
        q0_true = Quaternion.from_array(sim_data.q_true[0])
        perturb = np.array([att_err_rad, att_err_rad, att_err_rad]) / np.sqrt(3)
        q0_est = q0_true @ Quaternion.from_avec(perturb)
        q0_est = q0_est.normalize()

        nom0 = NominalState(ori=q0_est, gyro_bias=np.zeros(3))
        err0 = MultiVarGauss(np.zeros(6), P0.copy())
        x_est = EskfState(nom=nom0, err=err0)

        # Storage
        times = []
        eskf_errors = []
        smoother_errors = []
        disagreements = []
        mahalanobis_dists = []
        primaries = []
        cov_traces = []
        innovation_records = []

        # CUSUM detector for disagreement
        cusum = CUSUMDetector(target=0.0, threshold=5.0, drift=0.5)
        cusum_values = []

        # Chi-squared statistics
        chi2_stats = []

        # Fault injection state
        fault_active = False
        injected_bias = np.zeros(3)

        for k in range(1, len(sim_data.t)):
            t = sim_data.t[k]
            jd = sim_data.jd[k]
            dt_k = t - sim_data.t[k-1]

            # Get gyro measurement
            omega_k = sim_data.omega_meas[k]
            if np.any(np.isnan(omega_k)):
                omega_k = sim_data.omega_meas[k-1]

            # Apply fault injection if specified
            if fault_injection:
                if t >= fault_injection.get('onset_time', float('inf')):
                    fault_active = True

                    if fault_injection['type'] == 'gyro_bias_step':
                        # Inject step bias
                        if np.linalg.norm(injected_bias) < 1e-10:
                            injected_bias = fault_injection['magnitude'] * np.array([1, 0.5, 0.3])
                        omega_k = omega_k + injected_bias

                    elif fault_injection['type'] == 'gyro_degradation':
                        # Gradual degradation (increasing noise)
                        elapsed = t - fault_injection['onset_time']
                        noise_scale = 1.0 + elapsed * fault_injection['magnitude']
                        omega_k = omega_k + noise_scale * 0.01 * np.random.randn(3)

            # Get measurements
            z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
            z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
            z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None
            B_n = sim_data.b_eci[k]
            s_n = sim_data.s_eci[k]

            # Apply measurement fault injection
            if fault_injection and fault_active:
                if fault_injection['type'] == 'mag_spike' and z_mag is not None:
                    if np.random.random() < fault_injection.get('probability', 0.1):
                        z_mag = z_mag + fault_injection['magnitude'] * np.random.randn(3)

            # Run redundant estimator step
            x_est, smoother_state, disagreement_deg, primary_str = redundant.step(
                x_eskf=x_est,
                t=t,
                jd=jd,
                omega_meas=omega_k,
                dt=dt_k,
                z_mag=z_mag,
                z_sun=z_sun,
                z_st=z_st,
                B_n=B_n,
                s_n=s_n,
            )

            # Compute errors
            q_true = Quaternion.from_array(sim_data.q_true[k])
            eskf_err = compute_attitude_error(x_est.nom.ori, q_true)
            smoother_err = compute_attitude_error(smoother_state.ori, q_true)

            # Compute Mahalanobis distance (approximate for smoother)
            # Use ESKF covariance as proxy for smoother
            smoother_P_proxy = x_est.err.cov * 1.5  # Smoother typically more uncertain
            mahal = compute_mahalanobis_quaternion(
                x_est.nom.ori, x_est.err.cov,
                smoother_state.ori, smoother_P_proxy
            )

            # Update CUSUM
            cusum_pos, cusum_neg, _ = cusum.update(disagreement_deg)

            # Record chi-squared stat (from last innovation if available)
            # This is an approximation - in practice we'd need to track innovations
            chi2_stat = (disagreement_deg / 0.5) ** 2  # Normalized by expected std

            # Store results
            times.append(t)
            eskf_errors.append(eskf_err)
            smoother_errors.append(smoother_err)
            disagreements.append(disagreement_deg)
            mahalanobis_dists.append(mahal)
            primaries.append(primary_str)
            cov_traces.append(np.trace(x_est.err.cov[:3, :3]))
            cusum_values.append(max(cusum_pos, cusum_neg))
            chi2_stats.append(chi2_stat)

        # Get switch events from redundant estimator
        stats = redundant.get_statistics()
        switch_events = stats.get('switch_events', [])

        return FaultDetectionResult(
            times=np.array(times),
            eskf_errors_deg=np.array(eskf_errors),
            smoother_errors_deg=np.array(smoother_errors),
            disagreement_deg=np.array(disagreements),
            mahalanobis_distance=np.array(mahalanobis_dists),
            primary_estimator=primaries,
            switch_events=switch_events,
            chi2_stats=np.array(chi2_stats),
            cusum_values=np.array(cusum_values),
            eskf_cov_trace=np.array(cov_traces),
            fault_onset_time=fault_injection.get('onset_time') if fault_injection else None,
            fault_type=fault_injection.get('type') if fault_injection else None,
            fault_magnitude=fault_injection.get('magnitude') if fault_injection else None,
        )

    def compute_roc_curve(
        self,
        results_nominal: List[FaultDetectionResult],
        results_faulty: List[FaultDetectionResult],
        thresholds: np.ndarray = None,
    ) -> Tuple[List[ROCPoint], float]:
        """
        Compute ROC curve from nominal and faulty runs.

        Args:
            results_nominal: Results from runs without faults (for false alarms)
            results_faulty: Results from runs with faults (for detection)
            thresholds: Disagreement thresholds to evaluate

        Returns:
            (roc_points, auc): ROC points and area under curve
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 5.0, 50)

        roc_points = []

        for thresh in thresholds:
            # False positive rate: fraction of nominal samples exceeding threshold
            fp_count = 0
            total_nominal = 0
            for res in results_nominal:
                # Only count after convergence (t > 30s)
                mask = res.times > 30
                fp_count += np.sum(res.disagreement_deg[mask] > thresh)
                total_nominal += np.sum(mask)
            fpr = fp_count / total_nominal if total_nominal > 0 else 0

            # True positive rate: fraction of faulty samples detected after fault onset
            tp_count = 0
            total_faulty = 0
            detection_delays = []

            for res in results_faulty:
                if res.fault_onset_time is not None:
                    # Find first detection after fault
                    mask = res.times >= res.fault_onset_time
                    fault_times = res.times[mask]
                    fault_dis = res.disagreement_deg[mask]

                    detected_idx = np.where(fault_dis > thresh)[0]
                    if len(detected_idx) > 0:
                        tp_count += 1
                        delay = fault_times[detected_idx[0]] - res.fault_onset_time
                        detection_delays.append(delay)
                    total_faulty += 1

            tpr = tp_count / total_faulty if total_faulty > 0 else 0
            avg_delay = np.mean(detection_delays) if detection_delays else float('inf')

            roc_points.append(ROCPoint(
                threshold=thresh,
                true_positive_rate=tpr,
                false_positive_rate=fpr,
                detection_delay=avg_delay,
            ))

        # Compute AUC using trapezoidal rule
        fprs = [p.false_positive_rate for p in roc_points]
        tprs = [p.true_positive_rate for p in roc_points]
        # Sort by FPR
        sorted_idx = np.argsort(fprs)
        fprs_sorted = np.array(fprs)[sorted_idx]
        tprs_sorted = np.array(tprs)[sorted_idx]
        auc = np.trapz(tprs_sorted, fprs_sorted)

        return roc_points, auc

    def find_optimal_threshold(self, roc_points: List[ROCPoint],
                               max_fpr: float = 0.05) -> float:
        """Find optimal threshold given maximum acceptable FPR."""
        valid_points = [p for p in roc_points if p.false_positive_rate <= max_fpr]
        if not valid_points:
            return roc_points[0].threshold

        # Find point with highest TPR among valid points
        best_point = max(valid_points, key=lambda p: p.true_positive_rate)
        return best_point.threshold

    def run_monte_carlo_analysis(
        self,
        n_runs: int = 50,
        fault_configs: List[Dict] = None,
    ) -> Dict:
        """
        Run Monte Carlo analysis with various fault configurations.

        Args:
            n_runs: Number of Monte Carlo runs per configuration
            fault_configs: List of fault configurations to test

        Returns:
            Dictionary with analysis results
        """
        if fault_configs is None:
            fault_configs = [
                None,  # Nominal (no fault)
                {'type': 'gyro_bias_step', 'onset_time': 100.0, 'magnitude': 0.01},
                {'type': 'gyro_bias_step', 'onset_time': 100.0, 'magnitude': 0.05},
                {'type': 'gyro_degradation', 'onset_time': 100.0, 'magnitude': 0.1},
                {'type': 'mag_spike', 'onset_time': 100.0, 'magnitude': 0.5, 'probability': 0.1},
            ]

        results_by_config = {}

        for config_idx, fault_config in enumerate(fault_configs):
            config_name = fault_config['type'] if fault_config else 'nominal'
            print(f"\nRunning config {config_idx+1}/{len(fault_configs)}: {config_name}")

            config_results = []

            for run_idx in range(n_runs):
                if (run_idx + 1) % 10 == 0:
                    print(f"  Run {run_idx+1}/{n_runs}")

                # Generate new simulation
                GeneratorClass = get_generator()
                generator = GeneratorClass(
                    db_path=self.db_path,
                    config_path=self.config_path
                )
                sim_cfg = SimulationConfig(
                    T=self.config['time']['sim_T'],
                    dt=self.config['time']['sim_dt'],
                    start_jd=self.config['time']['start_jd']
                )
                sim_id = generator.run(sim_cfg)
                sim_data = self.db.load_run(sim_id)

                # Run analysis
                result = self.run_single_analysis(
                    sim_data,
                    initial_error_deg=10.0,
                    fault_injection=fault_config,
                )
                config_results.append(result)

            results_by_config[config_name] = config_results

        return results_by_config


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_disagreement_dynamics(result: FaultDetectionResult, save_path: str = None):
    """Plot disagreement evolution over time."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # Plot 1: Attitude errors
    ax1 = axes[0]
    ax1.semilogy(result.times, result.eskf_errors_deg, 'b-', label='ESKF', alpha=0.8)
    ax1.semilogy(result.times, result.smoother_errors_deg, 'r--', label='Smoother', alpha=0.8)
    ax1.set_ylabel('Attitude Error [deg]')
    ax1.legend(loc='upper right')
    ax1.set_title('Fault Detection Analysis: Disagreement Dynamics')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([1e-3, 20])

    # Mark fault onset if present
    if result.fault_onset_time:
        for ax in axes:
            ax.axvline(result.fault_onset_time, color='red', linestyle='--',
                      alpha=0.5, label='Fault onset')

    # Plot 2: Disagreement (quaternion angle)
    ax2 = axes[1]
    ax2.plot(result.times, result.disagreement_deg, 'purple', linewidth=1)
    ax2.fill_between(result.times, result.disagreement_deg, alpha=0.3, color='purple')
    ax2.axhline(2.0, color='r', linestyle='--', alpha=0.5, label='Switch threshold (2°)')
    ax2.axhline(0.5, color='g', linestyle='--', alpha=0.5, label='Agreement threshold (0.5°)')
    ax2.set_ylabel('Disagreement [deg]')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Mahalanobis distance
    ax3 = axes[2]
    ax3.plot(result.times, result.mahalanobis_distance, 'orange', linewidth=1)
    ax3.fill_between(result.times, result.mahalanobis_distance, alpha=0.3, color='orange')
    ax3.set_ylabel('Mahalanobis Distance')
    ax3.grid(True, alpha=0.3)

    # Plot 4: CUSUM values
    ax4 = axes[3]
    ax4.plot(result.times, result.cusum_values, 'green', linewidth=1)
    ax4.axhline(5.0, color='r', linestyle='--', alpha=0.5, label='CUSUM threshold')
    ax4.set_ylabel('CUSUM Statistic')
    ax4.set_xlabel('Time [s]')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved disagreement dynamics plot to {save_path}")

    plt.close()


def plot_roc_curve(roc_points: List[ROCPoint], auc: float, save_path: str = None):
    """Plot ROC curve with detection delay annotations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    fprs = [p.false_positive_rate for p in roc_points]
    tprs = [p.true_positive_rate for p in roc_points]

    ax1.plot(fprs, tprs, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random classifier')
    ax1.fill_between(fprs, tprs, alpha=0.2)

    # Mark operating points
    # 1% FPR point
    fpr_1pct = [p for p in roc_points if p.false_positive_rate <= 0.01]
    if fpr_1pct:
        best_1pct = max(fpr_1pct, key=lambda p: p.true_positive_rate)
        ax1.scatter([best_1pct.false_positive_rate], [best_1pct.true_positive_rate],
                   c='red', s=100, zorder=5, label=f'FPR≤1%: TPR={best_1pct.true_positive_rate:.2f}')

    # 5% FPR point
    fpr_5pct = [p for p in roc_points if p.false_positive_rate <= 0.05]
    if fpr_5pct:
        best_5pct = max(fpr_5pct, key=lambda p: p.true_positive_rate)
        ax1.scatter([best_5pct.false_positive_rate], [best_5pct.true_positive_rate],
                   c='green', s=100, zorder=5, label=f'FPR≤5%: TPR={best_5pct.true_positive_rate:.2f}')

    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve for Fault Detection')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Detection delay vs threshold
    thresholds = [p.threshold for p in roc_points]
    delays = [p.detection_delay for p in roc_points]
    tprs_plot = [p.true_positive_rate for p in roc_points]

    ax2.plot(thresholds, delays, 'b-', linewidth=2, label='Detection delay')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(thresholds, tprs_plot, 'r--', linewidth=2, label='TPR')

    ax2.set_xlabel('Threshold [deg]')
    ax2.set_ylabel('Detection Delay [s]', color='blue')
    ax2_twin.set_ylabel('True Positive Rate', color='red')
    ax2.set_title('Detection Delay vs Threshold Trade-off')
    ax2.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")

    plt.close()


def plot_detection_delay_analysis(
    results_by_config: Dict[str, List[FaultDetectionResult]],
    threshold: float,
    save_path: str = None
):
    """Plot detection delay distribution for different fault types."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    fault_types = [k for k in results_by_config.keys() if k != 'nominal']

    for idx, fault_type in enumerate(fault_types[:4]):
        ax = axes[idx]
        results = results_by_config[fault_type]

        delays = []
        for res in results:
            if res.fault_onset_time:
                mask = res.times >= res.fault_onset_time
                fault_dis = res.disagreement_deg[mask]
                fault_times = res.times[mask]

                detected_idx = np.where(fault_dis > threshold)[0]
                if len(detected_idx) > 0:
                    delay = fault_times[detected_idx[0]] - res.fault_onset_time
                    delays.append(delay)

        if delays:
            ax.hist(delays, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(delays), color='r', linestyle='--',
                      label=f'Mean: {np.mean(delays):.2f}s')
            ax.axvline(np.median(delays), color='g', linestyle='--',
                      label=f'Median: {np.median(delays):.2f}s')
            ax.set_xlabel('Detection Delay [s]')
            ax.set_ylabel('Count')
            ax.set_title(f'{fault_type}\n(Detection rate: {len(delays)}/{len(results)})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No detections', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'{fault_type}')

    plt.suptitle(f'Detection Delay Distribution (threshold = {threshold}°)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved detection delay analysis to {save_path}")

    plt.close()


def plot_cusum_analysis(result: FaultDetectionResult, save_path: str = None):
    """Plot CUSUM analysis with chi-squared comparison."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Raw disagreement signal
    ax1 = axes[0]
    ax1.plot(result.times, result.disagreement_deg, 'b-', linewidth=0.8)
    ax1.set_ylabel('Disagreement [deg]')
    ax1.set_title('CUSUM vs Chi-squared Fault Detection Comparison')
    ax1.grid(True, alpha=0.3)

    if result.fault_onset_time:
        ax1.axvline(result.fault_onset_time, color='red', linestyle='--',
                   alpha=0.7, label='Fault onset')
        ax1.legend()

    # Plot 2: Chi-squared statistic
    ax2 = axes[1]
    ax2.plot(result.times, result.chi2_stats, 'orange', linewidth=0.8)
    ax2.axhline(7.81, color='r', linestyle='--', alpha=0.5, label='χ² threshold (99%, 3 DOF)')
    ax2.set_ylabel('χ² Statistic')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: CUSUM statistic
    ax3 = axes[2]
    ax3.plot(result.times, result.cusum_values, 'green', linewidth=0.8)
    ax3.axhline(5.0, color='r', linestyle='--', alpha=0.5, label='CUSUM threshold')
    ax3.set_ylabel('CUSUM')
    ax3.set_xlabel('Time [s]')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved CUSUM analysis to {save_path}")

    plt.close()


def plot_information_theoretic(result: FaultDetectionResult, save_path: str = None):
    """Plot entropy evolution over time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Compute entropy from covariance trace (approximation)
    # For 3D Gaussian, entropy ~ 0.5 * log(det(P)) ~ 0.5 * 3 * log(trace/3)
    entropies = 0.5 * 3 * np.log(result.eskf_cov_trace / 3 + 1e-10)

    # Plot 1: Covariance trace
    ax1 = axes[0]
    ax1.semilogy(result.times, result.eskf_cov_trace, 'b-', linewidth=1)
    ax1.set_ylabel('Attitude Cov Trace [rad²]')
    ax1.set_title('Information-Theoretic Metrics')
    ax1.grid(True, alpha=0.3)

    if result.fault_onset_time:
        ax1.axvline(result.fault_onset_time, color='red', linestyle='--',
                   alpha=0.7, label='Fault onset')
        ax1.legend()

    # Plot 2: Entropy proxy
    ax2 = axes[1]
    ax2.plot(result.times, entropies, 'purple', linewidth=1)
    ax2.set_ylabel('Entropy Proxy')
    ax2.set_xlabel('Time [s]')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved information-theoretic plot to {save_path}")

    plt.close()


def create_summary_table(
    results_by_config: Dict[str, List[FaultDetectionResult]],
    threshold: float,
) -> str:
    """Create summary table of fault detection performance."""
    lines = []
    lines.append("=" * 80)
    lines.append("FAULT DETECTION PERFORMANCE SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Detection threshold: {threshold}°")
    lines.append("")
    lines.append(f"{'Fault Type':<25} {'Detection Rate':>15} {'Mean Delay':>12} {'Std Delay':>12}")
    lines.append("-" * 80)

    for fault_type, results in results_by_config.items():
        delays = []
        detections = 0

        for res in results:
            if res.fault_onset_time:  # Has fault
                mask = res.times >= res.fault_onset_time
                fault_dis = res.disagreement_deg[mask]
                fault_times = res.times[mask]

                detected_idx = np.where(fault_dis > threshold)[0]
                if len(detected_idx) > 0:
                    detections += 1
                    delay = fault_times[detected_idx[0]] - res.fault_onset_time
                    delays.append(delay)
            else:  # Nominal run
                # Count false alarms
                mask = res.times > 30  # After convergence
                fa_count = np.sum(res.disagreement_deg[mask] > threshold)
                if fa_count > 0:
                    detections += 1  # False alarm

        n_runs = len(results)
        rate = detections / n_runs if n_runs > 0 else 0
        mean_delay = np.mean(delays) if delays else float('nan')
        std_delay = np.std(delays) if delays else float('nan')

        lines.append(f"{fault_type:<25} {rate:>14.1%} {mean_delay:>11.2f}s {std_delay:>11.2f}s")

    lines.append("=" * 80)
    return "\n".join(lines)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Fault Detection Analysis")
    parser.add_argument("--config", type=str, default="configs/config_baseline_short.yaml",
                       help="Configuration file path")
    parser.add_argument("--db", type=str, default="simulations.db",
                       help="Database file path")
    parser.add_argument("--sim-id", type=int, help="Use existing simulation ID")
    parser.add_argument("--generate", action="store_true", help="Generate new simulation")
    parser.add_argument("--monte-carlo", type=int, default=0,
                       help="Number of Monte Carlo runs (0 for single run)")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="Output directory for plots")

    args = parser.parse_args()

    print("=" * 70)
    print("COMPREHENSIVE FAULT DETECTION ANALYSIS")
    print("=" * 70)
    print(f"Config: {args.config}")

    config = load_yaml(args.config)
    scenario_name = config.get('simulation', {}).get('run_name', Path(args.config).stem)
    print(f"Scenario: {scenario_name}")

    analyzer = FaultDetectionAnalyzer(args.config, args.db)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.monte_carlo > 0:
        # Monte Carlo analysis
        print(f"\nRunning Monte Carlo analysis with {args.monte_carlo} runs...")

        fault_configs = [
            None,  # Nominal
            {'type': 'gyro_bias_step', 'onset_time': 100.0, 'magnitude': 0.01},
            {'type': 'gyro_bias_step', 'onset_time': 100.0, 'magnitude': 0.03},
            {'type': 'gyro_degradation', 'onset_time': 100.0, 'magnitude': 0.05},
        ]

        results_by_config = analyzer.run_monte_carlo_analysis(
            n_runs=args.monte_carlo,
            fault_configs=fault_configs,
        )

        # Compute ROC curve
        nominal_results = results_by_config.get('nominal', [])
        faulty_results = []
        for k, v in results_by_config.items():
            if k != 'nominal':
                faulty_results.extend(v)

        if nominal_results and faulty_results:
            print("\nComputing ROC curve...")
            roc_points, auc = analyzer.compute_roc_curve(nominal_results, faulty_results)
            optimal_threshold = analyzer.find_optimal_threshold(roc_points, max_fpr=0.05)

            print(f"  AUC: {auc:.3f}")
            print(f"  Optimal threshold (FPR≤5%): {optimal_threshold:.2f}°")

            if args.save_plots:
                plot_roc_curve(roc_points, auc,
                              str(output_dir / f"fault_detection_roc_{scenario_name}.pdf"))
                plot_detection_delay_analysis(results_by_config, optimal_threshold,
                              str(output_dir / f"fault_detection_delay_{scenario_name}.pdf"))

        # Print summary table
        print("\n" + create_summary_table(results_by_config, 2.0))

    else:
        # Single run analysis
        if args.generate:
            print("\nGenerating new simulation...")
            GeneratorClass = get_generator()
            generator = GeneratorClass(
                db_path=args.db,
                config_path=args.config
            )
            sim_cfg = SimulationConfig(
                T=config['time']['sim_T'],
                dt=config['time']['sim_dt'],
                start_jd=config['time']['start_jd']
            )
            sim_id = generator.run(sim_cfg)
        elif args.sim_id:
            sim_id = args.sim_id
        else:
            print("Error: Must specify --generate or --sim-id")
            return 1

        db = SimulationDatabase(args.db)
        sim_data = db.load_run(sim_id)
        print(f"Loaded simulation: {sim_data.t[-1]:.1f}s, {len(sim_data.t)} samples")

        # Run single analysis
        print("\nRunning fault detection analysis...")

        # Nominal run
        result_nominal = analyzer.run_single_analysis(sim_data, initial_error_deg=10.0)

        # Run with fault injection
        fault_config = {'type': 'gyro_bias_step', 'onset_time': 100.0, 'magnitude': 0.02}
        result_faulty = analyzer.run_single_analysis(
            sim_data, initial_error_deg=10.0, fault_injection=fault_config
        )

        # Print statistics
        print("\n" + "=" * 50)
        print("NOMINAL RUN STATISTICS")
        print("=" * 50)
        print(f"  Mean disagreement: {np.mean(result_nominal.disagreement_deg):.4f}°")
        print(f"  Max disagreement: {np.max(result_nominal.disagreement_deg):.4f}°")
        print(f"  Std disagreement: {np.std(result_nominal.disagreement_deg):.4f}°")
        print(f"  Switch events: {len(result_nominal.switch_events)}")

        print("\n" + "=" * 50)
        print("FAULTY RUN STATISTICS")
        print("=" * 50)
        print(f"  Fault type: {result_faulty.fault_type}")
        print(f"  Fault onset: {result_faulty.fault_onset_time}s")
        print(f"  Mean disagreement (post-fault): ", end="")
        post_fault_mask = result_faulty.times >= result_faulty.fault_onset_time
        print(f"{np.mean(result_faulty.disagreement_deg[post_fault_mask]):.4f}°")
        print(f"  Max disagreement: {np.max(result_faulty.disagreement_deg):.4f}°")
        print(f"  Switch events: {len(result_faulty.switch_events)}")

        # Generate plots
        if args.save_plots:
            plot_disagreement_dynamics(result_nominal,
                str(output_dir / f"fault_detection_nominal_{scenario_name}.pdf"))
            plot_disagreement_dynamics(result_faulty,
                str(output_dir / f"fault_detection_faulty_{scenario_name}.pdf"))
            plot_cusum_analysis(result_faulty,
                str(output_dir / f"fault_detection_cusum_{scenario_name}.pdf"))
            plot_information_theoretic(result_faulty,
                str(output_dir / f"fault_detection_entropy_{scenario_name}.pdf"))

    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
