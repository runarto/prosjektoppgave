#!/usr/bin/env python3
"""
Comprehensive Fault Detection Test for Attitude Estimation

Tests three estimators (ESKF, Fixed-Lag Smoother, Redundant Architecture)
under three fault scenarios:
1. Sun sensor fault (spikes)
2. Magnetometer fault (spikes)
3. Star tracker fault (spikes)

Uses:
- GTSAM integrated preintegration
- Gating for outlier detection
- Huber M-estimator with k=0.1

Goal: Check if the hybrid/redundant method can successfully detect faults
and visualize its response.

Generates report-quality plots.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import copy
import yaml
import os

# Set up logging
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from utilities.sensors import SensorGyro, SensorMagnetometer, SensorSunVector, SensorStarTracker
from utilities.process_model import ProcessModel
from utilities.utils import load_yaml
from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from estimation.redundant_estimator import RedundantEstimator
from environment.environment import OrbitEnvironmentModel
from logging_config import get_logger

logger = get_logger(__name__)

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


@dataclass
class FaultScenario:
    """Configuration for a fault scenario."""
    name: str
    sensor: str  # 'sun', 'mag', or 'star'
    fault_start: float  # seconds
    fault_end: float  # seconds
    spike_probability: float
    spike_magnitude: float  # magnitude for vector sensors, angle (rad) for star tracker


@dataclass
class EstimatorResult:
    """Results from a single estimator run."""
    times: np.ndarray
    attitude_errors_deg: np.ndarray
    bias_errors: np.ndarray
    raw_states: List[NominalState]
    # For redundant estimator
    disagreement_deg: Optional[np.ndarray] = None
    primary_estimator: Optional[List[str]] = None
    switch_events: Optional[List[Tuple[float, str]]] = None


def create_temp_config(base_config_path: str, scenario: FaultScenario) -> str:
    """Create a temporary config file with fault injection enabled."""
    config = load_yaml(base_config_path)

    # Enable spikes for the target sensor
    if scenario.sensor == 'sun':
        config['sensors']['sun']['spikes'] = {
            'enabled': True,
            'magnitude': scenario.spike_magnitude,
            'probability': scenario.spike_probability,
        }
    elif scenario.sensor == 'mag':
        config['sensors']['mag']['spikes'] = {
            'enabled': True,
            'magnitude': scenario.spike_magnitude,
            'probability': scenario.spike_probability,
        }
    elif scenario.sensor == 'star':
        config['sensors']['star']['spikes'] = {
            'enabled': True,
            'angle': scenario.spike_magnitude,
            'probability': scenario.spike_probability,
        }

    # Write to temp file
    temp_path = f'/tmp/config_fault_{scenario.sensor}.yaml'
    with open(temp_path, 'w') as f:
        yaml.dump(config, f)

    return temp_path


def get_simple_B_eci(r_eci: np.ndarray, jd: float) -> np.ndarray:
    """Simple dipole magnetic field model as fallback when IGRF fails."""
    # Simple tilted dipole model
    r = np.linalg.norm(r_eci)
    r_hat = r_eci / r

    # Earth's magnetic dipole axis (approximately)
    dipole_axis = np.array([0.0, 0.1, -0.99])
    dipole_axis = dipole_axis / np.linalg.norm(dipole_axis)

    # Dipole field: B = (3 * (m . r_hat) * r_hat - m) / r^3
    B = 3 * np.dot(dipole_axis, r_hat) * r_hat - dipole_axis
    B = B / np.linalg.norm(B)  # Normalize to unit vector
    return B


def generate_simulation_data(
    config_path: str,
    scenario: FaultScenario,
    T: float = 300.0,
    dt: float = 0.02,
) -> Dict:
    """Generate simulation data with fault injection during specified window."""
    config = load_yaml(config_path)
    env = OrbitEnvironmentModel()
    use_simple_B = False  # Will be set to True if IGRF fails

    # Create sensors
    gyro = SensorGyro(config_path)
    mag = SensorMagnetometer(config_path)
    sun = SensorSunVector(config_path)
    star = SensorStarTracker(config_path)

    n_steps = int(T / dt) + 1
    t_vec = np.linspace(0.0, T, n_steps)
    start_jd = config['time']['start_jd']
    jd_vec = start_jd + t_vec / 86400.0

    # Get omega profile parameters
    A_x = config['omega_profile']['amplitude']['x']
    A_y = config['omega_profile']['amplitude']['y']
    A_z = config['omega_profile']['amplitude']['z']
    f_x = config['omega_profile']['frequency']['x']
    f_y = config['omega_profile']['frequency']['y']
    f_z = config['omega_profile']['frequency']['z']

    # Initialize truth
    s_eci_init = env.get_sun_eci(jd_vec[0])
    s_eci_init = s_eci_init / np.linalg.norm(s_eci_init)
    z_body = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_body, s_eci_init)
    axis_norm = np.linalg.norm(axis)

    if axis_norm > 1e-6:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(z_body, s_eci_init), -1.0, 1.0))
        q_true = Quaternion.from_avec(axis * angle)
    else:
        q_true = Quaternion(1.0, np.zeros(3))

    b_g_true = np.zeros(3)

    # Bias random walk parameters
    RRW_deg = config['sensors']['gyro']['noise']['rrw_deg']
    sigma_bg_true = RRW_deg * (np.pi / 180.0) / 3600.0 / 60.0

    # Measurement divisors
    gyro_div = max(1, int(round(gyro.dt / dt)))
    mag_div = max(1, int(round(mag.dt / dt)))
    sun_div = max(1, int(round(sun.dt / dt)))
    st_div = max(1, int(round(star.dt / dt)))

    # Allocate arrays
    q_true_log = np.zeros((n_steps, 4))
    b_g_true_log = np.zeros((n_steps, 3))
    omega_true_log = np.zeros((n_steps, 3))
    b_eci_log = np.zeros((n_steps, 3))
    s_eci_log = np.zeros((n_steps, 3))
    omega_meas_log = np.full((n_steps, 3), np.nan)
    mag_meas_log = np.full((n_steps, 3), np.nan)
    sun_meas_log = np.full((n_steps, 3), np.nan)
    st_meas_log = np.full((n_steps, 4), np.nan)
    fault_active_log = np.zeros(n_steps, dtype=bool)

    print(f"Generating simulation data for {scenario.name}...")
    print(f"  Fault window: {scenario.fault_start}s - {scenario.fault_end}s")
    print(f"  Faulty sensor: {scenario.sensor}")

    for k in range(n_steps):
        t = t_vec[k]
        jd = jd_vec[k]

        # Check if fault is active
        fault_active = scenario.fault_start <= t <= scenario.fault_end
        fault_active_log[k] = fault_active

        # Environment
        r_eci = env.get_r_eci(jd)
        B_eci = env.get_B_eci(r_eci, jd)
        s_eci = env.get_sun_eci(jd)

        # True angular velocity
        omega_true = np.array([
            A_x * np.sin(2 * np.pi * f_x * t),
            A_y * np.sin(2 * np.pi * f_y * t),
            A_z * np.sin(2 * np.pi * f_z * t),
        ])
        omega_true_log[k] = omega_true

        # Propagate truth
        if k > 0:
            q_true = q_true.propagate(omega_true, dt)
            w_bg = sigma_bg_true * np.sqrt(dt) * np.random.randn(3)
            b_g_true = b_g_true + w_bg

        # Gyro measurement
        if k % gyro_div == 0:
            omega_meas = gyro.sample(omega_true + b_g_true)
            omega_meas_log[k] = omega_meas

        # Magnetometer measurement - inject spikes during fault window
        if k % mag_div == 0:
            if fault_active and scenario.sensor == 'mag':
                # Temporarily enable spikes
                orig_enabled = mag.spikes_enabled
                orig_prob = mag.spike_probability
                orig_mag = mag.spike_magnitude
                mag.spikes_enabled = True
                mag.spike_probability = scenario.spike_probability
                mag.spike_magnitude = scenario.spike_magnitude

                mag_meas = mag.sample(q_true=q_true, B_n=B_eci, t=t)

                mag.spikes_enabled = orig_enabled
                mag.spike_probability = orig_prob
                mag.spike_magnitude = orig_mag
            else:
                mag_meas = mag.sample(q_true=q_true, B_n=B_eci, t=t)

            if mag_meas is not None:
                mag_meas_log[k] = mag_meas

        # Sun sensor measurement - inject spikes during fault window
        if k % sun_div == 0:
            if fault_active and scenario.sensor == 'sun':
                orig_enabled = sun.spikes_enabled
                orig_prob = sun.spike_probability
                orig_mag = sun.spike_magnitude
                sun.spikes_enabled = True
                sun.spike_probability = scenario.spike_probability
                sun.spike_magnitude = scenario.spike_magnitude

                sun_meas = sun.sample(q_true=q_true, s_n=s_eci)

                sun.spikes_enabled = orig_enabled
                sun.spike_probability = orig_prob
                sun.spike_magnitude = orig_mag
            else:
                sun_meas = sun.sample(q_true=q_true, s_n=s_eci)

            if sun_meas is not None:
                sun_meas_log[k] = sun_meas

        # Star tracker measurement - inject spikes during fault window
        if k % st_div == 0:
            if fault_active and scenario.sensor == 'star':
                orig_enabled = star.spikes_enabled
                orig_prob = star.spike_probability
                orig_angle = star.spike_angle
                star.spikes_enabled = True
                star.spike_probability = scenario.spike_probability
                star.spike_angle = scenario.spike_magnitude

                st_meas = star.sample(q_true=q_true, omega_body=omega_true)

                star.spikes_enabled = orig_enabled
                star.spike_probability = orig_prob
                star.spike_angle = orig_angle
            else:
                st_meas = star.sample(q_true=q_true, omega_body=omega_true)

            if st_meas is not None:
                st_meas_log[k] = st_meas.as_array()

        # Log truth
        q_true_log[k] = q_true.as_array()
        b_g_true_log[k] = b_g_true
        b_eci_log[k] = B_eci
        s_eci_log[k] = s_eci

    print(f"  Generated {n_steps} samples")

    return {
        't': t_vec,
        'jd': jd_vec,
        'q_true': q_true_log,
        'b_g_true': b_g_true_log,
        'omega_true': omega_true_log,
        'omega_meas': omega_meas_log,
        'mag_meas': mag_meas_log,
        'sun_meas': sun_meas_log,
        'st_meas': st_meas_log,
        'b_eci': b_eci_log,
        's_eci': s_eci_log,
        'fault_active': fault_active_log,
    }


def compute_attitude_error(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_true @ q_est.conjugate()
    angle = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle)


def run_eskf(
    sim_data: Dict,
    config_path: str,
    use_gating: bool = True,
    gating_threshold: float = 9.21,  # chi2 with 3 DOF, 99% confidence
) -> EstimatorResult:
    """Run ESKF on simulation data with optional gating."""
    print("Running ESKF...")

    config = load_yaml(config_path)
    process = ProcessModel(config_path)

    # Initial covariance
    P0 = np.diag([0.1**2, 0.1**2, 0.1**2, 1e-8, 1e-8, 1e-8])

    eskf = ESKF(P0=P0, config_path=config_path)
    eskf.use_gating = use_gating
    eskf.gating_threshold = gating_threshold

    # Initialize from first star tracker or perturbed truth
    q_init = Quaternion.from_array(sim_data['q_true'][0])
    q_init = q_init @ Quaternion.from_avec(np.random.randn(3) * 0.1)  # Small perturbation

    x = EskfState(
        nom=NominalState(ori=q_init, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy()),
    )

    times = []
    errors = []
    bias_errors = []
    states = []

    n_steps = len(sim_data['t'])

    for k in range(n_steps):
        t = sim_data['t'][k]
        jd = sim_data['jd'][k]

        # Gyro measurement
        omega = sim_data['omega_meas'][k]
        if np.any(np.isnan(omega)):
            if k > 0:
                omega = sim_data['omega_meas'][k-1]
            else:
                omega = np.zeros(3)

        # Prediction
        dt = sim_data['t'][1] - sim_data['t'][0] if k > 0 else 0.02
        if k > 0:
            x = eskf.predict(x, omega, dt)

        # Measurement updates
        z_mag = sim_data['mag_meas'][k]
        z_sun = sim_data['sun_meas'][k]
        z_st = sim_data['st_meas'][k]
        B_n = sim_data['b_eci'][k]
        s_n = sim_data['s_eci'][k]

        if not np.any(np.isnan(z_mag)):
            try:
                x = eskf.update(x, z_mag, SensorType.MAGNETOMETER, B_n=B_n)
            except ValueError:
                pass

        if not np.any(np.isnan(z_sun)):
            try:
                x = eskf.update(x, z_sun, SensorType.SUN_VECTOR, s_n=s_n)
            except ValueError:
                pass

        if not np.any(np.isnan(z_st)):
            try:
                q_st = Quaternion.from_array(z_st)
                x = eskf.update(x, q_st, SensorType.STAR_TRACKER)
            except ValueError:
                pass

        # Compute errors
        q_true = Quaternion.from_array(sim_data['q_true'][k])
        err_deg = compute_attitude_error(x.nom.ori, q_true)
        bias_err = x.nom.gyro_bias - sim_data['b_g_true'][k]

        times.append(t)
        errors.append(err_deg)
        bias_errors.append(bias_err)
        states.append(NominalState(ori=x.nom.ori.copy(), gyro_bias=x.nom.gyro_bias.copy()))

    print(f"  ESKF complete: mean error = {np.mean(errors):.4f} deg")

    return EstimatorResult(
        times=np.array(times),
        attitude_errors_deg=np.array(errors),
        bias_errors=np.array(bias_errors),
        raw_states=states,
    )


def run_smoother(
    sim_data: Dict,
    config_path: str,
    use_robust: bool = True,
    robust_param: float = 0.1,
    lag: float = 60.0,
) -> EstimatorResult:
    """Run Fixed-Lag Smoother on simulation data."""
    print("Running Fixed-Lag Smoother...")

    # Initialize smoother
    smoother = FixedLagAttitudeSmoother(
        config_path=config_path,
        lag=lag,
        use_robust=use_robust,
        normalize_mag=True,
    )

    # Override robust parameter if needed
    smoother.factors.robust_param = robust_param

    # Initialize from first star tracker or perturbed truth
    q_init = Quaternion.from_array(sim_data['q_true'][0])
    q_init = q_init @ Quaternion.from_avec(np.random.randn(3) * 0.1)

    t0 = sim_data['t'][0]
    smoother.initialize(t0, q_init, np.zeros(3))

    times = []
    errors = []
    bias_errors = []
    states = []

    n_steps = len(sim_data['t'])
    dt = sim_data['t'][1] - sim_data['t'][0]

    for k in range(n_steps):
        t = sim_data['t'][k]
        jd = sim_data['jd'][k]

        # Gyro measurement
        omega = sim_data['omega_meas'][k]
        if np.any(np.isnan(omega)):
            if k > 0:
                omega = sim_data['omega_meas'][k-1]
            else:
                omega = np.zeros(3)

        # Integrate gyro
        smoother.integrate_gyro(omega, dt, t=t)

        # Measurement updates
        z_mag = sim_data['mag_meas'][k]
        z_sun = sim_data['sun_meas'][k]
        z_st = sim_data['st_meas'][k]
        B_n = sim_data['b_eci'][k]
        s_n = sim_data['s_eci'][k]

        has_meas = (not np.any(np.isnan(z_mag)) or
                   not np.any(np.isnan(z_sun)) or
                   not np.any(np.isnan(z_st)))

        if has_meas:
            z_mag_arg = z_mag if not np.any(np.isnan(z_mag)) else None
            z_sun_arg = z_sun if not np.any(np.isnan(z_sun)) else None
            z_st_arg = Quaternion.from_array(z_st) if not np.any(np.isnan(z_st)) else None

            smoother.add_measurement(
                t=t, jd=jd,
                z_mag=z_mag_arg, z_sun=z_sun_arg, z_st=z_st_arg,
                B_eci=B_n, s_eci=s_n,
            )

        # Get current state
        state = smoother.get_propagated_state()
        if state is None:
            state = NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3))

        # Compute errors
        q_true = Quaternion.from_array(sim_data['q_true'][k])
        err_deg = compute_attitude_error(state.ori, q_true)
        bias_err = state.gyro_bias - sim_data['b_g_true'][k]

        times.append(t)
        errors.append(err_deg)
        bias_errors.append(bias_err)
        states.append(NominalState(ori=state.ori.copy(), gyro_bias=state.gyro_bias.copy()))

    print(f"  Smoother complete: mean error = {np.mean(errors):.4f} deg")

    return EstimatorResult(
        times=np.array(times),
        attitude_errors_deg=np.array(errors),
        bias_errors=np.array(bias_errors),
        raw_states=states,
    )


def run_redundant(
    sim_data: Dict,
    config_path: str,
    use_robust: bool = True,
    robust_param: float = 0.1,
    smoother_lag: float = 60.0,
    disagreement_threshold_deg: float = 2.0,
    consecutive_to_switch: int = 3,
) -> EstimatorResult:
    """Run Redundant Estimator (ESKF + Smoother with fault detection)."""
    print("Running Redundant Estimator...")

    P0 = np.diag([0.1**2, 0.1**2, 0.1**2, 1e-8, 1e-8, 1e-8])

    redundant = RedundantEstimator(
        P0=P0,
        config_path=config_path,
        smoother_lag=smoother_lag,
        use_robust=use_robust,
        normalize_mag=True,
        disagreement_threshold_deg=disagreement_threshold_deg,
        consecutive_disagreements_to_switch=consecutive_to_switch,
        consecutive_agreements_to_recover=10,
    )

    # Override robust parameter
    redundant.smoother.factors.robust_param = robust_param

    # Initialize from perturbed truth
    q_init = Quaternion.from_array(sim_data['q_true'][0])
    q_init = q_init @ Quaternion.from_avec(np.random.randn(3) * 0.1)

    x = EskfState(
        nom=NominalState(ori=q_init, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy()),
    )

    redundant.initialize(sim_data['t'][0], q_init, np.zeros(3))

    times = []
    errors = []
    bias_errors = []
    states = []
    disagreements = []
    primaries = []

    n_steps = len(sim_data['t'])
    dt = sim_data['t'][1] - sim_data['t'][0]

    for k in range(n_steps):
        t = sim_data['t'][k]
        jd = sim_data['jd'][k]

        # Gyro measurement
        omega = sim_data['omega_meas'][k]
        if np.any(np.isnan(omega)):
            if k > 0:
                omega = sim_data['omega_meas'][k-1]
            else:
                omega = np.zeros(3)

        # Measurement data
        z_mag = sim_data['mag_meas'][k]
        z_sun = sim_data['sun_meas'][k]
        z_st = sim_data['st_meas'][k]
        B_n = sim_data['b_eci'][k]
        s_n = sim_data['s_eci'][k]

        z_mag_arg = z_mag if not np.any(np.isnan(z_mag)) else None
        z_sun_arg = z_sun if not np.any(np.isnan(z_sun)) else None
        z_st_arg = Quaternion.from_array(z_st) if not np.any(np.isnan(z_st)) else None

        # Run step
        x, smoother_state, disagreement_deg, primary_str = redundant.step(
            x_eskf=x,
            t=t, jd=jd,
            omega_meas=omega,
            dt=dt,
            z_mag=z_mag_arg,
            z_sun=z_sun_arg,
            z_st=z_st_arg,
            B_n=B_n,
            s_n=s_n,
        )

        # Get output from primary estimator
        output_state = redundant.get_primary_state(x)

        # Compute errors
        q_true = Quaternion.from_array(sim_data['q_true'][k])
        err_deg = compute_attitude_error(output_state.ori, q_true)
        bias_err = output_state.gyro_bias - sim_data['b_g_true'][k]

        times.append(t)
        errors.append(err_deg)
        bias_errors.append(bias_err)
        states.append(NominalState(ori=output_state.ori.copy(), gyro_bias=output_state.gyro_bias.copy()))
        disagreements.append(disagreement_deg)
        primaries.append(primary_str)

    stats = redundant.get_statistics()
    print(f"  Redundant complete: mean error = {np.mean(errors):.4f} deg")
    print(f"  Switch events: {stats['total_switch_events']}")
    print(f"  Max disagreement: {stats['max_disagreement_deg']:.4f} deg")

    return EstimatorResult(
        times=np.array(times),
        attitude_errors_deg=np.array(errors),
        bias_errors=np.array(bias_errors),
        raw_states=states,
        disagreement_deg=np.array(disagreements),
        primary_estimator=primaries,
        switch_events=stats['switch_events'],
    )


def plot_fault_detection_results(
    scenario: FaultScenario,
    eskf_result: EstimatorResult,
    smoother_result: EstimatorResult,
    redundant_result: EstimatorResult,
    sim_data: Dict,
    output_prefix: str = "fault_detection",
) -> None:
    """Create publication-quality plots for fault detection analysis."""

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 1, height_ratios=[1.5, 1, 1, 0.8], hspace=0.3)

    # Get fault window
    fault_start = scenario.fault_start
    fault_end = scenario.fault_end

    # Colors
    colors = {
        'eskf': '#1f77b4',      # Blue
        'smoother': '#ff7f0e',  # Orange
        'redundant': '#2ca02c', # Green
        'fault': '#d62728',     # Red
    }

    # ====== Plot 1: Attitude Error Comparison ======
    ax1 = fig.add_subplot(gs[0])
    ax1.semilogy(eskf_result.times, eskf_result.attitude_errors_deg,
                 color=colors['eskf'], label='ESKF', alpha=0.8, linewidth=0.8)
    ax1.semilogy(smoother_result.times, smoother_result.attitude_errors_deg,
                 color=colors['smoother'], label='Smoother', alpha=0.8, linewidth=0.8)
    ax1.semilogy(redundant_result.times, redundant_result.attitude_errors_deg,
                 color=colors['redundant'], label='Redundant', alpha=0.8, linewidth=0.8)

    # Shade fault window
    ax1.axvspan(fault_start, fault_end, alpha=0.2, color=colors['fault'], label='Fault Active')

    ax1.set_ylabel('Attitude Error [deg]')
    ax1.set_title(f'Fault Detection Test: {scenario.name}')
    ax1.legend(loc='upper right', ncol=4, framealpha=0.9)
    ax1.set_xlim([0, eskf_result.times[-1]])
    ax1.set_ylim([1e-4, 100])
    ax1.grid(True, alpha=0.3, which='both')

    # ====== Plot 2: Redundant Estimator Disagreement ======
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(redundant_result.times, redundant_result.disagreement_deg,
             color='purple', linewidth=0.8, label='ESKF-Smoother Disagreement')
    ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Switch Threshold')
    ax2.axvspan(fault_start, fault_end, alpha=0.2, color=colors['fault'])

    # Mark switch events
    if redundant_result.switch_events:
        for t_switch, event_type in redundant_result.switch_events:
            ax2.axvline(x=t_switch, color='green', linestyle=':', alpha=0.7)
            ax2.annotate(event_type.split('->')[-1], xy=(t_switch, ax2.get_ylim()[1]*0.9),
                        fontsize=7, ha='center', rotation=90)

    ax2.set_ylabel('Disagreement [deg]')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_ylim([0, max(10, np.percentile(redundant_result.disagreement_deg, 99) * 1.2)])
    ax2.grid(True, alpha=0.3)

    # ====== Plot 3: Primary Estimator Selection ======
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Convert primary strings to numeric for plotting
    primary_map = {'ESKF': 0, 'SMOOTHER': 1, 'CONSERVATIVE': 0.5}
    primary_numeric = [primary_map.get(p, 0) for p in redundant_result.primary_estimator]

    ax3.fill_between(redundant_result.times, 0, primary_numeric,
                     step='post', alpha=0.5, color=colors['redundant'])
    ax3.axvspan(fault_start, fault_end, alpha=0.2, color=colors['fault'])
    ax3.set_ylabel('Primary')
    ax3.set_yticks([0, 0.5, 1])
    ax3.set_yticklabels(['ESKF', 'CONS.', 'SMOOTHER'])
    ax3.set_ylim([-0.1, 1.1])
    ax3.grid(True, alpha=0.3)

    # ====== Plot 4: Fault Indicator ======
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.fill_between(sim_data['t'], 0, sim_data['fault_active'].astype(float),
                     step='post', alpha=0.5, color=colors['fault'], label='Fault Active')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Fault')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Normal', 'Faulty'])
    ax4.set_ylim([-0.1, 1.1])
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fname = f"{output_prefix}_{scenario.sensor}_fault.pdf"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.savefig(fname.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def plot_comparison_summary(
    scenarios: List[FaultScenario],
    results: Dict[str, Dict[str, EstimatorResult]],
    output_prefix: str = "fault_detection_summary",
) -> None:
    """Create a summary comparison plot across all scenarios."""

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    colors = {
        'eskf': '#1f77b4',
        'smoother': '#ff7f0e',
        'redundant': '#2ca02c',
    }

    for i, scenario in enumerate(scenarios):
        scenario_results = results[scenario.name]
        fault_start = scenario.fault_start
        fault_end = scenario.fault_end

        # Column 1: Attitude error during fault
        ax = axes[i, 0]
        t = scenario_results['eskf'].times
        mask = (t >= fault_start - 10) & (t <= fault_end + 30)

        ax.semilogy(t[mask], scenario_results['eskf'].attitude_errors_deg[mask],
                   color=colors['eskf'], label='ESKF', alpha=0.8)
        ax.semilogy(t[mask], scenario_results['smoother'].attitude_errors_deg[mask],
                   color=colors['smoother'], label='Smoother', alpha=0.8)
        ax.semilogy(t[mask], scenario_results['redundant'].attitude_errors_deg[mask],
                   color=colors['redundant'], label='Redundant', alpha=0.8)
        ax.axvspan(fault_start, fault_end, alpha=0.2, color='red')

        if i == 0:
            ax.set_title('Attitude Error During Fault')
            ax.legend(loc='upper right', fontsize=8)
        ax.set_ylabel(f'{scenario.sensor.upper()} Fault\n[deg]')
        if i == 2:
            ax.set_xlabel('Time [s]')
        ax.grid(True, alpha=0.3, which='both')

        # Column 2: Disagreement
        ax = axes[i, 1]
        ax.plot(t[mask], scenario_results['redundant'].disagreement_deg[mask],
               color='purple', alpha=0.8)
        ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.7)
        ax.axvspan(fault_start, fault_end, alpha=0.2, color='red')

        if i == 0:
            ax.set_title('ESKF-Smoother Disagreement')
        ax.set_ylabel('[deg]')
        if i == 2:
            ax.set_xlabel('Time [s]')
        ax.grid(True, alpha=0.3)

        # Column 3: Statistics bar chart
        ax = axes[i, 2]

        # Compute statistics during fault window
        during_fault = (t >= fault_start) & (t <= fault_end)
        after_fault = (t >= fault_end) & (t <= fault_end + 30)

        stats = {
            'During Fault': {
                'ESKF': np.mean(scenario_results['eskf'].attitude_errors_deg[during_fault]),
                'Smoother': np.mean(scenario_results['smoother'].attitude_errors_deg[during_fault]),
                'Redundant': np.mean(scenario_results['redundant'].attitude_errors_deg[during_fault]),
            },
            'Recovery': {
                'ESKF': np.mean(scenario_results['eskf'].attitude_errors_deg[after_fault]),
                'Smoother': np.mean(scenario_results['smoother'].attitude_errors_deg[after_fault]),
                'Redundant': np.mean(scenario_results['redundant'].attitude_errors_deg[after_fault]),
            },
        }

        x = np.arange(2)
        width = 0.25

        ax.bar(x - width, [stats['During Fault']['ESKF'], stats['Recovery']['ESKF']],
               width, label='ESKF', color=colors['eskf'])
        ax.bar(x, [stats['During Fault']['Smoother'], stats['Recovery']['Smoother']],
               width, label='Smoother', color=colors['smoother'])
        ax.bar(x + width, [stats['During Fault']['Redundant'], stats['Recovery']['Redundant']],
               width, label='Redundant', color=colors['redundant'])

        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(['During Fault', 'Recovery'])
        if i == 0:
            ax.set_title('Mean Error [deg]')
            ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    fname = f"{output_prefix}.pdf"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.savefig(fname.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def generate_results_table(
    scenarios: List[FaultScenario],
    results: Dict[str, Dict[str, EstimatorResult]],
    output_file: str = "fault_detection_results.tex",
) -> None:
    """Generate a LaTeX table of results."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Fault Detection Performance Comparison}",
        r"\label{tab:fault_detection}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Fault Type & Period & ESKF & Smoother & Redundant \\",
        r"\midrule",
    ]

    for scenario in scenarios:
        r = results[scenario.name]
        t = r['eskf'].times
        fault_mask = (t >= scenario.fault_start) & (t <= scenario.fault_end)
        normal_mask = t < scenario.fault_start

        # During fault
        eskf_fault = np.mean(r['eskf'].attitude_errors_deg[fault_mask])
        smth_fault = np.mean(r['smoother'].attitude_errors_deg[fault_mask])
        red_fault = np.mean(r['redundant'].attitude_errors_deg[fault_mask])

        # Normal operation
        eskf_norm = np.mean(r['eskf'].attitude_errors_deg[normal_mask])
        smth_norm = np.mean(r['smoother'].attitude_errors_deg[normal_mask])
        red_norm = np.mean(r['redundant'].attitude_errors_deg[normal_mask])

        sensor_name = scenario.sensor.capitalize()
        lines.append(f"{sensor_name} & Normal & {eskf_norm:.4f} & {smth_norm:.4f} & {red_norm:.4f} \\\\")
        lines.append(f" & During Fault & {eskf_fault:.4f} & {smth_fault:.4f} & {red_fault:.4f} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved: {output_file}")


def main():
    """Main entry point for fault detection test."""
    print("=" * 70)
    print("COMPREHENSIVE FAULT DETECTION TEST")
    print("=" * 70)
    print("Testing: ESKF vs Smoother vs Redundant Architecture")
    print("Faults: Sun sensor, Magnetometer, Star tracker spikes")
    print("Config: GTSAM preintegration, Gating, Huber k=0.1")
    print("=" * 70)

    # Base config path
    base_config = "configs/config_fault_detection_test.yaml"

    # Define fault scenarios - severe persistent faults to trigger detection
    scenarios = [
        FaultScenario(
            name="Sun Sensor Fault",
            sensor="sun",
            fault_start=100.0,
            fault_end=150.0,
            spike_probability=1.0,  # 100% of measurements are faulty
            spike_magnitude=2.0,    # Very large spike
        ),
        FaultScenario(
            name="Magnetometer Fault",
            sensor="mag",
            fault_start=100.0,
            fault_end=150.0,
            spike_probability=1.0,  # 100% faulty
            spike_magnitude=2.0,    # Very large spike
        ),
        FaultScenario(
            name="Star Tracker Fault",
            sensor="star",
            fault_start=100.0,
            fault_end=150.0,
            spike_probability=1.0,  # 100% faulty
            spike_magnitude=1.5,    # ~86 degrees - very large error
        ),
    ]

    all_results = {}

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario.name}")
        print(f"{'='*60}")

        # Generate simulation data with fault injection
        sim_data = generate_simulation_data(
            config_path=base_config,
            scenario=scenario,
            T=300.0,
            dt=0.02,
        )

        # Run all three estimators
        # ESKF with chi-squared gating enabled
        eskf_result = run_eskf(
            sim_data, base_config,
            use_gating=True,  # Chi-squared gate active
            gating_threshold=9.21,  # 99% confidence for 3 DOF
        )

        smoother_result = run_smoother(
            sim_data, base_config,
            use_robust=True,
            robust_param=0.1,  # Huber k=0.1
            lag=60.0,
        )

        redundant_result = run_redundant(
            sim_data, base_config,
            use_robust=True,
            robust_param=0.1,  # Huber k=0.1
            smoother_lag=60.0,
            disagreement_threshold_deg=0.5,  # Lower threshold for better detection
            consecutive_to_switch=3,  # Faster switching
        )

        # Store results
        all_results[scenario.name] = {
            'eskf': eskf_result,
            'smoother': smoother_result,
            'redundant': redundant_result,
            'sim_data': sim_data,
        }

        # Create individual scenario plot
        plot_fault_detection_results(
            scenario=scenario,
            eskf_result=eskf_result,
            smoother_result=smoother_result,
            redundant_result=redundant_result,
            sim_data=sim_data,
            output_prefix="fault_detection",
        )

    # Create summary comparison
    plot_comparison_summary(
        scenarios=scenarios,
        results={s.name: all_results[s.name] for s in scenarios},
    )

    # Generate LaTeX table
    generate_results_table(
        scenarios=scenarios,
        results={s.name: all_results[s.name] for s in scenarios},
    )

    print("\n" + "=" * 70)
    print("FAULT DETECTION TEST COMPLETE")
    print("=" * 70)

    # Print summary statistics
    print("\nSUMMARY:")
    for scenario in scenarios:
        r = all_results[scenario.name]
        t = r['eskf'].times
        fault_mask = (t >= scenario.fault_start) & (t <= scenario.fault_end)

        print(f"\n{scenario.name}:")
        print(f"  ESKF mean error during fault:      {np.mean(r['eskf'].attitude_errors_deg[fault_mask]):.4f} deg")
        print(f"  Smoother mean error during fault:  {np.mean(r['smoother'].attitude_errors_deg[fault_mask]):.4f} deg")
        print(f"  Redundant mean error during fault: {np.mean(r['redundant'].attitude_errors_deg[fault_mask]):.4f} deg")

        if r['redundant'].switch_events:
            print(f"  Switch events: {r['redundant'].switch_events}")


if __name__ == "__main__":
    main()
