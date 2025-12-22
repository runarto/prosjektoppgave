"""
Fault Scenario Comparison: Isolate the value of switching.

Compare:
1. ESKF-only
2. Smoother-only
3. Redundant with switching DISABLED
4. Redundant with switching ENABLED

For a magnetometer bias fault scenario.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.size'] = 18
rcParams['axes.titlesize'] = 20
rcParams['axes.labelsize'] = 18
rcParams['legend.fontsize'] = 14
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['lines.linewidth'] = 1.5
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.width'] = 1.2
rcParams['xtick.major.size'] = 6
rcParams['ytick.major.size'] = 6
rcParams['grid.linewidth'] = 0.6
rcParams['legend.framealpha'] = 0.95
rcParams['legend.edgecolor'] = 'gray'
rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

from data.db import SimulationDatabase
from data.classes import SimulationResult
from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from estimation.redundant_estimator import RedundantEstimator, PrimaryEstimator
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss


# =============================================================================
# Centralized Test Parameters - Ensure consistency across all estimators
# =============================================================================
TEST_PARAMS = {
    # Initial covariance
    "P0_attitude_deg": 5.0,  # degrees
    "P0_bias": 1e-6,

    # Smoother parameters
    "smoother_lag": 60.0,  # seconds
    "use_robust": True,
    "robust_kernel": "cauchy",  # cauchy k=0.1 performs best across all scenarios
    "robust_param": 0.1,

    # Redundant estimator switching parameters
    "disagreement_threshold_deg": 2.0,
    "consecutive_to_switch": 5,
    "consecutive_to_recover": 10,

    # Config path
    "config_path": "configs/config_baseline_short.yaml",
}


def load_nominal_data(db_path: str = "simulations.db"):
    """Load a nominal simulation run."""
    db = SimulationDatabase(db_path)
    cur = db.conn.cursor()
    cur.execute("SELECT id, name FROM runs ORDER BY id DESC;")
    runs = cur.fetchall()

    run_id = None
    for rid, name in runs:
        if 'baseline' in name.lower() or 'nominal' in name.lower():
            run_id = rid
            break

    if run_id is None:
        run_id = runs[0][0]

    print(f"Loading run {run_id}")
    return db.load_run(run_id)


def inject_magnetometer_fault(data, fault_start: float, fault_end: float,
                               bias_magnitude: float = 0.3):
    """Inject magnetometer bias fault into data."""
    # Create a copy of the data
    mag_meas_faulty = data.mag_meas.copy()

    # Bias direction (fixed in body frame)
    bias_dir = np.array([1.0, 0.5, 0.3])
    bias_dir = bias_dir / np.linalg.norm(bias_dir)
    bias = bias_magnitude * bias_dir

    fault_mask = (data.t >= fault_start) & (data.t <= fault_end)

    for k in range(len(data.t)):
        if fault_mask[k] and not np.isnan(mag_meas_faulty[k, 0]):
            mag_meas_faulty[k] += bias

    print(f"Injected magnetometer bias: magnitude={bias_magnitude}, "
          f"window=[{fault_start}, {fault_end}]s")

    return mag_meas_faulty, fault_mask


def inject_star_tracker_fault(data, fault_start: float, fault_end: float,
                               error_deg: float = 5.0):
    """Inject star tracker attitude error."""
    st_meas_faulty = data.st_meas.copy()

    # Fixed rotation axis
    axis = np.array([0.5, 0.5, 0.707])
    axis = axis / np.linalg.norm(axis)
    error_rad = np.deg2rad(error_deg)

    q_error = Quaternion.from_avec(axis * error_rad)

    fault_mask = (data.t >= fault_start) & (data.t <= fault_end)

    for k in range(len(data.t)):
        if fault_mask[k] and not np.isnan(st_meas_faulty[k, 0]):
            q_true = Quaternion.from_array(st_meas_faulty[k])
            q_faulty = q_true @ q_error
            st_meas_faulty[k] = q_faulty.as_array()

    print(f"Injected star tracker fault: {error_deg}° error, "
          f"window=[{fault_start}, {fault_end}]s")

    return st_meas_faulty, fault_mask


def inject_eclipse_and_st_dropouts(data, eclipse_start: float, eclipse_end: float,
                                    st_dropout_probability: float = 0.7,
                                    st_dropout_duration: float = 10.0):
    """
    Inject eclipse (sun sensor dropout) and frequent star tracker dropouts.

    Args:
        data: Simulation data
        eclipse_start: Start time of eclipse
        eclipse_end: End time of eclipse
        st_dropout_probability: Probability of star tracker being unavailable
        st_dropout_duration: Average duration of each dropout period
    """
    sun_meas_eclipse = data.sun_meas.copy()
    st_meas_dropout = data.st_meas.copy()

    eclipse_mask = (data.t >= eclipse_start) & (data.t <= eclipse_end)

    # Remove sun sensor during eclipse
    for k in range(len(data.t)):
        if eclipse_mask[k]:
            sun_meas_eclipse[k] = np.nan

    # Create random star tracker dropouts during eclipse
    np.random.seed(42)  # Reproducibility
    in_dropout = False
    dropout_end_time = 0
    st_available_count = 0
    st_dropout_count = 0

    for k in range(len(data.t)):
        if eclipse_mask[k] and not np.isnan(st_meas_dropout[k, 0]):
            if not in_dropout:
                # Randomly start a dropout
                if np.random.random() < st_dropout_probability:
                    in_dropout = True
                    dropout_end_time = data.t[k] + np.random.exponential(st_dropout_duration)

            if in_dropout:
                st_meas_dropout[k] = np.nan
                st_dropout_count += 1
                if data.t[k] >= dropout_end_time:
                    in_dropout = False
            else:
                st_available_count += 1

    print(f"Injected eclipse: window=[{eclipse_start}, {eclipse_end}]s")
    print(f"  Sun sensor: unavailable during eclipse")
    print(f"  Star tracker: {st_available_count} available, {st_dropout_count} dropped")

    return sun_meas_eclipse, st_meas_dropout, eclipse_mask


def compute_attitude_error(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_true.conjugate() @ q_est
    angle = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle)


def run_eskf(data, mag_meas, st_meas, sun_meas=None):
    """Run ESKF with potentially faulty measurements."""
    if sun_meas is None:
        sun_meas = data.sun_meas

    att_var = np.deg2rad(TEST_PARAMS["P0_attitude_deg"])**2
    P0 = np.diag([att_var, att_var, att_var,
                  TEST_PARAMS["P0_bias"], TEST_PARAMS["P0_bias"], TEST_PARAMS["P0_bias"]])

    eskf = ESKF(P0=P0, config_path=TEST_PARAMS["config_path"])

    q_init = Quaternion.from_array(data.q_true[0])
    x = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy())
    )

    errors = []
    nis_values = []
    dt = data.t[1] - data.t[0]

    for k in range(len(data.t)):
        q_true = Quaternion.from_array(data.q_true[k])
        err = compute_attitude_error(x.nom.ori, q_true)
        errors.append(err)

        # Prediction
        if not np.isnan(data.omega_meas[k, 0]):
            x = eskf.predict(x, data.omega_meas[k], dt)

        # Magnetometer update
        if not np.isnan(mag_meas[k, 0]):
            try:
                x = eskf.update(x, mag_meas[k], SensorType.MAGNETOMETER, B_n=data.b_eci[k])
                nis_values.append(eskf.last_nis)
            except:
                pass

        # Sun sensor update
        if not np.isnan(sun_meas[k, 0]):
            try:
                x = eskf.update(x, sun_meas[k], SensorType.SUN_VECTOR, s_n=data.s_eci[k])
            except:
                pass

        # Star tracker update
        if not np.isnan(st_meas[k, 0]):
            try:
                q_st = Quaternion.from_array(st_meas[k])
                x = eskf.update(x, q_st, SensorType.STAR_TRACKER)
            except:
                pass

    return np.array(errors), np.array(nis_values)


def run_smoother(data, mag_meas, st_meas, sun_meas=None):
    """Run fixed-lag smoother with potentially faulty measurements.

    Uses interpolation between keyframes to get state at all timesteps.
    """
    if sun_meas is None:
        sun_meas = data.sun_meas

    smoother = FixedLagAttitudeSmoother(
        config_path=TEST_PARAMS["config_path"],
        lag=TEST_PARAMS["smoother_lag"],
        use_robust=TEST_PARAMS["use_robust"],
        robust_kernel=TEST_PARAMS["robust_kernel"],
        robust_param=TEST_PARAMS["robust_param"],
        normalize_mag=True,
    )

    q_init = Quaternion.from_array(data.q_true[0])
    smoother.initialize(data.t[0], q_init.copy(), np.zeros(3))

    # Store keyframe times and states for interpolation
    keyframe_times = [data.t[0]]
    keyframe_states = [q_init.copy()]

    dt = data.t[1] - data.t[0]

    for k in range(len(data.t)):
        if not np.isnan(data.omega_meas[k, 0]):
            smoother.integrate_gyro(data.omega_meas[k], dt, t=data.t[k])

        z_mag = mag_meas[k] if not np.isnan(mag_meas[k, 0]) else None
        z_sun = sun_meas[k] if not np.isnan(sun_meas[k, 0]) else None
        z_st = Quaternion.from_array(st_meas[k]) if not np.isnan(st_meas[k, 0]) else None

        if z_mag is not None or z_sun is not None or z_st is not None:
            smoother.add_measurement(
                t=data.t[k],
                jd=data.jd[k],
                z_mag=z_mag,
                z_sun=z_sun,
                z_st=z_st,
                B_eci=data.b_eci[k],
                s_eci=data.s_eci[k],
            )
            # Store keyframe
            state = smoother.get_state()
            if state is not None:
                keyframe_times.append(data.t[k])
                keyframe_states.append(state.ori.copy())

    # Interpolate states at all timesteps
    keyframe_times = np.array(keyframe_times)
    errors = []

    for k in range(len(data.t)):
        t_k = data.t[k]
        q_true = Quaternion.from_array(data.q_true[k])

        # Find surrounding keyframes
        idx = np.searchsorted(keyframe_times, t_k)

        if idx == 0:
            # Before first keyframe
            q_interp = keyframe_states[0]
        elif idx >= len(keyframe_times):
            # After last keyframe
            q_interp = keyframe_states[-1]
        else:
            # Interpolate between keyframes using SLERP
            t0 = keyframe_times[idx - 1]
            t1 = keyframe_times[idx]
            q0 = keyframe_states[idx - 1]
            q1 = keyframe_states[idx]

            if t1 > t0:
                alpha = (t_k - t0) / (t1 - t0)
                q_interp = q0.slerp(q1, alpha)
            else:
                q_interp = q0

        err = compute_attitude_error(q_interp, q_true)
        errors.append(err)

    return np.array(errors)


def run_redundant(data, mag_meas, st_meas, enable_switching: bool = True, sun_meas=None):
    """Run redundant estimator with or without switching."""
    if sun_meas is None:
        sun_meas = data.sun_meas

    att_var = np.deg2rad(TEST_PARAMS["P0_attitude_deg"])**2
    P0 = np.diag([att_var, att_var, att_var,
                  TEST_PARAMS["P0_bias"], TEST_PARAMS["P0_bias"], TEST_PARAMS["P0_bias"]])

    redundant = RedundantEstimator(
        P0=P0,
        config_path=TEST_PARAMS["config_path"],
        smoother_lag=TEST_PARAMS["smoother_lag"],
        use_robust=TEST_PARAMS["use_robust"],
        robust_kernel=TEST_PARAMS["robust_kernel"],
        robust_param=TEST_PARAMS["robust_param"],
        disagreement_threshold_deg=TEST_PARAMS["disagreement_threshold_deg"],
        consecutive_disagreements_to_switch=TEST_PARAMS["consecutive_to_switch"],
    )

    q_init = Quaternion.from_array(data.q_true[0])
    x_eskf = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy())
    )

    redundant.initialize(data.t[0], q_init.copy(), np.zeros(3))

    errors_output = []
    disagreements = []
    modes = []
    dt = data.t[1] - data.t[0]

    for k in range(len(data.t)):
        q_true = Quaternion.from_array(data.q_true[k])

        z_mag = mag_meas[k] if not np.isnan(mag_meas[k, 0]) else None
        z_sun = sun_meas[k] if not np.isnan(sun_meas[k, 0]) else None
        z_st = Quaternion.from_array(st_meas[k]) if not np.isnan(st_meas[k, 0]) else None

        x_eskf, smoother_state, disagreement, primary = redundant.step(
            x_eskf=x_eskf,
            t=data.t[k],
            jd=data.jd[k],
            omega_meas=data.omega_meas[k],
            dt=dt,
            z_mag=z_mag,
            z_sun=z_sun,
            z_st=z_st,
            B_n=data.b_eci[k],
            s_n=data.s_eci[k],
        )

        # If switching disabled, always use ESKF output
        if not enable_switching:
            output_state = x_eskf.nom
            primary = "ESKF (no switch)"
        else:
            output_state = redundant.get_primary_state(x_eskf)

        err_output = compute_attitude_error(output_state.ori, q_true)
        errors_output.append(err_output)
        disagreements.append(disagreement)
        modes.append(primary)

    return np.array(errors_output), np.array(disagreements), modes


def plot_fault_comparison(data, fault_mask, fault_type: str,
                          errors_eskf, errors_smoother,
                          errors_redundant_no_switch, errors_redundant_switch,
                          disagreements, modes,
                          output_dir: Path):
    """Plot comprehensive fault scenario comparison."""
    t = data.t

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Panel 1: Attitude errors
    ax = axes[0]
    ax.semilogy(t, errors_eskf, 'C0-', label='ESKF only', linewidth=1.5)
    ax.semilogy(t, errors_smoother, 'C1-', label='Smoother only', linewidth=1.5)
    ax.semilogy(t, errors_redundant_no_switch, 'C2--', label='Redundant (no switching)', linewidth=1.5)
    ax.semilogy(t, errors_redundant_switch, 'C3-', label='Redundant (with switching)', linewidth=2)

    # Shade fault period
    fault_start = t[fault_mask][0] if np.any(fault_mask) else 0
    fault_end = t[fault_mask][-1] if np.any(fault_mask) else 0
    ax.axvspan(fault_start, fault_end, alpha=0.2, color='red', label='Fault active')

    ax.set_ylabel('Attitude error [deg]')
    ax.set_title(f'Fault Scenario: {fault_type}')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([1e-3, 50])

    # Panel 2: Disagreement
    ax = axes[1]
    ax.plot(t, disagreements, 'k-', linewidth=1)
    ax.axhline(2.0, color='r', linestyle='--', label='Switch threshold (2°)')
    ax.axvspan(fault_start, fault_end, alpha=0.2, color='red')
    ax.set_ylabel('Disagreement [deg]')
    ax.set_title('ESKF-Smoother Disagreement')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 3: Mode indicator
    ax = axes[2]
    mode_numeric = []
    for m in modes:
        if 'SMOOTHER' in m:
            mode_numeric.append(2)
        elif 'CONSERVATIVE' in m:
            mode_numeric.append(1)
        else:
            mode_numeric.append(0)

    ax.fill_between(t, 0, np.array(mode_numeric), step='mid', alpha=0.7)
    ax.axvspan(fault_start, fault_end, alpha=0.2, color='red')
    ax.set_ylabel('Mode')
    ax.set_xlabel('Time [s]')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['ESKF', 'CONSERVATIVE', 'SMOOTHER'])
    ax.set_title('Redundant Estimator Mode (with switching)')
    ax.set_ylim([-0.1, 2.5])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'fault_comparison_{fault_type.lower().replace(" ", "_")}.{ext}')
    plt.close(fig)
    print(f"Saved fault_comparison_{fault_type.lower().replace(' ', '_')}.pdf/png")


def plot_steady_state_comparison(data, fault_mask, fault_type: str,
                                  errors_eskf, errors_smoother,
                                  errors_redundant_no_switch, errors_redundant_switch,
                                  output_dir: Path):
    """Compare steady-state performance during and after fault."""
    t = data.t

    # Define periods
    fault_start = t[fault_mask][0] if np.any(fault_mask) else 100
    fault_end = t[fault_mask][-1] if np.any(fault_mask) else 200

    # During fault
    during_fault = fault_mask
    # After fault (recovery period)
    after_fault = (t > fault_end) & (t <= fault_end + 50)
    # Long after fault (steady state)
    steady_state = t > fault_end + 50

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    periods = [
        ('During Fault', during_fault),
        ('Recovery (0-50s after)', after_fault),
        ('Steady State (>50s after)', steady_state),
    ]

    for idx, (period_name, mask) in enumerate(periods):
        ax = axes[idx]

        if not np.any(mask):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        data_box = [
            errors_eskf[mask],
            errors_smoother[mask],
            errors_redundant_no_switch[mask],
            errors_redundant_switch[mask],
        ]

        bp = ax.boxplot(data_box, tick_labels=['ESKF', 'Smoother', 'Red.\n(no sw)', 'Red.\n(sw)'],
                        patch_artist=True)
        colors = ['C0', 'C1', 'C2', 'C3']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Attitude error [deg]')
        ax.set_title(period_name)
        ax.grid(True, alpha=0.3)

        # Add RMS values as text
        for i, errs in enumerate(data_box):
            rms = np.sqrt(np.mean(errs**2))
            ax.text(i+1, ax.get_ylim()[1]*0.95, f'{rms:.3f}°',
                    ha='center', va='top', fontsize=10)

    plt.suptitle(f'Performance Comparison: {fault_type}', fontsize=16, y=1.02)
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'fault_boxplot_{fault_type.lower().replace(" ", "_")}.{ext}')
    plt.close(fig)
    print(f"Saved fault_boxplot_{fault_type.lower().replace(' ', '_')}.pdf/png")


def print_statistics(data, fault_mask, fault_type: str,
                     errors_eskf, errors_smoother, errors_redundant):
    """Print performance statistics for 3 methods."""
    t = data.t
    fault_end = t[fault_mask][-1] if np.any(fault_mask) else 200

    print(f"\n{'='*70}")
    print(f"Performance Statistics: {fault_type}")
    print(f"{'='*70}")

    periods = [
        ('During Fault', fault_mask),
        ('After Fault (t > fault_end)', t > fault_end),
    ]

    methods = [
        ('ESKF', errors_eskf),
        ('Smoother', errors_smoother),
        ('Redundant', errors_redundant),
    ]

    for period_name, mask in periods:
        if not np.any(mask):
            continue
        print(f"\n{period_name}:")
        print(f"{'Method':<20} {'Mean':>10} {'RMS':>10} {'Max':>10}")
        print("-" * 50)

        for method_name, errors in methods:
            errs = errors[mask]
            print(f"{method_name:<20} {np.mean(errs):>10.4f} "
                  f"{np.sqrt(np.mean(errs**2)):>10.4f} {np.max(errs):>10.4f}")


def generate_latex_table(results: dict, output_path: Path):
    """Generate a LaTeX table comparing methods across fault scenarios."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Fault scenario comparison: attitude error (degrees)}",
        r"\label{tab:fault_scenario_comparison}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Scenario & Period & ESKF & Smoother & Redundant \\",
        r"\midrule",
    ]

    for scenario_name, scenario_data in results.items():
        for i, (period_name, stats) in enumerate(scenario_data.items()):
            if i == 0:
                scenario_col = scenario_name
            else:
                scenario_col = ""

            eskf_rms = stats['ESKF']['rms']
            smoother_rms = stats['Smoother']['rms']
            redundant_rms = stats['Redundant']['rms']

            # Bold the best (lowest) value
            values = [eskf_rms, smoother_rms, redundant_rms]
            min_val = min(values)

            def fmt(v):
                if v == min_val:
                    return f"\\textbf{{{v:.3f}}}"
                return f"{v:.3f}"

            lines.append(f"{scenario_col} & {period_name} & {fmt(eskf_rms)} & {fmt(smoother_rms)} & {fmt(redundant_rms)} \\\\")

        lines.append(r"\midrule")

    # Remove last midrule and add bottomrule
    lines[-1] = r"\bottomrule"
    lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved LaTeX table to {output_path}")


def compute_statistics(errors, mask):
    """Compute statistics for errors over a mask."""
    errs = errors[mask]
    return {
        'mean': np.mean(errs),
        'rms': np.sqrt(np.mean(errs**2)),
        'max': np.max(errs),
    }


def plot_fault_comparison_simple(data, fault_mask, fault_type: str,
                                  errors_eskf, errors_smoother, errors_redundant,
                                  disagreements, modes, output_dir: Path):
    """Plot fault scenario comparison for 3 methods."""
    t = data.t

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Panel 1: Attitude errors
    ax = axes[0]
    ax.semilogy(t, errors_eskf, 'C0-', label='ESKF', linewidth=1.5)
    ax.semilogy(t, errors_smoother, 'C1-', label='Smoother', linewidth=1.5)
    ax.semilogy(t, errors_redundant, 'C3-', label='Redundant', linewidth=2)

    # Shade fault period
    fault_start = t[fault_mask][0] if np.any(fault_mask) else 0
    fault_end = t[fault_mask][-1] if np.any(fault_mask) else 0
    ax.axvspan(fault_start, fault_end, alpha=0.2, color='red', label='Fault active')

    ax.set_ylabel('Attitude error [deg]')
    ax.set_title(f'Fault Scenario: {fault_type}')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([1e-3, 50])

    # Panel 2: Disagreement
    ax = axes[1]
    ax.plot(t, disagreements, 'k-', linewidth=1)
    ax.axhline(2.0, color='r', linestyle='--', label='Switch threshold (2°)')
    ax.axvspan(fault_start, fault_end, alpha=0.2, color='red')
    ax.set_ylabel('Disagreement [deg]')
    ax.set_title('ESKF-Smoother Disagreement')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 3: Mode indicator
    ax = axes[2]
    mode_numeric = []
    for m in modes:
        if 'SMOOTHER' in str(m):
            mode_numeric.append(2)
        elif 'CONSERVATIVE' in str(m):
            mode_numeric.append(1)
        else:
            mode_numeric.append(0)

    ax.fill_between(t, 0, np.array(mode_numeric), step='mid', alpha=0.7)
    ax.axvspan(fault_start, fault_end, alpha=0.2, color='red')
    ax.set_ylabel('Mode')
    ax.set_xlabel('Time [s]')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['ESKF', 'CONSERVATIVE', 'SMOOTHER'])
    ax.set_title('Redundant Estimator Mode')
    ax.set_ylim([-0.1, 2.5])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'fault_comparison_{fault_type.lower().replace(" ", "_")}.{ext}')
    plt.close(fig)
    print(f"Saved fault_comparison_{fault_type.lower().replace(' ', '_')}.pdf/png")


def main():
    output_dir = Path(__file__).parent.parent / "figures"
    output_dir.mkdir(exist_ok=True)

    # Print test parameters for verification
    print("="*70)
    print("TEST PARAMETERS:")
    for k, v in TEST_PARAMS.items():
        print(f"  {k}: {v}")
    print("="*70)

    print("\nLoading simulation data...")
    data = load_nominal_data()

    # Results dictionary for LaTeX table
    all_results = {}

    # ===== Magnetometer Bias Fault =====
    print("\n" + "="*70)
    print("SCENARIO 1: Magnetometer Bias Fault")
    print("="*70)

    mag_faulty, mag_fault_mask = inject_magnetometer_fault(
        data, fault_start=100.0, fault_end=200.0, bias_magnitude=0.62
    )

    print("\nRunning ESKF...")
    errors_eskf_mag, _ = run_eskf(data, mag_faulty, data.st_meas)

    print("Running Smoother...")
    errors_smoother_mag = run_smoother(data, mag_faulty, data.st_meas)

    print("Running Redundant...")
    errors_redundant_mag, disagree_mag, modes_mag = run_redundant(
        data, mag_faulty, data.st_meas, enable_switching=True
    )

    plot_fault_comparison_simple(
        data, mag_fault_mask, "Magnetometer Bias",
        errors_eskf_mag, errors_smoother_mag, errors_redundant_mag,
        disagree_mag, modes_mag, output_dir
    )

    print_statistics(
        data, mag_fault_mask, "Magnetometer Bias",
        errors_eskf_mag, errors_smoother_mag, errors_redundant_mag
    )

    # Collect statistics for table
    fault_end_mag = data.t[mag_fault_mask][-1]
    all_results['Mag. bias'] = {
        'During fault': {
            'ESKF': compute_statistics(errors_eskf_mag, mag_fault_mask),
            'Smoother': compute_statistics(errors_smoother_mag, mag_fault_mask),
            'Redundant': compute_statistics(errors_redundant_mag, mag_fault_mask),
        },
        'After fault': {
            'ESKF': compute_statistics(errors_eskf_mag, data.t > fault_end_mag),
            'Smoother': compute_statistics(errors_smoother_mag, data.t > fault_end_mag),
            'Redundant': compute_statistics(errors_redundant_mag, data.t > fault_end_mag),
        },
    }

    # ===== Star Tracker Fault =====
    print("\n" + "="*70)
    print("SCENARIO 2: Star Tracker Fault")
    print("="*70)

    st_faulty, st_fault_mask = inject_star_tracker_fault(
        data, fault_start=100.0, fault_end=150.0, error_deg=5.0
    )

    print("\nRunning ESKF...")
    errors_eskf_st, _ = run_eskf(data, data.mag_meas, st_faulty)

    print("Running Smoother...")
    errors_smoother_st = run_smoother(data, data.mag_meas, st_faulty)

    print("Running Redundant...")
    errors_redundant_st, disagree_st, modes_st = run_redundant(
        data, data.mag_meas, st_faulty, enable_switching=True
    )

    plot_fault_comparison_simple(
        data, st_fault_mask, "Star Tracker Fault",
        errors_eskf_st, errors_smoother_st, errors_redundant_st,
        disagree_st, modes_st, output_dir
    )

    print_statistics(
        data, st_fault_mask, "Star Tracker Fault",
        errors_eskf_st, errors_smoother_st, errors_redundant_st
    )

    # Collect statistics for table
    fault_end_st = data.t[st_fault_mask][-1]
    all_results['ST fault'] = {
        'During fault': {
            'ESKF': compute_statistics(errors_eskf_st, st_fault_mask),
            'Smoother': compute_statistics(errors_smoother_st, st_fault_mask),
            'Redundant': compute_statistics(errors_redundant_st, st_fault_mask),
        },
        'After fault': {
            'ESKF': compute_statistics(errors_eskf_st, data.t > fault_end_st),
            'Smoother': compute_statistics(errors_smoother_st, data.t > fault_end_st),
            'Redundant': compute_statistics(errors_redundant_st, data.t > fault_end_st),
        },
    }

    # ===== Eclipse (sun sensor dropout only) =====
    print("\n" + "="*70)
    print("SCENARIO 3: Eclipse (Sun Sensor Dropout)")
    print("="*70)

    # Eclipse only - no star tracker dropouts
    sun_eclipse = data.sun_meas.copy()
    eclipse_mask = (data.t >= 100.0) & (data.t <= 200.0)
    for k in range(len(data.t)):
        if eclipse_mask[k]:
            sun_eclipse[k] = np.nan
    print(f"Injected eclipse: window=[100.0, 200.0]s (sun sensor unavailable)")

    print("\nRunning ESKF...")
    errors_eskf_ecl, _ = run_eskf(data, data.mag_meas, data.st_meas, sun_meas=sun_eclipse)

    print("Running Smoother...")
    errors_smoother_ecl = run_smoother(data, data.mag_meas, data.st_meas, sun_meas=sun_eclipse)

    print("Running Redundant...")
    errors_redundant_ecl, disagree_ecl, modes_ecl = run_redundant(
        data, data.mag_meas, data.st_meas, enable_switching=True, sun_meas=sun_eclipse
    )

    plot_fault_comparison_simple(
        data, eclipse_mask, "Eclipse",
        errors_eskf_ecl, errors_smoother_ecl, errors_redundant_ecl,
        disagree_ecl, modes_ecl, output_dir
    )

    print_statistics(
        data, eclipse_mask, "Eclipse",
        errors_eskf_ecl, errors_smoother_ecl, errors_redundant_ecl
    )

    # Collect statistics for table
    eclipse_end_t = data.t[eclipse_mask][-1]
    all_results['Eclipse'] = {
        'During eclipse': {
            'ESKF': compute_statistics(errors_eskf_ecl, eclipse_mask),
            'Smoother': compute_statistics(errors_smoother_ecl, eclipse_mask),
            'Redundant': compute_statistics(errors_redundant_ecl, eclipse_mask),
        },
        'After eclipse': {
            'ESKF': compute_statistics(errors_eskf_ecl, data.t > eclipse_end_t),
            'Smoother': compute_statistics(errors_smoother_ecl, data.t > eclipse_end_t),
            'Redundant': compute_statistics(errors_redundant_ecl, data.t > eclipse_end_t),
        },
    }

    # Generate LaTeX table
    generate_latex_table(all_results, output_dir.parent / "fault_scenario_comparison.tex")

    print("\n" + "="*70)
    print("All plots and table saved!")
    print("="*70)


if __name__ == "__main__":
    main()
