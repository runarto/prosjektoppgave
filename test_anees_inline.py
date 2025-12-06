#!/usr/bin/env python3
"""
Inline ANEES test script - the exact code that produces ANEES ≈ 6.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from estimation.eskf import ESKF


def main():
    db = SimulationDatabase('simulations.db')
    sim = db.load_run(1)  # Use latest run with STIM300 parameters

    # Check current R values being used
    P0 = np.diag([0.001, 0.001, 0.001, 1e-4, 1e-4, 1e-4])
    eskf = ESKF(P0=P0, config_path='configs/config_baseline_short.yaml', chi2_threshold=1e10)

    print('Current R values in filter:')
    print(f'  Mag R:  {eskf.sens_mag.R[0,0]:.2e}')
    print(f'  Sun R:  {eskf.sv.R[0,0]:.2e}')
    print(f'  Star R: {eskf.st.R[0,0]:.2e}')
    print()

    # Initial state with some error
    q0_true = Quaternion.from_array(sim.q_true[0])
    q0_est = Quaternion.from_avec(np.array([0.01, 0.01, 0.01])).multiply(q0_true).normalize()

    nom0 = NominalState(ori=q0_est, gyro_bias=np.zeros(3))
    err0 = MultiVarGauss(np.zeros(6), P0)
    x_est = EskfState(nom=nom0, err=err0)

    nees_vals = []
    N = len(sim.t)

    print(f'Running ESKF on {N} samples...')

    for k in range(1, N):
        omega_k = sim.omega_meas[k]
        dt_k = sim.t[k] - sim.t[k-1]
        x_est = eskf.predict(x_est, omega_k, dt_k)

        # Magnetometer update
        mag_meas = sim.mag_meas[k]
        if not np.any(np.isnan(mag_meas)):
            try:
                x_est = eskf.update(x_est, mag_meas, SensorType.MAGNETOMETER, B_n=sim.b_eci[k])
            except:
                pass

        # Sun sensor update
        sun_meas = sim.sun_meas[k]
        if not np.any(np.isnan(sun_meas)):
            try:
                x_est = eskf.update(x_est, sun_meas, SensorType.SUN_VECTOR, s_n=sim.s_eci[k])
            except:
                pass

        # Star tracker update
        st_meas = sim.st_meas[k]
        if not np.any(np.isnan(st_meas)):
            try:
                q_meas = Quaternion.from_array(st_meas)
                x_est = eskf.update(x_est, q_meas, SensorType.STAR_TRACKER)
            except:
                pass

        # Compute NEES
        # The filter injection is: q_new = δq ⊗ q_nom (left multiply)
        # So the error is: δq = q_true ⊗ q_est^{-1}
        q_true = Quaternion.from_array(sim.q_true[k])
        q_err = q_true.multiply(x_est.nom.ori.conjugate()).normalize()

        # Use abs(mu) to handle quaternion double-cover
        theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
        if theta < 1e-8:
            att_err = np.zeros(3)
        else:
            att_err = theta * q_err.eta / np.sin(theta / 2.0)

        bias_err = x_est.nom.gyro_bias - sim.b_g_true[k]
        delta_x = np.concatenate([att_err, bias_err])

        try:
            nees = delta_x.T @ np.linalg.solve(x_est.err.cov, delta_x)
        except:
            nees = np.inf
        nees_vals.append(nees)

    # Final attitude error
    q_true_final = Quaternion.from_array(sim.q_true[N-1])
    q_err_final = q_true_final.multiply(x_est.nom.ori.conjugate()).normalize()
    final_att_err = np.rad2deg(2.0 * np.arccos(np.clip(abs(q_err_final.mu), 0.0, 1.0)))

    nees_arr = np.array(nees_vals)
    n = 6
    lower_bound = chi2.ppf(0.25, n)
    upper_bound = chi2.ppf(0.75, n)

    print()
    print('='*60)
    print('RESULTS - FULL DATASET')
    print('='*60)
    print(f'  Mean NEES:  {np.mean(nees_arr):.2f} (expected: 6)')
    print(f'  Median NEES: {np.median(nees_arr):.2f} (expected: {chi2.ppf(0.5, n):.2f})')
    print(f'  Final attitude error: {final_att_err:.4f} deg')
    print()

    below = np.sum(nees_arr < lower_bound) / len(nees_arr) * 100
    within = np.sum((nees_arr >= lower_bound) & (nees_arr <= upper_bound)) / len(nees_arr) * 100
    above = np.sum(nees_arr > upper_bound) / len(nees_arr) * 100
    print('NEES distribution (full):')
    print(f'  Below 25th:  {below:.1f}% (expected: 25%)')
    print(f'  Within:      {within:.1f}% (expected: 50%)')
    print(f'  Above 75th:  {above:.1f}% (expected: 25%)')

    # Steady-state analysis (skip first 1000 samples = 20 seconds)
    ss_nees = nees_arr[1000:]
    print()
    print('='*60)
    print('RESULTS - STEADY STATE (after 20s)')
    print('='*60)
    print(f'  Mean NEES:  {np.mean(ss_nees):.2f} (expected: 6)')
    print(f'  Median NEES: {np.median(ss_nees):.2f} (expected: {chi2.ppf(0.5, n):.2f})')

    below_ss = np.sum(ss_nees < lower_bound) / len(ss_nees) * 100
    within_ss = np.sum((ss_nees >= lower_bound) & (ss_nees <= upper_bound)) / len(ss_nees) * 100
    above_ss = np.sum(ss_nees > upper_bound) / len(ss_nees) * 100
    print('NEES distribution (steady state):')
    print(f'  Below 25th:  {below_ss:.1f}% (expected: 25%)')
    print(f'  Within:      {within_ss:.1f}% (expected: 50%)')
    print(f'  Above 75th:  {above_ss:.1f}% (expected: 25%)')

    # Plot NEES with chi-squared bounds (steady-state only)
    plot_nees(sim.t[1001:], nees_vals[1000:])


def plot_nees(t: np.ndarray, nees_vals: list):
    """Plot NEES with chi-squared percentile bounds."""
    nees = np.array(nees_vals)
    n = 6  # State dimension

    # Chi-squared bounds for 25th and 75th percentiles
    lower_bound = chi2.ppf(0.25, n)  # 25th percentile
    upper_bound = chi2.ppf(0.75, n)  # 75th percentile

    # Calculate percentages
    below = np.sum(nees < lower_bound) / len(nees) * 100
    within = np.sum((nees >= lower_bound) & (nees <= upper_bound)) / len(nees) * 100
    above = np.sum(nees > upper_bound) / len(nees) * 100

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t, nees, alpha=0.7, linewidth=0.5, label='NEES')
    ax.axhline(lower_bound, color='C2', linestyle='--', linewidth=1.5,
               label=f'25th percentile ({lower_bound:.2f})')
    ax.axhline(upper_bound, color='C3', linestyle='--', linewidth=1.5,
               label=f'75th percentile ({upper_bound:.2f})')
    ax.axhline(n, color='C1', linestyle='-', linewidth=1.5, alpha=0.7,
               label=f'Expected (n={n})')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('NEES')
    ax.set_title('Normalized Estimation Error Squared (NEES) - Steady State')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Add text box with percentages
    textstr = f'Below 25th: {below:.1f}%\nWithin 25-75th: {within:.1f}%\nAbove 75th: {above:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('nees_inline.png', dpi=300)
    plt.savefig('nees_inline.pdf')
    print('\nPlot saved to nees_inline.png and nees_inline.pdf')
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()

    print()
    print('Chi-squared bounds (n=6):')
    print(f'  25th percentile: {lower_bound:.2f}')
    print(f'  75th percentile: {upper_bound:.2f}')
    print()
    print('NEES distribution:')
    print(f'  Below 25th percentile: {below:.1f}%')
    print(f'  Within 25-75th percentile: {within:.1f}%')
    print(f'  Above 75th percentile: {above:.1f}%')


if __name__ == "__main__":
    main()
