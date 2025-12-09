"""
Hybrid Estimator: ESKF Frontend + Keyframe FGO Backend

This module implements a dual-mode estimation architecture:
1. ESKF runs continuously for real-time state estimates
2. Keyframe FGO with preintegration runs periodically for refined estimates
3. FGO results are fed back to ESKF

Uses the KeyframeFGO with RK4 preintegration for the backend.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from collections import deque

import numpy as np
import copy

from estimation.eskf import ESKF
from estimation.keyframe_fgo import KeyframeFGO, quat_from_rot3
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from environment.environment import OrbitEnvironmentModel
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class WindowSample:
    """A single sample stored in the sliding window."""
    t: float
    jd: float
    omega_meas: np.ndarray
    z_mag: Optional[np.ndarray] = None
    z_sun: Optional[np.ndarray] = None
    z_st: Optional[Quaternion] = None
    B_eci: Optional[np.ndarray] = None
    s_eci: Optional[np.ndarray] = None


@dataclass
class WindowData:
    """Simulation-like data structure for FGO processing."""
    t: np.ndarray
    jd: np.ndarray
    omega_meas: np.ndarray
    mag_meas: np.ndarray
    sun_meas: np.ndarray
    st_meas: np.ndarray
    b_eci: np.ndarray
    s_eci: np.ndarray
    q_true: np.ndarray  # Not actually true - used for initialization


class HybridEstimator:
    """
    Hybrid estimator combining ESKF (frontend) and KeyframeFGO (backend).

    Architecture:
    - ESKF runs continuously, providing real-time state estimates
    - Measurements stored in sliding window
    - KeyframeFGO optimizes over window periodically with preintegration
    - Optimized state fed back to ESKF
    """

    def __init__(
        self,
        P0: np.ndarray,
        config_path: str = "configs/config_baseline_short.yaml",
        fgo_window_size: int = 500,  # ~10 seconds at 50 Hz
        fgo_optimize_interval: float = 10.0,  # seconds
        use_robust: bool = True,
        use_isam2: bool = False,
    ):
        """
        Initialize hybrid estimator.

        Args:
            P0: Initial error covariance (6x6)
            config_path: Path to configuration file
            fgo_window_size: Size of FGO sliding window (samples)
            fgo_optimize_interval: Time between FGO optimizations (seconds)
            use_robust: Use M-estimator in FGO
            use_isam2: Use iSAM2 instead of batch optimization
        """
        self.config_path = config_path
        self.fgo_window_size = fgo_window_size
        self.fgo_optimize_interval = fgo_optimize_interval
        self.P0 = P0

        # Initialize ESKF
        self.eskf = ESKF(P0=P0, config_path=config_path)

        # Initialize KeyframeFGO
        self.fgo = KeyframeFGO(
            config_path=config_path,
            use_robust=use_robust,
            use_isam2=use_isam2,
            use_rk4=True,
        )

        # Sliding window for samples
        self.window: deque[WindowSample] = deque(maxlen=fgo_window_size)

        # Environment model
        self.env = OrbitEnvironmentModel()

        # State
        self.last_fgo_time = 0.0
        self.fgo_count = 0

        logger.info(f"HybridEstimator initialized:")
        logger.info(f"  FGO window: {fgo_window_size} samples")
        logger.info(f"  FGO interval: {fgo_optimize_interval}s")
        logger.info(f"  Use robust: {use_robust}")
        logger.info(f"  Use iSAM2: {use_isam2}")

    def _window_to_sim_data(self) -> WindowData:
        """Convert window samples to simulation-like data structure for FGO."""
        n = len(self.window)

        t = np.array([s.t for s in self.window])
        jd = np.array([s.jd for s in self.window])
        omega_meas = np.array([s.omega_meas for s in self.window])

        mag_meas = np.full((n, 3), np.nan)
        sun_meas = np.full((n, 3), np.nan)
        st_meas = np.full((n, 4), np.nan)
        b_eci = np.zeros((n, 3))
        s_eci = np.zeros((n, 3))

        for i, s in enumerate(self.window):
            if s.z_mag is not None:
                mag_meas[i] = s.z_mag
            if s.z_sun is not None:
                sun_meas[i] = s.z_sun
            if s.z_st is not None:
                st_meas[i] = s.z_st.as_array()
            if s.B_eci is not None:
                b_eci[i] = s.B_eci
            if s.s_eci is not None:
                s_eci[i] = s.s_eci

        # Use identity quaternion as placeholder for q_true (FGO will use its own initialization)
        q_true = np.zeros((n, 4))
        q_true[:, 0] = 1.0  # Identity quaternion

        return WindowData(
            t=t,
            jd=jd,
            omega_meas=omega_meas,
            mag_meas=mag_meas,
            sun_meas=sun_meas,
            st_meas=st_meas,
            b_eci=b_eci,
            s_eci=s_eci,
            q_true=q_true,
        )

    def should_optimize_fgo(self, current_time: float) -> bool:
        """Check if it's time to run FGO optimization."""
        if len(self.window) < 50:  # Need minimum samples
            return False

        time_since_last = current_time - self.last_fgo_time
        return time_since_last >= self.fgo_optimize_interval

    def feedback_fgo_to_eskf(
        self,
        x_eskf: EskfState,
        optimized_state: NominalState,
    ) -> EskfState:
        """
        Feed optimized state from FGO back to ESKF.

        Args:
            x_eskf: Current ESKF state
            optimized_state: Optimized state from FGO

        Returns:
            Updated ESKF state
        """
        # Update nominal state
        x_eskf.nom.ori = optimized_state.ori.copy()
        x_eskf.nom.gyro_bias = optimized_state.gyro_bias.copy()

        # Reduce covariance after FGO optimization
        # FGO should have reduced uncertainty
        x_eskf.err.cov = 0.5 * x_eskf.err.cov
        x_eskf.err.cov = 0.5 * (x_eskf.err.cov + x_eskf.err.cov.T)  # Ensure symmetry

        # Reset error mean
        x_eskf.err.mean[:] = 0.0

        return x_eskf

    def step(
        self,
        x_eskf: EskfState,
        t: float,
        jd: float,
        omega_meas: np.ndarray,
        dt: float,
        z_mag: Optional[np.ndarray] = None,
        z_sun: Optional[np.ndarray] = None,
        z_st: Optional[Quaternion] = None,
        B_n: Optional[np.ndarray] = None,
        s_n: Optional[np.ndarray] = None
    ) -> Tuple[EskfState, bool]:
        """
        Perform one step of hybrid estimation.

        Args:
            x_eskf: Current ESKF state
            t: Current time (seconds)
            jd: Julian date
            omega_meas: Gyro measurement
            dt: Time step
            z_mag: Magnetometer measurement (optional)
            z_sun: Sun sensor measurement (optional)
            z_st: Star tracker measurement (optional)
            B_n: Magnetic field in navigation frame (optional)
            s_n: Sun vector in navigation frame (optional)

        Returns:
            (updated_state, fgo_updated): Updated state and flag indicating FGO update
        """
        fgo_updated = False

        # --- ESKF Prediction ---
        x_eskf = self.eskf.predict(x_eskf, omega_meas, dt)

        # --- ESKF Updates ---
        if z_mag is not None:
            try:
                x_eskf = self.eskf.update(x_eskf, z_mag, SensorType.MAGNETOMETER, B_n=B_n)
            except ValueError:
                pass

        if z_sun is not None:
            try:
                x_eskf = self.eskf.update(x_eskf, z_sun, SensorType.SUN_VECTOR, s_n=s_n)
            except ValueError:
                pass

        if z_st is not None:
            try:
                x_eskf = self.eskf.update(x_eskf, z_st, SensorType.STAR_TRACKER)
            except ValueError:
                pass

        # --- Store sample in window ---
        sample = WindowSample(
            t=float(t),
            jd=float(jd),
            omega_meas=omega_meas.copy(),
            z_mag=z_mag.copy() if z_mag is not None else None,
            z_sun=z_sun.copy() if z_sun is not None else None,
            z_st=z_st if z_st is not None else None,
            B_eci=B_n.copy() if B_n is not None else None,
            s_eci=s_n.copy() if s_n is not None else None,
        )
        self.window.append(sample)

        # --- FGO Optimization (periodic) ---
        if self.should_optimize_fgo(t):
            try:
                logger.debug(f"Running KeyframeFGO optimization at t={t:.2f}s")

                # Convert window to simulation-like data
                window_data = self._window_to_sim_data()

                # Create fresh FGO instance for this window
                fgo = KeyframeFGO(
                    config_path=self.config_path,
                    use_robust=self.fgo.factors.use_robust,
                    use_isam2=self.fgo.use_isam2,
                    use_rk4=self.fgo.use_rk4,
                )

                # Process window with FGO
                times, states = fgo.process_simulation(window_data, self.env)

                if len(states) > 0:
                    # Get the last optimized state
                    optimized_state = states[-1]

                    # Feed back to ESKF
                    x_eskf = self.feedback_fgo_to_eskf(x_eskf, optimized_state)

                    self.fgo_count += 1
                    fgo_updated = True
                    logger.debug(f"FGO update #{self.fgo_count} successful at t={t:.2f}s")

                self.last_fgo_time = t

            except Exception as e:
                logger.warning(f"FGO optimization failed: {e}")

        return x_eskf, fgo_updated
