"""
Hybrid Estimator: ESKF Frontend + FGO Backend

This module implements a dual-mode estimation architecture:
1. Normal mode: ESKF runs continuously with periodic FGO optimization
2. Degraded mode: FGO-only when measurements are sparse

The FGO backend provides refined estimates back to the ESKF frontend.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

import numpy as np
import gtsam

from estimation.eskf import ESKF
from estimation.gtsam_fg import GtsamFGO, WindowSample, SlidingWindow
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from environment.environment import OrbitEnvironmentModel
from logging_config import get_logger
from estimation.gtsam_fg import quat_from_rot3

logger = get_logger(__name__)


class EstimatorMode(Enum):
    """Operating mode of the hybrid estimator."""
    DUAL = "dual"           # ESKF frontend + FGO backend
    FGO_ONLY = "fgo_only"   # FGO-only during degraded measurements


@dataclass
class MeasurementHealth:
    """Tracks measurement availability and quality."""
    mag_count: int = 0
    sun_count: int = 0
    star_count: int = 0
    samples_with_measurement: int = 0  # Samples with at least one measurement
    total_samples: int = 0
    consecutive_degraded: int = 0  # Consecutive samples below threshold
    consecutive_healthy: int = 0    # Consecutive samples above threshold

    @property
    def measurement_rate(self) -> float:
        """Fraction of samples with at least one measurement."""
        if self.total_samples == 0:
            return 1.0  # Assume healthy until proven otherwise
        return self.samples_with_measurement / self.total_samples

    @property
    def is_degraded(self) -> bool:
        """Check if measurements are degraded."""
        return self.measurement_rate < 0.2  # Less than 20% sample availability


class HybridEstimator:
    """
    Hybrid estimator combining ESKF (frontend) and FGO (backend).

    Architecture:
    - ESKF runs continuously, providing real-time state estimates
    - Measurements stored in sliding window
    - FGO optimizes over window periodically (every 10-20 seconds)
    - Optimized state and covariance fed back to ESKF
    - Automatic mode switching during degraded measurements
    """

    def __init__(
        self,
        P0: np.ndarray,
        config_path: str = "configs/config_baseline.yaml",
        fgo_window_size: int = 100,
        fgo_optimize_interval: float = 10.0,  # seconds
        degraded_threshold: float = 0.2,       # measurement rate threshold (20%)
        mode_switch_hysteresis: int = 50,      # samples of sustained condition before switching
        warmup_samples: int = 100,             # samples before allowing mode switches
        use_robust: bool = True,
    ):
        """
        Initialize hybrid estimator.

        Args:
            P0: Initial error covariance (6x6)
            config_path: Path to configuration file
            fgo_window_size: Size of FGO sliding window
            fgo_optimize_interval: Time between FGO optimizations (seconds)
            degraded_threshold: Measurement rate threshold for mode switching
            mode_switch_hysteresis: Number of consecutive samples required before mode switch
            warmup_samples: Number of samples before allowing mode switches (stay in DUAL)
            use_robust: Use M-estimator in FGO
        """
        self.config_path = config_path
        self.fgo_window_size = fgo_window_size
        self.fgo_optimize_interval = fgo_optimize_interval
        self.degraded_threshold = degraded_threshold
        self.mode_switch_hysteresis = mode_switch_hysteresis
        self.warmup_samples = warmup_samples

        # Initialize estimators
        self.eskf = ESKF(P0=P0, config_path=config_path)
        self.fgo = GtsamFGO(
            config_path=config_path,
            use_robust=use_robust,
            robust_kernel="Huber",
            robust_param=1.345
        )

        # Sliding window for FGO
        self.window = SlidingWindow(max_len=fgo_window_size)

        # State
        self.mode = EstimatorMode.DUAL
        self.last_fgo_time = 0.0
        self.measurement_health = MeasurementHealth()
        self.health_window_size = 100  # Samples to track for health

        # Environment model for FGO
        self.env = OrbitEnvironmentModel()

        logger.info(f"HybridEstimator initialized:")
        logger.info(f"  Mode: {self.mode.value}")
        logger.info(f"  FGO window: {fgo_window_size} samples")
        logger.info(f"  FGO interval: {fgo_optimize_interval}s")
        logger.info(f"  Degraded threshold: {degraded_threshold}")
        logger.info(f"  Mode switch hysteresis: {mode_switch_hysteresis} samples")
        logger.info(f"  Warmup period: {warmup_samples} samples")

    def update_measurement_health(
        self,
        has_mag: bool,
        has_sun: bool,
        has_star: bool
    ):
        """Track measurement availability for mode switching."""
        self.measurement_health.total_samples += 1

        # Track individual sensor counts
        if has_mag:
            self.measurement_health.mag_count += 1
        if has_sun:
            self.measurement_health.sun_count += 1
        if has_star:
            self.measurement_health.star_count += 1

        # Track samples with at least one measurement
        if has_mag or has_sun or has_star:
            self.measurement_health.samples_with_measurement += 1

        # Reset counters periodically to track recent health
        if self.measurement_health.total_samples > self.health_window_size:
            # Decay old measurements
            decay = 0.95
            self.measurement_health.mag_count = int(self.measurement_health.mag_count * decay)
            self.measurement_health.sun_count = int(self.measurement_health.sun_count * decay)
            self.measurement_health.star_count = int(self.measurement_health.star_count * decay)
            self.measurement_health.samples_with_measurement = int(self.measurement_health.samples_with_measurement * decay)
            self.measurement_health.total_samples = int(self.measurement_health.total_samples * decay)

    def check_mode_switch(self) -> bool:
        """
        Check if mode should be switched with warmup period and hysteresis.

        Returns:
            True if mode was changed
        """
        # During warmup period, stay in DUAL mode
        if self.measurement_health.total_samples < self.warmup_samples:
            return False

        # Current measurement condition
        current_rate = self.measurement_health.measurement_rate
        is_currently_degraded = current_rate < self.degraded_threshold

        # Update consecutive counters
        if is_currently_degraded:
            self.measurement_health.consecutive_degraded += 1
            self.measurement_health.consecutive_healthy = 0
        else:
            self.measurement_health.consecutive_healthy += 1
            self.measurement_health.consecutive_degraded = 0

        # Check for mode switch with hysteresis
        if self.mode == EstimatorMode.DUAL:
            # Switch to FGO_ONLY if degraded for sustained period
            if self.measurement_health.consecutive_degraded >= self.mode_switch_hysteresis:
                logger.warning(
                    f"Switching to FGO_ONLY mode (rate: {current_rate:.2%} < {self.degraded_threshold:.2%} "
                    f"for {self.measurement_health.consecutive_degraded} samples)"
                )
                self.mode = EstimatorMode.FGO_ONLY
                self.measurement_health.consecutive_degraded = 0
                return True

        elif self.mode == EstimatorMode.FGO_ONLY:
            # Switch back to DUAL if healthy for sustained period
            # Use higher threshold (degraded_threshold + 0.05) to avoid rapid switching
            recovery_threshold = self.degraded_threshold + 0.05
            if current_rate >= recovery_threshold and self.measurement_health.consecutive_healthy >= self.mode_switch_hysteresis:
                logger.info(
                    f"Switching back to DUAL mode (rate: {current_rate:.2%} >= {recovery_threshold:.2%} "
                    f"for {self.measurement_health.consecutive_healthy} samples)"
                )
                self.mode = EstimatorMode.DUAL
                self.measurement_health.consecutive_healthy = 0
                return True

        return False

    def extract_fgo_covariance(
        self,
        graph: gtsam.NonlinearFactorGraph,
        result: gtsam.Values,
        state_idx: int
    ) -> np.ndarray:
        """
        Extract marginal covariance from FGO optimization.

        Args:
            graph: Factor graph
            result: Optimization result
            state_idx: Index of state to extract covariance for

        Returns:
            6x6 covariance matrix [rotation(3), bias(3)]
        """
        try:
            # Compute marginal covariances
            marginals = gtsam.Marginals(graph, result)

            # Get covariance for rotation and bias
            R_key = self.fgo.X(state_idx)
            b_key = self.fgo.B(state_idx)

            P_R = marginals.marginalCovariance(R_key)  # 3x3
            P_b = marginals.marginalCovariance(b_key)  # 3x3

            # Combine into 6x6 block diagonal (assuming independence)
            P = np.zeros((6, 6))
            P[0:3, 0:3] = P_R
            P[3:6, 3:6] = P_b

            return P

        except Exception as e:
            logger.warning(f"Failed to extract FGO covariance: {e}, using ESKF covariance")
            return None

    def feedback_fgo_to_eskf(
        self,
        x_eskf: EskfState,
        optimized_state: NominalState,
        P_fgo: Optional[np.ndarray] = None,
        use_optimized_covariance: bool = True
    ) -> EskfState:
        """
        Feed optimized state from FGO back to ESKF.

        Args:
            x_eskf: Current ESKF state
            optimized_state: Optimized state from FGO
            P_fgo: Covariance from FGO (if available)
            use_optimized_covariance: If True, reset covariance after FGO optimization

        Returns:
            Updated ESKF state
        """
        # Update nominal state
        x_eskf.nom.ori = optimized_state.ori.normalize()
        x_eskf.nom.gyro_bias = optimized_state.gyro_bias.copy()

        # Update error covariance
        if P_fgo is not None:
            # Use covariance extracted from FGO
            P_fgo = 0.5 * (P_fgo + P_fgo.T)  # Ensure symmetry
            eigenvalues = np.linalg.eigvals(P_fgo)
            if np.all(eigenvalues > 0):
                x_eskf.err.cov = P_fgo
                logger.debug("Using FGO marginal covariance")
            else:
                logger.warning("FGO covariance not positive definite, using reduced ESKF covariance")
                x_eskf.err.cov = 0.5 * x_eskf.err.cov  # Reduce uncertainty after optimization
        else:
            # FGO covariance extraction failed (underconstrained system)
            # After batch optimization, uncertainty should be lower than accumulated ESKF uncertainty
            if use_optimized_covariance:
                # Reduce covariance to reflect the optimization
                # Use a conservative reduction: take 50% of current covariance or P0, whichever is smaller
                P_reduced = 0.5 * x_eskf.err.cov
                P_min = self.eskf.P0  # Initial covariance as lower bound
                x_eskf.err.cov = np.minimum(P_reduced, P_min)
                logger.debug("FGO covariance extraction failed, using reduced ESKF covariance")
            else:
                # Keep ESKF covariance as-is
                logger.debug("FGO covariance extraction failed, keeping ESKF covariance")

        # Reset error mean
        x_eskf.err.mean[:] = 0.0

        return x_eskf

    def should_optimize_fgo(self, current_time: float) -> bool:
        """Check if it's time to run FGO optimization."""
        if not self.window.ready:
            return False

        time_since_last = current_time - self.last_fgo_time
        return time_since_last >= self.fgo_optimize_interval

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

        # Track measurement health
        has_mag = z_mag is not None and not np.any(np.isnan(z_mag))
        has_sun = z_sun is not None and not np.any(np.isnan(z_sun))
        has_star = z_st is not None
        self.update_measurement_health(has_mag, has_sun, has_star)

        # Check for mode switch
        self.check_mode_switch()

        # --- ESKF Prediction (always run) ---
        x_eskf = self.eskf.predict(x_eskf, omega_meas, dt)

        # --- ESKF Update (only in DUAL mode) ---
        if self.mode == EstimatorMode.DUAL:
            # Magnetometer
            if has_mag:
                try:
                    x_eskf = self.eskf.update(x_eskf, z_mag, SensorType.MAGNETOMETER, B_n=B_n)
                except ValueError:
                    pass  # Skip outliers

            # Sun sensor
            if has_sun:
                try:
                    x_eskf = self.eskf.update(x_eskf, z_sun, SensorType.SUN_VECTOR, s_n=s_n)
                except ValueError:
                    pass

            # Star tracker
            if has_star:
                try:
                    x_eskf = self.eskf.update(x_eskf, z_st, SensorType.STAR_TRACKER)
                except ValueError:
                    pass

        # --- Store sample in window ---
        sample = WindowSample(
            t=float(t),
            jd=float(jd),
            x_nom=NominalState(ori=x_eskf.nom.ori, gyro_bias=x_eskf.nom.gyro_bias),
            omega_meas=omega_meas.copy(),
            z_mag=z_mag.copy() if has_mag else None,
            z_sun=z_sun.copy() if has_sun else None,
            z_st=z_st if has_star else None
        )
        self.window.add(sample)

        # --- FGO Optimization (periodic or in FGO_ONLY mode) ---
        if self.should_optimize_fgo(t) or self.mode == EstimatorMode.FGO_ONLY:
            try:
                logger.debug(f"Running FGO optimization at t={t:.2f}s, mode={self.mode.value}")

                # Build and optimize graph
                graph, values = self.fgo.build_window_graph(list(self.window.samples), self.env)

                params = gtsam.LevenbergMarquardtParams()
                params.setMaxIterations(self.fgo.max_iters)
                params.setAbsoluteErrorTol(self.fgo.tol)

                optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
                result = optimizer.optimize()

                # Extract optimized state at current time (last sample in window)
                last_idx = len(self.window.samples) - 1
                R_opt = result.atRot3(self.fgo.X(last_idx))
                b_opt = result.atPoint3(self.fgo.B(last_idx))

                q_opt = quat_from_rot3(R_opt)
                optimized_state = NominalState(ori=q_opt, gyro_bias=np.array(b_opt))

                # Extract covariance
                P_fgo = self.extract_fgo_covariance(graph, result, last_idx)

                # Feed back to ESKF
                x_eskf = self.feedback_fgo_to_eskf(x_eskf, optimized_state, P_fgo)

                self.last_fgo_time = t
                fgo_updated = True

                logger.debug(f"FGO update successful, error cov trace: {np.trace(x_eskf.err.cov):.6e}")

            except Exception as e:
                logger.warning(f"FGO optimization failed: {e}")

        return x_eskf, fgo_updated
