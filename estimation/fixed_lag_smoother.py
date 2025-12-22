"""
Fixed-Lag Attitude Smoother using GTSAM's IncrementalFixedLagSmoother.

This module implements a proper fixed-lag smoother with automatic marginalization,
suitable for running in parallel with an ESKF for redundant attitude estimation.

Key features:
1. Uses GTSAM's IncrementalFixedLagSmoother (iSAM2-based) with automatic marginalization
2. Creates keyframes only at measurements (efficient)
3. Uses GTSAM's PreintegratedAhrsMeasurements with AHRSFactor for gyro preintegration
   - This connects the bias node to the IMU factor for proper bias estimation
4. Properly marginalizes out old states to keep memory bounded
5. Right-multiply quaternion convention (GTSAM compatible)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque
import numpy as np
import gtsam
from gtsam import (
    Rot3, Point3, Symbol,
    NonlinearFactorGraph, Values,
    IncrementalFixedLagSmoother,
    ISAM2Params,
)

from utilities.quaternion import Quaternion
from utilities.states import NominalState
from utilities.utils import load_yaml
from utilities.process_model import ProcessModel
from estimation.keyframe_fgo import (
    KeyframeFactorBuilders,
    rot3_from_quat,
    quat_from_rot3,
)
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SmootherKeyframe:
    """A keyframe in the fixed-lag smoother."""
    index: int
    time: float
    key_R: int  # Symbol key for attitude
    key_B: int  # Symbol key for bias


class FixedLagAttitudeSmoother:
    """
    Fixed-lag attitude smoother using GTSAM's IncrementalFixedLagSmoother.

    This smoother:
    1. Runs incrementally, adding new keyframes as measurements arrive
    2. Automatically marginalizes out states older than the lag window
    3. Provides smoothed attitude estimates within the lag window
    4. Uses GTSAM's PreintegratedAhrsMeasurements with AHRSFactor for gyro preintegration
       - AHRSFactor connects rotation states AND bias node for proper bias estimation

    The lag parameter controls the trade-off:
    - Larger lag = better accuracy (more future measurements inform past states)
    - Smaller lag = lower latency to final estimate

    Memory is bounded: O(keyframes_in_lag_window) regardless of total runtime.
    """

    def __init__(
        self,
        config_path: str = "configs/config_baseline_short.yaml",
        lag: float = 60.0,  # seconds - fixed-lag window duration
        use_robust: bool = True,
        robust_kernel: str = "huber",  # huber, cauchy, welsch, tukey, etc.
        robust_param: float = 0.1,     # kernel-specific parameter (k for Huber)
        relinearize_threshold: float = 0.001,
        relinearize_skip: int = 1,
        normalize_mag: bool = True,
    ):
        """
        Initialize the fixed-lag attitude smoother.

        Args:
            config_path: Path to configuration file
            lag: Duration of the fixed-lag window in seconds
            use_robust: Use M-estimator for outlier rejection
            robust_kernel: Type of robust kernel (huber, cauchy, etc.)
            robust_param: Kernel parameter (k for Huber, c for Cauchy)
            relinearize_threshold: iSAM2 relinearization threshold
            relinearize_skip: iSAM2 relinearization skip
            normalize_mag: If True, normalize magnetometer measurements (direction-only)
        """
        self.config_path = config_path
        self.lag = lag
        self.use_robust = use_robust
        self.robust_kernel = robust_kernel
        self.robust_param = robust_param
        self.normalize_mag = normalize_mag

        # Load config
        self.config = load_yaml(config_path)
        self.process = ProcessModel(config_path)

        # Create factor builders
        # Use standard noise assumptions (noise_scale = 1.0) for factor graph
        # This allows robust kernels to work properly with normalized residuals
        gyro_sensor_var = self.process.sigma_g ** 2
        discretization_var = (0.10 * 0.0175)**2  # discretization error
        gyro_cov = np.eye(3) * (gyro_sensor_var + discretization_var)

        # Standard noise assumptions for factor graph (no scaling)
        self.factors = KeyframeFactorBuilders(
            gyro_cov=gyro_cov,
            sigma_mag=self.config["sensors"]["mag"]["mag_std"],
            sigma_sun=self.config["sensors"]["sun"]["noise"]["sun_std"],
            sigma_star=self.config["sensors"]["star"]["noise"]["st_std"],
            sigma_bias=self.process.sigma_bg,
            use_robust=use_robust,
            robust_kernel=robust_kernel,
            robust_param=robust_param,
        )

        # iSAM2 parameters for the fixed-lag smoother
        isam2_params = ISAM2Params()
        isam2_params.setRelinearizeThreshold(relinearize_threshold)
        isam2_params.relinearizeSkip = relinearize_skip

        # Create the IncrementalFixedLagSmoother
        # Note: lag is in seconds, IncrementalFixedLagSmoother uses a timestamp-based interface
        self.smoother = IncrementalFixedLagSmoother(lag, isam2_params)

        # State tracking
        self.keyframes: List[SmootherKeyframe] = []
        self.keyframe_index = 0
        self.current_bias = np.zeros(3)
        self.current_preint: Optional[gtsam.PreintegratedAhrsMeasurements] = None
        self.last_keyframe_time: float = 0.0

        # Current state estimate
        self.current_state: Optional[NominalState] = None

        # Initialized flag
        self.initialized = False

        # Statistics
        self.total_keyframes_added = 0
        self.total_keyframes_marginalized = 0

        logger.info(f"FixedLagAttitudeSmoother initialized:")
        logger.info(f"  Lag: {lag}s")
        logger.info(f"  Robust: {use_robust} ({robust_kernel}, k={robust_param})")
        logger.info(f"  Relinearize threshold: {relinearize_threshold}")

    @staticmethod
    def X(k: int) -> int:
        """Symbol key for rotation at keyframe k."""
        return Symbol('x', k).key()

    @staticmethod
    def B(k: int) -> int:
        """Symbol key for bias at keyframe k."""
        return Symbol('b', k).key()

    def initialize(self, t0: float, q_init: Quaternion, b_init: Optional[np.ndarray] = None):
        """
        Initialize the smoother with the first state.

        Args:
            t0: Initial time
            q_init: Initial attitude quaternion
            b_init: Initial gyro bias (default: zeros)
        """
        if self.initialized:
            logger.warning("Smoother already initialized, reinitializing...")

        b_init = b_init if b_init is not None else np.zeros(3)

        # Create first keyframe
        kf = SmootherKeyframe(
            index=0,
            time=t0,
            key_R=self.X(0),
            key_B=self.B(0),
        )
        self.keyframes.append(kf)
        self.keyframe_index = 1

        # Build initial factor graph
        graph = NonlinearFactorGraph()
        values = Values()

        R0 = rot3_from_quat(q_init)
        values.insert(kf.key_R, R0)
        values.insert(kf.key_B, Point3(*b_init))

        # Add priors - relatively loose to allow FGO to correct
        prior_R_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
        graph.add(gtsam.PriorFactorRot3(kf.key_R, R0, prior_R_noise))
        graph.add(self.factors.make_bias_prior(kf.key_B, b_init, 1e-4))

        # Create timestamp map for fixed-lag smoother
        # Maps symbol keys to their timestamps
        timestamps = gtsam.FixedLagSmootherKeyTimestampMap()
        timestamps[kf.key_R] = t0
        timestamps[kf.key_B] = t0

        # Update smoother
        self.smoother.update(graph, values, timestamps)

        # Initialize preintegration for next interval (GTSAM's built-in preintegration)
        self.current_preint = self.factors.create_preintegration(b_init)
        self.current_bias = b_init.copy()
        self.last_keyframe_time = t0

        # Store initial state
        self.current_state = NominalState(
            ori=q_init.copy(),
            gyro_bias=b_init.copy(),
        )

        self.initialized = True
        self.total_keyframes_added = 1

        logger.info(f"FixedLagSmoother initialized at t={t0:.2f}s")

    def integrate_gyro(self, omega_meas: np.ndarray, dt: float, t: float = None,
                       max_preint_time: float = 2.0):
        """
        Integrate a gyroscope measurement.

        Call this for every gyro sample. When a measurement arrives (mag/sun/star),
        call add_measurement() which will use the accumulated preintegration.

        If preintegration exceeds max_preint_time, automatically creates a
        preintegration-only keyframe to avoid accumulating too much drift.

        Args:
            omega_meas: Measured angular velocity (rad/s)
            dt: Time step (seconds)
            t: Current time (optional, needed for auto-keyframe creation)
            max_preint_time: Maximum preintegration time before creating keyframe
        """
        if not self.initialized:
            logger.warning("Smoother not initialized, ignoring gyro")
            return

        if self.current_preint is not None:
            self.current_preint.integrateMeasurement(omega_meas, dt)

            # Auto-create keyframe if preintegration too long
            if t is not None and self.current_preint.deltaTij() > max_preint_time:
                self._add_preintegration_keyframe(t)

    def _add_preintegration_keyframe(self, t: float) -> Optional[NominalState]:
        """
        Create a keyframe with only preintegration factor (no measurements).

        Used during measurement gaps to maintain graph structure and
        prevent excessive preintegration accumulation.
        """
        if not self.initialized or self.current_preint is None:
            return None

        if self.current_preint.deltaTij() < 0.1:
            return self.current_state  # Too short, skip

        # Create new keyframe
        kf = SmootherKeyframe(
            index=self.keyframe_index,
            time=t,
            key_R=self.X(self.keyframe_index),
            key_B=self.B(self.keyframe_index),
        )

        graph = NonlinearFactorGraph()
        values = Values()

        prev_kf = self.keyframes[-1]

        # Get previous estimate
        try:
            current_estimate = self.smoother.calculateEstimate()
            R_prev = current_estimate.atRot3(prev_kf.key_R)
            b_prev = np.array(current_estimate.atPoint3(prev_kf.key_B))
        except Exception:
            R_prev = rot3_from_quat(self.current_state.ori)
            b_prev = self.current_state.gyro_bias.copy()

        # Predict using preintegration
        R_pred = R_prev.compose(self.current_preint.deltaRij())
        values.insert(kf.key_R, R_pred)
        values.insert(kf.key_B, Point3(*b_prev))

        # Add AHRS factor (connects R_i, R_j, and B_i - bias at start of interval)
        graph.add(self.factors.make_ahrs_factor(
            prev_kf.key_R, kf.key_R, prev_kf.key_B, self.current_preint
        ))
        # Add bias random walk factor (models bias evolution between keyframes)
        graph.add(self.factors.make_bias_between(
            prev_kf.key_B, kf.key_B, self.current_preint.deltaTij()
        ))

        # Timestamps
        timestamps = gtsam.FixedLagSmootherKeyTimestampMap()
        timestamps[kf.key_R] = t
        timestamps[kf.key_B] = t

        try:
            self.smoother.update(graph, values, timestamps)
        except Exception as e:
            logger.warning(f"Preintegration keyframe update failed: {e}")
            return self.current_state

        # Update tracking
        self.keyframes.append(kf)
        self.keyframe_index += 1
        self.total_keyframes_added += 1

        # Extract state
        try:
            estimate = self.smoother.calculateEstimate()
            R_est = estimate.atRot3(kf.key_R)
            b_est = np.array(estimate.atPoint3(kf.key_B))
            self.current_state = NominalState(
                ori=quat_from_rot3(R_est),
                gyro_bias=b_est,
            )
            self.current_bias = b_est.copy()
        except Exception:
            pass

        # Reset preintegration
        self.current_preint = self.factors.create_preintegration(self.current_bias)
        self.last_keyframe_time = t

        # Clean up old keyframes
        cutoff_time = t - self.lag
        old_count = len(self.keyframes)
        self.keyframes = [kf for kf in self.keyframes if kf.time >= cutoff_time]
        marginalized = old_count - len(self.keyframes)
        if marginalized > 0:
            self.total_keyframes_marginalized += marginalized

        return self.current_state

    def add_measurement(
        self,
        t: float,
        jd: float,
        z_mag: Optional[np.ndarray] = None,
        z_sun: Optional[np.ndarray] = None,
        z_st: Optional[Quaternion] = None,
        B_eci: Optional[np.ndarray] = None,
        s_eci: Optional[np.ndarray] = None,
    ) -> Optional[NominalState]:
        """
        Add a measurement and create a new keyframe.

        This creates a new keyframe at the measurement time, adds the appropriate
        factors (preintegration + measurements), and updates the smoother.
        Old states outside the lag window are automatically marginalized.

        Args:
            t: Measurement time
            jd: Julian date
            z_mag: Magnetometer measurement in body frame (optional)
            z_sun: Sun sensor measurement in body frame (optional)
            z_st: Star tracker quaternion measurement (optional)
            B_eci: Magnetic field reference in ECI frame (optional)
            s_eci: Sun direction reference in ECI frame (optional)

        Returns:
            Updated state estimate, or None if smoother not initialized
        """
        if not self.initialized:
            logger.warning("Smoother not initialized")
            return None

        # Must have at least one measurement
        if z_mag is None and z_sun is None and z_st is None:
            return self.current_state

        # Create new keyframe
        kf = SmootherKeyframe(
            index=self.keyframe_index,
            time=t,
            key_R=self.X(self.keyframe_index),
            key_B=self.B(self.keyframe_index),
        )

        # Build incremental graph
        graph = NonlinearFactorGraph()
        values = Values()

        # Get previous keyframe
        prev_kf = self.keyframes[-1]

        # Get current estimate for prediction
        try:
            current_estimate = self.smoother.calculateEstimate()
            R_prev = current_estimate.atRot3(prev_kf.key_R)
            b_prev = np.array(current_estimate.atPoint3(prev_kf.key_B))
        except Exception:
            # Fall back to current state
            R_prev = rot3_from_quat(self.current_state.ori)
            b_prev = self.current_state.gyro_bias.copy()

        # Predict new state using preintegration
        if self.current_preint is not None and self.current_preint.deltaTij() > 0:
            R_pred = R_prev.compose(self.current_preint.deltaRij())

            # Add AHRS factor (connects R_i, R_j, and B_i - bias node at start of interval)
            graph.add(self.factors.make_ahrs_factor(
                prev_kf.key_R, kf.key_R, prev_kf.key_B, self.current_preint
            ))

            # Add bias random walk factor (models bias evolution between keyframes)
            graph.add(self.factors.make_bias_between(
                prev_kf.key_B, kf.key_B, self.current_preint.deltaTij()
            ))
        else:
            R_pred = R_prev

        # Add initial values
        values.insert(kf.key_R, R_pred)
        values.insert(kf.key_B, Point3(*b_prev))

        # Add measurement factors
        if z_mag is not None and B_eci is not None:
            graph.add(self.factors.make_magnetometer_factor(kf.key_R, z_mag, B_eci, normalize=self.normalize_mag))

        if z_sun is not None and s_eci is not None:
            graph.add(self.factors.make_sun_factor(kf.key_R, z_sun, s_eci))

        if z_st is not None:
            graph.add(self.factors.make_star_factor(kf.key_R, z_st))

        # Create timestamp map
        timestamps = gtsam.FixedLagSmootherKeyTimestampMap()
        timestamps[kf.key_R] = t
        timestamps[kf.key_B] = t

        # Update smoother (this handles marginalization automatically)
        try:
            result = self.smoother.update(graph, values, timestamps)

            # Track marginalized keys
            # Note: IncrementalFixedLagSmoother handles marginalization internally
            # We can query which keys are still in the smoother

        except Exception as e:
            logger.warning(f"Smoother update failed: {e}")
            return self.current_state

        # Update keyframe tracking
        self.keyframes.append(kf)
        self.keyframe_index += 1
        self.total_keyframes_added += 1

        # Extract updated state estimate
        try:
            estimate = self.smoother.calculateEstimate()
            R_est = estimate.atRot3(kf.key_R)
            b_est = np.array(estimate.atPoint3(kf.key_B))

            self.current_state = NominalState(
                ori=quat_from_rot3(R_est),
                gyro_bias=b_est,
            )
            self.current_bias = b_est.copy()

        except Exception as e:
            logger.warning(f"Failed to extract estimate: {e}")

        # Reset preintegration for next interval
        self.current_preint = self.factors.create_preintegration(self.current_bias)
        self.last_keyframe_time = t

        # Clean up old keyframes that have been marginalized
        # Keep only keyframes within the lag window
        cutoff_time = t - self.lag
        old_count = len(self.keyframes)
        self.keyframes = [kf for kf in self.keyframes if kf.time >= cutoff_time]
        marginalized = old_count - len(self.keyframes)
        if marginalized > 0:
            self.total_keyframes_marginalized += marginalized
            logger.debug(f"Marginalized {marginalized} old keyframes")

        return self.current_state

    def get_state(self) -> Optional[NominalState]:
        """Get the current smoothed state estimate."""
        return self.current_state

    def get_propagated_state(self) -> Optional[NominalState]:
        """
        Get the state estimate propagated forward using accumulated gyro data.

        This provides a smoother output between measurement updates by using
        the preintegrated gyro rotation to propagate from the last keyframe.

        Returns:
            Propagated state, or current_state if no preintegration available
        """
        if self.current_state is None:
            return None

        if self.current_preint is None or self.current_preint.deltaTij() < 1e-6:
            return self.current_state

        # Propagate rotation using preintegrated delta
        # R_propagated = R_keyframe * delta_R (right multiply)
        R_keyframe = rot3_from_quat(self.current_state.ori)
        R_propagated = R_keyframe.compose(self.current_preint.deltaRij())
        q_propagated = quat_from_rot3(R_propagated)

        return NominalState(
            ori=q_propagated,
            gyro_bias=self.current_state.gyro_bias.copy(),
        )

    def get_state_at_time(self, t: float) -> Optional[NominalState]:
        """
        Get the smoothed state estimate at a specific time.

        Only works for times within the lag window that have keyframes.

        Args:
            t: Query time

        Returns:
            State at time t, or None if not available
        """
        if not self.initialized:
            return None

        # Find closest keyframe
        closest_kf = None
        min_dt = float('inf')
        for kf in self.keyframes:
            dt = abs(kf.time - t)
            if dt < min_dt:
                min_dt = dt
                closest_kf = kf

        if closest_kf is None or min_dt > 1.0:  # Max 1 second tolerance
            return None

        try:
            estimate = self.smoother.calculateEstimate()
            R_est = estimate.atRot3(closest_kf.key_R)
            b_est = np.array(estimate.atPoint3(closest_kf.key_B))

            return NominalState(
                ori=quat_from_rot3(R_est),
                gyro_bias=b_est,
            )
        except Exception:
            return None

    def get_statistics(self) -> dict:
        """Get smoother statistics."""
        return {
            "lag": self.lag,
            "active_keyframes": len(self.keyframes),
            "total_keyframes_added": self.total_keyframes_added,
            "total_keyframes_marginalized": self.total_keyframes_marginalized,
            "initialized": self.initialized,
        }

    def reset(self):
        """Reset the smoother to uninitialized state."""
        # Create new smoother instance
        isam2_params = ISAM2Params()
        isam2_params.setRelinearizeThreshold(0.001)
        isam2_params.relinearizeSkip = 1
        self.smoother = IncrementalFixedLagSmoother(self.lag, isam2_params)

        self.keyframes = []
        self.keyframe_index = 0
        self.current_bias = np.zeros(3)
        self.current_preint = None
        self.last_keyframe_time = 0.0
        self.current_state = None
        self.initialized = False

        logger.info("FixedLagSmoother reset")
