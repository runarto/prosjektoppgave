"""
Hybrid Estimator: ESKF Frontend + FGO Backend with Mode Switching

This module implements a three-tier estimation architecture:

NORMAL OPERATION:
1. ESKF runs continuously for real-time state estimates (primary)
2. Batch FGO runs periodically for smoothed corrections
3. FGO corrections applied to ESKF when significant

DEGRADED OPERATION (iSAM2 mode):
4. When ESKF is unreliable (high rates, no star tracker, divergence),
   switch to iSAM2 as primary estimator
5. iSAM2 maintains incremental solution, always available for switchover

Key design principle: FGO should be able to CORRECT ESKF, not just confirm it.
Therefore, FGO is initialized from its own previous state, NOT from ESKF.

Uses QUEST algorithm for deterministic attitude initialization from vector measurements.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from collections import deque
from enum import Enum, auto

import numpy as np
import copy

from estimation.eskf import ESKF
from estimation.keyframe_fgo import KeyframeFGO, quat_from_rot3, rot3_from_quat
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from environment.environment import OrbitEnvironmentModel
from logging_config import get_logger

logger = get_logger(__name__)


class EstimatorMode(Enum):
    """Primary estimator mode."""
    ESKF = auto()      # ESKF primary with periodic batch FGO smoothing
    ISAM2 = auto()     # iSAM2 primary (degraded mode)


def quest(
    b_vectors: List[np.ndarray],
    r_vectors: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> Optional[Quaternion]:
    """
    QUEST algorithm for attitude determination from vector observations.

    Solves Wahba's problem: find R that minimizes Σ w_i ||b_i - R @ r_i||²
    where b_i are body-frame measurements and r_i are reference vectors.

    Args:
        b_vectors: List of body-frame unit vectors (measurements)
        r_vectors: List of reference unit vectors (inertial frame)
        weights: Optional weights for each measurement (default: equal weights)

    Returns:
        Estimated quaternion (body-to-inertial), or None if estimation fails
    """
    if len(b_vectors) < 2 or len(r_vectors) < 2:
        return None

    n = min(len(b_vectors), len(r_vectors))
    if weights is None:
        weights = [1.0 / n] * n

    # Normalize vectors
    b_norm = [b / np.linalg.norm(b) for b in b_vectors[:n]]
    r_norm = [r / np.linalg.norm(r) for r in r_vectors[:n]]

    # Compute attitude profile matrix B = Σ w_i * b_i * r_i^T
    B = np.zeros((3, 3))
    for i in range(n):
        B += weights[i] * np.outer(b_norm[i], r_norm[i])

    # Form K matrix (Davenport's K)
    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([
        B[1, 2] - B[2, 1],
        B[2, 0] - B[0, 2],
        B[0, 1] - B[1, 0]
    ])

    # K = [[sigma, Z^T], [Z, S - sigma*I]]
    K = np.zeros((4, 4))
    K[0, 0] = sigma
    K[0, 1:4] = Z
    K[1:4, 0] = Z
    K[1:4, 1:4] = S - sigma * np.eye(3)

    # Find eigenvector corresponding to largest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    max_idx = np.argmax(eigenvalues)
    q_opt = eigenvectors[:, max_idx]

    # q_opt is [q_w, q_x, q_y, q_z] format
    # Convert to our Quaternion class (mu=scalar, eta=vector)
    return Quaternion(mu=q_opt[0], eta=q_opt[1:4]).normalize()


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
    q_true: np.ndarray  # Initial attitude for FGO (from last FGO state)


class HybridEstimator:
    """
    Hybrid estimator combining ESKF (frontend) and sliding window FGO (backend).

    Architecture:
    - ESKF runs continuously, providing real-time state estimates
    - Measurements stored in sliding window
    - Batch FGO optimizes over window periodically with preintegration
    - FGO maintains its OWN state history (does NOT use ESKF estimates)
    - FGO corrections applied to ESKF when difference is significant

    This design allows FGO to CORRECT systematic ESKF errors, not just confirm them.
    """

    def __init__(
        self,
        P0: np.ndarray,
        config_path: str = "configs/config_baseline_short.yaml",
        fgo_window_duration: float = 120.0,  # seconds - sliding window duration (longer = more ST)
        fgo_optimize_interval: float = 60.0,  # seconds - batch optimize every N seconds
        use_robust: bool = True,
        correction_mode: str = "normal",  # "normal", "aggressive", "conservative"
        # Adaptive FGO triggering based on angular rate
        omega_threshold_deg_s: float = 5.0,  # trigger faster FGO above this rate
        fast_fgo_interval: float = 10.0,  # FGO interval during high rates
        # Mode switching parameters
        enable_isam2_fallback: bool = False,  # Enable iSAM2 as fallback primary
        isam2_switch_threshold_deg: float = 2.0,  # Switch to iSAM2 if ESKF error exceeds this
        isam2_return_threshold_st: int = 3,  # Return to ESKF after this many star tracker measurements
    ):
        """
        Initialize hybrid estimator.

        Args:
            P0: Initial error covariance (6x6)
            config_path: Path to configuration file
            fgo_window_duration: Duration of sliding window (seconds)
            fgo_optimize_interval: Time between FGO batch optimizations (seconds)
            use_robust: Use Huber M-estimator in FGO
            correction_mode: FGO correction strategy
                - "normal": Standard correction strategy
                - "aggressive": Trust FGO more, apply larger corrections
                - "conservative": Smaller corrections, smoother transitions
            omega_threshold_deg_s: Angular rate threshold for fast FGO (deg/s)
            fast_fgo_interval: FGO interval when above threshold (seconds)
            enable_isam2_fallback: Whether to allow switching to iSAM2 as primary
            isam2_switch_threshold_deg: Switch to iSAM2 if disagreement exceeds this
            isam2_return_threshold_st: Return to ESKF mode after N star tracker updates
        """
        self.config_path = config_path
        self.fgo_window_duration = fgo_window_duration
        self.fgo_optimize_interval = fgo_optimize_interval
        self.P0 = P0
        self.correction_mode = correction_mode
        self.use_robust = use_robust

        # Adaptive FGO parameters
        self.omega_threshold_rad_s = np.deg2rad(omega_threshold_deg_s)
        self.fast_fgo_interval = fast_fgo_interval
        self.high_rate_mode = False
        self.omega_history: deque[float] = deque(maxlen=50)  # ~1 second at 50Hz

        # Mode switching parameters
        self.enable_isam2_fallback = enable_isam2_fallback
        self.isam2_switch_threshold_deg = isam2_switch_threshold_deg
        self.isam2_return_threshold_st = isam2_return_threshold_st
        self.primary_mode = EstimatorMode.ESKF
        self.star_tracker_count_since_switch = 0

        # Initialize ESKF
        self.eskf = ESKF(P0=P0, config_path=config_path)

        # Environment model
        self.env = OrbitEnvironmentModel()

        # Sliding window for samples (store enough for window duration)
        # At 50 Hz, 60s window = 3000 samples
        max_samples = int(fgo_window_duration * 60)  # ~20% buffer
        self.window: deque[WindowSample] = deque(maxlen=max_samples)

        # FGO state tracking (independent of ESKF)
        self.last_fgo_state: Optional[NominalState] = None
        self.last_fgo_time: float = 0.0
        self.fgo_count = 0

        # iSAM2 instance (lazy initialized when needed)
        self.isam2_fgo: Optional[KeyframeFGO] = None
        self.isam2_initialized = False

        # Statistics
        self.correction_history: List[float] = []
        self.mode_switches: List[Tuple[float, str]] = []  # (time, new_mode)

        logger.info(f"HybridEstimator initialized, correction mode: {correction_mode}")
        logger.info(f"  FGO window duration: {fgo_window_duration}s")
        logger.info(f"  FGO optimize interval: {fgo_optimize_interval}s (fast: {fast_fgo_interval}s)")
        logger.info(f"  Omega threshold: {omega_threshold_deg_s} deg/s")
        logger.info(f"  Use robust: {use_robust}")
        if enable_isam2_fallback:
            logger.info(f"  iSAM2 fallback: ENABLED (threshold: {isam2_switch_threshold_deg}°)")

    def initialize(self, t0: float, q_init: Quaternion):
        """
        Initialize the hybrid estimator with starting state.

        Args:
            t0: Initial time
            q_init: Initial attitude (from ESKF or perturbed)
        """
        self.last_fgo_state = NominalState(
            ori=q_init.copy(),
            gyro_bias=np.zeros(3),
        )
        self.last_fgo_time = t0
        logger.info(f"HybridEstimator initialized at t={t0:.2f}s")

    def _window_to_sim_data(self, current_eskf_state: Optional[NominalState] = None) -> WindowData:
        """Convert window samples to simulation-like data structure for FGO.

        Args:
            current_eskf_state: Current ESKF state as fallback for initialization
        """
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

        # Get best initial attitude using priority:
        # 1. Star tracker, 2. Previous FGO, 3. QUEST, 4. ESKF
        q_init_quat = self._get_best_initial_attitude(current_eskf_state)
        q_init = q_init_quat.as_array()

        q_true = np.zeros((n, 4))
        q_true[0] = q_init
        for i in range(1, n):
            q_true[i] = q_init  # FGO will propagate from here

        window_data = WindowData(
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

        return window_data

    def _update_rate_mode(self, omega_meas: np.ndarray):
        """Update angular rate tracking and high-rate mode detection."""
        omega_norm = np.linalg.norm(omega_meas)
        self.omega_history.append(omega_norm)

        # Use moving average to avoid spurious triggers
        if len(self.omega_history) >= 10:
            avg_omega = np.mean(list(self.omega_history))
            was_high_rate = self.high_rate_mode
            self.high_rate_mode = avg_omega > self.omega_threshold_rad_s

            if self.high_rate_mode and not was_high_rate:
                logger.info(f"High rate mode ENABLED (avg ω={np.rad2deg(avg_omega):.1f} deg/s)")
            elif not self.high_rate_mode and was_high_rate:
                logger.info(f"High rate mode DISABLED (avg ω={np.rad2deg(avg_omega):.1f} deg/s)")

    def should_optimize_fgo(self, current_time: float) -> bool:
        """Check if it's time to run FGO optimization.

        Uses adaptive interval: faster updates during high angular rates.
        """
        # Need minimum samples for meaningful optimization
        min_samples = 50
        if len(self.window) < min_samples:
            return False

        time_since_last = current_time - self.last_fgo_time

        # Use shorter interval during high angular rates
        if self.high_rate_mode:
            return time_since_last >= self.fast_fgo_interval
        else:
            return time_since_last >= self.fgo_optimize_interval

    def _count_keyframes(self) -> int:
        """Count approximate number of keyframes in window."""
        count = 0
        for s in self.window:
            if s.z_mag is not None or s.z_sun is not None or s.z_st is not None:
                count += 1
        return max(count, 1)

    def _count_star_tracker_in_window(self) -> int:
        """Count star tracker measurements in current window."""
        count = 0
        for s in self.window:
            if s.z_st is not None:
                count += 1
        return count

    def _compute_quest_attitude(self) -> Optional[Quaternion]:
        """
        Compute initial attitude estimate using QUEST from window measurements.

        Looks for co-located magnetometer and sun sensor measurements in the window
        and uses QUEST to compute a deterministic attitude estimate.

        Returns:
            Quaternion from QUEST, or None if insufficient measurements
        """
        # Look for samples with both magnetometer and sun sensor
        b_vectors = []
        r_vectors = []

        for s in self.window:
            if s.z_mag is not None and s.B_eci is not None:
                b_vectors.append(s.z_mag / np.linalg.norm(s.z_mag))
                r_vectors.append(s.B_eci / np.linalg.norm(s.B_eci))
            if s.z_sun is not None and s.s_eci is not None:
                b_vectors.append(s.z_sun / np.linalg.norm(s.z_sun))
                r_vectors.append(s.s_eci / np.linalg.norm(s.s_eci))

            # If we have 2+ pairs, we can compute QUEST
            if len(b_vectors) >= 2:
                break

        if len(b_vectors) < 2:
            return None

        # Use QUEST to compute attitude
        q_quest = quest(b_vectors, r_vectors)
        if q_quest is not None:
            logger.info(f"QUEST attitude computed from {len(b_vectors)} vectors")
        return q_quest

    def _get_best_initial_attitude(
        self,
        current_eskf_state: Optional[NominalState] = None
    ) -> Quaternion:
        """
        Get the best initial attitude for FGO optimization.

        Priority:
        1. Star tracker measurement (if available in window) - most accurate
        2. Previous FGO state (if FGO has run successfully before)
        3. QUEST from vector measurements - deterministic, no convergence needed
        4. Current ESKF state - may be converging

        Args:
            current_eskf_state: Current ESKF state as fallback

        Returns:
            Best available initial attitude estimate
        """
        # 1. Check for star tracker measurement
        for s in self.window:
            if s.z_st is not None:
                logger.info("Using star tracker for FGO initialization")
                return s.z_st.copy()

        # 2. Use previous FGO state if available
        if self.last_fgo_state is not None and self.fgo_count > 0:
            logger.info("Using previous FGO state for initialization")
            return self.last_fgo_state.ori.copy()

        # 3. Try QUEST
        q_quest = self._compute_quest_attitude()
        if q_quest is not None:
            logger.info("Using QUEST for FGO initialization")
            return q_quest

        # 4. Fall back to ESKF
        if current_eskf_state is not None:
            logger.info("Using ESKF state for FGO initialization")
            return current_eskf_state.ori.copy()

        # 5. Last resort - identity
        logger.warning("No good initial attitude available, using identity")
        return Quaternion(mu=1.0, eta=np.zeros(3))

    def run_batch_fgo(
        self, current_eskf_state: Optional[NominalState] = None
    ) -> Tuple[Optional[NominalState], float]:
        """
        Run batch FGO optimization over the current window.

        Args:
            current_eskf_state: Current ESKF state for initialization on first run

        Returns:
            (optimized_state, final_error): Optimized state and FGO's final error
            Returns (None, inf) if optimization failed
        """
        if len(self.window) < 50:
            return None, float("inf")

        try:
            # Convert window to sim data
            window_data = self._window_to_sim_data(current_eskf_state)

            # Create fresh FGO instance for this window
            # perturb_initial=False since we're using our own last_fgo_state
            fgo = KeyframeFGO(
                config_path=self.config_path,
                use_robust=self.use_robust,
                use_isam2=False,  # Use batch for sliding window
                use_rk4=True,
                perturb_initial=False,  # Use provided initial state
            )

            # Process window with FGO
            # Note: NOT passing kf_estimates - FGO initializes from q_true[0]
            times, states = fgo.process_simulation(window_data, self.env)

            # Get final error from FGO for confidence estimation
            final_error = fgo.get_final_error() if hasattr(fgo, "get_final_error") else float("inf")

            if len(states) > 0:
                return states[-1], final_error

        except Exception as e:
            logger.warning(f"Batch FGO failed: {e}")

        return None, float("inf")

    def feedback_fgo_to_eskf(
        self,
        x_eskf: EskfState,
        fgo_state: NominalState,
        n_star_tracker: int = 0,
    ) -> Tuple[EskfState, float]:
        """
        Feed optimized state from FGO back to ESKF.

        Args:
            x_eskf: Current ESKF state
            fgo_state: Optimized state from FGO
            n_star_tracker: Number of star tracker measurements in window

        Returns:
            (updated_state, correction_deg): Updated ESKF state and correction magnitude
        """
        # Compute attitude correction
        q_eskf = x_eskf.nom.ori
        q_fgo = fgo_state.ori

        # Compute rotation from ESKF to FGO: dq = q_eskf^-1 * q_fgo
        dq = q_eskf.conjugate() @ q_fgo
        correction_angle = 2 * np.arccos(np.clip(abs(dq.mu), 0, 1))
        correction_deg = np.rad2deg(correction_angle)

        # Determine confidence based on star tracker availability
        # Without star tracker, FGO may not be more accurate than ESKF
        has_good_reference = n_star_tracker >= 2

        # Log correction
        if correction_deg > 0.001:
            if n_star_tracker >= 10:
                conf_str = "VERY HIGH"
            elif has_good_reference:
                conf_str = "HIGH"
            else:
                conf_str = "LOW"
            logger.info(f"FGO correction: {correction_deg:.4f} deg (confidence: {conf_str}, ST={n_star_tracker})")

        # If no star tracker and large correction, be very cautious
        # FGO without star tracker is not necessarily better than ESKF
        if not has_good_reference and correction_deg > 0.5:
            logger.warning(f"Skipping large FGO correction ({correction_deg:.2f}°) - no star tracker reference")
            return x_eskf, correction_deg

        # Mode-dependent correction strategy
        if self.correction_mode == "aggressive":
            # Trust FGO completely
            x_eskf.nom.ori = fgo_state.ori.copy()
            x_eskf.nom.gyro_bias = fgo_state.gyro_bias.copy()
            cov_scale = 0.5
        elif self.correction_mode == "conservative":
            # Blend: 70% FGO, 30% ESKF
            alpha = 0.7
            if correction_deg > 0.01:
                x_eskf.nom.ori = fgo_state.ori.copy()
            x_eskf.nom.gyro_bias = alpha * fgo_state.gyro_bias + (1-alpha) * x_eskf.nom.gyro_bias
            cov_scale = 0.8
        else:  # "normal"
            # Scale trust based on star tracker availability
            # More star tracker = more trust in FGO (it can reject outliers with Huber)
            if n_star_tracker >= 10:
                # Very high confidence - trust FGO completely
                # FGO with robust cost and many ST can correct systematic errors
                x_eskf.nom.ori = fgo_state.ori.copy()
                x_eskf.nom.gyro_bias = fgo_state.gyro_bias.copy()
                cov_scale = 0.5
                logger.info(f"High-confidence FGO correction ({n_star_tracker} ST)")
            elif has_good_reference:
                # Good confidence - apply FGO state directly
                x_eskf.nom.ori = fgo_state.ori.copy()
                x_eskf.nom.gyro_bias = 0.9 * fgo_state.gyro_bias + 0.1 * x_eskf.nom.gyro_bias
                cov_scale = 0.6
            else:
                # Low confidence - be more cautious with large corrections
                if correction_deg < 0.3:
                    x_eskf.nom.ori = fgo_state.ori.copy()
                    x_eskf.nom.gyro_bias = 0.5 * fgo_state.gyro_bias + 0.5 * x_eskf.nom.gyro_bias
                    cov_scale = 0.85
                else:
                    # Large correction without star tracker - skip attitude, only update bias slightly
                    x_eskf.nom.gyro_bias = 0.3 * fgo_state.gyro_bias + 0.7 * x_eskf.nom.gyro_bias
                    cov_scale = 0.95
                    logger.info(f"Reduced FGO correction weight (no star tracker)")

        # Scale covariance based on correction magnitude
        if correction_deg > 1.0:
            cov_scale *= 0.5
        elif correction_deg < 0.01:
            cov_scale = 0.95

        x_eskf.err.cov = cov_scale * x_eskf.err.cov
        x_eskf.err.cov = 0.5 * (x_eskf.err.cov + x_eskf.err.cov.T)  # Ensure symmetry

        # Reset error mean
        x_eskf.err.mean[:] = 0.0

        return x_eskf, correction_deg

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

        # Initialize FGO state if needed
        if self.last_fgo_state is None:
            self.initialize(t, x_eskf.nom.ori)

        # --- Update rate mode for adaptive FGO triggering ---
        self._update_rate_mode(omega_meas)

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

        # --- Trim window to duration ---
        # Remove old samples outside window duration
        while len(self.window) > 1 and (t - self.window[0].t) > self.fgo_window_duration:
            self.window.popleft()

        # --- FGO Optimization (periodic batch) ---
        if self.should_optimize_fgo(t):
            n_keyframes = self._count_keyframes()
            n_star_tracker = self._count_star_tracker_in_window()
            logger.info(f"Batch smoothing at t={t:.2f}s ({n_keyframes} keyframes, {n_star_tracker} ST)")

            # Pass current ESKF state for initialization on first run
            optimized_state, fgo_final_error = self.run_batch_fgo(current_eskf_state=x_eskf.nom)

            if optimized_state is not None:
                # Feed back to ESKF with star tracker count for confidence weighting
                x_eskf, correction_deg = self.feedback_fgo_to_eskf(
                    x_eskf, optimized_state, n_star_tracker=n_star_tracker
                )

                # Update FGO state history
                self.last_fgo_state = optimized_state
                self.last_fgo_time = t
                self.fgo_count += 1
                self.correction_history.append(correction_deg)
                fgo_updated = True

                logger.info(f"Batch smoothing applied correction from {n_keyframes} keyframes")
            else:
                # FGO failed, just update timestamp
                self.last_fgo_time = t

        return x_eskf, fgo_updated

    def _initialize_isam2(self, t0: float, q_init: Quaternion):
        """Initialize iSAM2 for redundant operation."""
        import gtsam
        from gtsam import ISAM2, ISAM2Params, NonlinearFactorGraph, Values, Point3

        # Create iSAM2 instance
        isam2_params = ISAM2Params()
        isam2_params.setRelinearizeThreshold(0.001)
        isam2_params.relinearizeSkip = 1

        self.isam2 = ISAM2(isam2_params)
        self.isam2_keyframe_idx = 0

        # Create factor builders (reuse from KeyframeFGO)
        from estimation.keyframe_fgo import KeyframeFactorBuilders, RK4Preintegration
        from utilities.process_model import ProcessModel

        process = ProcessModel(self.config_path)
        gyro_cov = np.eye(3) * (process.sigma_g ** 2 + (0.10 * 0.0175)**2)

        from utilities.utils import load_yaml
        config = load_yaml(self.config_path)

        self.isam2_factors = KeyframeFactorBuilders(
            gyro_cov=gyro_cov,
            sigma_mag=config["sensors"]["mag"]["mag_std"],
            sigma_sun=config["sensors"]["sun"]["noise"]["sun_std"],
            sigma_star=config["sensors"]["star"]["noise"]["st_std"],
            sigma_bias=process.sigma_bg,
            use_robust=self.use_robust,
        )

        # Initialize first node
        R0 = rot3_from_quat(q_init)
        b0 = np.zeros(3)

        graph0 = NonlinearFactorGraph()
        values0 = Values()

        key_R = gtsam.Symbol('x', 0).key()
        key_B = gtsam.Symbol('b', 0).key()

        values0.insert(key_R, R0)
        values0.insert(key_B, Point3(*b0))

        # Add priors
        prior_R_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
        graph0.add(gtsam.PriorFactorRot3(key_R, R0, prior_R_noise))
        graph0.add(self.isam2_factors.make_bias_prior(key_B, b0, 1e-4))

        self.isam2.update(graph0, values0)
        self.isam2_current_estimate = self.isam2.calculateEstimate()

        # Preintegration state
        self.isam2_preint = self.isam2_factors.create_rk4_preintegration(b0)
        self.isam2_last_key_R = key_R
        self.isam2_last_key_B = key_B
        self.isam2_bias = b0.copy()

        self.isam2_initialized = True
        self.isam2_state = NominalState(ori=q_init.copy(), gyro_bias=b0.copy())

        logger.info(f"iSAM2 redundant estimator initialized at t={t0:.2f}s")

    def _update_isam2(
        self,
        t: float,
        jd: float,
        omega_meas: np.ndarray,
        dt: float,
        z_mag: Optional[np.ndarray] = None,
        z_sun: Optional[np.ndarray] = None,
        z_st: Optional[Quaternion] = None,
        B_n: Optional[np.ndarray] = None,
        s_n: Optional[np.ndarray] = None,
    ) -> Optional[NominalState]:
        """
        Update iSAM2 redundant estimator incrementally.

        Returns the current iSAM2 state estimate.
        """
        import gtsam
        from gtsam import NonlinearFactorGraph, Values, Point3

        if not self.isam2_initialized:
            return None

        # Integrate gyro measurement
        self.isam2_preint.integrate(omega_meas, dt)

        # Check if we have a measurement to create a new keyframe
        has_measurement = z_mag is not None or z_sun is not None or z_st is not None

        if has_measurement:
            self.isam2_keyframe_idx += 1

            key_R = gtsam.Symbol('x', self.isam2_keyframe_idx).key()
            key_B = gtsam.Symbol('b', self.isam2_keyframe_idx).key()

            # Predict state
            R_prev = self.isam2_current_estimate.atRot3(self.isam2_last_key_R)
            R_pred = R_prev.compose(self.isam2_preint.deltaRij())

            graph_inc = NonlinearFactorGraph()
            values_inc = Values()

            values_inc.insert(key_R, R_pred)
            values_inc.insert(key_B, Point3(*self.isam2_bias))

            # Add preintegration factor
            graph_inc.add(self.isam2_factors.make_between_factor_rot3(
                self.isam2_last_key_R, key_R, self.isam2_preint
            ))

            # Add bias random walk
            graph_inc.add(self.isam2_factors.make_bias_between(
                self.isam2_last_key_B, key_B, self.isam2_preint.deltaTij()
            ))

            # Add measurement factors
            if z_mag is not None and B_n is not None:
                graph_inc.add(self.isam2_factors.make_magnetometer_factor(key_R, z_mag, B_n))
            if z_sun is not None and s_n is not None:
                graph_inc.add(self.isam2_factors.make_sun_factor(key_R, z_sun, s_n))
            if z_st is not None:
                graph_inc.add(self.isam2_factors.make_star_factor(key_R, z_st))

            # Update iSAM2
            self.isam2.update(graph_inc, values_inc)
            self.isam2_current_estimate = self.isam2.calculateEstimate()

            # Extract state
            R_est = self.isam2_current_estimate.atRot3(key_R)
            b_est = np.array(self.isam2_current_estimate.atPoint3(key_B))

            self.isam2_state = NominalState(
                ori=quat_from_rot3(R_est),
                gyro_bias=b_est,
            )

            # Update for next iteration
            self.isam2_last_key_R = key_R
            self.isam2_last_key_B = key_B
            self.isam2_bias = b_est.copy()
            self.isam2_preint = self.isam2_factors.create_rk4_preintegration(self.isam2_bias)

        return self.isam2_state

    def compute_disagreement(self, eskf_state: NominalState, isam2_state: NominalState) -> float:
        """
        Compute disagreement between ESKF and iSAM2 estimates in degrees.

        This is useful for fault detection - large disagreement may indicate
        one estimator has failed.
        """
        q_eskf = eskf_state.ori
        q_isam2 = isam2_state.ori

        # Compute rotation between estimates
        dq = q_eskf.conjugate() @ q_isam2
        angle = 2 * np.arccos(np.clip(abs(dq.mu), 0, 1))

        return np.rad2deg(angle)

    def select_output(
        self,
        eskf_state: EskfState,
        isam2_state: Optional[NominalState],
        disagreement_deg: float,
    ) -> Tuple[NominalState, str]:
        """
        Select which estimator output to use based on diagnostics.

        Selection criteria:
        1. If disagreement is small (<1°), trust ESKF (lower latency)
        2. If ESKF covariance is large, consider iSAM2
        3. If iSAM2 unavailable, use ESKF

        Returns:
            (selected_state, source): The selected state and which estimator it came from
        """
        if isam2_state is None:
            return eskf_state.nom, "ESKF"

        # Check ESKF covariance (attitude part)
        att_cov_trace = np.trace(eskf_state.err.cov[:3, :3])
        att_std_deg = np.rad2deg(np.sqrt(att_cov_trace / 3))

        # Selection logic
        if disagreement_deg < 1.0:
            # Good agreement - use ESKF (faster updates)
            return eskf_state.nom, "ESKF"
        elif att_std_deg > 5.0:
            # ESKF uncertain - consider iSAM2
            logger.info(f"ESKF uncertainty high ({att_std_deg:.1f}°), using iSAM2")
            return isam2_state, "iSAM2"
        else:
            # Disagreement but ESKF seems confident - flag for investigation
            logger.warning(f"Estimator disagreement: {disagreement_deg:.2f}° (ESKF σ={att_std_deg:.2f}°)")
            return eskf_state.nom, "ESKF (flagged)"

    def get_statistics(self) -> dict:
        """Get hybrid estimator statistics."""
        stats = {
            "fgo_count": self.fgo_count,
            "mean_correction_deg": np.mean(self.correction_history) if self.correction_history else 0.0,
            "max_correction_deg": np.max(self.correction_history) if self.correction_history else 0.0,
            "window_size": len(self.window),
            "primary_mode": self.primary_mode.name,
            "mode_switches": len(self.mode_switches),
        }
        if self.isam2_initialized:
            stats["isam2_keyframes"] = self.isam2_keyframe_idx
        return stats

    def step_redundant(
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
    ) -> Tuple[EskfState, Optional[NominalState], float, str]:
        """
        Perform one step with redundant estimation (ESKF + iSAM2 in parallel).

        This method:
        1. Updates ESKF (always)
        2. Updates iSAM2 incrementally (always)
        3. Runs batch FGO smoothing periodically (corrects ESKF)
        4. Computes disagreement between ESKF and iSAM2
        5. Selects output based on diagnostics

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
            (eskf_state, isam2_state, disagreement_deg, selected_source):
            - Updated ESKF state
            - Current iSAM2 state (or None)
            - Disagreement between estimators in degrees
            - Which estimator was selected for output
        """
        # Initialize if needed
        if self.last_fgo_state is None:
            self.initialize(t, x_eskf.nom.ori)

        if self.enable_isam2_fallback and not self.isam2_initialized:
            self._initialize_isam2(t, x_eskf.nom.ori)

        # --- Update rate mode for adaptive FGO triggering ---
        self._update_rate_mode(omega_meas)

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
            # Track star tracker updates for mode switching
            self.star_tracker_count_since_switch += 1

        # --- iSAM2 Update (parallel) ---
        isam2_state = None
        if self.enable_isam2_fallback and self.isam2_initialized:
            isam2_state = self._update_isam2(
                t, jd, omega_meas, dt,
                z_mag, z_sun, z_st, B_n, s_n
            )

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

        # --- Trim window to duration ---
        while len(self.window) > 1 and (t - self.window[0].t) > self.fgo_window_duration:
            self.window.popleft()

        # --- Batch FGO Smoothing (periodic) ---
        if self.should_optimize_fgo(t):
            n_keyframes = self._count_keyframes()
            n_star_tracker = self._count_star_tracker_in_window()
            logger.info(f"Batch smoothing at t={t:.2f}s ({n_keyframes} keyframes, {n_star_tracker} ST)")

            optimized_state, fgo_final_error = self.run_batch_fgo(current_eskf_state=x_eskf.nom)

            if optimized_state is not None:
                x_eskf, correction_deg = self.feedback_fgo_to_eskf(
                    x_eskf, optimized_state, n_star_tracker=n_star_tracker
                )
                self.last_fgo_state = optimized_state
                self.last_fgo_time = t
                self.fgo_count += 1
                self.correction_history.append(correction_deg)
                logger.info(f"Batch smoothing applied correction from {n_keyframes} keyframes")
            else:
                self.last_fgo_time = t

        # --- Compute disagreement ---
        disagreement_deg = 0.0
        selected_source = "ESKF"

        if isam2_state is not None:
            disagreement_deg = self.compute_disagreement(x_eskf.nom, isam2_state)
            _, selected_source = self.select_output(x_eskf, isam2_state, disagreement_deg)

        return x_eskf, isam2_state, disagreement_deg, selected_source
