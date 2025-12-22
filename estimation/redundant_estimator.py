"""
Redundant Attitude Estimator: ESKF + Fixed-Lag Smoother with Fault Detection.

This module implements a redundant estimation architecture for spacecraft attitude:

Architecture:
1. ESKF = primary estimator for real-time control (low latency)
2. IncrementalFixedLagSmoother = parallel estimator with proper marginalization
3. Disagreement monitor = track ||q_ESKF - q_smoother|| over time
4. Switching logic = if disagreement exceeds threshold for N consecutive updates, switch primary
5. Recovery logic = switch back to ESKF when disagreement naturally drops (ESKF re-converges)

Key design principles:
- Both estimators run in parallel, always
- Disagreement history is tracked to avoid spurious switches
- When switching to smoother, ESKF is NOT reset (continues running with current state)
- This ensures disagreement stays high during persistent faults (no premature switch-back)
- Switch back to ESKF only when ESKF naturally re-converges (fault cleared)

Uses QUEST algorithm for deterministic attitude initialization from vector measurements.
Right-multiply quaternion convention (GTSAM compatible).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from collections import deque
from enum import Enum, auto

import numpy as np

from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from logging_config import get_logger

logger = get_logger(__name__)


class PrimaryEstimator(Enum):
    """Which estimator is currently primary."""
    ESKF = auto()
    SMOOTHER = auto()
    CONSERVATIVE = auto()  # Fallback: use ESKF but with inflated covariance


@dataclass
class DisagreementEvent:
    """Record of a disagreement event."""
    time: float
    disagreement_deg: float
    exceeds_threshold: bool


class RedundantEstimator:
    """
    Redundant attitude estimator combining ESKF and Fixed-Lag Smoother.

    Architecture:
    - ESKF runs continuously for real-time estimates (low latency)
    - Fixed-lag smoother runs in parallel with automatic marginalization
    - Disagreement is tracked over time for fault detection
    - If disagreement exceeds threshold for N consecutive measurements,
      switch output to smoother (ESKF continues running in background)
    - Switch back to ESKF when disagreement naturally drops (ESKF re-converges)

    This provides:
    1. Real-time estimates from ESKF (primary in normal operation)
    2. Fault detection via disagreement monitoring
    3. Bounded output via smoother when ESKF diverges
    4. Automatic recovery when fault clears and ESKF re-converges
    """

    def __init__(
        self,
        P0: np.ndarray,
        config_path: str = "configs/config_baseline_short.yaml",
        # Fixed-lag smoother parameters
        smoother_lag: float = 60.0,  # seconds
        use_robust: bool = True,
        robust_kernel: str = "huber",  # huber, cauchy, etc.
        robust_param: float = 0.1,     # kernel parameter (k for Huber)
        normalize_mag: bool = True,
        # Disagreement monitoring parameters
        disagreement_threshold_deg: float = 2.0,  # degrees
        consecutive_disagreements_to_switch: int = 5,  # N measurements
        disagreement_history_length: int = 100,  # samples to track
        # Recovery parameters
        consecutive_agreements_to_recover: int = 10,  # N measurements
        agreement_threshold_deg: float = 0.5,  # degrees
        # Innovation-based health monitoring
        nis_threshold: float = 5.0,  # NIS threshold for "healthy" ESKF (chi2 with 3 DOF)
        nis_history_length: int = 20,  # Number of NIS samples to track
        eskf_healthy_fraction: float = 0.8,  # Fraction of NIS below threshold to consider ESKF healthy
    ):
        """
        Initialize redundant estimator with innovation-based health monitoring.

        Args:
            P0: Initial error covariance (6x6)
            config_path: Path to configuration file
            smoother_lag: Duration of fixed-lag window (seconds)
            use_robust: Use Huber M-estimator in smoother
            normalize_mag: If True, normalize magnetometer measurements
            disagreement_threshold_deg: Switch threshold (degrees)
            consecutive_disagreements_to_switch: Number of consecutive disagreements to trigger switch
            disagreement_history_length: Number of disagreement samples to track
            consecutive_agreements_to_recover: Number of agreements to switch back to ESKF
            agreement_threshold_deg: Threshold for "agreement" (degrees)
            nis_threshold: NIS threshold for healthy ESKF
            nis_history_length: Number of NIS samples to average
            eskf_healthy_fraction: Fraction of recent NIS below threshold to consider ESKF healthy
        """
        self.config_path = config_path
        self.P0 = P0

        # Thresholds
        self.disagreement_threshold_deg = disagreement_threshold_deg
        self.consecutive_disagreements_to_switch = consecutive_disagreements_to_switch
        self.consecutive_agreements_to_recover = consecutive_agreements_to_recover
        self.agreement_threshold_deg = agreement_threshold_deg

        # Innovation health monitoring parameters
        self.nis_threshold = nis_threshold
        self.eskf_healthy_fraction = eskf_healthy_fraction

        # Initialize ESKF
        self.eskf = ESKF(P0=P0, config_path=config_path)

        # Initialize Fixed-Lag Smoother
        self.smoother = FixedLagAttitudeSmoother(
            config_path=config_path,
            lag=smoother_lag,
            use_robust=use_robust,
            robust_kernel=robust_kernel,
            robust_param=robust_param,
            normalize_mag=normalize_mag,
        )

        # State tracking
        self.primary = PrimaryEstimator.ESKF
        self.initialized = False

        # Disagreement history
        self.disagreement_history: deque[DisagreementEvent] = deque(
            maxlen=disagreement_history_length
        )
        self.consecutive_disagreements = 0
        self.consecutive_agreements = 0

        # NIS (Normalized Innovation Squared) history for ESKF health monitoring
        # Per-sensor tracking allows identifying which sensor is faulty
        self.nis_history: deque[float] = deque(maxlen=nis_history_length)  # Combined (legacy)
        self.nis_history_mag: deque[float] = deque(maxlen=nis_history_length)
        self.nis_history_sun: deque[float] = deque(maxlen=nis_history_length)
        self.nis_history_st: deque[float] = deque(maxlen=nis_history_length)

        # Statistics
        self.switch_events: List[Tuple[float, str]] = []
        self.smoother_switch_time: Optional[float] = None  # Time when we switched to smoother
        self.total_disagreements = 0
        self.max_disagreement_deg = 0.0

        logger.info(f"RedundantEstimator initialized:")
        logger.info(f"  Smoother lag: {smoother_lag}s")
        logger.info(f"  Disagreement threshold: {disagreement_threshold_deg} deg")
        logger.info(f"  Consecutive to switch: {consecutive_disagreements_to_switch}")
        logger.info(f"  Consecutive to recover: {consecutive_agreements_to_recover}")
        logger.info(f"  NIS threshold: {nis_threshold}")
        logger.info(f"  ESKF healthy fraction: {eskf_healthy_fraction}")

    def initialize(self, t0: float, q_init: Quaternion, b_init: Optional[np.ndarray] = None):
        """
        Initialize both estimators.

        Args:
            t0: Initial time
            q_init: Initial attitude
            b_init: Initial gyro bias (default: zeros)
        """
        b_init = b_init if b_init is not None else np.zeros(3)

        # Initialize smoother
        self.smoother.initialize(t0, q_init, b_init)

        self.initialized = True
        self.primary = PrimaryEstimator.ESKF

        logger.info(f"RedundantEstimator initialized at t={t0:.2f}s")

    def compute_disagreement(self, q_eskf: Quaternion, q_smoother: Quaternion) -> float:
        """
        Compute disagreement between ESKF and smoother estimates.

        Args:
            q_eskf: ESKF attitude estimate
            q_smoother: Smoother attitude estimate

        Returns:
            Disagreement angle in degrees
        """
        # Compute rotation between estimates: dq = q_eskf^{-1} * q_smoother
        dq = q_eskf.conjugate() @ q_smoother
        angle = 2 * np.arccos(np.clip(abs(dq.mu), 0, 1))
        return np.rad2deg(angle)

    def _is_eskf_healthy(self) -> bool:
        """
        Check if ESKF is healthy based on recent innovation history.

        Returns True if the fraction of recent NIS values below threshold
        exceeds eskf_healthy_fraction.
        """
        if len(self.nis_history) < 5:
            # Not enough history, assume healthy
            return True

        healthy_count = sum(1 for nis in self.nis_history if nis < self.nis_threshold)
        fraction_healthy = healthy_count / len(self.nis_history)
        return fraction_healthy >= self.eskf_healthy_fraction

    def _get_eskf_health_score(self) -> float:
        """
        Get a health score for ESKF (0 = unhealthy, 1 = healthy).
        """
        if len(self.nis_history) < 3:
            return 1.0  # Assume healthy with insufficient data

        healthy_count = sum(1 for nis in self.nis_history if nis < self.nis_threshold)
        return healthy_count / len(self.nis_history)

    def _sensor_healthy(self, nis_history: deque, min_samples: int = 5) -> bool:
        """
        Check if a specific sensor is healthy based on its NIS history.

        Args:
            nis_history: NIS history deque for the sensor
            min_samples: Minimum samples required for reliable assessment

        Returns:
            True if the sensor appears healthy, False otherwise.
            Returns True if insufficient data (assume healthy).
        """
        if len(nis_history) < min_samples:
            return True  # Not enough history, assume healthy

        healthy_count = sum(1 for nis in nis_history if nis < self.nis_threshold)
        fraction_healthy = healthy_count / len(nis_history)
        return fraction_healthy >= self.eskf_healthy_fraction

    def _get_sensor_health_scores(self) -> dict:
        """
        Get health scores for each sensor type.

        Returns:
            Dict with health scores (0-1) for 'mag', 'sun', 'st'.
            Score of 1.0 means healthy, 0.0 means unhealthy.
        """
        def compute_score(history: deque) -> float:
            if len(history) < 3:
                return 1.0
            healthy = sum(1 for nis in history if nis < self.nis_threshold)
            return healthy / len(history)

        return {
            "mag": compute_score(self.nis_history_mag),
            "sun": compute_score(self.nis_history_sun),
            "st": compute_score(self.nis_history_st),
        }

    def _get_faulty_sensor(self) -> Optional[str]:
        """
        Identify which sensor has degraded NIS (if any).

        Logic:
        - A sensor is "the outlier" if it's unhealthy while others are healthy
        - If multiple sensors are unhealthy, returns None (systematic issue)
        - If all sensors are healthy, returns None

        Returns:
            "magnetometer", "sun_sensor", "star_tracker", or None
        """
        mag_healthy = self._sensor_healthy(self.nis_history_mag)
        sun_healthy = self._sensor_healthy(self.nis_history_sun)
        st_healthy = self._sensor_healthy(self.nis_history_st)

        # Count unhealthy sensors
        unhealthy_count = sum(1 for h in [mag_healthy, sun_healthy, st_healthy] if not h)

        if unhealthy_count == 0:
            return None  # All healthy

        if unhealthy_count >= 2:
            return None  # Multiple unhealthy = systematic issue

        # Exactly one unhealthy sensor - identify it
        if not st_healthy and mag_healthy and sun_healthy:
            return "star_tracker"
        elif not mag_healthy and st_healthy and sun_healthy:
            return "magnetometer"
        elif not sun_healthy and st_healthy and mag_healthy:
            return "sun_sensor"

        return None  # Shouldn't reach here

    def _update_disagreement_tracking(self, t: float, disagreement_deg: float):
        """
        Update disagreement history and consecutive counters.

        Args:
            t: Current time
            disagreement_deg: Current disagreement in degrees
        """
        exceeds_threshold = disagreement_deg > self.disagreement_threshold_deg
        is_agreement = disagreement_deg < self.agreement_threshold_deg

        # Record event
        event = DisagreementEvent(
            time=t,
            disagreement_deg=disagreement_deg,
            exceeds_threshold=exceeds_threshold,
        )
        self.disagreement_history.append(event)

        # Update statistics
        if exceeds_threshold:
            self.total_disagreements += 1
        self.max_disagreement_deg = max(self.max_disagreement_deg, disagreement_deg)

        # Update consecutive counters
        if exceeds_threshold:
            self.consecutive_disagreements += 1
            self.consecutive_agreements = 0
        elif is_agreement:
            self.consecutive_agreements += 1
            self.consecutive_disagreements = 0
        else:
            # In between - reset both
            self.consecutive_disagreements = 0
            self.consecutive_agreements = 0

    def _check_for_switch(self, t: float, x_eskf: EskfState) -> bool:
        """
        Check if we should switch primary estimator using per-sensor innovation-based logic.

        Per-sensor logic:
        1. If star tracker NIS is poor but mag/sun are good → trust smoother (ST faulty)
        2. If magnetometer NIS is poor but ST/sun are good → trust ESKF (ST dominates)
        3. If sun sensor NIS is poor but ST/mag are good → trust ESKF (marginal impact)
        4. If multiple sensors unhealthy → conservative mode (systematic issue)
        5. If all sensors healthy but disagreement → conservative mode (smoother issue)

        Returns True if a switch occurred.
        """
        switched = False
        eskf_healthy = self._is_eskf_healthy()
        health_score = self._get_eskf_health_score()
        faulty_sensor = self._get_faulty_sensor()
        sensor_scores = self._get_sensor_health_scores()

        if self.primary == PrimaryEstimator.ESKF:
            # Check if we should switch away from ESKF
            if self.consecutive_disagreements >= self.consecutive_disagreements_to_switch:
                if faulty_sensor == "star_tracker":
                    # Star tracker is faulty → smoother might be corrupted too,
                    # but ESKF has poor innovations. Trust smoother if it uses
                    # other sensors, otherwise conservative.
                    logger.warning(
                        f"Switching to SMOOTHER at t={t:.2f}s "
                        f"(star tracker faulty, scores: mag={sensor_scores['mag']:.2f}, "
                        f"sun={sensor_scores['sun']:.2f}, st={sensor_scores['st']:.2f})"
                    )
                    self._switch_to_smoother(t, x_eskf)
                    switched = True
                elif faulty_sensor == "magnetometer":
                    # Magnetometer is faulty but ST is good → ESKF should be fine
                    # since star tracker dominates. Use conservative mode.
                    logger.warning(
                        f"Entering CONSERVATIVE mode at t={t:.2f}s "
                        f"(magnetometer faulty, ST dominates: scores: mag={sensor_scores['mag']:.2f}, "
                        f"sun={sensor_scores['sun']:.2f}, st={sensor_scores['st']:.2f})"
                    )
                    self._switch_to_conservative(t)
                    switched = True
                elif faulty_sensor == "sun_sensor":
                    # Sun sensor faulty but ST good → ESKF should be fine
                    logger.warning(
                        f"Entering CONSERVATIVE mode at t={t:.2f}s "
                        f"(sun sensor faulty: scores: mag={sensor_scores['mag']:.2f}, "
                        f"sun={sensor_scores['sun']:.2f}, st={sensor_scores['st']:.2f})"
                    )
                    self._switch_to_conservative(t)
                    switched = True
                elif not eskf_healthy:
                    # Multiple sensors unhealthy or systematic issue
                    logger.warning(
                        f"Entering CONSERVATIVE mode at t={t:.2f}s "
                        f"(multiple sensors unhealthy, health={health_score:.2f}, "
                        f"scores: mag={sensor_scores['mag']:.2f}, "
                        f"sun={sensor_scores['sun']:.2f}, st={sensor_scores['st']:.2f})"
                    )
                    self._switch_to_conservative(t)
                    switched = True
                else:
                    # All sensors healthy but disagreement → smoother likely wrong
                    logger.warning(
                        f"Entering CONSERVATIVE mode at t={t:.2f}s "
                        f"(all sensors healthy but disagreement, health={health_score:.2f})"
                    )
                    self._switch_to_conservative(t)
                    switched = True

        elif self.primary == PrimaryEstimator.SMOOTHER:
            # Check if we should switch back to ESKF
            # Require: consecutive agreements AND minimum time in smoother AND healthy sensors
            time_in_smoother = t - self.smoother_switch_time if self.smoother_switch_time else 0
            min_smoother_duration = 30.0  # seconds

            if (self.consecutive_agreements >= self.consecutive_agreements_to_recover
                and time_in_smoother >= min_smoother_duration
                and self._is_eskf_healthy()):
                logger.info(
                    f"Switching back to ESKF at t={t:.2f}s "
                    f"({self.consecutive_agreements} agreements, {time_in_smoother:.1f}s in smoother, "
                    f"ESKF health={self._get_eskf_health_score():.2f})"
                )
                self._switch_to_eskf(t)
                switched = True

        elif self.primary == PrimaryEstimator.CONSERVATIVE:
            # In conservative mode: check if we can return to normal ESKF
            if self.consecutive_agreements >= self.consecutive_agreements_to_recover:
                logger.info(
                    f"Exiting CONSERVATIVE mode at t={t:.2f}s "
                    f"({self.consecutive_agreements} consecutive agreements)"
                )
                self._switch_to_eskf(t)
                switched = True
            elif faulty_sensor == "star_tracker" and not self._sensor_healthy(self.nis_history_st):
                # Star tracker became definitively faulty, switch to smoother
                logger.warning(
                    f"CONSERVATIVE → SMOOTHER at t={t:.2f}s "
                    f"(star tracker definitively faulty, st_score={sensor_scores['st']:.2f})"
                )
                self._switch_to_smoother(t, x_eskf)
                switched = True

        return switched

    def _switch_to_smoother(self, t: float, x_eskf: EskfState) -> EskfState:
        """
        Switch primary estimator to smoother and reset ESKF state.

        When switching to smoother, we reset the ESKF state to match the smoother's
        estimate. This allows the ESKF to start fresh with a good state, enabling
        proper NIS tracking and eventual recovery.

        We also clear the NIS histories to prevent stale bad values from affecting
        future health assessments.
        """
        smoother_state = self.smoother.get_state()
        if smoother_state is None:
            logger.warning("Cannot switch to smoother - no state available")
            return x_eskf

        # Update primary (output will now come from smoother)
        self.primary = PrimaryEstimator.SMOOTHER
        self.smoother_switch_time = t  # Track when we switched

        # Record event
        self.switch_events.append((t, "ESKF->SMOOTHER"))

        # Reset counters
        self.consecutive_disagreements = 0
        self.consecutive_agreements = 0

        # Reset ESKF state to match smoother estimate
        x_eskf.nom.ori = smoother_state.ori.copy()
        x_eskf.nom.gyro_bias = smoother_state.gyro_bias.copy()
        # Keep covariance but could optionally reset it too

        # Clear NIS histories to start fresh
        self.nis_history.clear()
        self.nis_history_mag.clear()
        self.nis_history_sun.clear()
        self.nis_history_st.clear()

        logger.info(f"Switched to smoother output (ESKF state reset to smoother estimate)")

        return x_eskf

    def _switch_to_conservative(self, t: float):
        """
        Switch to conservative mode.

        In conservative mode, we use the ESKF output but flag that there is
        uncertainty. This is used when disagreement is high but ESKF innovations
        are healthy (suggesting the smoother is wrong, not the ESKF).
        """
        self.primary = PrimaryEstimator.CONSERVATIVE

        # Record event
        self.switch_events.append((t, "ESKF->CONSERVATIVE"))

        # Reset counters
        self.consecutive_disagreements = 0
        self.consecutive_agreements = 0

        logger.info(f"Entered conservative mode (trusting ESKF despite disagreement)")

    def _switch_to_eskf(self, t: float):
        """Switch primary estimator back to ESKF."""
        prev_mode = self.primary.name
        self.primary = PrimaryEstimator.ESKF
        self.smoother_switch_time = None  # Clear smoother switch time

        # Record event
        self.switch_events.append((t, f"{prev_mode}->ESKF"))

        # Reset counters
        self.consecutive_disagreements = 0
        self.consecutive_agreements = 0

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
        s_n: Optional[np.ndarray] = None,
    ) -> Tuple[EskfState, NominalState, float, str]:
        """
        Perform one step of redundant estimation.

        Both ESKF and smoother are updated. Disagreement is computed and
        tracked. If conditions warrant, primary estimator may switch.

        Args:
            x_eskf: Current ESKF state
            t: Current time (seconds)
            jd: Julian date
            omega_meas: Gyro measurement (rad/s)
            dt: Time step (seconds)
            z_mag: Magnetometer measurement (optional)
            z_sun: Sun sensor measurement (optional)
            z_st: Star tracker measurement (optional)
            B_n: Magnetic field reference vector (optional)
            s_n: Sun direction reference vector (optional)

        Returns:
            Tuple of:
            - Updated ESKF state
            - Current smoother state
            - Disagreement in degrees
            - Selected primary ("ESKF" or "SMOOTHER")
        """
        # Initialize if needed
        if not self.initialized:
            self.initialize(t, x_eskf.nom.ori, x_eskf.nom.gyro_bias)

        # === ESKF Update ===
        # Prediction
        x_eskf = self.eskf.predict(x_eskf, omega_meas, dt)

        # Measurement updates and NIS tracking (per-sensor and combined)
        # Record NIS even when measurements are rejected for fault detection
        if z_mag is not None:
            try:
                x_eskf = self.eskf.update(x_eskf, z_mag, SensorType.MAGNETOMETER, B_n=B_n)
            except ValueError:
                pass  # Measurement rejected by chi-squared gate
            # Record NIS regardless of acceptance (ESKF stores it before rejection check)
            self.nis_history.append(self.eskf.last_nis)
            self.nis_history_mag.append(self.eskf.last_nis)

        if z_sun is not None:
            try:
                x_eskf = self.eskf.update(x_eskf, z_sun, SensorType.SUN_VECTOR, s_n=s_n)
            except ValueError:
                pass
            self.nis_history.append(self.eskf.last_nis)
            self.nis_history_sun.append(self.eskf.last_nis)

        if z_st is not None:
            try:
                x_eskf = self.eskf.update(x_eskf, z_st, SensorType.STAR_TRACKER)
            except ValueError:
                pass
            self.nis_history.append(self.eskf.last_nis)
            self.nis_history_st.append(self.eskf.last_nis)

        # === Smoother Update ===
        # Integrate gyro (pass time to enable auto-keyframe during measurement gaps)
        self.smoother.integrate_gyro(omega_meas, dt, t=t)

        # Add measurements if present
        smoother_state = None
        has_measurement = z_mag is not None or z_sun is not None or z_st is not None

        if has_measurement:
            smoother_state = self.smoother.add_measurement(
                t=t,
                jd=jd,
                z_mag=z_mag,
                z_sun=z_sun,
                z_st=z_st,
                B_eci=B_n,
                s_eci=s_n,
            )
        else:
            # Use propagated state between measurements for smooth output
            smoother_state = self.smoother.get_propagated_state()

        # === Disagreement Computation ===
        disagreement_deg = 0.0
        if smoother_state is not None:
            disagreement_deg = self.compute_disagreement(x_eskf.nom.ori, smoother_state.ori)

            # Update tracking (only when we have a measurement for meaningful comparison)
            if has_measurement:
                self._update_disagreement_tracking(t, disagreement_deg)

                # Check for switch
                self._check_for_switch(t, x_eskf)

        # === Determine Output ===
        if self.primary == PrimaryEstimator.ESKF:
            primary_str = "ESKF"
        elif self.primary == PrimaryEstimator.SMOOTHER:
            primary_str = "SMOOTHER"
        else:  # CONSERVATIVE
            primary_str = "CONSERVATIVE"

        # If smoother_state is None, create a dummy
        if smoother_state is None:
            smoother_state = NominalState(
                ori=x_eskf.nom.ori.copy(),
                gyro_bias=x_eskf.nom.gyro_bias.copy(),
            )

        return x_eskf, smoother_state, disagreement_deg, primary_str

    def get_primary_state(self, x_eskf: EskfState) -> NominalState:
        """
        Get the state from the current primary estimator.

        Args:
            x_eskf: Current ESKF state

        Returns:
            State from primary estimator
        """
        if self.primary == PrimaryEstimator.ESKF or self.primary == PrimaryEstimator.CONSERVATIVE:
            # In both ESKF and CONSERVATIVE modes, use ESKF output
            return x_eskf.nom
        else:
            # Use propagated state for smooth output between measurements
            state = self.smoother.get_propagated_state()
            if state is None:
                return x_eskf.nom
            return state

    def get_statistics(self) -> dict:
        """Get redundant estimator statistics."""
        smoother_stats = self.smoother.get_statistics()

        recent_disagreements = list(self.disagreement_history)
        avg_disagreement = (
            np.mean([e.disagreement_deg for e in recent_disagreements])
            if recent_disagreements else 0.0
        )

        sensor_scores = self._get_sensor_health_scores()
        faulty_sensor = self._get_faulty_sensor()

        return {
            "primary": self.primary.name,
            "total_switch_events": len(self.switch_events),
            "switch_events": self.switch_events,
            "total_disagreements_over_threshold": self.total_disagreements,
            "max_disagreement_deg": self.max_disagreement_deg,
            "avg_recent_disagreement_deg": avg_disagreement,
            "consecutive_disagreements": self.consecutive_disagreements,
            "consecutive_agreements": self.consecutive_agreements,
            "smoother": smoother_stats,
            # Per-sensor health info
            "sensor_health_scores": sensor_scores,
            "faulty_sensor": faulty_sensor,
            "eskf_health_score": self._get_eskf_health_score(),
        }

    def get_disagreement_history(self) -> List[Tuple[float, float, bool]]:
        """
        Get disagreement history.

        Returns:
            List of (time, disagreement_deg, exceeds_threshold) tuples
        """
        return [
            (e.time, e.disagreement_deg, e.exceeds_threshold)
            for e in self.disagreement_history
        ]
