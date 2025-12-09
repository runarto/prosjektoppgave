"""
Keyframe-based Factor Graph Optimization with Gyro Preintegration.

Uses RK4 preintegration by default for higher accuracy with time-varying angular velocity.

Key design principles:
1. Create attitude nodes ONLY when measurements arrive (mag, sun, or star)
2. Preintegrate gyro measurements between keyframes using RK4
3. One bias state per keyframe with random walk between them

This dramatically reduces graph size:
- Original: ~50 nodes/second (one per gyro sample)
- Keyframe: ~5 nodes/second (one per measurement)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
import numpy as np
import gtsam
from gtsam import Rot3, Point3, Symbol, NonlinearFactorGraph, Values, ISAM2, ISAM2Params

from utilities.quaternion import Quaternion
from utilities.states import NominalState
from utilities.utils import load_yaml
from utilities.process_model import ProcessModel
from environment.environment import OrbitEnvironmentModel
from logging_config import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Utility Functions
# ==============================================================================

def skew(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from 3-vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def rot3_from_quat(q: Quaternion) -> Rot3:
    """Convert Quaternion to gtsam.Rot3."""
    return Rot3.Quaternion(q.mu, *(q.eta.reshape(3)))


def quat_from_rot3(R: Rot3) -> Quaternion:
    """Convert gtsam.Rot3 to Quaternion."""
    q = R.toQuaternion()
    return Quaternion(mu=q.w(), eta=np.array([q.x(), q.y(), q.z()])).normalize()


# ==============================================================================
# Keyframe
# ==============================================================================

@dataclass
class Keyframe:
    """A keyframe in the factor graph."""
    index: int
    time: float
    jd: float
    key_R: int  # Symbol for attitude
    key_B: int  # Symbol for bias

    # Measurements at this keyframe (any/all may be present)
    z_mag: Optional[np.ndarray] = None
    z_sun: Optional[np.ndarray] = None
    z_st: Optional[Quaternion] = None

    # Reference vectors (computed from environment)
    B_eci: Optional[np.ndarray] = None
    s_eci: Optional[np.ndarray] = None


# ==============================================================================
# RK4 Preintegration (Higher-order than GTSAM's Euler method)
# ==============================================================================

@dataclass
class RK4Preintegration:
    """
    Custom gyro preintegration using RK4 integration.

    GTSAM's PreintegratedAhrsMeasurements uses Euler's method internally.
    This class provides higher-order (4th order) integration for better
    accuracy with rapidly changing angular velocity.

    The result can be used with BetweenFactorRot3 instead of AHRSFactor.
    """
    gyro_cov: np.ndarray  # 3x3 per-sample gyro covariance
    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.delta_R: Rot3 = Rot3.Identity()
        self.delta_t: float = 0.0
        self.covariance: np.ndarray = np.zeros((3, 3))  # Accumulated rotation covariance
        self._prev_omega: Optional[np.ndarray] = None

    def reset(self, bias: np.ndarray):
        """Reset preintegration with new bias."""
        self.bias = bias.copy()
        self.delta_R = Rot3.Identity()
        self.delta_t = 0.0
        self.covariance = np.zeros((3, 3))
        self._prev_omega = None

    def integrate(self, omega: np.ndarray, dt: float):
        """
        Integrate a gyro measurement using RK4.

        Args:
            omega: Measured angular velocity (3,) in rad/s
            dt: Time step in seconds
        """
        omega = np.asarray(omega, float).reshape(3)
        omega_corrected = omega - self.bias

        if self._prev_omega is None:
            # First sample: use simple exponential map
            theta = np.linalg.norm(omega_corrected) * dt
            if theta > 1e-10:
                axis = omega_corrected / np.linalg.norm(omega_corrected)
                delta = Rot3.Rodrigues(axis * theta)
            else:
                delta = Rot3.Identity()
        else:
            # Use RK4 with previous omega for interpolation
            prev_corrected = self._prev_omega - self.bias
            delta = self._rk4_rotation(prev_corrected, omega_corrected, dt)

        # Right-multiply: delta_R_new = delta_R_old * delta (body-frame convention)
        self.delta_R = self.delta_R.compose(delta)
        self.delta_t += dt

        # Propagate covariance (first-order approximation)
        # Cov(k+1) = Cov(k) + dt * gyro_cov
        self.covariance += dt * self.gyro_cov

        self._prev_omega = omega.copy()

    def _rk4_rotation(self, omega_start: np.ndarray, omega_end: np.ndarray, dt: float) -> Rot3:
        """
        Compute rotation increment using RK4.

        For rotation kinematics: R_dot = R @ skew(omega) (right-multiply convention)
        """
        omega_mid = 0.5 * (omega_start + omega_end)

        def rot_dot(R: np.ndarray, omega: np.ndarray) -> np.ndarray:
            """R_dot = R @ skew(omega) for right-multiply convention."""
            return R @ skew(omega)

        R = np.eye(3)

        # RK4 stages
        k1 = rot_dot(R, omega_start)
        k2 = rot_dot(R + 0.5 * dt * k1, omega_mid)
        k3 = rot_dot(R + 0.5 * dt * k2, omega_mid)
        k4 = rot_dot(R + dt * k3, omega_end)

        R_new = R + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Project back to SO(3) via SVD
        U, _, Vt = np.linalg.svd(R_new)
        R_proj = U @ Vt
        if np.linalg.det(R_proj) < 0:
            R_proj = -R_proj

        return Rot3(R_proj)

    def deltaRij(self) -> Rot3:
        """Get accumulated rotation."""
        return self.delta_R

    def deltaTij(self) -> float:
        """Get accumulated time."""
        return self.delta_t

    def preint_cov(self) -> np.ndarray:
        """Get accumulated covariance (3x3)."""
        return self.covariance


# ==============================================================================
# Keyframe Factor Builders
# ==============================================================================

class KeyframeFactorBuilders:
    """Factor builders for keyframe-based FGO using GTSAM classes."""

    def __init__(
        self,
        gyro_cov: np.ndarray,  # 3x3 gyro noise covariance (includes discretization error)
        sigma_mag: float,
        sigma_sun: float,
        sigma_star: float,
        sigma_bias: float,
        use_robust: bool = True,
    ):
        self.gyro_cov = gyro_cov
        self.sigma_mag = sigma_mag
        self.sigma_sun = sigma_sun
        self.sigma_star = sigma_star
        self.sigma_bias = sigma_bias
        self.use_robust = use_robust

        # Setup GTSAM preintegration params (for Euler fallback)
        self.preint_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
        self.preint_params.setGyroscopeCovariance(gyro_cov)

    def _robust_noise(self, base_noise):
        """Apply Huber robust kernel."""
        if not self.use_robust:
            return base_noise
        return gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345),
            base_noise
        )

    def create_preintegration(self, bias: np.ndarray) -> gtsam.PreintegratedAhrsMeasurements:
        """Create a new GTSAM preintegration object (Euler method)."""
        return gtsam.PreintegratedAhrsMeasurements(self.preint_params, bias)

    def create_rk4_preintegration(self, bias: np.ndarray) -> RK4Preintegration:
        """Create a new RK4 preintegration object."""
        return RK4Preintegration(
            gyro_cov=self.gyro_cov,
            bias=bias.copy(),
        )

    def make_ahrs_factor(
        self,
        key_R_i: int,
        key_R_j: int,
        key_B: int,
        preint: gtsam.PreintegratedAhrsMeasurements,
    ) -> gtsam.AHRSFactor:
        """Create GTSAM AHRSFactor from preintegrated measurements (Euler)."""
        return gtsam.AHRSFactor(key_R_i, key_R_j, key_B, preint)

    def make_between_factor_rot3(
        self,
        key_R_i: int,
        key_R_j: int,
        preint: RK4Preintegration,
    ) -> gtsam.BetweenFactorRot3:
        """
        Create a BetweenFactorRot3 from RK4 preintegrated measurements.

        This is an alternative to AHRSFactor that uses higher-order (RK4)
        integration instead of GTSAM's Euler method. The trade-off is that
        this factor does not support automatic bias Jacobian computation,
        so bias must be re-linearized manually (which batch mode already does).

        Args:
            key_R_i: Key for rotation at start of interval
            key_R_j: Key for rotation at end of interval
            preint: RK4 preintegration result

        Returns:
            BetweenFactorRot3 constraining R_j = R_i * delta_R (right-multiply)
        """
        delta_R = preint.deltaRij()

        # Noise model from accumulated covariance
        cov = preint.preint_cov()
        # Ensure covariance is positive definite
        cov = cov + 1e-10 * np.eye(3)
        noise = gtsam.noiseModel.Gaussian.Covariance(cov)

        return gtsam.BetweenFactorRot3(key_R_i, key_R_j, delta_R, noise)

    def make_magnetometer_factor(
        self,
        key_R: int,
        z_mag: np.ndarray,
        B_eci: np.ndarray,
    ) -> gtsam.MagFactor1:
        """Vector measurement factor for magnetometer.

        Measurement model: z_body = R_bn @ B_nav / |B_nav| + noise
        We use direction-only (normalized) measurements with scale=1.0.
        """
        v_meas = z_mag / np.linalg.norm(z_mag)  # Normalize measurement
        v_ref = B_eci / np.linalg.norm(B_eci)   # Normalize reference

        # Use scale=1.0 since we're comparing unit vectors
        scale = 1.0
        direction = gtsam.Unit3(*v_ref)

        base_noise = gtsam.noiseModel.Gaussian.Covariance(np.eye(3) * (self.sigma_mag ** 2))
        noise = self._robust_noise(base_noise)

        bias = gtsam.Point3(0, 0, 0)  # No hard-iron bias estimation

        factor = gtsam.MagFactor1(
            key_R,
            v_meas,
            scale,
            direction,
            bias,
            noise
        )

        return factor


    def make_sun_factor(
        self,
        key_R: int,
        z_sun: np.ndarray,
        s_eci: np.ndarray,
    ) -> gtsam.CustomFactor:
        """Vector measurement factor for sun sensor.

        Measurement model: z_body = R_nb^T @ s_nav + noise
        where R_nb is body-to-nav rotation (what we estimate)

        Error: e = z_meas - R_nb^T @ s_nav

        Right-multiply convention: R_nb stores body-to-nav rotation
        Jacobian: de/dδθ = -skew(z_pred) for right perturbation R_nb * exp(δθ)
        """
        v_meas = z_sun / np.linalg.norm(z_sun)
        v_ref = s_eci / np.linalg.norm(s_eci)

        base_noise = gtsam.noiseModel.Gaussian.Covariance(np.eye(3) * (self.sigma_sun ** 2))
        noise = self._robust_noise(base_noise)
        keys = [key_R]

        def error_fn(this, values, jacobians=None):
            R = values.atRot3(key_R)
            R_nb = R.matrix()  # body-to-nav rotation matrix
            v_pred = R_nb.T @ v_ref  # predicted measurement in body frame
            e = v_meas - v_pred

            if jacobians is not None:
                # Jacobian for right perturbation: de/dδθ = -skew(v_pred)
                # Since v_pred = R^T @ v_ref, and we perturb R -> R * exp(δθ)
                # Then v_pred_new = (R * exp(δθ))^T @ v_ref = exp(-δθ) @ R^T @ v_ref
                # ≈ (I - skew(δθ)) @ v_pred = v_pred - skew(δθ) @ v_pred = v_pred + skew(v_pred) @ δθ
                # So d(v_pred)/dδθ = skew(v_pred), and de/dδθ = -skew(v_pred)
                jacobians[0] = -skew(v_pred)
            return e

        return gtsam.CustomFactor(noise, keys, error_fn)

    def make_star_factor(
        self,
        key_R: int,
        q_meas: Quaternion,
    ) -> gtsam.PriorFactorRot3:
        """Star tracker measurement - direct attitude measurement."""
        R_meas = rot3_from_quat(q_meas)
        base_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([self.sigma_star] * 3)
        )
        noise = self._robust_noise(base_noise)
        return gtsam.PriorFactorRot3(key_R, R_meas, noise)

    def make_bias_prior(
        self,
        key_B: int,
        b_prior: np.ndarray,
        sigma: float,
    ) -> gtsam.PriorFactorPoint3:
        """Prior on bias."""
        noise = gtsam.noiseModel.Gaussian.Covariance(np.eye(3) * (sigma ** 2))
        return gtsam.PriorFactorPoint3(key_B, Point3(*b_prior), noise)

    def make_bias_between(
        self,
        key_B_i: int,
        key_B_j: int,
        dt: float,
    ) -> gtsam.BetweenFactorPoint3:
        """Bias random walk factor."""
        sigma = self.sigma_bias * np.sqrt(dt)
        noise = gtsam.noiseModel.Gaussian.Covariance(np.eye(3) * (sigma ** 2))
        return gtsam.BetweenFactorPoint3(key_B_i, key_B_j, Point3(0, 0, 0), noise)


# ==============================================================================
# Keyframe FGO Estimator
# ==============================================================================

class KeyframeFGO:
    """
    Keyframe-based Factor Graph Optimizer with gyro preintegration.

    Uses RK4 preintegration by default for higher accuracy.
    Right-multiply quaternion convention: q_new = q_old * delta_q (body-frame perturbation)
    """

    def __init__(
        self,
        config_path: str = "configs/config_baseline_short.yaml",
        use_robust: bool = True,
        discretization_factor: float = 0.10,
        use_isam2: bool = False,
        use_rk4: bool = True,  # RK4 is default
        isam2_relinearize_threshold: float = 0.0001,
        isam2_relinearize_skip: int = 1,
        isam2_finalize_updates: int = 5,
    ):
        self.config = load_yaml(config_path)
        self.process = ProcessModel(config_path)
        self.use_isam2 = use_isam2
        self.use_rk4 = use_rk4
        self.isam2_relinearize_threshold = isam2_relinearize_threshold
        self.isam2_relinearize_skip = isam2_relinearize_skip
        self.isam2_finalize_updates = isam2_finalize_updates

        # Extract noise parameters
        mag_inflation = self.config["sensors"]["mag"].get("fgo_noise_inflation", 1.0)

        # Gyro covariance for preintegration
        gyro_sensor_var = self.process.sigma_g ** 2

        # Additional discretization error covariance
        typical_omega = 0.0175  # rad/s (~1 deg/s typical rotation rate)
        discretization_var = (discretization_factor * typical_omega)**2

        gyro_cov = np.eye(3) * (gyro_sensor_var + discretization_var)

        self.factors = KeyframeFactorBuilders(
            gyro_cov=gyro_cov,
            sigma_mag=self.config["sensors"]["mag"]["mag_std"] * mag_inflation,
            sigma_sun=self.config["sensors"]["sun"]["noise"]["sun_std"],
            sigma_star=self.config["sensors"]["star"]["noise"]["st_std"],
            sigma_bias=self.process.sigma_bg,
            use_robust=use_robust,
        )

        # State
        self.keyframes: List[Keyframe] = []
        self.keyframe_index = 0
        self.current_bias = np.zeros(3)

        # iSAM2 instance (created lazily in process_simulation)
        self.isam2: Optional[ISAM2] = None

        # KF estimates for initialization
        self.kf_estimates = None

        mode = "iSAM2 (incremental)" if use_isam2 else "Batch LM"
        preint_mode = "RK4" if use_rk4 else "GTSAM (Euler)"
        logger.info(f"KeyframeFGO initialized, mode: {mode}, preintegration: {preint_mode}")

    @staticmethod
    def X(k: int) -> int:
        return Symbol('x', k).key()

    @staticmethod
    def B(k: int) -> int:
        return Symbol('b', k).key()

    def process_simulation(
        self,
        sim_data,
        env: OrbitEnvironmentModel,
        kf_estimates: Optional[Tuple[np.ndarray, List[NominalState]]] = None,
    ) -> Tuple[List[float], List[NominalState]]:
        """
        Process simulation data with keyframe-based FGO.

        Args:
            sim_data: Simulation data with t, jd, omega_meas, mag_meas, sun_meas, st_meas
            env: Environment model
            kf_estimates: Optional tuple of (times, states) from Kalman filter.

        Returns:
            Tuple of (times, estimated_states) at keyframe times
        """
        self.kf_estimates = kf_estimates
        if kf_estimates is not None:
            logger.info("Using Kalman filter estimates as initial values")

        if self.use_isam2:
            return self._process_simulation_isam2(sim_data, env)
        else:
            return self._process_simulation_batch(sim_data, env)

    def _get_kf_estimate_at_time(self, t: float) -> Optional[NominalState]:
        """Get interpolated KF estimate at time t."""
        if self.kf_estimates is None:
            return None

        kf_times, kf_states = self.kf_estimates

        idx = np.searchsorted(kf_times, t)

        if idx == 0:
            if abs(kf_times[0] - t) < 0.1:
                return kf_states[0]
            return None
        elif idx >= len(kf_times):
            if abs(kf_times[-1] - t) < 0.1:
                return kf_states[-1]
            return None

        if abs(kf_times[idx] - t) < abs(kf_times[idx-1] - t):
            if abs(kf_times[idx] - t) < 0.1:
                return kf_states[idx]
        else:
            if abs(kf_times[idx-1] - t) < 0.1:
                return kf_states[idx-1]

        return None

    def _process_simulation_batch(
        self,
        sim_data,
        env: OrbitEnvironmentModel,
        max_relinearizations: int = 5,
        bias_convergence_tol: float = 1e-8,
    ) -> Tuple[List[float], List[NominalState]]:
        """Process simulation data using batch optimization with bias re-linearization."""
        N = len(sim_data.t)
        logger.info(f"Processing {N} samples with batch FGO")

        # Initialize bias linearization point
        self.current_bias = np.zeros(3)

        # Build keyframes (only need to do this once - measurements don't change)
        self._build_keyframes(sim_data, env)

        logger.info(f"Created {len(self.keyframes)} keyframes from {N} samples")
        logger.info(f"Reduction ratio: {N / len(self.keyframes):.1f}x")

        # Iterative re-linearization loop
        for iteration in range(max_relinearizations):
            logger.info(f"Re-linearization iteration {iteration + 1}/{max_relinearizations}, "
                       f"bias linearization point: {self.current_bias * 3600 * 180 / np.pi} deg/h")

            # Preintegrate with current bias estimate
            if self.use_rk4:
                preintegrations = self._preintegrate_gyro_rk4(sim_data, self.current_bias)
            else:
                preintegrations = self._preintegrate_gyro(sim_data, self.current_bias)

            # Build and solve factor graph
            graph, values = self._build_graph(preintegrations, sim_data.q_true[0])

            if iteration == 0:
                logger.info(f"Factor graph: {graph.size()} factors, {values.size()} variables")

            # Optimize
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(50)
            params.setAbsoluteErrorTol(1e-6)

            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
            result = optimizer.optimize()

            logger.info(f"  Iteration {iteration + 1} complete, final error: {optimizer.error():.6f}")

            # Extract average bias estimate for re-linearization
            bias_estimates = []
            for kf in self.keyframes:
                b_est = np.array(result.atPoint3(kf.key_B))
                bias_estimates.append(b_est)

            new_bias = np.mean(bias_estimates, axis=0)
            bias_change = np.linalg.norm(new_bias - self.current_bias)

            logger.info(f"  New bias estimate: {new_bias * 3600 * 180 / np.pi} deg/h")
            logger.info(f"  Bias change: {bias_change * 3600 * 180 / np.pi:.6f} deg/h")

            # Check convergence
            if bias_change < bias_convergence_tol:
                logger.info(f"  Bias converged after {iteration + 1} iterations")
                break

            # Update linearization point for next iteration
            self.current_bias = new_bias

        logger.info(f"Optimization complete after {iteration + 1} re-linearization(s)")

        # Extract final results
        times = []
        states = []

        for kf in self.keyframes:
            R_est = result.atRot3(kf.key_R)
            b_est = np.array(result.atPoint3(kf.key_B))

            times.append(kf.time)
            states.append(NominalState(
                ori=quat_from_rot3(R_est),
                gyro_bias=b_est,
            ))

        return times, states

    def _build_keyframes(self, sim_data, env: OrbitEnvironmentModel) -> None:
        """Build keyframes from simulation data (measurements only, no preintegration)."""
        self.keyframes = []
        self.keyframe_index = 0

        # First keyframe at t=0
        kf0 = Keyframe(
            index=0,
            time=sim_data.t[0],
            jd=sim_data.jd[0],
            key_R=self.X(0),
            key_B=self.B(0),
        )

        # Check for measurements at t=0
        if not np.any(np.isnan(sim_data.mag_meas[0])):
            kf0.z_mag = sim_data.mag_meas[0]
            r_eci = env.get_r_eci(sim_data.jd[0])
            kf0.B_eci = env.get_B_eci(r_eci, sim_data.jd[0])
        if not np.any(np.isnan(sim_data.sun_meas[0])):
            kf0.z_sun = sim_data.sun_meas[0]
            kf0.s_eci = env.get_sun_eci(sim_data.jd[0])
        if not np.any(np.isnan(sim_data.st_meas[0])):
            kf0.z_st = Quaternion.from_array(sim_data.st_meas[0])

        self.keyframes.append(kf0)
        self.keyframe_index = 1

        N = len(sim_data.t)
        for k in range(1, N):
            t = sim_data.t[k]
            jd = sim_data.jd[k]

            # Check if we have any measurement
            has_mag = not np.any(np.isnan(sim_data.mag_meas[k]))
            has_sun = not np.any(np.isnan(sim_data.sun_meas[k]))
            has_st = not np.any(np.isnan(sim_data.st_meas[k]))

            if has_mag or has_sun or has_st:
                kf = Keyframe(
                    index=self.keyframe_index,
                    time=t,
                    jd=jd,
                    key_R=self.X(self.keyframe_index),
                    key_B=self.B(self.keyframe_index),
                )

                if has_mag:
                    kf.z_mag = sim_data.mag_meas[k]
                    r_eci = env.get_r_eci(jd)
                    kf.B_eci = env.get_B_eci(r_eci, jd)
                if has_sun:
                    kf.z_sun = sim_data.sun_meas[k]
                    kf.s_eci = env.get_sun_eci(jd)
                if has_st:
                    kf.z_st = Quaternion.from_array(sim_data.st_meas[k])

                self.keyframes.append(kf)
                self.keyframe_index += 1

    def _preintegrate_gyro(
        self,
        sim_data,
        bias_linearization: np.ndarray,
    ) -> List[gtsam.PreintegratedAhrsMeasurements]:
        """Preintegrate gyro measurements between keyframes (GTSAM/Euler)."""
        preintegrations = []
        current_preint = self.factors.create_preintegration(bias_linearization)

        keyframe_idx = 1
        N = len(sim_data.t)

        for k in range(1, N):
            t = sim_data.t[k]
            dt = t - sim_data.t[k-1]

            omega = sim_data.omega_meas[k]
            if not np.any(np.isnan(omega)):
                current_preint.integrateMeasurement(omega, dt)

            if keyframe_idx < len(self.keyframes) and abs(t - self.keyframes[keyframe_idx].time) < 1e-6:
                preintegrations.append(current_preint)
                current_preint = self.factors.create_preintegration(bias_linearization)
                keyframe_idx += 1

        return preintegrations

    def _preintegrate_gyro_rk4(
        self,
        sim_data,
        bias_linearization: np.ndarray,
    ) -> List[RK4Preintegration]:
        """Preintegrate gyro measurements between keyframes using RK4 integration."""
        preintegrations = []
        current_preint = self.factors.create_rk4_preintegration(bias_linearization)

        keyframe_idx = 1
        N = len(sim_data.t)

        for k in range(1, N):
            t = sim_data.t[k]
            dt = t - sim_data.t[k-1]

            omega = sim_data.omega_meas[k]
            if not np.any(np.isnan(omega)):
                current_preint.integrate(omega, dt)

            if keyframe_idx < len(self.keyframes) and abs(t - self.keyframes[keyframe_idx].time) < 1e-6:
                preintegrations.append(current_preint)
                current_preint = self.factors.create_rk4_preintegration(bias_linearization)
                keyframe_idx += 1

        return preintegrations

    def _process_simulation_isam2(
        self,
        sim_data,
        env: OrbitEnvironmentModel,
    ) -> Tuple[List[float], List[NominalState]]:
        """Process simulation data using iSAM2 incremental optimization."""
        N = len(sim_data.t)
        logger.info(f"Processing {N} samples with iSAM2")

        # Initialize iSAM2
        isam2_params = ISAM2Params()
        isam2_params.setRelinearizeThreshold(self.isam2_relinearize_threshold)
        isam2_params.relinearizeSkip = self.isam2_relinearize_skip
        self.isam2 = ISAM2(isam2_params)
        logger.info(f"iSAM2 params: threshold={self.isam2_relinearize_threshold}, skip={self.isam2_relinearize_skip}")

        # Initialize
        self.current_bias = np.zeros(3)
        b0 = np.zeros(3)

        self.keyframes = []
        self.keyframe_index = 0

        times = []
        states = []

        # Initial attitude
        q0 = Quaternion.from_array(sim_data.q_true[0])
        # Perturb initial attitude - right-multiply convention
        q0 = q0.multiply(Quaternion.from_avec(np.array([0.1, 0.1, 0.1])))
        R0 = rot3_from_quat(q0)

        # First keyframe
        kf0 = Keyframe(
            index=0,
            time=sim_data.t[0],
            jd=sim_data.jd[0],
            key_R=self.X(0),
            key_B=self.B(0),
        )

        if not np.any(np.isnan(sim_data.mag_meas[0])):
            kf0.z_mag = sim_data.mag_meas[0]
            r_eci = env.get_r_eci(sim_data.jd[0])
            kf0.B_eci = env.get_B_eci(r_eci, sim_data.jd[0])
        if not np.any(np.isnan(sim_data.sun_meas[0])):
            kf0.z_sun = sim_data.sun_meas[0]
            kf0.s_eci = env.get_sun_eci(sim_data.jd[0])
        if not np.any(np.isnan(sim_data.st_meas[0])):
            kf0.z_st = Quaternion.from_array(sim_data.st_meas[0])

        self.keyframes.append(kf0)
        self.keyframe_index = 1

        # Add initial keyframe to iSAM2
        graph0 = NonlinearFactorGraph()
        values0 = Values()

        values0.insert(self.X(0), R0)
        values0.insert(self.B(0), Point3(*b0))

        prior_b_sigma = 1e-4
        graph0.add(self.factors.make_bias_prior(self.B(0), b0, prior_b_sigma))

        if kf0.z_mag is not None:
            graph0.add(self.factors.make_magnetometer_factor(kf0.key_R, kf0.z_mag, kf0.B_eci))
        if kf0.z_sun is not None:
            graph0.add(self.factors.make_sun_factor(kf0.key_R, kf0.z_sun, kf0.s_eci))
        if kf0.z_st is not None:
            graph0.add(self.factors.make_star_factor(kf0.key_R, kf0.z_st))
        else:
            prior_R_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01]))
            graph0.add(gtsam.PriorFactorRot3(self.X(0), R0, prior_R_noise))

        self.isam2.update(graph0, values0)
        current_estimate = self.isam2.calculateEstimate()

        R_est = current_estimate.atRot3(kf0.key_R)
        b_est = np.array(current_estimate.atPoint3(kf0.key_B))
        times.append(kf0.time)
        states.append(NominalState(ori=quat_from_rot3(R_est), gyro_bias=b_est))

        # Current preintegration
        if self.use_rk4:
            current_preint = self.factors.create_rk4_preintegration(self.current_bias)
        else:
            current_preint = self.factors.create_preintegration(self.current_bias)

        # Process remaining samples
        for k in range(1, N):
            t = sim_data.t[k]
            jd = sim_data.jd[k]
            dt = t - sim_data.t[k-1]

            omega = sim_data.omega_meas[k]
            if not np.any(np.isnan(omega)):
                if self.use_rk4:
                    current_preint.integrate(omega, dt)
                else:
                    current_preint.integrateMeasurement(omega, dt)

            has_mag = not np.any(np.isnan(sim_data.mag_meas[k]))
            has_sun = not np.any(np.isnan(sim_data.sun_meas[k]))
            has_st = not np.any(np.isnan(sim_data.st_meas[k]))

            if has_mag or has_sun or has_st:
                kf = Keyframe(
                    index=self.keyframe_index,
                    time=t,
                    jd=jd,
                    key_R=self.X(self.keyframe_index),
                    key_B=self.B(self.keyframe_index),
                )

                if has_mag:
                    kf.z_mag = sim_data.mag_meas[k]
                    r_eci = env.get_r_eci(jd)
                    kf.B_eci = env.get_B_eci(r_eci, jd)
                if has_sun:
                    kf.z_sun = sim_data.sun_meas[k]
                    kf.s_eci = env.get_sun_eci(jd)
                if has_st:
                    kf.z_st = Quaternion.from_array(sim_data.st_meas[k])

                self.keyframes.append(kf)

                graph_inc = NonlinearFactorGraph()
                values_inc = Values()

                prev_kf = self.keyframes[-2]
                R_prev_updated = current_estimate.atRot3(prev_kf.key_R)
                b_prev_updated = np.array(current_estimate.atPoint3(prev_kf.key_B))

                R_pred = R_prev_updated.compose(current_preint.deltaRij())
                values_inc.insert(kf.key_R, R_pred)
                values_inc.insert(kf.key_B, Point3(*b_prev_updated))

                # Rotation factor
                if self.use_rk4:
                    graph_inc.add(self.factors.make_between_factor_rot3(
                        prev_kf.key_R, kf.key_R, current_preint
                    ))
                else:
                    graph_inc.add(self.factors.make_ahrs_factor(
                        prev_kf.key_R, kf.key_R, prev_kf.key_B, current_preint
                    ))

                graph_inc.add(self.factors.make_bias_between(
                    prev_kf.key_B, kf.key_B, current_preint.deltaTij()
                ))

                if kf.z_mag is not None:
                    graph_inc.add(self.factors.make_magnetometer_factor(
                        kf.key_R, kf.z_mag, kf.B_eci
                    ))
                if kf.z_sun is not None:
                    graph_inc.add(self.factors.make_sun_factor(
                        kf.key_R, kf.z_sun, kf.s_eci
                    ))
                if kf.z_st is not None:
                    graph_inc.add(self.factors.make_star_factor(
                        kf.key_R, kf.z_st
                    ))

                self.isam2.update(graph_inc, values_inc)
                current_estimate = self.isam2.calculateEstimate()

                R_est = current_estimate.atRot3(kf.key_R)
                b_est = np.array(current_estimate.atPoint3(kf.key_B))

                times.append(kf.time)
                states.append(NominalState(ori=quat_from_rot3(R_est), gyro_bias=b_est))

                self.current_bias = b_est
                if self.use_rk4:
                    current_preint = self.factors.create_rk4_preintegration(self.current_bias)
                else:
                    current_preint = self.factors.create_preintegration(self.current_bias)
                self.keyframe_index += 1

        logger.info(f"Created {len(self.keyframes)} keyframes from {N} samples")
        logger.info(f"Reduction ratio: {N / len(self.keyframes):.1f}x")

        if self.isam2_finalize_updates > 0:
            logger.info(f"Running {self.isam2_finalize_updates} finalization updates...")
            for _ in range(self.isam2_finalize_updates):
                self.isam2.update()

        result = self.isam2.calculateEstimate()
        times = []
        states = []
        for kf in self.keyframes:
            R_est = result.atRot3(kf.key_R)
            b_est = np.array(result.atPoint3(kf.key_B))
            times.append(kf.time)
            states.append(NominalState(ori=quat_from_rot3(R_est), gyro_bias=b_est))

        logger.info(f"iSAM2 processing complete")

        return times, states

    def _build_graph(
        self,
        preintegrations: List,
        q_init: np.ndarray,
    ) -> Tuple[NonlinearFactorGraph, Values]:
        """
        Build the factor graph from keyframes.

        When use_rk4 is True, uses BetweenFactorRot3 with RK4-preintegrated rotations.
        When use_rk4 is False, uses AHRSFactor with GTSAM's Euler-based preintegration.
        """
        graph = NonlinearFactorGraph()
        values = Values()

        kf0 = self.keyframes[0]
        kf_est_0 = self._get_kf_estimate_at_time(kf0.time)

        if kf_est_0 is not None:
            R0 = rot3_from_quat(kf_est_0.ori)
            b0 = kf_est_0.gyro_bias
        else:
            q0 = Quaternion.from_array(q_init)
            R0 = rot3_from_quat(q0)
            b0 = np.zeros(3)

        values.insert(self.X(0), R0)
        values.insert(self.B(0), Point3(*b0))

        prior_b_sigma = 1e-6
        graph.add(self.factors.make_bias_prior(self.B(0), b0, prior_b_sigma))

        if kf0.z_mag is not None:
            graph.add(self.factors.make_magnetometer_factor(kf0.key_R, kf0.z_mag, kf0.B_eci))
        if kf0.z_sun is not None:
            graph.add(self.factors.make_sun_factor(kf0.key_R, kf0.z_sun, kf0.s_eci))
        if kf0.z_st is not None:
            graph.add(self.factors.make_star_factor(kf0.key_R, kf0.z_st))
        else:
            prior_R_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, 0.5]))
            graph.add(gtsam.PriorFactorRot3(self.X(0), R0, prior_R_noise))

        R_prev = R0
        b_prev = b0

        for i, kf in enumerate(self.keyframes[1:], 1):
            preint = preintegrations[i-1]

            kf_est = self._get_kf_estimate_at_time(kf.time)

            if kf_est is not None:
                R_init = rot3_from_quat(kf_est.ori)
                b_init = kf_est.gyro_bias
            else:
                R_init = R_prev.compose(preint.deltaRij())
                b_init = b_prev

            values.insert(kf.key_R, R_init)
            values.insert(kf.key_B, Point3(*b_init))

            prev_kf = self.keyframes[i-1]
            if self.use_rk4:
                graph.add(self.factors.make_between_factor_rot3(
                    prev_kf.key_R, kf.key_R, preint
                ))
            else:
                graph.add(self.factors.make_ahrs_factor(
                    prev_kf.key_R, kf.key_R, prev_kf.key_B, preint
                ))

            graph.add(self.factors.make_bias_between(
                prev_kf.key_B, kf.key_B, preint.deltaTij()
            ))

            if kf.z_mag is not None:
                graph.add(self.factors.make_magnetometer_factor(
                    kf.key_R, kf.z_mag, kf.B_eci
                ))

            if kf.z_sun is not None:
                graph.add(self.factors.make_sun_factor(
                    kf.key_R, kf.z_sun, kf.s_eci
                ))

            if kf.z_st is not None:
                graph.add(self.factors.make_star_factor(
                    kf.key_R, kf.z_st
                ))

            R_prev = R_init
            b_prev = b_init

        return graph, values


# ==============================================================================
# Test/Run Function
# ==============================================================================

def run_keyframe_fgo(
    sim_run_id: int = 1,
    db_path: str = "simulations.db",
    config_path: str = "configs/config_baseline_short.yaml",
    use_isam2: bool = False,
    use_rk4: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run keyframe-based FGO on simulation data.

    Args:
        sim_run_id: Simulation run ID in database
        db_path: Path to simulation database
        config_path: Path to config file
        use_isam2: If True, use iSAM2 incremental optimization; else batch LM
        use_rk4: If True, use RK4 preintegration; else GTSAM's Euler method

    Returns:
        Tuple of (times, errors_deg)
    """
    from data.db import SimulationDatabase

    mode = "iSAM2 (INCREMENTAL)" if use_isam2 else "BATCH LM"
    preint = "RK4" if use_rk4 else "GTSAM (Euler)"
    print("=" * 70)
    print(f"KEYFRAME-BASED FGO - {mode}, Preintegration: {preint}")
    print("=" * 70)

    db = SimulationDatabase(db_path)
    sim = db.load_run(sim_run_id)
    env = OrbitEnvironmentModel()

    print(f"Simulation: {len(sim.t)} samples ({sim.t[-1]:.1f}s)")

    fgo = KeyframeFGO(config_path=config_path, use_isam2=use_isam2, use_rk4=use_rk4)
    times, states = fgo.process_simulation(sim, env)

    errors = []
    for i, (t, state) in enumerate(zip(times, states)):
        idx = np.searchsorted(sim.t, t)
        idx = min(idx, len(sim.q_true) - 1)

        q_true = Quaternion.from_array(sim.q_true[idx])
        q_est = state.ori
        # Right-multiply convention: error = q_true * q_est^{-1}
        q_err = q_true.multiply(q_est.conjugate())
        angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
        errors.append(np.rad2deg(angle_err))

    errors = np.array(errors)
    times = np.array(times)

    print(f"\nAttitude Error Statistics:")
    print(f"  Mean:  {np.mean(errors):.6f} deg")
    print(f"  Std:   {np.std(errors):.6f} deg")
    print(f"  Max:   {np.max(errors):.6f} deg")
    print(f"  Final: {errors[-1]:.6f} deg")

    print(f"\nError at different times:")
    for t_check in [10, 30, 60, 120, 180, 240, 300]:
        idx = np.searchsorted(times, t_check)
        if idx < len(errors):
            print(f"  t={t_check}s: {errors[idx]:.6f} deg")

    return times, errors


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Keyframe-based FGO with GTSAM")
    parser.add_argument("--isam2", action="store_true", help="Use iSAM2 incremental optimization")
    parser.add_argument("--euler", action="store_true", help="Use GTSAM Euler preintegration instead of RK4 (default)")
    args = parser.parse_args()

    times, errors = run_keyframe_fgo(use_isam2=args.isam2, use_rk4=not args.euler)

    # Plot
    mode_str = "iSAM2" if args.isam2 else "Batch LM"
    preint_str = "Euler" if args.euler else "RK4"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, errors, 'b-', linewidth=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title(f'Keyframe FGO ({mode_str}, {preint_str} preintegration)')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    if args.isam2:
        fname = f'keyframe_fgo_isam2_{preint_str.lower()}.png'
    else:
        fname = f'keyframe_fgo_batch_{preint_str.lower()}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"\nPlot saved to {fname}")
