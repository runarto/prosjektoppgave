from dataclasses import dataclass, field
import numpy as np
from utilities.states import NominalState, EskfState
from utilities.gaussian import MultiVarGauss
from utilities.quaternion import Quaternion
from utilities.utils import get_skew_matrix, load_yaml

from logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class SensorGyro:
    
    def __init__(self, config_path: str = "config.yaml"):
        config = load_yaml(config_path)
        self.dt = config["sensors"]["gyro"]["dt"]
        ARW_deg = config["sensors"]["gyro"]["noise"]["arw_deg"]  # e.g. 0.15 deg / sqrt(h)

        self.noise_rw = ARW_deg * np.pi/180 / np.sqrt(3600.0)  # rad / sqrt(s)
        self.gyro_std = self.noise_rw / np.sqrt(self.dt)                 # rad / s (per 

    def __post_init__(self):
        self.R = np.eye(3) * self.gyro_std**2
    
    def sample(self, omega_true: np.ndarray) -> np.ndarray:
        """Sample a gyro measurement from the nominal state."""
        noise = np.random.normal(0.0, self.gyro_std, size=3)
        return omega_true + noise
class SensorMagnetometer:
    def __init__(self, config_path: str = "config.yaml"):
        config = load_yaml(config_path)
        mag_cfg = config["sensors"]["mag"]

        self.mag_std: float = mag_cfg["mag_std"]
        self.dt: float = mag_cfg["dt"]

        # Hard-iron and soft-iron calibration errors
        self.hard_iron: np.ndarray = np.asarray(mag_cfg.get("hard_iron", [0.0, 0.0, 0.0]), float).reshape(3)
        self.soft_iron: np.ndarray = np.asarray(mag_cfg.get("soft_iron", np.eye(3)), float).reshape(3, 3)

        # Spike config
        spikes_cfg = mag_cfg.get("spikes", {})
        self.spikes_enabled: bool = spikes_cfg.get("enabled", False)
        self.spike_magnitude: float = spikes_cfg.get("magnitude", 0.0)
        self.spike_probability: float = spikes_cfg.get("probability", 0.0)

        # Set measurement noise covariance
        self.R = np.eye(3) * self.mag_std**2
        logger.debug(f"Magnetometer measurement noise covariance R set to {self.mag_std**2:.2e} * I")

    # ---- internal helpers -------------------------------------------------

    def _maybe_spike(self) -> np.ndarray:
        """Generate a spike vector or zero."""
        if (
            not self.spikes_enabled
            or self.spike_probability <= 0.0
            or self.spike_magnitude <= 0.0
        ):
            return np.zeros(3)

        if np.random.rand() >= self.spike_probability:
            return np.zeros(3)

        # Random spike direction with magnitude ~ spike_magnitude
        direction = np.random.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            return np.zeros(3)
        direction /= norm
        return self.spike_magnitude * direction

    # ---- sensor model -----------------------------------------------------

    def sample(self, q_true: Quaternion, B_n: np.ndarray) -> np.ndarray:
        """Sample a magnetometer measurement from the true state.

        Args:
            q_true: True body-to-navigation quaternion (b->n)
            B_n:    Magnetic field in navigation/inertial frame, shape (3,)

        Returns:
            Magnetometer measurement in body frame, shape (3,).
        """
        B_n = np.asarray(B_n, float).reshape(3)

        # Rotate magnetic field into body frame
        R_nb = q_true.as_rotmat()   # body-to-nav
        R_bn = R_nb.T               # nav-to-body
        B_b = R_bn @ B_n            # true field in body frame

        # Apply hard- and soft-iron effects
        # z_ideal = S * (B_b + hard_iron)
        z_ideal = self.soft_iron @ (B_b + self.hard_iron) 

        # Add white measurement noise
        noise = np.random.normal(0.0, self.mag_std, size=3)

        # Add occasional spikes
        spike = self._maybe_spike()

        return z_ideal + noise + spike

    # ---- ESKF-related methods --------------------------------------------

    def H(self, x_nom: NominalState, B_n: np.ndarray) -> np.ndarray:
        """Jacobian wrt error state.

        Measurement model used by the filter (ideal, calibrated magnetometer):

            z = S (R_bn B_n + h_HI)

        with S = soft_iron, h_HI = hard_iron.

        With attitude error δθ in body frame and
            R_bn_true ≈ (I + [δθ×]) R_bn,
        we have:
            z_true = R_bn_true · B_n ≈ (I + [δθ×]) R_bn · B_n
                   = R_bn · B_n + [δθ×] · B_b
                   = z_nom + [δθ×] · B_b

        Using [a]× · b = -[b]× · a:
            δz = z_true - z_nom ≈ [δθ×] · B_b = -[B_b]× · δθ

        so:
            ∂z/∂(δθ) = -[B_b]×
        """
        B_n = np.asarray(B_n, float).reshape(3)

        R_nb = x_nom.ori.as_rotmat()
        R_bn = R_nb.T

        B_b = R_bn @ B_n  # field in body frame before iron effects

        H = np.zeros((3, 6))
        H[0:3, 0:3] = get_skew_matrix(B_b)  # REVERTED: Original sign
        return H

    def pred_from_est(self, x_est: EskfState, B_n: np.ndarray) -> MultiVarGauss[np.ndarray]:
        """Predict magnetometer measurement (mean and covariance) from ESKF state.

        Note:
            We use the *ideal calibrated* model here; bias, spikes and
            miscalibration can be injected in the simulator by using a
            different SensorMagnetometer instance/config for generation.
        """
        x_nom = x_est.nom
        B_n = np.asarray(B_n, float).reshape(3)

        R_nb = x_nom.ori.as_rotmat()
        R_bn = R_nb.T

        B_b = R_bn @ B_n
        z_pred = B_b
        # Don't normalize - B_n is already unit norm and rotation preserves norm
        # z_pred /= np.linalg.norm(z_pred)  # REMOVED: Inconsistent with Jacobian

        H = self.H(x_nom, B_n)
        S = H @ x_est.err.cov @ H.T + self.R

        return MultiVarGauss(mean=z_pred, cov=S)

    def innovation(self, x_est: EskfState, y: np.ndarray, B_n: np.ndarray) -> np.ndarray:
        """Compute innovation ν = y - h(x_est) for magnetometer."""
        z_pred = self.pred_from_est(x_est, B_n).mean
        return np.asarray(y, float).reshape(3) - z_pred

    def kalman_gain(self, x_est: EskfState, B_n: np.ndarray) -> np.ndarray:
        """Compute Kalman gain K for magnetometer measurement."""
        H = self.H(x_est.nom, B_n)
        S = H @ x_est.err.cov @ H.T + self.R
        K = x_est.err.cov @ H.T @ np.linalg.inv(S)
        return K
class SensorSunVector:
    def __init__(self, config_path: str = "config.yaml"):
        config = load_yaml(config_path)
        sun_cfg = config["sensors"]["sun"]

        self.dt = sun_cfg["dt"]
        self.sun_std = sun_cfg["noise"]["sun_std"]

        self.fov_half_angle_rad = np.deg2rad(0.5 * sun_cfg.get("fov_deg", 180.0))
        self.cos_fov_half = np.cos(self.fov_half_angle_rad)

        bias_cfg = sun_cfg.get("bias", {})
        self.bias_rw_std = bias_cfg.get("rw_std", 0.0)
        self.bias = np.asarray(bias_cfg.get("init", [0.0, 0.0, 0.0]), float).reshape(3)

        drop_cfg = sun_cfg.get("dropout", {})
        self.dropout_prob = drop_cfg.get("probability", 0.0)
        self.use_measurements = drop_cfg.get("use_measurements", True)

        spikes_cfg = sun_cfg.get("spikes", {})
        self.spikes_enabled = spikes_cfg.get("enabled", False)
        self.spike_magnitude = spikes_cfg.get("magnitude", 0.0)
        self.spike_probability = spikes_cfg.get("probability", 0.0)

        eps_R = 1e-12
        var = max((self.sun_std**2) * sun_cfg.get("scaling", {}).get("noise_scale", 1.0), eps_R)
        self.R = var * np.eye(3)


    # ---- internal helpers -------------------------------------------------

    def _update_bias(self) -> None:
        """Random-walk update of bias on the sphere (approximate)."""
        if self.bias_rw_std <= 0.0:
            return
        step = np.random.normal(0.0, self.bias_rw_std * np.sqrt(self.dt), size=3)
        self.bias += step

    def _tangential_noise(self, s_b: np.ndarray) -> np.ndarray:
        """Generate noise mostly tangential to the unit sphere at s_b."""
        e = np.random.normal(0.0, self.sun_std, size=3)
        # Remove component along s_b to make it mostly angular
        e -= s_b * (e @ s_b)
        return e

    def _maybe_spike(self) -> np.ndarray:
        """Generate a spike vector or zero."""
        if (
            not self.spikes_enabled
            or self.spike_probability <= 0.0
            or self.spike_magnitude <= 0.0
        ):
            return np.zeros(3)

        if np.random.rand() >= self.spike_probability:
            return np.zeros(3)

        direction = np.random.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            return np.zeros(3)
        direction /= norm
        return self.spike_magnitude * direction

    # ---- sensor model -----------------------------------------------------

    def sample(
        self,
        q_true: Quaternion,
        s_n: np.ndarray,
        in_eclipse: bool = False,
    ) -> np.ndarray | None:
        """Sample a sun vector measurement from the true state.

        Args:
            q_true:    true body-to-navigation quaternion
            s_n:       sun vector in navigation/inertial frame (unit vector)
            in_eclipse: if True, force dropout (no sun)

        Returns:
            Measured unit sun vector in body frame, or None if no measurement.
        """
        if not self.use_measurements:
            return None

        s_n = np.asarray(s_n, float).reshape(3)

        # Rotate sun vector into body frame
        R_nb = q_true.as_rotmat()
        R_bn = R_nb.T
        s_b = R_bn @ s_n

        # Normalize in case of numerical drift
        nrm = np.linalg.norm(s_b)
        if nrm < 1e-12:
            return None
        s_b /= nrm

        # Eclipse or probabilistic dropout
        if in_eclipse:
            return None
        if np.random.rand() < self.dropout_prob:
            return None

        # Field of view: assume boresight is +z_body
        # Only provide measurement if sun is within FOV cone
        cos_incidence = s_b[2]  # dot([0,0,1], s_b)
        if cos_incidence < self.cos_fov_half:
            return None

        spike = self._maybe_spike()

        # Apply bias, noise, spikes
        z = s_b + self.bias + spike

        # Renormalize to unit vector
        nrm = np.linalg.norm(z)
        if nrm < 1e-12:
            # Fallback to noise-free vector
            return s_b
        return z / nrm

    # ---- ESKF related methods (unchanged interface) -----------------------

    def H(self, x_nom: NominalState, s_n: np.ndarray) -> np.ndarray:
        """Jacobian wrt 6D error state [δθ; δb_g].

        Same derivation as magnetometer:
            δz = -[s_b]× · δθ
        so:
            ∂z/∂(δθ) = -[s_b]×
        """
        s_n = np.asarray(s_n, float).reshape(3)
        R_nb = x_nom.ori.as_rotmat()
        R_bn = R_nb.T
        z_pred = R_bn @ s_n

        H = np.zeros((3, 6))
        H[0:3, 0:3] = get_skew_matrix(z_pred)  # REVERTED: Original sign
        return H

    def pred_from_est(self, x_est: EskfState, s_n: np.ndarray) -> MultiVarGauss[np.ndarray]:
        """Predict sun vector measurement (ideal calibrated sensor)."""
        x_nom = x_est.nom
        s_n = np.asarray(s_n, float).reshape(3)
        R_nb = x_nom.ori.as_rotmat()
        R_bn = R_nb.T

        z_pred = R_bn @ s_n

        H = self.H(x_nom, s_n)
        S = H @ x_est.err.cov @ H.T + self.R

        return MultiVarGauss(mean=z_pred, cov=S)

    def innovation(self, x_est: EskfState, y: np.ndarray, s_n: np.ndarray) -> np.ndarray:
        """Compute innovation ν = y - h(x_est) for sun vector."""
        z_pred = self.pred_from_est(x_est, s_n).mean
        return np.asarray(y, float).reshape(3) - z_pred

    def kalman_gain(self, x_est: EskfState, s_n: np.ndarray) -> np.ndarray:
        """Compute Kalman gain K for sun vector measurement."""
        H = self.H(x_est.nom, s_n)
        S = H @ x_est.err.cov @ H.T + self.R
        K = x_est.err.cov @ H.T @ np.linalg.inv(S)
        return K    
class SensorStarTracker:
    def __init__(self, config_path: str = "config.yaml"):
        config = load_yaml(config_path)
        st_cfg = config["sensors"]["star"]

        self.dt: float = st_cfg["dt"]

        noise_cfg = st_cfg.get("noise", {})
        self.st_std: float = noise_cfg["st_std"]

        bias_cfg = st_cfg.get("bias", {})
        self.bias_rw_std: float = bias_cfg.get("rw_std", 0.0)  # rad/√s
        self.bias: np.ndarray = np.asarray(
            bias_cfg.get("init", [0.0, 0.0, 0.0]), float
        ).reshape(3)

        drop_cfg = st_cfg.get("dropout", {})
        self.dropout_prob: float = drop_cfg.get("probability", 0.0)
        self.use_measurements: bool = drop_cfg.get("use_measurements", True)
        self.max_rate_deg_s: float = drop_cfg.get("max_rate_deg_s", np.inf)

        spikes_cfg = st_cfg.get("spikes", {})
        self.spikes_enabled: bool = spikes_cfg.get("enabled", False)
        self.spike_angle: float = spikes_cfg.get("angle", 0.0)
        self.spike_probability: float = spikes_cfg.get("probability", 0.0)

        scale_cfg = st_cfg.get("scaling", {})
        noise_scale = scale_cfg.get("noise_scale", 1.0)

        eps_R = 1e-12
        var = max((self.st_std * noise_scale) ** 2, eps_R)
        self.R = var * np.eye(3)
        logger.debug(f"Star tracker measurement noise covariance R set to:\n{self.R}")

    # ---------- internal helpers ----------

    def _update_bias(self) -> None:
        """Random-walk bias in small-angle space."""
        if self.bias_rw_std <= 0.0:
            return
        step = np.random.normal(0.0, self.bias_rw_std * np.sqrt(self.dt), size=3)
        self.bias += step

    def _maybe_dropout(self, omega_body: np.ndarray | None) -> bool:
        """Return True if we should drop this measurement."""
        if not self.use_measurements:
            return True

        # Rate-dependent dropout (cannot track during fast slew)
        if omega_body is not None:
            rate_deg_s = np.linalg.norm(omega_body) * 180.0 / np.pi
            if rate_deg_s > self.max_rate_deg_s:
                return True

        # Probabilistic dropout
        if np.random.rand() < self.dropout_prob:
            return True

        return False

    def _maybe_spike(self) -> np.ndarray:
        """Return a large-angle error vector (false solution) or zero."""
        if (
            not self.spikes_enabled
            or self.spike_probability <= 0.0
            or self.spike_angle <= 0.0
        ):
            return np.zeros(3)

        if np.random.rand() >= self.spike_probability:
            return np.zeros(3)

        # Random axis with fixed angle self.spike_angle
        axis = np.random.normal(size=3)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return np.zeros(3)
        axis /= norm
        return axis * self.spike_angle

    # ---------- sensor model ----------

    def sample(
        self,
        q_true: Quaternion,
        omega_body: np.ndarray | None = None,
    ) -> Quaternion | None:
        """Sample a star tracker quaternion measurement.

        Args:
            q_true:      true attitude (body-to-nav quaternion)
            omega_body:  body angular rate [rad/s], optional (for rate-based dropout)

        Returns:
            Measured quaternion or None if no measurement.
        """
        if self._maybe_dropout(omega_body):
            return None

        # Small-angle white noise
        delta_theta = np.random.normal(0.0, self.st_std, size=3)

        # Add occasional large spike
        delta_theta += self._maybe_spike()

        # Add bias
        delta_theta += self.bias

        angle = np.linalg.norm(delta_theta)
        if angle < 1e-12:
            q_error = Quaternion(1.0, np.zeros(3))
        else:
            axis = delta_theta / angle
            half_angle = 0.5 * angle
            mu = np.cos(half_angle)
            eta = axis * np.sin(half_angle)
            q_error = Quaternion(mu, eta)

        return q_error.multiply(q_true)

    # ---------- ESKF interface (unchanged) ----------

    def H(self, x_nom: NominalState) -> np.ndarray:
        """Jacobian wrt 6D error state [δθ; δb_g]."""
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)
        return H

    def quat_error(self, q_nom: Quaternion, q_meas: Quaternion) -> np.ndarray:
        """Map quaternion difference to small-angle error δθ."""

        arr_nom = q_nom.as_array()
        arr_meas = q_meas.as_array()

        # enforce same hemisphere
        if np.dot(arr_nom, arr_meas) < 0.0:
            arr_meas = -arr_meas
            q_meas = Quaternion(arr_meas[0], arr_meas[1:4])

        # Error quaternion: δq = q_meas ⊗ q_nom^{-1}
        q_err = q_meas.multiply(q_nom.conjugate())
        return 2.0 * q_err.eta

    def pred_from_est(self, x_est: EskfState, q_meas: Quaternion) -> MultiVarGauss[np.ndarray]:
        """Predict error measurement distribution (δθ space)."""
        H = self.H(x_est.nom)
        S = H @ x_est.err.cov @ H.T + self.R
        z_pred = np.zeros(3)
        return MultiVarGauss(mean=z_pred, cov=S)

    def innovation(self, x_est: EskfState, q_meas: Quaternion) -> np.ndarray:
        """Innovation y = δθ_meas in small-angle space."""
        q_nom = x_est.nom.ori
        return self.quat_error(q_nom, q_meas)

    def kalman_gain(self, x_est: EskfState) -> np.ndarray:
        """Kalman gain for star tracker measurement."""
        H = self.H(x_est.nom)
        S = H @ x_est.err.cov @ H.T + self.R
        return x_est.err.cov @ H.T @ np.linalg.inv(S)

