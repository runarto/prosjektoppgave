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
    
    def __init__(self):
        config = load_yaml("config.yaml")
        self.gyro_std = config["sensors"]["gyro"]["gyro_std"]
        self.dt = config["sensors"]["gyro"]["dt"]

    def __post_init__(self):
        self.R = np.eye(3) * self.gyro_std**2
    
    def sample(self, omega_true: np.ndarray) -> np.ndarray:
        """Sample a gyro measurement from the nominal state."""
        noise = np.random.normal(0.0, self.gyro_std, size=3)
        return omega_true + noise


@dataclass
class SensorMagnetometer:

    def __init__(self):
        config = load_yaml("config.yaml")
        self.mag_std = config["sensors"]["mag"]["mag_std"]
        self.dt = config["sensors"]["mag"]["dt"]
        eps_R = 1e-12
        var = max(self.mag_std**2, eps_R)
        self.R = var * np.eye(3)
        logger.debug(f"Magnetometer measurement noise covariance R set to:\n{self.R}")

    def __post_init__(self):
        eps_R = 1e-12
        var = max(self.mag_std**2, eps_R)
        self.R = var * np.eye(3)
        logger.debug(f"Magnetometer measurement noise covariance R set to:\n{self.R}")
        
        
        
    def sample(self, q_true: Quaternion, b_eci: np.ndarray) -> np.ndarray:
        """Sample a magnetometer measurement from the nominal state.

        Args:
            b_eci: Magnetic field in navigation/inertial frame
        """
        R_nb = q_true.as_rotmat()
        R_bn = R_nb.T
        z_true = R_bn @ b_eci
        noise = np.random.normal(0.0, self.mag_std, size=3)
        return z_true + noise

    def H(self, x_nom: NominalState, B_n: np.ndarray[3]) -> 'np.ndarray[3, 6]':
        """Jacobian wrt error state.

        z = R_bn * B_n

        With attitude error δθ in body frame and R_nb_true ≈ R_nb (I - [δθ×]),
        you get δz ≈ [z_pred×] δθ, so:

            ∂z/∂(δθ) = [z_pred×]
        """
        B_n = np.asarray(B_n).reshape(3)
        R_nb = x_nom.ori.as_rotmat()
        R_bn = R_nb.T
        z_pred = R_bn @ B_n

        H = np.zeros((3, 6))
        H[:, 0:3] = get_skew_matrix(z_pred)   # attitude error block
        return H

    def pred_from_est(self, x_est: EskfState, B_n: 'np.ndarray[3]') -> MultiVarGauss[np.ndarray]:
        """Predict magnetometer measurement.

        Args:
            x_est: ESKF state
            B_n: magnetic field in navigation/inertial frame
        """
        x_nom = x_est.nom
        B_n = np.asarray(B_n, float).reshape(3)
        R_nb = x_nom.ori.as_rotmat()
        R_bn = R_nb.T

        z_pred = R_bn @ B_n

        H = self.H(x_nom, B_n)
        S = H @ x_est.err.cov @ H.T + self.R

        return MultiVarGauss(mean=z_pred, cov=S)
    
    def innovation(self, x_est: EskfState, y: 'np.ndarray[3]', B_n: 'np.ndarray[3]') -> np.ndarray:
        """Compute innovation y = z - h(x_est) for magnetometer.

        Args:
            x_est: ESKF state
            y: actual magnetometer measurement
            B_n: magnetic field in navigation/inertial frame
        """
        z_pred = self.pred_from_est(x_est, B_n).mean
        return y.reshape(3) - z_pred
    
    def kalman_gain(self, x_est: EskfState, B_n: 'np.ndarray[3]') -> 'np.ndarray[6, 3]':
        """Compute Kalman gain K for magnetometer measurement.

        Args:
            x_est: ESKF state
            B_n: magnetic field in navigation/inertial frame
        """
        H = self.H(x_est.nom, B_n)
        S = H @ x_est.err.cov @ H.T + self.R
        K = x_est.err.cov @ H.T @ np.linalg.inv(S)
        return K
    
    


@dataclass
class SensorSunVector:

    def __init__(self):
        config = load_yaml("config.yaml")
        self.sun_std = config["sensors"]["sun"]["sun_std"]
        self.dt = config["sensors"]["sun"]["dt"]
        eps_R = 1e-12
        var = max(self.sun_std**2, eps_R)
        self.R = var * np.eye(3)
        logger.debug(f"Sun sensor measurement noise covariance R set to:\n{self.R}")
        
    def __post_init__(self):
        eps_R = 1e-12
        var = max(self.sun_std**2, eps_R)
        self.R = var * np.eye(3)
        logger.debug(f"Sun sensor measurement noise covariance R set to:\n{self.R}")
        

    def H(self, x_nom: NominalState, s_n: np.ndarray) -> np.ndarray:
        """Jacobian wrt 6D error state [δθ; δb_g].

        z = R_bn * s_n

        With attitude error δθ in body frame and
        R_nb_true ≈ R_nb (I - [δθ×]), you get

            δz ≈ [z_pred×] δθ

        so

            ∂z/∂(δθ) = [z_pred×],   ∂z/∂(δb_g) = 0
        """
        s_n = np.asarray(s_n, float).reshape(3)
        R_nb = x_nom.ori.as_rotmat()
        R_bn = R_nb.T
        z_pred = R_bn @ s_n

        H = np.zeros((3, 6))
        H[:, 0:3] = get_skew_matrix(z_pred)   # attitude block
        # H[:, 3:6] stays zero (no direct dependence on gyro bias)
        return H

    
    def sample(self, q_true: Quaternion, s_eci: np.ndarray) -> np.ndarray:
        """Sample a sun vector measurement from the nominal state.

        Args:
            s_eci: Sun vector in navigation/inertial frame (unit vector)
        """
        R_nb = q_true.as_rotmat()
        R_bn = R_nb.T
        z_true = R_bn @ s_eci
        noise = np.random.normal(0.0, self.sun_std, size=3)
        return z_true + noise

    def pred_from_est(self, x_est: EskfState, s_n: 'np.ndarray[3]') -> MultiVarGauss[np.ndarray]:
        """Predict sun vector measurement."""
        x_nom = x_est.nom
        s_n = np.asarray(s_n, float).reshape(3)
        R_nb = x_nom.ori.as_rotmat()
        R_bn = R_nb.T

        z_pred = R_bn @ s_n

        H = self.H(x_nom, s_n)
        S = H @ x_est.err.cov @ H.T + self.R

        return MultiVarGauss(mean=z_pred, cov=S)
    
    def innovation(self, x_est: EskfState, y: 'np.ndarray[3]', s_n: 'np.ndarray[3]') -> np.ndarray:
        """Compute innovation y = z - h(x_est) for sun vector."""
        z_pred = self.pred_from_est(x_est, s_n).mean
        return y.reshape(3) - z_pred
    
    def kalman_gain(self, x_est: EskfState, s_n: 'np.ndarray[3]') -> 'np.ndarray[6, 3]':
        """Compute Kalman gain K for sun vector measurement."""
        H = self.H(x_est.nom, s_n)
        S = H @ x_est.err.cov @ H.T + self.R
        K = x_est.err.cov @ H.T @ np.linalg.inv(S)
        return K
    
@dataclass
class SensorStarTracker:

    def __init__(self):
        config = load_yaml("config.yaml")
        self.st_std = config["sensors"]["star"]["st_std"]
        self.dt = config["sensors"]["star"]["dt"]
        eps_R = 1e-12
        var = max(self.st_std**2, eps_R)
        self.R = var * np.eye(3)
        logger.debug(f"Star tracker measurement noise covariance R set to:\n{self.R}")

    def __post_init__(self):
        eps_R = 1e-12
        var = max(self.st_std**2, eps_R)
        self.R = var * np.eye(3)
        logger.debug(f"Star tracker measurement noise covariance R set to:\n{self.R}")
        
    def sample(self, q_true: Quaternion) -> Quaternion:
        """Sample a star tracker measurement from the true quaternion."""
        # small-angle error
        delta_theta = np.random.normal(0.0, self.st_std, size=3)
        angle = np.linalg.norm(delta_theta)
        if angle < 1e-12:
            q_error = Quaternion(1.0, np.zeros(3))
        else:
            axis = delta_theta / angle
            half_angle = angle / 2.0
            mu = np.cos(half_angle)
            eta = axis * np.sin(half_angle)
            q_error = Quaternion(mu, eta)
        q_meas = q_error.multiply(q_true).normalize()
        return q_meas

    def H(self, x_nom: NominalState) -> np.ndarray:
        """
        Jacobian wrt 6D error state [δθ; δb_g].

        Star tracker measurement model in small-angle form:

            z = δθ + v

        so

            ∂z/∂(δθ) = I
            ∂z/∂(δb_g) = 0
        """
        H = np.zeros((3, 6))
        H[:, 0:3] = np.eye(3)   # attitude error
        # H[:, 3:6] stays zero   # gyro bias has no direct effect
        return H


    @staticmethod
    def quat_error(q_nom: Quaternion, q_meas: Quaternion) -> np.ndarray[3]:
        """Map quaternion difference to small-angle error δθ.

        q_err = q_meas ⊗ q_nom^{-1} ~ [1, 0.5 δθ] for small δθ.
        """
        q_nom_inv = q_nom.conjugate()
        # δq ≈ q_nom^{-1} ⊗ q_meas
        q_err = q_nom_inv.multiply(q_meas).normalize()

        w = q_err.mu
        v = q_err.eta
        return 2.0 * v

    def pred_from_est(self, x_est: EskfState, q_meas: Quaternion) -> MultiVarGauss[np.ndarray]:
        """Predict the *error* measurement for the star tracker.

        We choose the measurement space as δθ, so

            h(x_nom) = 0,   z = δθ_meas

        The Gaussian returned here is what your filter expects for the
        innovation model (zero mean, covariance S).
        """
        x_nom = x_est.nom

        # predicted measurement mean in δθ-space is zero
        z_pred = np.zeros(3)

        H = self.H(x_nom)
        S = H @ x_est.err.cov @ H.T + self.R

        return MultiVarGauss(mean=z_pred, cov=S)

    def innovation(self, x_est: EskfState, q_meas: Quaternion) -> np.ndarray:
        """Compute innovation y = z - h(x_est) in δθ-space.

        Here: h(x_est) = 0, so y = δθ_meas.
        """
        q_nom = x_est.nom.ori
        return self.quat_error(q_nom, q_meas)
    
    def kalman_gain(self, x_est: EskfState) -> 'np.ndarray[6, 3]':
        """Compute Kalman gain K for star tracker measurement."""
        H = self.H(x_est.nom)
        S = H @ x_est.err.cov @ H.T + self.R
        K = x_est.err.cov @ H.T @ np.linalg.inv(S)
        return K

