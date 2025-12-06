from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from logging_config import get_logger
from utilities.states import EskfState, SensorType
from utilities.quaternion import Quaternion
from utilities.process_model import ProcessModel
from utilities.sensors import SensorGyro, SensorMagnetometer, SensorStarTracker, SensorSunVector
from data.db import SimulationDatabase

logger = get_logger(__name__)

def regularize_covariance(P: np.ndarray, min_eigenvalue: float = 1e-14) -> np.ndarray:
    """
    Regularize covariance matrix to prevent numerical issues.

    Ensures all eigenvalues are above a minimum threshold by adding
    a small diagonal term if needed. This prevents the matrix from
    becoming numerically singular.

    Args:
        P: Covariance matrix (n x n)
        min_eigenvalue: Minimum allowed eigenvalue

    Returns:
        Regularized covariance matrix
    """
    eigvals = np.linalg.eigvals(P)
    min_eig = np.min(np.real(eigvals))

    if min_eig < min_eigenvalue:
        # Add diagonal regularization
        delta = min_eigenvalue - min_eig
        P_reg = P + delta * np.eye(P.shape[0])
        return 0.5 * (P_reg + P_reg.T)  # Ensure symmetry

    return P

@dataclass
class ESKF:
    P0: np.ndarray              # initial error covariance
    config_path: str = "config.yaml"
    chi2_threshold: float = 7.81  # 99.9% confidence, 3 DOF (chi2(3, 0.999))
    chi2_threshold_sun: float = 1e10   # Effectively disabled - let sun sensor correct drift
    chi2_threshold_star: float = 1e10  # Effectively disabled - let star tracker correct drift

    def __post_init__(self):
        self.sens_mag = SensorMagnetometer(config_path=self.config_path)
        self.sv       = SensorSunVector(config_path=self.config_path)
        self.st       = SensorStarTracker(config_path=self.config_path)
        self.gyro     = SensorGyro(config_path=self.config_path)
        self.process = ProcessModel(config_path=self.config_path)

    # ---------- PREDICT ----------

    def predict(self, x_est: EskfState, omega_meas: np.ndarray, dt: float) -> EskfState:
        # 1) propagate nominal attitude
        quat = x_est.nom.ori.propagate(omega_meas - x_est.nom.gyro_bias, dt)
        x_est.nom.ori = quat
        x_est.nom.gyro_bias = x_est.nom.gyro_bias

        # 2) propagate covariance for 6D error state

        P = x_est.err.cov
        P_pred = self.process.propagate_covariance(P, x_est.nom, omega_meas, dt)

        # Regularize to prevent numerical issues
        P_pred = regularize_covariance(P_pred, min_eigenvalue=1e-14)

        x_est.err.cov = P_pred
        x_est.err.mean[:] = 0.0  # stays zero in ESKF

        if not np.isfinite(P_pred).all():
            raise ValueError("Non-finite P_pred produced in predict")

        if not np.isfinite(x_est.nom.gyro_bias).all():
            raise ValueError("Non-finite gyro_bias produced in predict")


        return EskfState(nom=x_est.nom, err=x_est.err)

    # ---------- UPDATE (ALL SENSORS) ----------

    def update(self,
               x_est: EskfState,
               y: np.ndarray | Quaternion,
               sensor_type: SensorType,
               B_n: Optional[np.ndarray] = None,
               s_n: Optional[np.ndarray] = None) -> EskfState:
        """Generic ESKF measurement update with match-case dispatch.

        Args:
            x_est:       current ESKF state (nominal + error)
            y:           measurement
            sensor_type: which sensor produced y
            B_n:         magnetic field in navigation frame (for magnetometer)
            s_n:         sun vector in navigation frame (for sun sensor)

        Returns:
            Updated ESKF state (nominal updated, error mean reset, P updated).
        """
        P = x_est.err.cov
        n = x_est.err.ndim
        I = np.eye(n)

        # to be filled by each case
        innovation: np.ndarray
        H: np.ndarray
        R: np.ndarray

        match sensor_type:

            case SensorType.MAGNETOMETER:
                assert B_n is not None, "B_n must be provided for magnetometer update"
                # Normalize measurement (database stores normalized vectors)
                y = y / np.linalg.norm(y)

                # Also normalize B_n for consistent comparison
                B_n_norm = B_n / np.linalg.norm(B_n)

                # predicted measurement z_pred ~ N(mean, S)
                innovation = self.sens_mag.innovation(x_est, y, B_n_norm)
                H = self.sens_mag.H(x_est.nom, B_n_norm)
                R = self.sens_mag.R

                # Chi-squared test for outlier rejection
                S = H @ P @ H.T + R
                mahalanobis_dist_sq = innovation.T @ np.linalg.solve(S, innovation)
                if mahalanobis_dist_sq > self.chi2_threshold:
                    raise ValueError(f"Magnetometer innovation too large (chi2={mahalanobis_dist_sq:.2f}), possible bad measurement.")

            case SensorType.SUN_VECTOR:
                assert s_n is not None, "s_n must be provided for sun-vector update"

                innovation = self.sv.innovation(x_est, y, s_n)
                H = self.sv.H(x_est.nom, s_n)
                R = self.sv.R

                # Chi-squared test for outlier rejection (relaxed for sun sensor)
                S = H @ P @ H.T + R
                mahalanobis_dist_sq = innovation.T @ np.linalg.solve(S, innovation)
                if mahalanobis_dist_sq > self.chi2_threshold_sun:
                    raise ValueError(f"Sun-vector innovation too large (chi2={mahalanobis_dist_sq:.2f}), possible bad measurement.")

            case SensorType.STAR_TRACKER:
                # y is a Quaternion measurement
                assert isinstance(y, Quaternion)

                innovation = self.st.innovation(x_est, y)
                H = self.st.H(x_est.nom)
                R = self.st.R

                # Chi-squared test for outlier rejection (relaxed for star tracker)
                S = H @ P @ H.T + R
                mahalanobis_dist_sq = innovation.T @ np.linalg.solve(S, innovation)
                if mahalanobis_dist_sq > self.chi2_threshold_star:
                    raise ValueError(f"Star tracker innovation too large (chi2={mahalanobis_dist_sq:.2f}), possible bad measurement.")

            case _:
                raise ValueError(f"Unsupported sensor type: {sensor_type}")

        # Common Kalman update for error state

        # innovation covariance S = H P H^T + R
        S = H @ P @ H.T + R

        # Kalman gain: K = P H^T S^{-1}
        # use solve rather than explicit inverse
        K = P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))

        # error-state mean: δx = K * innovation
        delta_x = K @ innovation

        # Joseph form for covariance
        I_KH = I - K @ H
        P_upd = I_KH @ P @ I_KH.T + K @ R @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)

        # Regularize to prevent numerical issues
        P_upd = regularize_covariance(P_upd, min_eigenvalue=1e-14)

        # Inject error into nominal state and reset error
        x_est.inject_error(delta_x)
        x_est.err.mean[:] = 0.0
        x_est.err.cov = P_upd

        return EskfState(nom=x_est.nom, err=x_est.err)
