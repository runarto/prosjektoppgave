from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from logging_config import get_logger
from utilities.states import EskfState, SensorType
from utilities.quaternion import Quaternion
from utilities.process_model import ProcessModel
from utilities.sensors import SensorGyro, SensorMagnetometer, SensorStarTracker, SensorSunVector

logger = get_logger(__name__)

@dataclass
class ESKF:
    P0: np.ndarray              # initial error covariance

    def __post_init__(self):
        self.sens_mag = SensorMagnetometer()
        self.sv       = SensorSunVector()
        self.st       = SensorStarTracker()
        self.gyro     = SensorGyro()
        self.process = ProcessModel()

    # ---------- PREDICT ----------

    def predict(self, x_est: EskfState, omega_meas: np.ndarray, dt: float) -> EskfState:
        # 1) propagate nominal attitude (you already do this)
        quat = x_est.nom.ori.propagate(omega_meas - x_est.nom.gyro_bias, dt)
        x_est.nom.ori = quat
        x_est.nom.gyro_bias = x_est.nom.gyro_bias  # stays the same

        # 2) propagate covariance for 6D error state
            
        P = x_est.err.cov
        P_pred = self.process.propagate_covariance(P, x_est.nom, omega_meas, dt)
        x_est.err.cov = P_pred
        x_est.err.mean[:] = 0.0  # stays zero in ESKF
        
        if not np.isfinite(P_pred).all():
            raise ValueError("Non-finite P_pred produced in predict")

        if not np.isfinite(x_est.nom.gyro_bias).all():
            raise ValueError("Non-finite gyro_bias produced in predict")


        return x_est

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

                # predicted measurement z_pred ~ N(mean, S)
                z_pred_gauss = self.sens_mag.pred_from_est(x_est, B_n)
                z_pred = z_pred_gauss.mean.reshape(-1)

                y_vec = np.asarray(y, float).reshape(-1)
                innovation = y_vec - z_pred

                H = self.sens_mag.H(x_est.nom, B_n)
                R = self.sens_mag.R

            case SensorType.SUN_VECTOR:
                assert s_n is not None, "s_n must be provided for sun-vector update"

                z_pred_gauss = self.sv.pred_from_est(x_est, s_n)
                z_pred = z_pred_gauss.mean.reshape(-1)

                y_vec = np.asarray(y, float).reshape(-1)
                innovation = y_vec - z_pred

                H = self.sv.H(x_est.nom, s_n)
                R = self.sv.R

            case SensorType.STAR_TRACKER:
                # y is a Quaternion measurement
                assert isinstance(y, Quaternion)

                # measurement space is small-angle error δθ, predicted mean = 0
                innovation = self.st.innovation(x_est=x_est, q_meas=y).reshape(-1)

                H = self.st.H(x_est.nom)
                R = self.st.R

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

        # Inject error into nominal state and reset error
        x_est.inject_error(delta_x)          # you implement this
        x_est.err.mean[:] = 0.0
        x_est.err.cov = P_upd
        
        logging_msg = (f"Updated with {sensor_type.name}: "
                       f"innovation = {innovation}, "
                       f"K = {K}, "
                       f"P = {x_est.err.cov}")
        logger.debug(logging_msg)

        return x_est
