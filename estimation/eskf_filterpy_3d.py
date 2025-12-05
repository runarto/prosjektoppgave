"""
3D Attitude-Only ESKF using FilterPy (no bias estimation).

This is a simplified version for scenarios where gyro bias is zero or negligible.
The error state is just δθ ∈ R^3 (attitude error).
"""

from __future__ import annotations
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from dataclasses import dataclass
from typing import Optional

from utilities.quaternion import Quaternion
from utilities.states import NominalState, SensorType
from utilities.sensors import SensorMagnetometer, SensorStarTracker, SensorSunVector
from utilities.process_model import ProcessModel
from utilities.utils import get_skew_matrix


def regularize_covariance(P: np.ndarray, min_eigenvalue: float = 1e-9) -> np.ndarray:
    """
    Regularize covariance matrix to prevent numerical issues.

    Ensures all eigenvalues are above a minimum threshold by adding
    a small diagonal term if needed.

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
class ESKFFilterPy3D:
    """
    3D Attitude-only ESKF using FilterPy's EKF.

    The nominal state is just the orientation quaternion.
    FilterPy's EKF operates on the 3D error state: δθ
    No gyro bias estimation.
    """
    config_path: str = "config.yaml"

    def __post_init__(self):
        # Sensors
        self.sens_mag = SensorMagnetometer(config_path=self.config_path)
        self.sens_sun = SensorSunVector(config_path=self.config_path)
        self.sens_star = SensorStarTracker(config_path=self.config_path)

        # Process model
        self.process = ProcessModel(config_path=self.config_path)

        # Create FilterPy EKF for 3D error state (attitude only)
        self.ekf = ExtendedKalmanFilter(dim_x=3, dim_z=3)

        # Nominal state (orientation only)
        self.nom_ori = Quaternion(1.0, np.zeros(3))

    def initialize(self, q0: Quaternion, P0: np.ndarray):
        """Initialize the filter state.

        Args:
            q0: Initial orientation quaternion
            P0: Initial attitude error covariance (3x3)
        """
        self.nom_ori = q0.copy()

        # Error state starts at zero
        self.ekf.x = np.zeros(3)
        self.ekf.P = P0.copy()

    def predict(self, omega_meas: np.ndarray, dt: float):
        """
        Predict step.

        1. Propagate nominal orientation (no bias correction)
        2. Propagate error covariance using FilterPy EKF
        """
        omega_meas = np.asarray(omega_meas, float).reshape(3)

        # 1. Propagate nominal orientation (assume zero bias)
        self.nom_ori = self.nom_ori.propagate(omega_meas, dt)

        # 2. Propagate error state covariance
        # Get the 6D dynamics and extract the 3x3 attitude block
        A_full = self.process.A(
            NominalState(ori=self.nom_ori, gyro_bias=np.zeros(3)),
            omega_meas
        )
        A = A_full[0:3, 0:3]  # Just δθ̇ = -[ω×]δθ
        F = np.eye(3) + A * dt + 0.5 * (A @ A) * dt**2

        # Process noise - just the attitude block
        Q_full = self.process.Q_d(
            NominalState(ori=self.nom_ori, gyro_bias=np.zeros(3)),
            omega_meas,
            dt
        )
        Q = Q_full[0:3, 0:3]

        # Set F and Q on the EKF object
        self.ekf.F = F
        self.ekf.Q = Q

        # Predict (error state stays at zero, only covariance propagates)
        self.ekf.predict()

        # No regularization - let covariance evolve naturally
        # self.ekf.P = regularize_covariance(self.ekf.P, min_eigenvalue=1e-8)

    def update_magnetometer(self, mag_meas: np.ndarray, B_eci: np.ndarray):
        """Update with magnetometer measurement."""
        mag_meas = np.asarray(mag_meas, float).reshape(3)
        B_eci = np.asarray(B_eci, float).reshape(3)

        # Predicted measurement from nominal state
        R_bn = self.nom_ori.as_rotmat().T
        z_pred = R_bn @ B_eci

        # Measurement function: h(x) where x is the error state
        def Hx(x):
            """Predicted measurement given error state."""
            return z_pred

        # Jacobian function
        def HJacobian(x):
            """Compute measurement Jacobian H."""
            # H = [B_b]× for magnetometer
            B_b = R_bn @ B_eci
            H = get_skew_matrix(B_b)
            return H

        # Measurement noise
        R = self.sens_mag.R

        # FilterPy update
        self.ekf.update(z=mag_meas, HJacobian=HJacobian, Hx=Hx, R=R)

        # No regularization
        # self.ekf.P = regularize_covariance(self.ekf.P, min_eigenvalue=1e-8)

        # Inject error into nominal state
        self._inject_error()

    def update_star_tracker(self, q_meas: Quaternion):
        """Update with star tracker measurement."""
        # For star tracker, the measurement is a quaternion
        # We convert it to angle-axis error for the update

        # Expected measurement (zero error in angle-axis)
        def Hx(x):
            """Predicted measurement given error state."""
            return np.zeros(3)

        # Jacobian: H = I for attitude
        def HJacobian(x):
            """Compute measurement Jacobian H."""
            H = np.eye(3)
            return H

        # "Measurement" is the attitude error in angle-axis form
        z = self.sens_star.quat_error(self.nom_ori, q_meas)

        # Measurement noise
        R = self.sens_star.R

        # FilterPy update
        self.ekf.update(z=z, HJacobian=HJacobian, Hx=Hx, R=R)

        # No regularization
        # self.ekf.P = regularize_covariance(self.ekf.P, min_eigenvalue=1e-8)

        # Inject error into nominal state
        self._inject_error()

    def _inject_error(self):
        """Inject error state into nominal state and reset error to zero."""
        delta_theta = self.ekf.x

        # Update nominal orientation: q_new = Exp(δθ) ⊗ q_nom
        delta_q = Quaternion.from_avec(delta_theta)
        self.nom_ori = delta_q.multiply(self.nom_ori).normalize()

        # Reset error state to zero
        self.ekf.x = np.zeros(3)

    @property
    def orientation(self) -> Quaternion:
        """Get current orientation estimate."""
        return self.nom_ori

    @property
    def covariance(self) -> np.ndarray:
        """Get current error state covariance."""
        return self.ekf.P.copy()
