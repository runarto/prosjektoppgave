import numpy as np
from dataclasses import dataclass
from enum import Enum
from utilities.quaternion import Quaternion
from utilities.gaussian import MultiVarGauss


@dataclass
class NominalState:
    """Nominal state (reduced):

    Attributes:
        ori       : orientation as a quaternion
        gyro_bias : gyroscope bias (3,)
    """
    ori: Quaternion
    gyro_bias: np.ndarray  # shape (3,)

    def diff(self, other: 'NominalState') -> 'ErrorState':
        """Difference between two nominal states as an error state.

        Used e.g. for NEES evaluation.
        """
        return ErrorState(
            avec=self.ori.diff_as_avec(other.ori),
            gyro_bias=self.gyro_bias - other.gyro_bias,
        )

    @property
    def euler(self) -> np.ndarray:
        """Orientation as Euler angles (roll, pitch, yaw) in NED."""
        return self.ori.as_euler()


@dataclass
class ErrorState:
    """Error state (reduced):

    avec      : small-angle attitude error vector (3,)
    gyro_bias : gyroscope bias error (3,)
    """
    avec: np.ndarray  # shape (3,)
    gyro_bias: np.ndarray  # shape (3,)

    def as_vector(self) -> np.ndarray:
        """Pack error state into a 6×1 vector [avec; gyro_bias]."""
        return np.concatenate(
            (np.asarray(self.avec, float).reshape(3),
             np.asarray(self.gyro_bias, float).reshape(3))
        )
        
    def to_quaternion(self) -> Quaternion:
        """Convert the small-angle attitude error vector (avec) to a quaternion."""
        angle = np.linalg.norm(self.avec)
        if angle < 1e-12:
            return Quaternion(1.0, np.zeros(3))  # No rotation

        axis = self.avec / angle
        half_angle = angle / 2.0
        mu = np.cos(half_angle)
        eta = axis * np.sin(half_angle)
        return Quaternion(mu, eta)

    @staticmethod
    def from_vector(v: np.ndarray) -> 'ErrorState':
        """Unpack error state from 6×1 vector [avec; gyro_bias]."""
        v = np.asarray(v, float).reshape(6)
        return ErrorState(
            avec=v[0:3],
            gyro_bias=v[3:6],
        )
        
    


@dataclass
class EskfState:
    """Combined nominal and error state for the ESKF.

    Attributes:
        nom : nominal state
        err : Gaussian over the 6D error state vector
              (mean usually zero; covariance P)
    """
    nom: NominalState
    err: MultiVarGauss[np.ndarray]  # mean is a 6D error vector

    @property
    def ndim(self) -> int:
        """Dimension of the error state."""
        return self.err.mean.shape[0]

    def get_err_gauss(self, gt: NominalState) -> MultiVarGauss[np.ndarray]:
        """Gaussian over the error state relative to a ground truth nominal state.

        This is useful for NEES calculations: we construct the
        true error state (as ErrorState), convert it to a vector,
        and reuse the current covariance.
        """
        err_state = ErrorState(
            avec=gt.ori.diff_as_avec(self.nom.ori),
            gyro_bias=self.nom.gyro_bias - gt.gyro_bias,
        )
        err_vec = err_state.as_vector()
        return MultiVarGauss(mean=err_vec, cov=self.err.cov)
    
    def inject_error(self, delta_x: np.ndarray) -> 'EskfState':
        """Inject an error state into the nominal state.

        This updates the nominal state by applying the error, and resets
        the error state mean to zero. The covariance remains unchanged.
        """
        delta_x = np.asarray(delta_x, float).reshape(6)
        delta_avec = delta_x[0:3]
        delta_gyro_bias = delta_x[3:6]

        # 1) Update nominal orientation: q_new = δq ⊗ q_nom
        delta_q = Quaternion.from_avec(avec=delta_avec)
        self.nom.ori = delta_q.multiply(self.nom.ori)

        # 2) Update nominal gyro bias (additive)
        self.nom.gyro_bias += delta_gyro_bias

        # 3) Reset error-state mean (ESKF convention)
        self.err.mean[:] = 0.0

        return self

        
class SensorType(Enum):
    MAGNETOMETER = 1
    SUN_VECTOR = 2
    STAR_TRACKER = 3
    GYRO = 4