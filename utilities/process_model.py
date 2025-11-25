from dataclasses import dataclass
import numpy as np

from utilities.states import NominalState
from utilities.utils import get_skew_matrix


@dataclass
class ProcessModel:
    """
    Error-state process model for attitude + gyro bias:

        δx = [δθ; δb_g] ∈ R^6

    Continuous-time linearized error dynamics:

        δθ̇ = -[ω - b_g]× δθ - δb_g + n_g
        δḃ_g = w_bg

    with white Gaussian noise:

        n_g   ~ N(0, σ_g^2 I_3)
        w_bg  ~ N(0, σ_bg^2 I_3)

    Standard discrete-time approximation (small Δt):

        F ≈ I + A Δt
        Q_d ≈ Q_c Δt
        G = I

    Covariance prediction:

        P_{k+1} = F P_k Fᵀ + G Q_d Gᵀ
    """
    sigma_g: float   # gyro white noise std [rad/s]
    sigma_bg: float  # gyro bias random-walk std [rad/s^2]

    @property
    def Q_c(self) -> np.ndarray:
        """Continuous-time noise covariance Q_c (6x6)."""
        Qg  = (self.sigma_g ** 2) * np.eye(3)
        Qbg = (self.sigma_bg ** 2) * np.eye(3)
        return np.block([
            [Qg,                np.zeros((3, 3))],
            [np.zeros((3, 3)),  Qbg            ],
        ])

    @staticmethod
    def G() -> np.ndarray:
        """Noise input matrix G (6x6). Standard assumption: G = I."""
        return np.eye(6)

    @staticmethod
    def A(x_nom: NominalState, omega_meas: np.ndarray) -> np.ndarray:
        """
        Continuous-time error dynamics matrix A (6x6).

        Uses nominal gyro bias and measured ω to form ω̂ = ω_meas - b_g.
        """
        omega_meas = np.asarray(omega_meas, float).reshape(3)
        b_g = np.asarray(x_nom.gyro_bias, float).reshape(3)

        omega_hat = omega_meas - b_g
        Omega = get_skew_matrix(omega_hat)  # [ω̂×]

        A = np.zeros((6, 6))
        # δθ̇ block
        A[0:3, 0:3] = -Omega          # -[ω̂×] δθ
        A[0:3, 3:6] = -np.eye(3)      # -δb_g
        # δḃ_g block is zero
        return A

    @classmethod
    def F(cls, x_nom: NominalState, omega_meas: np.ndarray, dt: float) -> np.ndarray:
        """Discrete-time state transition matrix F ≈ I + A Δt."""
        A = cls.A(x_nom, omega_meas)
        return np.eye(6) + A * dt

    def Q_d(self, dt: float) -> np.ndarray:
        """Discrete-time process covariance Q_d ≈ Q_c Δt."""
        return self.Q_c * dt

    def propagate_covariance(self,
                             P: np.ndarray,
                             x_nom: NominalState,
                             omega_meas: np.ndarray,
                             dt: float) -> np.ndarray:
        """
        Covariance prediction:

            P⁺ = F P Fᵀ + G Q_d Gᵀ

        where F, G, Q_d use the standard small-Δt assumptions.
        """
        P = np.asarray(P, float).reshape(6, 6)

        F = self.F(x_nom, omega_meas, dt)
        G = self.G()
        Qd = self.Q_d(dt)

        P_pred = F @ P @ F.T + G @ Qd @ G.T
        # enforce symmetry
        P_pred = 0.5 * (P_pred + P_pred.T)
        return P_pred
