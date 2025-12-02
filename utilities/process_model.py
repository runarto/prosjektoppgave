from dataclasses import dataclass
import numpy as np

from utilities.states import NominalState
from utilities.utils import get_skew_matrix, load_yaml

from logging_config import get_logger

logger = get_logger(__name__)


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
    
    def __init__(self):
        config = load_yaml("config.yaml")

        # sample time used for process discretization
        self.dt = float(config["process_model"]["dt"])

        # ---- gyro noise from angle random walk ----
        # ARW in deg/√h from datasheet / config, e.g. 0.15
        ARW_deg = float(config["sensors"]["gyro"]["noise"]["arw_deg"])

        # continuous-time angle random-walk density [rad/√s]
        self.noise_rw = ARW_deg * np.pi / 180.0 / np.sqrt(3600.0)

        # this is the continuous-time driving noise for δθ
        self.sigma_g = self.noise_rw

        # ---- bias random walk from bias instability (if provided) ----
        # bias_instability_deg_per_h should be in deg/h (1σ)
        bias_cfg = config["sensors"]["gyro"]["noise"].get("bias_instability_deg_per_h", None)
        BI_deg_per_h = float(bias_cfg)
        # convert to rad/s (1σ change over 1 h)
        b_inst = BI_deg_per_h * np.pi / 180.0 / 3600.0  # rad/s
        # random-walk density σ_bg so that σ_bg * sqrt(3600) ≈ b_inst
        self.sigma_bg = b_inst / np.sqrt(3600.0)
        
        logger.info(f"ProcessModel initialized with σ_g={self.sigma_g:.6e} rad/√s, "
                     f"σ_bg={self.sigma_bg:.6e} rad/s/√s")
   


    @property
    def Q_c(self) -> np.ndarray:
        """Continuous-time process noise covariance Q_c (6x6)."""
        Qg  = (self.sigma_g  ** 2) * np.eye(3)  # attitude / rate driving noise
        Qbg = (self.sigma_bg ** 2) * np.eye(3)  # bias random-walk driving noise
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
        """Second–order discrete-time state transition F ≈ I + A dt + 0.5 A^2 dt^2."""
        A = cls.A(x_nom, omega_meas)           # (6,6)
        A2 = A @ A
        return np.eye(6) + A * dt + 0.5 * A2 * dt**2


    def Q_d(self, x_nom: NominalState, omega_meas: np.ndarray, dt: float) -> np.ndarray:
        """
        Higher-order discrete-time process covariance.

        Q_d ≈ Q_c dt
            + 0.5 (A Q_c + Q_c A^T) dt^2
            + 1/3 A Q_c A^T dt^3
        """
        A = self.A(x_nom, omega_meas)          # same A as in F
        Qc = self.Q_c

        term1 = Qc * dt
        term2 = 0.5 * (A @ Qc + Qc @ A.T) * dt**2
        term3 = (1.0 / 3.0) * (A @ Qc @ A.T) * dt**3

        return term1 + term2 + term3


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
        Qd = self.Q_d(x_nom, omega_meas, dt)

        P_pred = F @ P @ F.T + G @ Qd @ G.T
        # enforce symmetry
        P_pred = 0.5 * (P_pred + P_pred.T)
        return P_pred
