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

    Standard second and third-order discrete-time approximation (small Δt):

        F ≈ I + A Δt + 0.5 A² Δt²
        Q_d = exp(F) G Q_c Gᵀ ≈ Q_c Δt
              + 0.5 (A Q_c + Q_c Aᵀ) Δt²
              + 1/3 A Q_c Aᵀ Δt³
        G = I

    Covariance prediction:

        P_{k+1} = F P_k Fᵀ + G Q_d Gᵀ
    """
    
    def __init__(self, config_path: str = "config.yaml") -> None:
        config = load_yaml(config_path)

        # sample time used for process discretization
        self.dt = float(config["process_model"]["dt"])

        # ---- Angle Random Walk (gyro white noise) ----
        # ARW in deg/√h from datasheet (e.g., STIM300: 0.15 deg/√h)
        # Meaning: after integrating for time t, angle error std = ARW × √(t in hours)
        ARW_deg = float(config["sensors"]["gyro"]["noise"]["arw_deg"])

        # Convert to continuous-time noise density [rad/√s]
        # σ_g such that σ_angle = σ_g × √(t in seconds)
        # ARW [deg/√h] → σ_g [rad/√s]: multiply by (π/180) and divide by √3600 = 60
        self.sigma_g = ARW_deg * (np.pi / 180.0) / 60.0

        # ---- Rate Random Walk (bias drift) ----
        # RRW in deg/h/√h from datasheet or estimated (e.g., STIM300: ~0.5 deg/h/√h)
        # Meaning: after time t, bias std = RRW × √(t in hours) [deg/h]
        RRW_deg = float(config["sensors"]["gyro"]["noise"]["rrw_deg"])

        # Convert to continuous-time noise density [rad/s/√s]
        # σ_bg such that σ_bias = σ_bg × √(t in seconds) [rad/s]
        # RRW [deg/h/√h] → σ_bg [rad/s/√s]:
        #   - deg/h to rad/s: multiply by (π/180)/3600
        #   - 1/√h to 1/√s: divide by √3600 = 60
        self.sigma_bg = RRW_deg * (np.pi / 180.0) / 3600.0 / 60.0

        # ---- Discrete gyro sample noise ----
        # For sampling at interval dt, per-sample std = σ_g / √dt
        gyro_dt = float(config["sensors"]["gyro"]["dt"])
        self.gyro_std = self.sigma_g / np.sqrt(gyro_dt)

        # ---- Process noise scaling factor ----
        # Allows tuning filter confidence without changing physical gyro specs
        self.noise_scale = float(config["sensors"]["gyro"].get("noise_scale", 1.0))

        logger.info(f"ProcessModel initialized with σ_g={self.sigma_g:.6e} rad/√s, "
                    f"σ_bg={self.sigma_bg:.6e} rad/s/√s, gyro_std={self.gyro_std:.6e} rad/s")
   


    @property
    def Q_c(self) -> np.ndarray:
        """Continuous-time process noise covariance Q_c (6x6)."""
        # Apply noise scaling to prevent filter over-confidence
        Qg  = (self.sigma_g  ** 2) * self.noise_scale * np.eye(3)  # attitude / rate driving noise
        Qbg = (self.sigma_bg ** 2) * self.noise_scale * np.eye(3)  # bias random-walk driving noise
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
