from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import gtsam
from gtsam import Rot3, Point3

from environment.environment import OrbitEnvironmentModel
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState
from utilities.gaussian import MultiVarGauss
from utilities.process_model import ProcessModel
from logging_config import get_logger
from utilities.utils import load_yaml
from collections import deque

logger = get_logger(__name__)

def skew(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from 3-vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def right_jacobian_SO3(omega: np.ndarray) -> np.ndarray:
    """
    Right Jacobian of SO(3) for exponential map.
    Jr(ω) such that: Exp(ω + δω) ≈ Exp(ω) * Exp(Jr(ω) * δω)
    
    Args:
        omega: Rotation vector (3,)
    
    Returns:
        Jr: 3×3 right Jacobian matrix
    """
    omega = np.asarray(omega, dtype=float).reshape(3)
    theta = np.linalg.norm(omega)
    
    if theta < 1e-6:
        # Small angle approximation: Jr ≈ I - 0.5*[ω]×
        return np.eye(3) - 0.5 * skew(omega)
    
    axis = omega / theta
    s = np.sin(theta)
    c = np.cos(theta)
    
    # Jr(ω) = I - (1-cos(θ))/θ² [ω]× + (θ-sin(θ))/θ³ [ω]×²
    omega_skew = skew(omega)
    omega_skew_sq = omega_skew @ omega_skew
    
    Jr = (np.eye(3) - 
          ((1 - c) / (theta**2)) * omega_skew + 
          ((theta - s) / (theta**3)) * omega_skew_sq)
    
    return Jr


def right_jacobian_inv_SO3(omega: np.ndarray) -> np.ndarray:
    """
    Inverse of right Jacobian of SO(3).
    
    Args:
        omega: Rotation vector (3,)
    
    Returns:
        Jr_inv: 3×3 inverse right Jacobian matrix
    """
    omega = np.asarray(omega, dtype=float).reshape(3)
    theta = np.linalg.norm(omega)
    
    if theta < 1e-6:
        # Small angle: Jr^{-1} ≈ I + 0.5*[ω]×
        return np.eye(3) + 0.5 * skew(omega)
    
    axis = omega / theta
    half_theta = 0.5 * theta
    cot_half = 1.0 / np.tan(half_theta)
    
    # Jr^{-1}(ω) = I + 0.5*[ω]× + (1/θ² - (1+cos(θ))/(2θsin(θ))) [ω]×²
    omega_skew = skew(omega)
    omega_skew_sq = omega_skew @ omega_skew
    
    Jr_inv = (np.eye(3) + 
              0.5 * omega_skew + 
              (1.0/(theta**2) - cot_half/(2.0*theta)) * omega_skew_sq)
    
    return Jr_inv


# ----------------------------------------------------------------------
# Helper: convert Quaternion <-> Rot3
# ----------------------------------------------------------------------

def rot3_from_quat(q: Quaternion) -> Rot3:
    """Quaternion(mu, eta) -> gtsam.Rot3 (w,x,y,z convention)."""
    return Rot3.Quaternion(q.mu, *(q.eta.reshape(3)))


def quat_from_rot3(R: Rot3) -> Quaternion:
    """gtsam.Rot3 -> our Quaternion(mu, eta)."""
    q = R.toQuaternion()
    return Quaternion(mu=q.w(), eta=np.array([q.x(), q.y(), q.z()], float)).normalize()


# ----------------------------------------------------------------------
# Window sample type (must match your SlidingWindow)
# ----------------------------------------------------------------------

@dataclass
class WindowSample:
    t: float
    jd: float
    x_nom: NominalState          # KF nominal (ori + bias) or prediction
    omega_meas: np.ndarray       # 3x1 gyro measurement
    z_mag: Optional[np.ndarray]  # 3x1 or None
    z_sun: Optional[np.ndarray]  # 3x1 or None
    z_st: Optional[Quaternion]   # Quaternion or None
    
    
class SlidingWindow:
    def __init__(self, max_len: int):
        self.max_len = max_len
        self.samples: deque[WindowSample] = deque()

    @property
    def ready(self) -> bool:
        return len(self.samples) == self.max_len

    def add(self, sample: WindowSample):
        self.samples.append(sample)
        if len(self.samples) > self.max_len:
            self.samples.popleft()
            
    def clear_half(self):
        half_size = len(self.samples) // 2
        for _ in range(half_size):
            self.samples.popleft()




# ----------------------------------------------------------------------
# GTSAM-based factor-graph optimizer
# ----------------------------------------------------------------------

class GtsamFGO:
    """
    GTSAM factor graph for attitude + gyro bias.

    Nodes per time step i:
        X(i) : Rot3  (attitude R_bn)
        B(i) : Point3 (gyro bias)

    Factors:
        - gyro factor between X(i), B(i), X(i+1)
        - bias random-walk factor between B(i), B(i+1)
        - magnetometer factor on X(i)
        - sun-sensor factor on X(i)
        - star-tracker factor on X(i)
    """

    def __init__(
        self,
        max_iters: int = 30,
        tol: float = 1e-6,
        config_path: str = "configs/config_baseline.yaml",
        use_robust: bool = True,
        robust_kernel: str = "Huber",  # Options: "Huber", "Cauchy", "Tukey"
        robust_param: float = 1.345,   # Tuning parameter (k for Huber, etc.)
    ):
        self.config = load_yaml(config_path)
        self.process = ProcessModel(config_path)
        self.sigma_g = self.process.sigma_g      # rad / sqrt(s)
        self.sigma_bg = self.process.sigma_bg    # rad/s / sqrt(s)

        self.sigma_mag = self.config["sensors"]["mag"]["mag_std"]               # for mag + sun (unit-vector error)
        self.sigma_sv = self.config["sensors"]["sun"]["noise"]["sun_std"]                 # for sun sensor
        self.sigma_st = self.config["sensors"]["star"]["noise"]["st_std"]                 # for star-tracker
        self.max_iters = max_iters
        self.tol = tol

        # M-estimator settings for robust outlier rejection
        self.use_robust = use_robust
        self.robust_kernel = robust_kernel
        self.robust_param = robust_param

    # ------------- key helpers -------------

    @staticmethod
    def X(i: int) -> int:
        return gtsam.symbol("x", i)

    @staticmethod
    def B(i: int) -> int:
        return gtsam.symbol("b", i)

    def make_robust_noise(self, base_noise):
        """
        Wrap a noise model with a robust M-estimator kernel.

        Args:
            base_noise: Base Gaussian noise model

        Returns:
            Robust noise model if self.use_robust=True, else base_noise
        """
        if not self.use_robust:
            return base_noise

        # Create M-estimator based on kernel type
        if self.robust_kernel == "Huber":
            mestimator = gtsam.noiseModel.mEstimator.Huber.Create(self.robust_param)
        elif self.robust_kernel == "Cauchy":
            mestimator = gtsam.noiseModel.mEstimator.Cauchy.Create(self.robust_param)
        elif self.robust_kernel == "Tukey":
            mestimator = gtsam.noiseModel.mEstimator.Tukey.Create(self.robust_param)
        else:
            logger.warning(f"Unknown robust kernel '{self.robust_kernel}', using Huber")
            mestimator = gtsam.noiseModel.mEstimator.Huber.Create(self.robust_param)

        return gtsam.noiseModel.Robust.Create(mestimator, base_noise)

    # ------------- factor builders -------------

    def make_gyro_factor(
        self,
        key_R_prev: int,
        key_b_prev: int,
        key_R_curr: int,
        omega_meas: np.ndarray,
        dt: float,
    ) -> gtsam.CustomFactor:
        """
        Corrected gyro process factor with proper Jacobians.
        
        Error e ∈ R^3:
            R_inc = Exp((ω - b_{k-1}) dt)
            R_pred = R_{k-1} * R_inc
            e = Log(R_pred^{-1} * R_k)
        """
        omega = np.asarray(omega_meas, float).reshape(3)
        dt = float(dt)

        # Discrete gyro noise
        sigma_angle = self.sigma_g * np.sqrt(dt)
        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([sigma_angle, sigma_angle, sigma_angle])
        )

        keys = [key_R_prev, key_b_prev, key_R_curr]

        def error_fn(this: gtsam.CustomFactor, values: gtsam.Values, jacobians=None):
            rot_i = values.atRot3(keys[0])
            bias_i = np.asarray(values.atPoint3(keys[1]), float)
            rot_j = values.atRot3(keys[2])
            
            # Corrected angular velocity
            omega_hat = omega - bias_i
            
            # Predicted rotation
            delta_rot = gtsam.Rot3.Expmap(omega_hat * dt)
            pred_rot_j = rot_i.compose(delta_rot)

            # Error: e = Log(R_pred^{-1} * R_j)
            # In GTSAM: A.between(B) returns A^{-1} * B
            error_rot = pred_rot_j.between(rot_j)  # FIXED: was rot_j.between(pred_rot_j)
            att_err = gtsam.Rot3.Logmap(error_rot)
            e = np.array(att_err, float)
            
            if jacobians is not None:
                # Use right Jacobians for proper manifold derivatives
                Jr_inv_e = right_jacobian_inv_SO3(e)

                # --- Jacobian w.r.t R_j ---
                # For e = Log(R_pred^{-1} * R_j), perturbing R_j: ∂e/∂R_j = Jr_inv(e)
                # Approximation: Jr_inv(e) ≈ I for small errors
                J_rot_j = Jr_inv_e

                # --- Jacobian w.r.t R_i ---
                # ∂e/∂R_i = -Jr_inv(e) * Adj(R_pred^{-1} * R_j)
                # For SO(3), Adj(R) = R (the rotation matrix itself)
                error_rot_matrix = error_rot.matrix()
                J_rot_i = -Jr_inv_e @ error_rot_matrix
                
                # --- Jacobian w.r.t bias b_i ---
                # When b changes, ω̂ = ω - b changes, affecting R_pred = R_i * Exp(ω̂ * dt)
                #
                # Perturbation: b → b + δb causes ω̂ → ω̂ - δb
                # This changes Exp(ω̂ * dt) → Exp(ω̂ * dt) * Exp(-Jr(ω̂ * dt) * δb * dt)
                #
                # Chain rule: ∂e/∂b = -Jr_inv(e) * Jr(ω̂ * dt) * dt

                Jr_omega_dt = right_jacobian_SO3(omega_hat * dt)
                J_bias_i = -Jr_inv_e @ Jr_omega_dt * dt
                
                jacobians[0] = J_rot_i
                jacobians[1] = J_bias_i
                jacobians[2] = J_rot_j
            
            return e

        return gtsam.CustomFactor(noise, keys, error_fn)

    def make_bias_rw_factor(
        self,
        key_b_prev: int,
        key_b_curr: int,
        dt: float,
    ) -> gtsam.BetweenFactorPoint3:
        """
        Random walk on bias: b_k = b_{k-1} + w,  w ~ N(0, σ_bg^2 dt I).
        """
        sigma = self.sigma_bg * np.sqrt(dt)
        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([sigma, sigma, sigma])
        )
        return gtsam.BetweenFactorPoint3(
            key_b_prev,
            key_b_curr,
            gtsam.Point3(0.0, 0.0, 0.0),
            noise,
        )

    def make_vector_factor(
        self,
        key_R: int,
        v_meas_body: np.ndarray,
        v_n_eci: np.ndarray,
        sensor_type: str
    ) -> gtsam.CustomFactor:
        """
        Vector measurement factor with analytical Jacobian and robust M-estimator.

        Error: e = v_meas - R^T * v_n
        where R rotates from body to inertial, so R^T rotates inertial to body.
        """
        v_meas = np.asarray(v_meas_body, float).reshape(3) / np.linalg.norm(v_meas_body)
        v_n = np.asarray(v_n_eci, float).reshape(3) / np.linalg.norm(v_n_eci)

        if sensor_type == "magnetometer":
            sigma_vec = self.sigma_mag
        elif sensor_type == "sun_sensor":
            sigma_vec = self.sigma_sv
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

        base_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([sigma_vec, sigma_vec, sigma_vec])
        )
        # Apply robust M-estimator for outlier rejection
        noise = self.make_robust_noise(base_noise)
        keys = [key_R]

        def error_fn(
            this: gtsam.CustomFactor,
            values: gtsam.Values,
            jacobians=None
        ) -> np.ndarray:
            R: gtsam.Rot3 = values.atRot3(key_R)
            
            # Prediction: v_pred = R^T * v_n
            R_mat = R.matrix()
            v_pred = R_mat.T @ v_n
            
            # Residual
            e = v_meas - v_pred
            
            if jacobians is not None:
                # Analytical Jacobian: ∂e/∂θ where R ← R * Exp(θ)
                # 
                # When we perturb: R ← R * Exp(δθ)
                # Then: R^T ← (R * Exp(δθ))^T = Exp(δθ)^T * R^T
                #              ≈ (I - [δθ]×)^T * R^T
                #              = (I + [δθ]×) * R^T
                # 
                # So: v_pred ← (I + [δθ]×) * R^T * v_n
                #            = v_pred + [δθ]× * v_pred
                #            = v_pred - [v_pred]× * δθ
                # 
                # Therefore: e ← e - (- [v_pred]× * δθ) = e + [v_pred]× * δθ
                # So: ∂e/∂θ = [v_pred]× = -[v_pred]× * (-1) = [v_pred]×
                
                # Actually, let's be more careful:
                # e = v_meas - v_pred
                # ∂e/∂θ = -∂v_pred/∂θ
                # ∂v_pred/∂θ = -[v_pred]×
                # Therefore: ∂e/∂θ = -(-[v_pred]×) = [v_pred]×
                
                v_pred_skew = R_mat @ skew(v_n)  # [R^T * v_n]×
                J = v_pred_skew  # 3×3
                
                jacobians[0] = J

            return e
    
        return gtsam.CustomFactor(noise, keys, error_fn)

    def make_star_factor(self, key_R: int, q_meas: Quaternion) -> gtsam.PriorFactorRot3:
        """Star tracker measurement with robust M-estimator for outlier rejection."""
        R_meas = rot3_from_quat(q_meas)
        base_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([self.sigma_st, self.sigma_st, self.sigma_st])
        )
        # Apply robust M-estimator
        noise = self.make_robust_noise(base_noise)
        return gtsam.PriorFactorRot3(key_R, R_meas, noise)


    # ------------- graph builder -------------

    def build_window_graph(
        self,
        samples: List,  # List[WindowSample]
        env,  # OrbitEnvironmentModel
    ) -> tuple:
        """
        Improved graph building with better initial guesses.
        """
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        M = len(samples)
        if M == 0:
            return graph, values

        # --- First node (i=0) ---
        s0 = samples[0]
        q0 = s0.x_nom.ori
        b0 = s0.x_nom.gyro_bias

        R0 = rot3_from_quat(q0)
        b0_pt = Point3(*b0)

        values.insert(self.X(0), R0)
        values.insert(self.B(0), b0_pt)

        # Priors - loosened to avoid overconstrained optimization
        # Prior rotation uncertainty: ~0.01 rad (~0.57°)
        prior_R_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.01, 0.01, 0.01])
        )
        # Prior bias uncertainty: ~1e-3 rad/s
        prior_b_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-3, 1e-3, 1e-3])
        )
        graph.add(gtsam.PriorFactorRot3(self.X(0), R0, prior_R_noise))
        graph.add(gtsam.PriorFactorPoint3(self.B(0), b0_pt, prior_b_noise))

        # Measurement factors at i=0
        r_eci = env.get_r_eci(s0.jd)
        B_eci = env.get_B_eci(r_eci, s0.jd)
        s_eci = env.get_sun_eci(s0.jd)

        if s0.z_mag is not None and not np.any(np.isnan(s0.z_mag)):
            #print(f"Norm of mag meas at i=0: {np.linalg.norm(s0.z_mag)}")
            #print(f"Norm of B_eci at i=0: {np.linalg.norm(B_eci)}")
            graph.add(self.make_vector_factor(self.X(0), s0.z_mag, B_eci, "magnetometer"))
        if s0.z_sun is not None and not np.any(np.isnan(s0.z_sun)):
            #print(f"Norm of sun meas at i=0: {np.linalg.norm(s0.z_sun)}")
            #print(f"Norm of s_eci at i=0: {np.linalg.norm(s_eci)}")
            graph.add(self.make_vector_factor(self.X(0), s0.z_sun, s_eci, "sun_sensor"))
        if s0.z_st is not None:
            graph.add(self.make_star_factor(self.X(0), s0.z_st))

        # --- Remaining nodes ---
        for i in range(1, M):
            sp = samples[i - 1]
            sc = samples[i]
            dt = sc.t - sp.t

            # Better initial guess: use the nominal state directly
            # (it comes from your Kalman filter, which should be good)
            q_guess = sc.x_nom.ori
            b_guess = sc.x_nom.gyro_bias

            values.insert(self.X(i), rot3_from_quat(q_guess))
            values.insert(self.B(i), Point3(*b_guess))

            # Process factors (with corrected implementations)
            graph.add(
                self.make_gyro_factor(
                    key_R_prev=self.X(i - 1),
                    key_b_prev=self.B(i - 1),
                    key_R_curr=self.X(i),
                    omega_meas=sc.omega_meas,
                    dt=dt,
                )
            )
            graph.add(
                self.make_bias_rw_factor(
                    key_b_prev=self.B(i - 1),
                    key_b_curr=self.B(i),
                    dt=dt,
                )
            )

            # Measurement factors
            r_eci = env.get_r_eci(sc.jd)
            B_eci = env.get_B_eci(r_eci, sc.jd)
            s_eci = env.get_sun_eci(sc.jd)

            if sc.z_mag is not None and not np.any(np.isnan(sc.z_mag)):
                #print(f"Norm of mag meas at i={i}: {np.linalg.norm(sc.z_mag)}")
                #print(f"Norm of B_eci at i={i}: {np.linalg.norm(B_eci)}")
                graph.add(self.make_vector_factor(self.X(i), sc.z_mag, B_eci, "magnetometer"))
            if sc.z_sun is not None and not np.any(np.isnan(sc.z_sun)):
                #print(f"Norm of sun meas at i={i}: {np.linalg.norm(sc.z_sun)}")
                #print(f"Norm of s_eci at i={i}: {np.linalg.norm(s_eci)}")
                graph.add(self.make_vector_factor(self.X(i), sc.z_sun, s_eci, "sun_sensor"))
            if sc.z_st is not None:
                graph.add(self.make_star_factor(self.X(i), sc.z_st))

        return graph, values

    # ------------- window optimization -------------

    def optimize_window(
        self,
        samples: List[WindowSample],
        env: OrbitEnvironmentModel,
    ) -> List[NominalState]:
        """
        Build and solve the factor graph for one window.

        Returns a list of EskfState with unit covariance (you can
        overwrite covariances as you like).
        """
        graph, values = self.build_window_graph(samples, env)
        if graph.size() == 0:
            return []

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.max_iters)
        params.setAbsoluteErrorTol(self.tol)

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
        result = optimizer.optimize()

        states: List[NominalState] = []
        for i, s in enumerate(samples):
            R_opt = result.atRot3(self.X(i))
            b_opt = result.atPoint3(self.B(i))
            q_opt = quat_from_rot3(R_opt)

            x_nom = NominalState(ori=q_opt, gyro_bias=b_opt)
            states.append(x_nom)

        return states


