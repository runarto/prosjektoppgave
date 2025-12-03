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
    ):
        self.config = load_yaml("config.yaml")
        self.process = ProcessModel()
        self.sigma_g = self.process.sigma_g      # rad / sqrt(s)
        self.sigma_bg = self.process.sigma_bg    # rad/s / sqrt(s)

        self.sigma_mag = self.config["sensors"]["mag"]["mag_std"]               # for mag + sun (unit-vector error)
        self.sigma_sv = self.config["sensors"]["sun"]["noise"]["sun_std"]                 # for sun sensor
        self.sigma_st = self.config["sensors"]["star"]["noise"]["st_std"]                 # for star-tracker
        self.max_iters = max_iters
        self.tol = tol

    # ------------- key helpers -------------

    @staticmethod
    def X(i: int) -> int:
        return gtsam.symbol("x", i)

    @staticmethod
    def B(i: int) -> int:
        return gtsam.symbol("b", i)

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
        Gyro process factor on (R_{k-1}, b_{k-1}, R_k).

        Error e ∈ R^3:
            R_inc = Exp((ω - b_{k-1}) dt)
            R_pred = R_{k-1} * R_inc
            e      = Log( R_pred^{-1} * R_k )
        """
        omega = np.asarray(omega_meas, float).reshape(3)
        dt = float(dt)

        # Continuous gyro noise σ_g [rad/s/√Hz] -> discrete angle noise σ ≈ σ_g * sqrt(dt)
        sigma_angle = self.sigma_g * np.sqrt(dt)
        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([sigma_angle, sigma_angle, sigma_angle])
        )

        keys = [key_R_prev, key_b_prev, key_R_curr]

        def error_fn(this: gtsam.CustomFactor, values: gtsam.Values, jacobians=None):
            rot_i = values.atRot3(keys[0])
            bias_i = np.asarray(values.atPoint3(keys[1]), float)
            rot_j = values.atRot3(keys[2])

            omega_hat = omega - bias_i
            delta_rot = gtsam.Rot3.Expmap(omega_hat * dt)
            pred_rot_j = rot_i.compose(delta_rot)

            att_err = gtsam.Rot3.Logmap(pred_rot_j.between(rot_j))
            e = np.array(att_err, float)

            if jacobians is not None:
                # jacobians is a sequence with one entry per key: [J_R_prev, J_b_prev, J_R_curr]
                for k in range(len(jacobians)):
                    jacobians.insert(k, np.zeros((3, 3)))  # placeholder zeros

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
        v_meas = np.asarray(v_meas_body, float).reshape(3)
        v_n = np.asarray(v_n_eci, float).reshape(3)
        
        if sensor_type == "magnetometer":
            self.sigma_vec = self.sigma_mag
        elif sensor_type == "sun_sensor":
            self.sigma_vec = self.sigma_sv

        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([self.sigma_vec, self.sigma_vec, self.sigma_vec])
        )
        keys = [key_R]
        eps = 1e-6

        def error_fn(
            this: gtsam.CustomFactor,
            values: gtsam.Values,
            jacobians=None
        ) -> np.ndarray:
            R: gtsam.Rot3 = values.atRot3(key_R)

            # prediction: h(R) = R^T v_n
            def h(R_local: gtsam.Rot3) -> np.ndarray:
                return R_local.matrix().T @ v_n

            v_pred = h(R)
            e = v_meas - v_pred   # residual in R^3

            # If jacobians is non-empty, fill it
            if jacobians is not None:
                # Jacobian wrt Rot3 in tangent space R^3
                J = np.zeros((3, 3))
                for k in range(3):
                    delta = np.zeros(3)
                    delta[k] = eps
                    R_plus = R.retract(delta)
                    R_minus = R.retract(-delta)
                    h_plus = h(R_plus)
                    h_minus = h(R_minus)
                    # e = v_meas - h(R) -> ∂e/∂δ = -(h_plus - h_minus)/(2 eps)
                    J[:, k] = - (h_plus - h_minus) / (2.0 * eps)
                    
                jacobians.insert(0, J)

            return e

        return gtsam.CustomFactor(noise, keys, error_fn)

    def make_star_factor(self, key_R: int, q_meas: Quaternion) -> gtsam.PriorFactorRot3:
        R_meas = rot3_from_quat(q_meas)
        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([self.sigma_st, self.sigma_st, self.sigma_st])
        )
        return gtsam.PriorFactorRot3(key_R, R_meas, noise)


    # ------------- graph builder -------------

    def build_window_graph(
        self,
        samples: List[WindowSample],
        env: OrbitEnvironmentModel,
    ) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
        """
        Build factor graph + initial values for a given window of samples.

        samples[i].x_nom is only used for initialization (not as a factor).
        """
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        M = len(samples)
        if M == 0:
            return graph, values

        # --- initial node (i = 0) ---
        s0 = samples[0]
        q0 = s0.x_nom.ori
        b0 = s0.x_nom.gyro_bias

        R0 = rot3_from_quat(q0)
        b0_pt = Point3(*b0)

        values.insert(self.X(0), R0)
        values.insert(self.B(0), b0_pt)

        # Priors on first attitude and bias
        prior_R_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-2, 1e-2, 1e-2])
        )
        prior_b_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-3, 1e-3, 1e-3])
        )
        graph.add(gtsam.PriorFactorRot3(self.X(0), R0, prior_R_noise))
        graph.add(gtsam.PriorFactorPoint3(self.B(0), b0_pt, prior_b_noise))

        # Measurement factors at i = 0
        r_eci = env.get_r_eci(s0.jd)
        B_eci = env.get_B_eci(r_eci, s0.jd)
        s_eci = env.get_sun_eci(s0.jd)

        np.isnan
        
        
        if s0.z_mag is not None:
            graph.add(self.make_vector_factor(self.X(0), s0.z_mag, B_eci, sensor_type="magnetometer"))
        if s0.z_sun is not None:
            graph.add(self.make_vector_factor(self.X(0), s0.z_sun, s_eci, sensor_type="sun_sensor"))
        if s0.z_st is not None:
            graph.add(self.make_star_factor(self.X(0), s0.z_st))

        # --- remaining nodes ---
        for i in range(1, M):
            sp = samples[i - 1]
            sc = samples[i]

            dt = sc.t - sp.t

            # initial guesses by propagating previous attitude with current bias
            q_guess = sp.x_nom.ori.propagate(sp.omega_meas - sp.x_nom.gyro_bias, dt)
            b_guess = sc.x_nom.gyro_bias

            values.insert(self.X(i), rot3_from_quat(q_guess))
            values.insert(self.B(i), Point3(*b_guess))

            # process factors
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

            # environment at current time
            r_eci = env.get_r_eci(sc.jd)
            B_eci = env.get_B_eci(r_eci, sc.jd)
            s_eci = env.get_sun_eci(sc.jd)

            # measurement factors
            if sc.z_mag is not None:
                graph.add(self.make_vector_factor(self.X(i), sc.z_mag, B_eci, sensor_type="magnetometer"))
            if sc.z_sun is not None:
                graph.add(self.make_vector_factor(self.X(i), sc.z_sun, s_eci, sensor_type="sun_sensor"))
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


