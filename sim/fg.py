from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import gtsam
import numpy as np

from environment.environment import OrbitEnvironmentModel
from logging_config import get_logger
from utilities.gaussian import MultiVarGauss
from utilities.process_model import ProcessModel
from utilities.quaternion import Quaternion
from utilities.sensors import SensorMagnetometer, SensorSunVector, SensorStarTracker
from utilities.states import EskfState, NominalState


logger = get_logger(__name__)


def _quat_from_rot3(rot: gtsam.Rot3) -> Quaternion:
    w, x, y, z = rot.quaternion()
    return Quaternion(mu=w, eta=np.array([x, y, z], dtype=float)).normalize()


@dataclass
class BatchResult:
    states: List[EskfState]
    last_rot: gtsam.Rot3
    last_bias: np.ndarray


class GtsamFGO:
    """Factor graph optimizer backed by GTSAM for attitude + gyro bias."""

    def __init__(self, max_iters: int = 50, tol: float = 1e-6):
        self.max_iters = max_iters
        self.tol = tol

        self.process = ProcessModel()
        self.sens_mag = SensorMagnetometer()
        self.sens_sv = SensorSunVector()
        self.sens_st = SensorStarTracker()

    # ------------------------------------------------------------------
    # Factor builders
    # ------------------------------------------------------------------
    @staticmethod
    def _rot_key(idx: int) -> int:
        return gtsam.symbol("R", idx)

    @staticmethod
    def _bias_key(idx: int) -> int:
        return gtsam.symbol("b", idx)

    def _add_priors(
        self,
        graph: gtsam.NonlinearFactorGraph,
        rot_key: int,
        bias_key: int,
        rot_prior: gtsam.Rot3,
        bias_prior: np.ndarray,
    ) -> None:
        rot_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2, 1e-2, 1e-2]))
        bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-3]))

        graph.add(gtsam.PriorFactorRot3(rot_key, rot_prior, rot_noise))
        graph.add(
            gtsam.PriorFactorPoint3(
                bias_key, gtsam.Point3(*bias_prior.reshape(3)), bias_noise
            )
        )

    def _make_gyro_factor(
        self, omega_meas: np.ndarray, dt: float, keys: list[int]
    ) -> gtsam.CustomFactor:
        Qd = self.process.Q_d(dt)
        noise = gtsam.noiseModel.Gaussian.Covariance(Qd)

        def error_fn(this: gtsam.CustomFactor, values: gtsam.Values, jacobians=None):
            rot_i = values.atRot3(keys[0])
            bias_i = np.asarray(values.atPoint3(keys[1]), float)
            rot_j = values.atRot3(keys[2])
            bias_j = np.asarray(values.atPoint3(keys[3]), float)

            omega_hat = np.asarray(omega_meas, float).reshape(3) - bias_i
            delta_rot = gtsam.Rot3.Expmap(omega_hat * dt)
            pred_rot_j = rot_i.compose(delta_rot)

            att_err = gtsam.Rot3.Logmap(pred_rot_j.between(rot_j))
            bias_err = bias_j - bias_i

            if jacobians is not None:
                # Jacobians are omitted for simplicity; GTSAM will numerical differentiate.
                pass

            return np.hstack([att_err, bias_err])

        return gtsam.CustomFactor(noise, keys, error_fn)

    def _make_bias_random_walk(self, dt: float, keys: list[int]) -> gtsam.BetweenFactorPoint3:
        bias_noise = np.eye(3) * self.process.sigma_bg**2 * dt
        noise_model = gtsam.noiseModel.Gaussian.Covariance(bias_noise)
        return gtsam.BetweenFactorPoint3(keys[0], keys[1], gtsam.Point3(0, 0, 0), noise_model)

    def _add_magnetometer_factor(
        self,
        graph: gtsam.NonlinearFactorGraph,
        rot_key: int,
        y: np.ndarray,
        B_n: np.ndarray,
    ) -> None:
        noise = gtsam.noiseModel.Gaussian.Covariance(self.sens_mag.R)

        def error_fn(this: gtsam.CustomFactor, values: gtsam.Values, jacobians=None):
            rot = values.atRot3(rot_key)
            R_bn = rot.matrix().T
            z_pred = R_bn @ B_n.reshape(3)
            return y.reshape(3) - z_pred

        graph.add(gtsam.CustomFactor(noise, [rot_key], error_fn))

    def _add_sun_factor(
        self,
        graph: gtsam.NonlinearFactorGraph,
        rot_key: int,
        y: np.ndarray,
        s_n: np.ndarray,
    ) -> None:
        noise = gtsam.noiseModel.Gaussian.Covariance(self.sens_sv.R)

        def error_fn(this: gtsam.CustomFactor, values: gtsam.Values, jacobians=None):
            rot = values.atRot3(rot_key)
            R_bn = rot.matrix().T
            z_pred = R_bn @ s_n.reshape(3)
            return y.reshape(3) - z_pred

        graph.add(gtsam.CustomFactor(noise, [rot_key], error_fn))

    def _add_star_tracker_factor(
        self,
        graph: gtsam.NonlinearFactorGraph,
        rot_key: int,
        q_meas: Quaternion,
    ) -> None:
        noise = gtsam.noiseModel.Gaussian.Covariance(self.sens_st.R)
        rot_meas = gtsam.Rot3.Quaternion(q_meas.mu, *q_meas.eta.reshape(3))

        def error_fn(this: gtsam.CustomFactor, values: gtsam.Values, jacobians=None):
            rot = values.atRot3(rot_key)
            return gtsam.Rot3.Logmap(rot_meas.between(rot))

        graph.add(gtsam.CustomFactor(noise, [rot_key], error_fn))

    # ------------------------------------------------------------------
    # Batch optimizer
    # ------------------------------------------------------------------
    def _optimize_batch(
        self,
        t: np.ndarray,
        jd: np.ndarray,
        omega_meas: np.ndarray,
        mag_meas: np.ndarray,
        sun_meas: np.ndarray,
        st_meas: np.ndarray,
        env: OrbitEnvironmentModel,
        q0: Quaternion,
        b0: np.ndarray,
    ) -> BatchResult:
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        # initial state
        rot0 = gtsam.Rot3.Quaternion(q0.mu, *q0.eta.reshape(3))
        bias0 = np.asarray(b0, float).reshape(3)
        initial.insert(self._rot_key(0), rot0)
        initial.insert(self._bias_key(0), gtsam.Point3(*bias0))
        self._add_priors(graph, self._rot_key(0), self._bias_key(0), rot0, bias0)

        # add measurement factors at initial time
        r_eci = env.get_r_eci(jd[0])
        B_eci = env.get_B_eci(r_eci, jd[0])
        s_eci = env.get_sun_eci(jd[0])

        if not np.any(np.isnan(mag_meas[0])):
            self._add_magnetometer_factor(graph, self._rot_key(0), mag_meas[0], B_eci)

        if not np.any(np.isnan(sun_meas[0])):
            self._add_sun_factor(graph, self._rot_key(0), sun_meas[0], s_eci)

        if not np.any(np.isnan(st_meas[0])):
            q_meas0 = Quaternion.from_array(st_meas[0])
            self._add_star_tracker_factor(graph, self._rot_key(0), q_meas0)

        rot_guess = rot0
        bias_guess = bias0

        for k in range(1, len(t)):
            dt = t[k] - t[k - 1]
            omega_hat = np.asarray(omega_meas[k], float).reshape(3) - bias_guess
            rot_guess = rot_guess.compose(gtsam.Rot3.Expmap(omega_hat * dt))

            initial.insert(self._rot_key(k), rot_guess)
            initial.insert(self._bias_key(k), gtsam.Point3(*bias_guess))

            keys = [self._rot_key(k - 1), self._bias_key(k - 1), self._rot_key(k), self._bias_key(k)]
            graph.add(self._make_gyro_factor(omega_meas[k], dt, keys))
            graph.add(self._make_bias_random_walk(dt, [keys[1], keys[3]]))

            r_eci = env.get_r_eci(jd[k])
            B_eci = env.get_B_eci(r_eci, jd[k])
            s_eci = env.get_sun_eci(jd[k])

            if not np.any(np.isnan(mag_meas[k])):
                self._add_magnetometer_factor(graph, self._rot_key(k), mag_meas[k], B_eci)

            if not np.any(np.isnan(sun_meas[k])):
                self._add_sun_factor(graph, self._rot_key(k), sun_meas[k], s_eci)

            if not np.any(np.isnan(st_meas[k])):
                q_meas = Quaternion.from_array(st_meas[k])
                self._add_star_tracker_factor(graph, self._rot_key(k), q_meas)

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.max_iters)
        params.setAbsoluteErrorTol(self.tol)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = optimizer.optimize()

        states: List[EskfState] = []
        err_cov = self.process.Q_c
        for k in range(len(t)):
            rot = result.atRot3(self._rot_key(k))
            bias = np.asarray(result.atPoint3(self._bias_key(k)), float)
            q = _quat_from_rot3(rot)
            x_nom = NominalState(ori=q, gyro_bias=bias)
            err = MultiVarGauss(mean=np.zeros(6), cov=err_cov)
            states.append(EskfState(nom=x_nom, err=err))

        last_rot = result.atRot3(self._rot_key(len(t) - 1))
        last_bias = np.asarray(result.atPoint3(self._bias_key(len(t) - 1)), float)
        return BatchResult(states=states, last_rot=last_rot, last_bias=last_bias)

    # ------------------------------------------------------------------
    # Public runner
    # ------------------------------------------------------------------
    def run_on_simulation(
        self,
        sim_run_id: int,
        env: OrbitEnvironmentModel,
        db_path: str = "simulations.db",
        est_name: str = "gtsam_fgo",
        batch_duration: float = 5.0,
    ) -> int:
        """Build and solve factor graphs batchwise using GTSAM."""

        from data.db import SimulationDatabase

        db = SimulationDatabase(db_path)
        result = db.load_run(sim_run_id)

        t = result.t
        jd = result.jd
        omega_meas_log = result.omega_meas
        mag_meas_log = result.mag_meas
        sun_meas_log = result.sun_meas
        st_meas_log = result.st_meas

        N = t.shape[0]
        optimized_states: List[EskfState] = []
        k_start = 0
        prev_rot: Optional[gtsam.Rot3] = None
        prev_bias: Optional[np.ndarray] = None

        while k_start < N:
            k_end = k_start + 1
            while k_end < N and t[k_end] - t[k_start] < batch_duration:
                k_end += 1

            if prev_rot is None:
                q0 = Quaternion.from_array(result.q_true[k_start])
                b0 = np.zeros(3)
            else:
                q0 = _quat_from_rot3(prev_rot)
                b0 = prev_bias if prev_bias is not None else np.zeros(3)

            batch = self._optimize_batch(
                t[k_start:k_end],
                jd[k_start:k_end],
                omega_meas_log[k_start:k_end],
                mag_meas_log[k_start:k_end],
                sun_meas_log[k_start:k_end],
                st_meas_log[k_start:k_end],
                env,
                q0=q0,
                b0=b0,
            )

            if not optimized_states:
                optimized_states.extend(batch.states)
            else:
                optimized_states.extend(batch.states[1:])

            prev_rot = batch.last_rot
            prev_bias = batch.last_bias
            k_start = k_end - 1

        est_run_id = db.insert_estimation_run(
            sim_run_id=sim_run_id,
            t=t,
            jd=jd,
            states=optimized_states,
            name=est_name,
        )
        return est_run_id

