from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from environment.environment import OrbitEnvironmentModel
from logging_config import get_logger
from utilities.gaussian import MultiVarGauss
from utilities.process_model import ProcessModel
from utilities.quaternion import Quaternion
from utilities.sensors import SensorMagnetometer, SensorStarTracker, SensorSunVector
from utilities.states import EskfState, NominalState, SensorType

logger = get_logger(__name__)


@dataclass
class Factor:
    """Lightweight container describing a single factor in the graph."""

    factor_type: str
    state_indices: Tuple[int, ...]
    payload: dict


class FGO:
    """Simple batch factor graph optimizer for attitude + gyro bias."""

    def __init__(self, max_iters: int = 5, tol: float = 1e-6):
        self.max_iters = max_iters
        self.tol = tol

        self.process = ProcessModel()
        self.sens_mag = SensorMagnetometer()
        self.sens_sv = SensorSunVector()
        self.sens_st = SensorStarTracker()

        self.states: List[EskfState] = []
        self.factors: List[Factor] = []

    # -------------------- GRAPH BUILDING --------------------

    def add_state(self, state: EskfState) -> int:
        self.states.append(state)
        return len(self.states) - 1

    def add_prior(self, state_idx: int, prior: NominalState, cov: np.ndarray) -> None:
        self.factors.append(
            Factor(
                factor_type="prior",
                state_indices=(state_idx,),
                payload={"prior": prior, "cov": np.asarray(cov, float)},
            )
        )

    def add_gyro_factor(self, i: int, j: int, omega_meas: np.ndarray, dt: float) -> None:
        self.factors.append(
            Factor(
                factor_type="gyro",
                state_indices=(i, j),
                payload={"omega": np.asarray(omega_meas, float), "dt": float(dt)},
            )
        )

    def add_measurement(self,
                        idx: int,
                        y: np.ndarray | Quaternion,
                        sensor_type: SensorType,
                        B_n: Optional[np.ndarray] = None,
                        s_n: Optional[np.ndarray] = None) -> None:
        self.factors.append(
            Factor(
                factor_type="measurement",
                state_indices=(idx,),
                payload={
                    "y": y,
                    "sensor_type": sensor_type,
                    "B_n": None if B_n is None else np.asarray(B_n, float),
                    "s_n": None if s_n is None else np.asarray(s_n, float),
                },
            )
        )

    # -------------------- OPTIMIZATION --------------------

    def optimize(self) -> Sequence[EskfState]:
        if not self.states:
            logger.warning("No states in graph; nothing to optimize")
            return self.states

        total_dim = 6 * len(self.states)

        for _ in range(self.max_iters):
            residuals: List[np.ndarray] = []
            jacobian_blocks: List[np.ndarray] = []

            for factor in self.factors:
                if factor.factor_type == "prior":
                    res, jac = self._linearize_prior(factor)
                elif factor.factor_type == "gyro":
                    res, jac = self._linearize_gyro(factor, total_dim)
                elif factor.factor_type == "measurement":
                    res, jac = self._linearize_measurement(factor, total_dim)
                else:
                    raise ValueError(f"Unknown factor type {factor.factor_type}")

                residuals.append(res)
                jacobian_blocks.append(jac)

            r = np.concatenate(residuals)
            J = np.vstack(jacobian_blocks)

            # Solve normal equations J delta = r using least squares
            delta, *_ = np.linalg.lstsq(J, r, rcond=None)

            if np.linalg.norm(delta) < self.tol:
                break

            self._apply_delta(delta)

        return self.states

    # -------------------- GRAPH HELPERS --------------------

    def _linearize_prior(self, factor: Factor) -> Tuple[np.ndarray, np.ndarray]:
        state_idx = factor.state_indices[0]
        x = self.states[state_idx].nom
        prior: NominalState = factor.payload["prior"]
        cov: np.ndarray = factor.payload["cov"]

        err_vec = x.diff(prior).as_vector()
        sqrt_info = np.linalg.cholesky(np.linalg.inv(cov))
        residual = sqrt_info @ err_vec

        J = np.zeros((6, 6 * len(self.states)))
        J[:, 6 * state_idx:6 * (state_idx + 1)] = sqrt_info
        return residual, J

    def _linearize_gyro(self, factor: Factor, total_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        i, j = factor.state_indices
        omega = factor.payload["omega"]
        dt = factor.payload["dt"]

        prev_nom = self.states[i].nom
        next_nom = self.states[j].nom

        # Predict next nominal state using gyro measurement
        quat_pred = prev_nom.ori.propagate(omega - prev_nom.gyro_bias, dt)
        bg_pred = prev_nom.gyro_bias.copy()
        pred_nom = NominalState(ori=quat_pred, gyro_bias=bg_pred)

        err_vec = next_nom.diff(pred_nom).as_vector()
        F = self.process.F(prev_nom, omega, dt)
        Qd = self.process.Q_d(dt)
        sqrt_info = np.linalg.cholesky(np.linalg.inv(Qd))

        residual = sqrt_info @ err_vec

        J = np.zeros((6, total_dim))
        J[:, 6 * i:6 * (i + 1)] = -sqrt_info @ F
        J[:, 6 * j:6 * (j + 1)] = sqrt_info
        return residual, J

    def _linearize_measurement(self, factor: Factor, total_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        state_idx = factor.state_indices[0]
        x_est = self.states[state_idx]

        sensor_type: SensorType = factor.payload["sensor_type"]
        y = factor.payload["y"]
        B_n = factor.payload["B_n"]
        s_n = factor.payload["s_n"]

        match sensor_type:
            case SensorType.MAGNETOMETER:
                assert B_n is not None, "B_n must be provided for magnetometer factors"
                z_pred = self.sens_mag.pred_from_est(x_est, B_n).mean.reshape(-1)
                innovation = np.asarray(y, float).reshape(-1) - z_pred
                H = self.sens_mag.H(x_est.nom, B_n)
                R = self.sens_mag.R
            case SensorType.SUN_VECTOR:
                assert s_n is not None, "s_n must be provided for sun-vector factors"
                z_pred = self.sens_sv.pred_from_est(x_est, s_n).mean.reshape(-1)
                innovation = np.asarray(y, float).reshape(-1) - z_pred
                H = self.sens_sv.H(x_est.nom, s_n)
                R = self.sens_sv.R
            case SensorType.STAR_TRACKER:
                assert isinstance(y, Quaternion)
                innovation = self.sens_st.innovation(x_est=x_est, q_meas=y).reshape(-1)
                H = self.sens_st.H(x_est.nom)
                R = self.sens_st.R
            case _:
                raise ValueError(f"Unsupported sensor type {sensor_type}")

        sqrt_info = np.linalg.cholesky(np.linalg.inv(R))
        residual = sqrt_info @ innovation

        J = np.zeros((H.shape[0], total_dim))
        J[:, 6 * state_idx:6 * (state_idx + 1)] = sqrt_info @ H
        return residual, J

    def _apply_delta(self, delta: np.ndarray) -> None:
        for idx, state in enumerate(self.states):
            delta_i = delta[6 * idx:6 * (idx + 1)]
            state.inject_error(delta_i)

    # -------------------- CONVENIENCE RUNNER --------------------

    def run_on_simulation(
        self,
        sim_run_id: int,
        env: OrbitEnvironmentModel,
        db_path: str = "simulations.db",
        est_name: str = "fgo",
        batch_duration: float = 5.0,
    ) -> int:
        """Build factor graphs in batches from a stored simulation and optimize each.

        A full simulation can contain thousands of measurements because of the small
        sampling time (dt = 0.02). Optimizing all of them at once can consume too much
        memory, so this method windows the problem into batches of roughly
        ``batch_duration`` seconds. Adjacent batches overlap by one state to preserve
        continuity, using the final state from the previous batch as the prior for the
        next.
        """

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
        prev_last_state: Optional[EskfState] = None

        while k_start < N:
            # Determine the end index for this batch based on elapsed time.
            k_end = k_start + 1
            while k_end < N and t[k_end] - t[k_start] < batch_duration:
                k_end += 1

            batch_graph = FGO(max_iters=self.max_iters, tol=self.tol)

            # ----- initial state and prior for this batch -----
            if prev_last_state is None:
                q0 = Quaternion.from_array(result.q_true[k_start])
                b0 = np.zeros(3)
            else:
                q0 = prev_last_state.nom.ori
                b0 = prev_last_state.nom.gyro_bias
            err0 = MultiVarGauss(mean=np.zeros(6), cov=self.process.Q_c)
            x0 = EskfState(nom=NominalState(ori=q0, gyro_bias=b0), err=err0)
            batch_graph.add_state(x0)
            batch_graph.add_prior(0, x0.nom, cov=np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6]))

            for k in range(k_start + 1, k_end):
                q_init = Quaternion.from_array(result.q_true[k])
                bg_init = np.zeros(3)
                err_init = MultiVarGauss(mean=np.zeros(6), cov=self.process.Q_c)
                x_init = EskfState(nom=NominalState(ori=q_init, gyro_bias=bg_init), err=err_init)
                batch_graph.add_state(x_init)

                dt_k = t[k] - t[k - 1]
                batch_graph.add_gyro_factor(k - 1 - k_start, k - k_start, omega_meas_log[k], dt_k)

                r_eci = env.get_r_eci(jd[k])
                B_eci = env.get_B_eci(r_eci, jd[k])
                s_eci = env.get_sun_eci(jd[k])

                mag_meas = mag_meas_log[k]
                if not np.any(np.isnan(mag_meas)):
                    batch_graph.add_measurement(
                        idx=k - k_start,
                        y=mag_meas,
                        sensor_type=SensorType.MAGNETOMETER,
                        B_n=B_eci,
                    )

                sun_meas = sun_meas_log[k]
                if not np.any(np.isnan(sun_meas)):
                    batch_graph.add_measurement(
                        idx=k - k_start,
                        y=sun_meas,
                        sensor_type=SensorType.SUN_VECTOR,
                        s_n=s_eci,
                    )

                st_meas = st_meas_log[k]
                if not np.any(np.isnan(st_meas)):
                    q_meas = Quaternion.from_array(st_meas)
                    batch_graph.add_measurement(
                        idx=k - k_start,
                        y=q_meas,
                        sensor_type=SensorType.STAR_TRACKER,
                    )

            batch_states = batch_graph.optimize()

            # Skip the overlapping prior state for all but the first batch to avoid
            # duplicating timestamps.
            if prev_last_state is None:
                optimized_states.extend(batch_states)
            else:
                optimized_states.extend(batch_states[1:])

            prev_last_state = optimized_states[-1]
            k_start = k_end - 1  # overlap last state with next batch

        est_run_id = db.insert_estimation_run(
            sim_run_id=sim_run_id,
            t=t,
            jd=jd,
            states=optimized_states,
            name=est_name,
        )
        return est_run_id
