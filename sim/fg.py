from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from environment.environment import OrbitEnvironmentModel
from logging_config import get_logger
from utilities.gaussian import MultiVarGauss
from utilities.process_model import ProcessModel
from utilities.quaternion import Quaternion
from utilities.sensors import SensorMagnetometer, SensorStarTracker, SensorSunVector
from utilities.states import EskfState, NominalState, SensorType
from data.db import SimulationDatabase

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
            residuals: list[np.ndarray] = []
            jacobian_blocks: list[np.ndarray] = []

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

            # Gauss–Newton step: J δ ≈ -r
            JT = J.T
            JTJ = JT @ J
            JTr = JT @ (-r)

            try:
                delta = np.linalg.solve(JTJ, JTr)
            except np.linalg.LinAlgError:
                logger.error("JTJ is singular; aborting optimization for this window")
                break

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
        # Note the minus sign here
        J[:, 6 * state_idx:6 * (state_idx + 1)] = -sqrt_info @ H
        return residual, J

    def _apply_delta(self, delta: np.ndarray) -> None:
        for idx, state in enumerate(self.states):
            delta_i = delta[6 * idx:6 * (idx + 1)]
            state.inject_error(delta_i)

    # -------------------- INTERNAL HELPERS --------------------

    def _make_initial_state(self, q0_arr: np.ndarray) -> EskfState:
        """Create initial EskfState from a quaternion array and default bias."""
        q0 = Quaternion.from_array(q0_arr)
        b0 = np.zeros(3)
        err0 = MultiVarGauss(mean=np.zeros(6), cov=self.process.Q_c)
        return EskfState(nom=NominalState(ori=q0, gyro_bias=b0), err=err0)

    def _add_step_state_and_factors(
        self,
        k: int,
        t: np.ndarray,
        jd: np.ndarray,
        q_estimated: np.ndarray,
        omega_meas_log: np.ndarray,
        mag_meas_log: np.ndarray,
        sun_meas_log: np.ndarray,
        st_meas_log: np.ndarray,
        env: OrbitEnvironmentModel,
    ) -> None:
        """Add state and all process/measurement factors for time index k."""

        # new state, initialized from ESKF estimate
        q_init = Quaternion.from_array(q_estimated[k])
        bg_init = np.zeros(3)
        err_init = MultiVarGauss(mean=np.zeros(6), cov=self.process.Q_c)
        x_init = EskfState(nom=NominalState(ori=q_init, gyro_bias=bg_init), err=err_init)

        new_idx = self.add_state(x_init)
        prev_idx = new_idx - 1

        # process factor
        dt_k = t[k] - t[k - 1]
        self.add_gyro_factor(prev_idx, new_idx, omega_meas_log[k], dt_k)

        # environment vectors
        r_eci = env.get_r_eci(jd[k])
        B_eci = env.get_B_eci(r_eci, jd[k])
        s_eci = env.get_sun_eci(jd[k])

        # magnetometer
        mag_meas = mag_meas_log[k]
        if not np.any(np.isnan(mag_meas)):
            self.add_measurement(
                idx=new_idx,
                y=mag_meas,
                sensor_type=SensorType.MAGNETOMETER,
                B_n=B_eci,
            )

        # sun sensor
        sun_meas = sun_meas_log[k]
        if not np.any(np.isnan(sun_meas)):
            self.add_measurement(
                idx=new_idx,
                y=sun_meas,
                sensor_type=SensorType.SUN_VECTOR,
                s_n=s_eci,
            )

        # star tracker
        st_meas = st_meas_log[k]
        if not np.any(np.isnan(st_meas)):
            q_meas = Quaternion.from_array(st_meas)
            self.add_measurement(
                idx=new_idx,
                y=q_meas,
                sensor_type=SensorType.STAR_TRACKER,
            )

    @staticmethod
    def _attitude_errors_deg_for_states(
        states: list[EskfState],
        start_k: int,
        q_true: np.ndarray,
    ) -> list[float]:
        """Compute attitude error [deg] for a contiguous block of states."""
        errors_deg: list[float] = []
        for local_i, state in enumerate(states):
            global_i = start_k + local_i
            if global_i >= q_true.shape[0]:
                break
            q_true_k = Quaternion.from_array(q_true[global_i])
            q_est_k = state.nom.ori
            q_err = q_true_k.conjugate().multiply(q_est_k)
            w = np.clip(abs(q_err.mu), -1.0, 1.0)
            angle_rad = 2.0 * np.arccos(w)
            errors_deg.append(float(np.degrees(angle_rad)))
        return errors_deg

    def _commit_window_to_global(
        self,
        optimized_states: Sequence[EskfState],
        start_k: int,
        est_states: list[EskfState | None],
    ) -> None:
        """Store all but the last optimized state into the global trajectory."""
        for local_i in range(len(optimized_states) - 1):
            global_i = start_k + local_i
            if global_i >= len(est_states):
                break
            if est_states[global_i] is None:
                est_states[global_i] = optimized_states[local_i]

    # -------------------- CONVENIENCE RUNNER --------------------

    def run_on_simulation(
        self,
        est_run_id: int,
        sim_run_id: int,
        env: OrbitEnvironmentModel,
        db_path: str = "simulations.db",
        est_name: str = "fgo",
    ) -> int:
        """Sliding-window factor graph on stored simulation."""

        db = SimulationDatabase(db_path)
        result = db.load_run(sim_run_id)
        estimates = db.load_estimated_states(est_run_id)

        t = result.t
        jd = result.jd
        omega_meas_log = result.omega_meas
        mag_meas_log = result.mag_meas
        sun_meas_log = result.sun_meas
        st_meas_log = result.st_meas
        q_estimated = estimates.q_est
        q_true = result.q_true

        N = t.shape[0]
        window_size = 200  # tune for memory/speed

        # full estimated trajectory storage
        est_states: list[EskfState | None] = [None] * N

        # ----- initialize with first state from ESKF -----
        x0 = self._make_initial_state(q_estimated[0])

        self.states = []
        self.factors = []
        idx0 = self.add_state(x0)
        assert idx0 == 0

        # prior on first state
        self.add_prior(
            state_idx=0,
            prior=x0.nom,
            cov=np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6]),
        )

        start_k = 0  # global index of self.states[0]

        pbar = tqdm(range(1, N), desc="FGO Sliding Window", unit="step")
        for k in pbar:
            # 1) extend graph with new time step
            self._add_step_state_and_factors(
                k=k,
                t=t,
                jd=jd,
                q_estimated=q_estimated,
                omega_meas_log=omega_meas_log,
                mag_meas_log=mag_meas_log,
                sun_meas_log=sun_meas_log,
                st_meas_log=st_meas_log,
                env=env,
            )

            # 2) check window condition
            window_full = len(self.states) >= window_size
            last_step = (k == N - 1)

            if not (window_full or last_step):
                continue

            # 3) pre-optimization attitude error in this window
            init_errors_deg = self._attitude_errors_deg_for_states(
                states=self.states,
                start_k=start_k,
                q_true=q_true,
            )

            # 4) optimize current window
            optimized_states = self.optimize()

            # 5) post-optimization attitude error
            window_errors_deg = self._attitude_errors_deg_for_states(
                states=list(optimized_states),
                start_k=start_k,
                q_true=q_true,
            )

            if window_errors_deg:
                mean_err = float(np.mean(window_errors_deg))
                max_err = float(np.max(window_errors_deg))
                pbar.set_postfix({
                    "mean_err": f"{mean_err:.3f}",
                    "max_err": f"{max_err:.3f}",
                    "start": start_k,
                })

            # 6) store optimized states globally (except last, which is carried)
            self._commit_window_to_global(
                optimized_states=optimized_states,
                start_k=start_k,
                est_states=est_states,
            )

            # 7) carry last state as prior for next window
            last_state = optimized_states[-1]
            self.states = [last_state]
            self.factors = []
            start_k = k  # new window starts at global index k

            # fresh prior on carried state
            self.add_prior(
                state_idx=0,
                prior=last_state.nom,
                cov=np.diag([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8]),
            )

        # ensure final entry is filled
        if est_states[-1] is None:
            est_states[-1] = self.states[0]

        # type ignore: we know all entries are filled now
        filled_states: list[EskfState] = [s for s in est_states if s is not None]  # type: ignore

        est_run_id_out = db.insert_estimation_run(
            sim_run_id=sim_run_id,
            t=t,
            jd=jd,
            states=filled_states,
            name=est_name,
        )
        print(f"FGO estimation run stored with est_run_id={est_run_id_out}")
        return est_run_id_out





if __name__ == "__main__":
    fg = FGO()
    fg.run_on_simulation(
        est_run_id=1,
        sim_run_id=1,
        env=OrbitEnvironmentModel(),
        db_path="simulations.db",
        est_name="fgo_test",
    )