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
        self.window_size = 200

        self.process = ProcessModel()
        self.sens_mag = SensorMagnetometer()
        self.sens_sv = SensorSunVector()
        self.sens_st = SensorStarTracker()

        self.states: List[EskfState] = []
        self.factors: List[Factor] = []

    # -------------------- GRAPH BUILDING --------------------
    
    def set_window_size(self, size: int) -> None:
        """Set the maximum number of states in the sliding window."""
        self.window_size = size

    def add_state(self, state: EskfState) -> int:
        """Add a new state to the graph and return its index."""
        self.states.append(state)
        return len(self.states) - 1

    def add_prior(self, state_idx: int, prior: NominalState, cov: np.ndarray) -> None:
        """Add a prior factor on the specified state."""
        self.factors.append(
            Factor(
                factor_type="prior",
                state_indices=(state_idx,),
                payload={"prior": prior, "cov": np.asarray(cov, float)},
            )
        )

    def add_gyro_factor(self, i: int, j: int, omega_meas: np.ndarray, dt: float) -> None:
        """
        Add a gyro process factor between states i and j. 
        The factor predicts state j from state i using omega_meas and dt.
        """
        self.factors.append(
            Factor(
                factor_type="gyro",
                state_indices=(i, j),
                payload={"omega": np.asarray(omega_meas, float), 
                         "dt": float(dt)},
            )
        )

    def add_measurement(self,
                        idx: int,
                        y: np.ndarray | Quaternion,
                        sensor_type: SensorType,
                        B_n: Optional[np.ndarray] = None,
                        s_n: Optional[np.ndarray] = None) -> None:
        """Add a measurement factor for the specified state."""
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
        """Optimize the current graph and return the optimized states."""
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


    def reset(self) -> None:
        """Clear all states and factors from the graph."""
        last_state = self.states[-1]
        self.states = [last_state]
        self.factors = []
        self.add_prior(
            state_idx=0,
            prior=last_state.nom,
            cov=np.diag([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8]),
        )
    

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
        Qd = self.process.Q_d(x_nom=prev_nom, omega_meas=omega, dt=dt)
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
                assert B_n is not None, "B_n must be provided for magnetometer update"

                # predicted measurement z_pred ~ N(mean, S)
                z_pred_gauss = self.sens_mag.pred_from_est(x_est, B_n)
                z_pred = z_pred_gauss.mean.reshape(-1)

                y_vec = np.asarray(y, float).reshape(-1)
                innovation = y_vec - z_pred

                H = self.sens_mag.H(x_est.nom, B_n)
                R = self.sens_mag.R

            case SensorType.SUN_VECTOR:
                assert s_n is not None, "s_n must be provided for sun-vector update"

                z_pred_gauss = self.sens_sv.pred_from_est(x_est, s_n)
                z_pred = z_pred_gauss.mean.reshape(-1)

                y_vec = np.asarray(y, float).reshape(-1)
                innovation = y_vec - z_pred

                H = self.sens_sv.H(x_est.nom, s_n)
                R = self.sens_sv.R

            case SensorType.STAR_TRACKER:
                # y is a Quaternion measurement
                assert isinstance(y, Quaternion)

                # measurement space is small-angle error δθ, predicted mean = 0
                innovation = self.sens_st.innovation(x_est=x_est, q_meas=y).reshape(-1)

                H = self.sens_st.H(x_est.nom)
                R = self.sens_st.R
            case _:
                raise ValueError(f"Unsupported sensor type {sensor_type}")

        sqrt_info = np.linalg.cholesky(np.linalg.inv(R))
        residual = sqrt_info @ innovation

        J = np.zeros((H.shape[0], total_dim))
        J[:, 6 * state_idx:6 * (state_idx + 1)] = -sqrt_info @ H
        return residual, J

    def _apply_delta(self, delta: np.ndarray) -> None:
        for idx, state in enumerate(self.states):
            delta_i = delta[6 * idx:6 * (idx + 1)]
            state.inject_error(delta_i)
            state.nom.ori = state.nom.ori.canonical().normalize()

    # -------------------- INTERNAL HELPERS --------------------

    def create_esfk_state(
        self,
        q0_arr: Optional[np.ndarray] = None,
        x_nom: Optional[NominalState] = None,
    ) -> EskfState:
        """Create initial EskfState.

        Priority of initialization:
        - If `x_nom` is provided, use it.
        - Else if `q0_arr` is provided, create quaternion from array and zero bias.
        - Else use identity quaternion and zero bias.
        """
        if x_nom is not None:
            q0 = x_nom.ori
            b0 = x_nom.gyro_bias
        elif q0_arr is not None:
            q0 = Quaternion.from_array(q0_arr)
            b0 = np.zeros(3)
        else:
            q0 = Quaternion.from_array(np.array([1.0, 0.0, 0.0, 0.0]))
            b0 = np.zeros(3)

        P0 = np.diag([1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4])
        err0 = MultiVarGauss(mean=np.zeros(6), cov=P0)
        return EskfState(nom=NominalState(ori=q0, gyro_bias=b0), err=err0)

    def iterate(
        self,
        dt: float,
        jd: float,
        x_nom: NominalState,
        omega_meas: np.ndarray,
        mag_meas: np.ndarray,
        sun_meas: np.ndarray,
        st_meas: np.ndarray,
        env: OrbitEnvironmentModel,
    ) -> int:
        """
        Add a new state and all process/measurement factors for one step.

        Precondition: at least one state already exists in self.states.
        Returns: the current number of states in the graph (len(self.states)).
        """
        if not self.states:
            raise RuntimeError("FGO.iterate() called with empty state list; "
                            "add an initial state + prior before the loop.")

        # State index of previous node in graph
        prev_idx = len(self.states) - 1

        # New graph state from current KF nominal (deep copy)
        x_init = self.create_esfk_state(x_nom=x_nom)
        idx = self.add_state(x_init)

        # Process factor between prev_idx and idx
        self.add_gyro_factor(prev_idx, idx, omega_meas, dt)

        # Environment vectors at jd
        r_eci = env.get_r_eci(jd)
        B_eci = env.get_B_eci(r_eci, jd)
        s_eci = env.get_sun_eci(jd)

        # Magnetometer factor
        if not np.any(np.isnan(mag_meas)):
            self.add_measurement(
                idx=idx,
                y=mag_meas,
                sensor_type=SensorType.MAGNETOMETER,
                B_n=B_eci,
            )

        # Sun sensor factor
        if not np.any(np.isnan(sun_meas)):
            self.add_measurement(
                idx=idx,
                y=sun_meas,
                sensor_type=SensorType.SUN_VECTOR,
                s_n=s_eci,
            )

        # # Star tracker factor
        # if not np.any(np.isnan(st_meas)):
        #     q_meas = Quaternion.from_array(st_meas)
        #     self.add_measurement(
        #         idx=idx,
        #         y=q_meas,
        #         sensor_type=SensorType.STAR_TRACKER,
        #     )

        return len(self.states)


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