import numpy as np
import copy
from typing import Optional

from sim.eskf import ESKF
from sim.fg import FGO
from utilities.states import NominalState, EskfState
from utilities.gaussian import MultiVarGauss
from utilities.quaternion import Quaternion
from utilities.states import SensorType
from environment.environment import OrbitEnvironmentModel
from data.db import SimulationDatabase
from sim.temp import SlidingWindow, ModeSupervisor
from logging_config import get_logger
from utilities.utils import load_yaml
from sim.temp import error_angle_deg

logger = get_logger(__name__)

class HybridRunner:
    def __init__(
        self,
        env: OrbitEnvironmentModel,
        eskf: ESKF,
        fgo: FGO,
        db_path: str = "simulations.db",
        window_duration: float = 10.0,
        use_switching: bool = True,
    ):
        self.env = env
        self.eskf = eskf
        self.fgo = fgo
        self.db = SimulationDatabase(db_path)
        self.window = SlidingWindow(duration=window_duration)
        self.supervisor = ModeSupervisor()
        self.use_switching = use_switching
        self.config = load_yaml("config.yaml")
        

    def run(self, sim_run_id: int,
                eskf_name: str = "eskf_only",
                fgo_name: str = "fgo_only"):
        """
        Run ESKF and FGO side-by-side in the same time loop, but without any
        hybrid feedback. Returns (eskf_est_run_id, fgo_est_run_id).
        """

        sim = self.db.load_run(sim_run_id)

        t = sim.t
        jd = sim.jd
        omega_meas_log = sim.omega_meas
        mag_meas_log   = sim.mag_meas
        sun_meas_log   = sim.sun_meas
        st_meas_log    = sim.st_meas
        q_true         = sim.q_true

        N = t.shape[0]

        # ----- 1) initial KF state -----
        q0 = Quaternion.from_array(q_true[0])
        b0 = np.zeros(3)
        nom0 = NominalState(ori=q0, gyro_bias=b0)
        err0 = MultiVarGauss(mean=np.zeros(6), cov=self.eskf.P0.copy())
        x_kf = EskfState(nom=nom0, err=err0)

        kf_states: list[EskfState] = [copy.deepcopy(x_kf)]
        fgo_global: list[Optional[EskfState]] = [None] * N  # FGO estimates per time index

        # ----- 2) initialize FGO window -----
        self.fgo.states = []
        self.fgo.factors = []

        # first graph node from initial KF
        idx0 = self.fgo.add_state(copy.deepcopy(x_kf))
        self.fgo.add_prior(
            state_idx=idx0,
            prior=x_kf.nom,
            cov=np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6]),
        )

        window_start_k = 0  # global index corresponding to self.fgo.states[0]

        # ----- 3) main loop -----
        for k in range(1, N):
            dt_k = t[k] - t[k-1]

            # ---------- KF: predict ----------
            omega_meas = omega_meas_log[k]
            if np.any(np.isnan(omega_meas)):
                omega_meas = omega_meas_log[k-1]

            x_kf = self.eskf.predict(x_kf, omega_meas, dt_k)

            # environment
            r_eci = self.env.get_r_eci(jd[k])
            B_eci = self.env.get_B_eci(r_eci, jd[k])
            s_eci = self.env.get_sun_eci(jd[k])

            # ---------- KF: updates ----------
            mag_meas = mag_meas_log[k]
            if not np.any(np.isnan(mag_meas)):
                x_kf = self.eskf.update(
                    x_est=x_kf,
                    y=mag_meas,
                    sensor_type=SensorType.MAGNETOMETER,
                    B_n=B_eci,
                )

            sun_meas = sun_meas_log[k]
            if not np.any(np.isnan(sun_meas)):
                x_kf = self.eskf.update(
                    x_est=x_kf,
                    y=sun_meas,
                    sensor_type=SensorType.SUN_VECTOR,
                    s_n=s_eci,
                )

            st_meas = st_meas_log[k]
            if not np.any(np.isnan(st_meas)):
                q_meas = Quaternion.from_array(st_meas)
                x_kf = self.eskf.update(
                    x_est=x_kf,
                    y=q_meas,
                    sensor_type=SensorType.STAR_TRACKER,
                )

            # store KF
            if np.any(np.isnan(x_kf.nom.gyro_bias)):
                raise ValueError(f"NaN gyro-bias in KF at step {k}")
            kf_states.append(copy.deepcopy(x_kf))
            
            if k % 100 == 0 or k == N - 1:
                q_true_arr = q_true[k]
                err_angle = error_angle_deg(
                    Quaternion.from_array(q_true_arr),
                    x_kf.nom.ori,
                )
                logger.info(f"KF step {k}/{N-1}: attitude error = {err_angle:.3f} deg")

        #     # ---------- FGO: build window graph (no feedback) ----------
        #     # add new graph state from KF nominal, plus process + measurement factors
        #     _ = self.fgo.iterate(
        #         dt=dt_k,
        #         jd=jd[k],
        #         x_nom=x_kf.nom,
        #         omega_meas=omega_meas,
        #         mag_meas=mag_meas,
        #         sun_meas=sun_meas,
        #         st_meas=st_meas,
        #         env=self.env,
        #     )

        #     window_full = len(self.fgo.states) >= self.fgo.window_size
        #     last_step   = (k == N - 1)

        #     if not (window_full or last_step):
        #         continue

        #     # ---------- FGO: optimize current window ----------
        #     optimized_states = list(self.fgo.optimize())

        #     # ---------- FGO: commit optimized window to global FGO trajectory ----------
        #     # all but last state are "fixed" and can be written to global array
        #     for local_i in range(len(optimized_states) - 1):
        #         global_i = window_start_k + local_i
        #         if global_i >= N:
        #             break
        #         if fgo_global[global_i] is None:
        #             fgo_global[global_i] = optimized_states[local_i]

        #     # carry last state as start of next window
        #     last_state = optimized_states[-1]
        #     window_start_k = k  # new window starts at current time index

        #     # reset graph for next window
        #     self.fgo.states = [last_state]
        #     self.fgo.factors = []
        #     self.fgo.add_prior(
        #         state_idx=0,
        #         prior=last_state.nom,
        #         cov=np.diag([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8]),
        #     )

        # # ensure last FGO state is filled
        # if fgo_global[-1] is None:
        #     # carry last window state
        #     fgo_global[-1] = self.fgo.states[0]

        # # ---------- 4) write both runs to DB ----------
        # eskf_run_id = self.db.insert_estimation_run(
        #     sim_run_id=sim_run_id,
        #     t=t,
        #     jd=jd,
        #     states=kf_states,
        #     name=eskf_name,
        # )

        # # type ignore: assume all entries filled
        # fgo_states_filled = [s for s in fgo_global if s is not None]  # type: ignore
        # fgo_run_id = self.db.insert_estimation_run(
        #     sim_run_id=sim_run_id,
        #     t=t,
        #     jd=jd,
        #     states=fgo_states_filled,
        #     name=fgo_name,
        # )

        # return eskf_run_id, fgo_run_id

    
if __name__ == "__main__":
    env = OrbitEnvironmentModel()
    eskf = ESKF(
        P0=np.diag([1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4]),  # initial covariance
    )
    fgo = FGO()
    hybrid_runner = HybridRunner(
        env=env,
        eskf=eskf,
        fgo=fgo,
        db_path="simulations.db",
        window_duration=10.0,
        use_switching=True,
    )
    
    sim_run_id = 1  # Example simulation run ID
    est_run_id = hybrid_runner.run(sim_run_id=sim_run_id, eskf_name="hybrid_eskf", fgo_name="hybrid_fgo")
    print(f"Hybrid estimation run ID: {est_run_id}")
