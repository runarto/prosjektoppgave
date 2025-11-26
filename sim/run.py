import numpy as np
import copy

from sim.eskf import ESKF
from utilities.states import NominalState, EskfState
from utilities.gaussian import MultiVarGauss
from utilities.quaternion import Quaternion
from utilities.states import SensorType
from environment.environment import OrbitEnvironmentModel
from data.db import SimulationDatabase

from logging_config import get_logger

logger = get_logger(__name__)


class FilterRunner:
    def __init__(self,
                 env: OrbitEnvironmentModel,
                 eskf: ESKF,
                 db_path: str = "simulations.db"):
        self.env = env
        self.eskf = eskf
        self.db = SimulationDatabase(db_path)

    def run_on_simulation(self, sim_run_id: int, est_name: str = "eskf") -> int:
        """
        Run ESKF over one SimulationResult stored in the DB and save results.

        Args:
            sim_run_id: run id in 'runs' table
            est_name:   label for this estimation run

        Returns:
            est_run_id in est_runs.
        """
        result = self.db.load_run(sim_run_id)

        t = result.t
        jd = result.jd
        omega_meas_log = result.omega_meas
        mag_meas_log   = result.mag_meas
        sun_meas_log   = result.sun_meas
        st_meas_log    = result.st_meas

        N = t.shape[0]

        # ----- initial state -----
        q0_arr = result.q_true[0]
        q0 = Quaternion.from_array(q0_arr)
        b0 = np.zeros(3)

        nom0 = NominalState(ori=q0, gyro_bias=b0)
        err_mean0 = np.zeros(6)
        err_cov0 = self.eskf.P0.copy()
        err0 = MultiVarGauss(err_mean0, err_cov0)

        x_est = EskfState(nom=nom0, err=err0)

        states: list = [x_est]

        # ----- main loop -----
        for k in range(1, N):
            dt_k = t[k] - t[k-1]

            # prediction
            omega_meas = omega_meas_log[k]
            if np.any(np.isnan(omega_meas)):
                omega_meas = omega_meas_log[k-1]

            x_est = self.eskf.predict(x_est, omega_meas, dt_k)

            # environment fields
            r_eci = self.env.get_r_eci(jd[k])
            B_eci = self.env.get_B_eci(r_eci, jd[k])
            s_eci = self.env.get_sun_eci(jd[k])

            # magnetometer update
            mag_meas = mag_meas_log[k]
            if not np.any(np.isnan(mag_meas)):
                x_est = self.eskf.update(
                    x_est=x_est,
                    y=mag_meas,
                    sensor_type=SensorType.MAGNETOMETER,
                    B_n=B_eci,
                )

            # sun vector update
            sun_meas = sun_meas_log[k]
            if not np.any(np.isnan(sun_meas)):
                x_est = self.eskf.update(
                    x_est=x_est,
                    y=sun_meas,
                    sensor_type=SensorType.SUN_VECTOR,
                    s_n=s_eci,
                )

            # star tracker update
            st_meas = st_meas_log[k]
            if not np.any(np.isnan(st_meas)):
                q_meas = Quaternion.from_array(st_meas)
                x_est = self.eskf.update(
                    x_est=x_est,
                    y=q_meas,
                    sensor_type=SensorType.STAR_TRACKER,
                )

            # store snapshot       
            states.append(copy.deepcopy(x_est))

        # save estimation results in DB
        est_run_id = self.db.insert_estimation_run(
            sim_run_id=sim_run_id,
            t=t,
            jd=jd,
            states=states,
            name=est_name,
        )
        return est_run_id
    
    
if __name__ == "__main__":
    env = OrbitEnvironmentModel()
    eskf = ESKF(
        Q=np.diag([1e-7, 1e-7, 1e-7]),          # gyro noise
        P0=np.diag([1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4]),  # initial covariance
    )
    filter_runner = FilterRunner(env=env, eskf=eskf, db_path="simulations.db")
    
    sim_run_id = 2  # Example simulation run ID
    est_run_id = filter_runner.run_on_simulation(sim_run_id=sim_run_id, est_name="eskf_baseline")
    print(f"Estimation run ID: {est_run_id}")