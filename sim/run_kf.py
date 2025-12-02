import numpy as np
import copy

from sim.eskf import ESKF
from utilities.states import NominalState, EskfState
from utilities.gaussian import MultiVarGauss
from utilities.quaternion import Quaternion
from utilities.states import SensorType
from environment.environment import OrbitEnvironmentModel
from data.db import SimulationDatabase
from sim.temp import get_sensor_config

from logging_config import get_logger
from utilities.utils import load_yaml

logger = get_logger(__name__)

class FilterRunner:
    def __init__(self,
                 env: OrbitEnvironmentModel,
                 eskf: ESKF,
                 db_path: str = "simulations.db"):
        self.env = env
        self.eskf = eskf
        self.db = SimulationDatabase(db_path)
        self.config = load_yaml("config.yaml")
        
    def _if_drop_measurement(self, drop_cfg) -> bool:
        """Decide whether to drop measurement based on config.

        Args:
            drop_cfg: a dictionary containing values drop and use_measurements.

        Returns:
            True if measurement should be dropped.
        """
        use_measurements = drop_cfg.get("use_measurements", True)
        if not use_measurements:
            return True
        
        drop_prob = drop_cfg.get("drop", 0.0)
        if drop_prob <= 0.0:
            return False
        
        rand_val = np.random.rand()
        if rand_val < drop_prob:
            logger.info(f"Dropping measurement (rand={rand_val:.3f} < drop_prob={drop_prob:.3f})")
            return True
        else:
            return False
        

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
        
        # --- Get configs for dropout handling ---
        mag_cfg = get_sensor_config(SensorType.MAGNETOMETER, self.config)
        sun_cfg = get_sensor_config(SensorType.SUN_VECTOR, self.config)
        st_cfg  = get_sensor_config(SensorType.STAR_TRACKER, self.config)

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
        
        # True quaternions for error computation
        q_true = result.q_true
        
        logger.info(f"P before first update:\n{err_cov0}")

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
            if not np.any(np.isnan(mag_meas)) and not self._if_drop_measurement(mag_cfg):
                x_est = self.eskf.update(
                    x_est=x_est,
                    y=mag_meas,
                    sensor_type=SensorType.MAGNETOMETER,
                    B_n=B_eci,
                )

            # sun vector update
            sun_meas = sun_meas_log[k]
            if not np.any(np.isnan(sun_meas)) and not self._if_drop_measurement(sun_cfg):
                x_est = self.eskf.update(
                    x_est=x_est,
                    y=sun_meas,
                    sensor_type=SensorType.SUN_VECTOR,
                    s_n=s_eci,
                )

            # star tracker update
            st_meas = st_meas_log[k]
            if not np.any(np.isnan(st_meas)) and not self._if_drop_measurement(st_cfg):
                q_meas = Quaternion.from_array(st_meas)
                x_est = self.eskf.update(
                    x_est=x_est,
                    y=q_meas,
                    sensor_type=SensorType.STAR_TRACKER,
                )

            # store snapshot       
            states.append(copy.deepcopy(x_est))
            
            
            if k % 100 == 0 or k == N - 1:
                q_true_arr = q_true[k]
                q_est_arr  = x_est.nom.ori.as_array()

                # Enforce same hemisphere to avoid sign flips
                if np.dot(q_true_arr, q_est_arr) < 0.0:
                    q_est_arr = -q_est_arr

                q_t = Quaternion.from_array(q_true_arr)
                q_e = Quaternion.from_array(q_est_arr)

                q_err = q_t.conjugate().multiply(q_e).normalize()
                mu = np.clip(q_err.mu, -1.0, 1.0)
                angle = 2.0 * np.arccos(mu)
                if angle > np.pi:
                    angle -= 2.0 * np.pi
                logger.info(f"Step {k:4d}/{N-1}: attitude error = {np.degrees(np.abs(angle)):.3f} deg")
                if angle > np.radians(100.0):
                    logger.warning("Large attitude error detected!")
                    exit(1)

            


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
        P0=np.diag([1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4]),  # initial covariance
    )
    filter_runner = FilterRunner(env=env, eskf=eskf, db_path="simulations.db")
    
    sim_run_id = 1  # Example simulation run ID
    est_run_id = filter_runner.run_on_simulation(sim_run_id=sim_run_id, est_name="eskf_baseline")
    print(f"Estimation run ID: {est_run_id}")
    