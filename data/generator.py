import numpy as np
from utilities.sensors import SensorGyro, SensorMagnetometer, SensorSunVector, SensorStarTracker
from utilities.quaternion import Quaternion
from environment.environment import OrbitEnvironmentModel
from data.classes import SimulationConfig, SimulationResult
from data.db import SimulationDatabase  # adjust import to your path
from utilities.utils import load_yaml


class AttitudeDataGenerator:
    def __init__(self,
                 env: OrbitEnvironmentModel,
                 gyro: SensorGyro,
                 mag: SensorMagnetometer,
                 sun: SensorSunVector,
                 star_trk: SensorStarTracker,
                 db_path: str = "simulations.db"):
        self.env = env
        self.gyro = gyro
        self.mag = mag
        self.sun = sun
        self.star_trk = star_trk
        self.db = SimulationDatabase(db_path)
        self.config = load_yaml("config.yaml")

    @staticmethod
    def _dt_to_divisor(sensor_dt: float, base_dt: float, name: str) -> int:
        """Return integer n such that sensor fires every n*base_dt ≈ sensor_dt."""
        if sensor_dt <= 0.0:
            raise ValueError(f"{name}.dt must be > 0")
        if sensor_dt < base_dt - 1e-12:
            raise ValueError(f"{name}.dt={sensor_dt} < base dt={base_dt}")
        ratio = sensor_dt / base_dt
        n = int(round(ratio))
        if n < 1:
            n = 1
        if abs(n * base_dt - sensor_dt) > 1e-9:
            raise ValueError(
                f"{name}.dt={sensor_dt} not an integer multiple of base dt={base_dt}"
            )
        return n
    
    def _get_omega_profile(self, t: float) -> np.ndarray:   
        A_x = self.config["omega_profile"]["amplitude"]["x"]
        A_y = self.config["omega_profile"]["amplitude"]["y"]
        A_z = self.config["omega_profile"]["amplitude"]["z"]
        f_x = eval(self.config["omega_profile"]["frequency"]["x"])
        f_y = eval(self.config["omega_profile"]["frequency"]["y"])
        f_z = eval(self.config["omega_profile"]["frequency"]["z"])
        
        omega_x = A_x * np.sin(2 * np.pi * f_x * t)
        omega_y = A_y * np.sin(2 * np.pi * f_y * t)
        omega_z = A_z * np.sin(2 * np.pi * f_z * t)
        
        return np.array([omega_x, omega_y, omega_z])

    def run(self, cfg: SimulationConfig) -> None:
        T, dt = self.config["simulation"]["T"], self.config["simulation"]["dt"]
        n_steps = int(np.floor(T / dt)) + 1

        # compute sampling divisors
        gyro_div = self._dt_to_divisor(self.gyro.dt, dt, "gyro")
        mag_div = self._dt_to_divisor(self.mag.dt, dt, "magnetometer")
        sun_div = self._dt_to_divisor(self.sun.dt, dt, "sun_sensor")
        st_div = self._dt_to_divisor(self.star_trk.dt, dt, "star_tracker")

        t_vec = np.linspace(0.0, T, n_steps)
        jd_vec = cfg.start_jd + t_vec / 86400.0

        # truth logs
        q_true_log = np.zeros((n_steps, 4))
        b_g_true_log = np.zeros((n_steps, 3))
        omega_true_log = np.zeros((n_steps, 3))

        # measurement logs
        omega_meas_log = np.full((n_steps, 3), np.nan)
        mag_meas_log = np.full((n_steps, 3), np.nan)
        sun_meas_log = np.full((n_steps, 3), np.nan)
        st_meas_log = np.full((n_steps, 4), np.nan)

        # true states
        q_true = Quaternion(1.0, np.zeros(3))
        b_g_true = np.zeros(3)

        # true gyro-bias random walk std (could be moved into cfg)
        sigma_bg_true = np.deg2rad(self.config["sensors"]["gyro"]["noise_rw"])

        for k in range(n_steps):
            t = t_vec[k]
            jd = jd_vec[k]

            # 1) Environment: ECI position, B_eci, sun_eci
            r_eci = self.env.get_r_eci(jd)
            B_eci = self.env.get_B_eci(r_eci, jd)   # magnetic field in ECI
            s_eci = self.env.get_sun_eci(jd)        # Sun direction in ECI (unit)

            # 2) True body rate profile
            omega_true = self._get_omega_profile(t)
            omega_true_log[k] = omega_true

            # 3) Propagate true attitude
            if k > 0:
                q_true = q_true.propagate(omega_true, dt)

            # 4) Propagate true gyro bias (random walk)
            if k > 0:
                w_bg = sigma_bg_true * np.sqrt(dt) * np.random.randn(3)
                b_g_true = b_g_true + w_bg

            # 5) Sensor measurements with individual sampling times

            # 5a) Gyro (at gyro.dt)
            if k % gyro_div == 0:
                omega_truth_for_sensor = omega_true + b_g_true
                omega_meas = self.gyro.sample(omega_truth_for_sensor)
                omega_meas_log[k] = omega_meas

            # 5b) Magnetometer (at mag.dt)
            if k % mag_div == 0:
                mag_meas = self.mag.sample(q_true=q_true, b_eci=B_eci)
                mag_meas_log[k] = mag_meas

            # 5c) Sun sensor (at sun.dt)
            if k % sun_div == 0:
                sun_meas = self.sun.sample(q_true=q_true, s_eci=s_eci)
                # renormalise to emulate a unit-vector measurement
                nrm = np.linalg.norm(sun_meas)
                if nrm > 1e-12:
                    sun_meas /= nrm
                else:
                    sun_meas[:] = np.nan
                sun_meas_log[k] = sun_meas

            # 5d) Star tracker (at star_trk.dt)
            if k % st_div == 0:
                q_st_meas = self.star_trk.sample(q_true=q_true)
                if q_st_meas is not None:
                    st_meas_log[k] = q_st_meas.as_array()

            # 6) log truth
            q_true_log[k] = q_true.as_array()
            b_g_true_log[k] = b_g_true

        # store directly in database
        result = SimulationResult(
            t=t_vec,
            jd=jd_vec,
            q_true=q_true_log,
            b_g_true=b_g_true_log,
            omega_true=omega_true_log,
            omega_meas=omega_meas_log,
            mag_meas=mag_meas_log,
            sun_meas=sun_meas_log,
            st_meas=st_meas_log,
            config=cfg,
        )
        id = self.db.insert_run(result=result)
        print(f"Simulation run '{id}' with ID stored in database.")
        return

    

        
    
if __name__ == "__main__":
    config = load_yaml("config.yaml")
    env = OrbitEnvironmentModel()
    gyro = SensorGyro()
    mag = SensorMagnetometer()
    sun = SensorSunVector()
    star_trk = SensorStarTracker()
    attitudeDataGenerator = AttitudeDataGenerator(
        env=env,
        gyro=gyro,
        mag=mag,
        sun=sun,
        star_trk=star_trk
    )
    
    sim_cfg = SimulationConfig(
        T=config["simulation"]["T"],
        dt=config["simulation"]["dt"],
        start_jd=config["simulation"]["start_jd"],
        run_name=config["simulation"]["run_name"],
        gyro_noise_scale=config["simulation"]["gyro_noise_scale"],
        mag_noise_scale=config["simulation"]["mag_noise_scale"],
        sun_noise_scale=config["simulation"]["sun_noise_scale"],
    )
    
    attitudeDataGenerator.run(cfg=sim_cfg)