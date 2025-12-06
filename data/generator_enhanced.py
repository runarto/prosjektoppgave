"""
Enhanced data generator with support for anomaly scenarios:
- Measurement spikes (already in sensor models)
- Measurement freezes (stuck measurements)
- Eclipse periods (sun sensor dropout)
- High spin rates (star tracker dropout)
"""

import numpy as np
from utilities.sensors import SensorGyro, SensorMagnetometer, SensorSunVector, SensorStarTracker
from utilities.quaternion import Quaternion
from environment.environment import OrbitEnvironmentModel
from data.classes import SimulationConfig, SimulationResult
from data.db import SimulationDatabase
from utilities.utils import load_yaml

def quat_diff(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """Compute the quaternion difference q_err = q1 * q2_conjugate."""
    return q1.multiply(q2.conjugate())


class EnhancedAttitudeDataGenerator:
    """Extended generator with anomaly scenario support."""

    def __init__(self, db_path: str = "simulations.db", config_path: str = "config.yaml"):
        self.env = OrbitEnvironmentModel()
        self.gyro = SensorGyro(config_path)
        self.mag = SensorMagnetometer(config_path)
        self.sun = SensorSunVector(config_path)
        self.star_trk = SensorStarTracker(config_path)
        self.db = SimulationDatabase(db_path)
        self.config = load_yaml(config_path)
        self.config_file = config_path

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
        """Get angular velocity profile at time t."""
        A_x = self.config["omega_profile"]["amplitude"]["x"]
        A_y = self.config["omega_profile"]["amplitude"]["y"]
        A_z = self.config["omega_profile"]["amplitude"]["z"]
        f_x = self.config["omega_profile"]["frequency"]["x"]
        f_y = self.config["omega_profile"]["frequency"]["y"]
        f_z = self.config["omega_profile"]["frequency"]["z"]

        omega_x = A_x * np.sin(2 * np.pi * f_x * t)
        omega_y = A_y * np.sin(2 * np.pi * f_y * t)
        omega_z = A_z * np.sin(2 * np.pi * f_z * t)

        return np.array([omega_x, omega_y, omega_z])

    def _check_eclipse(self, t: float, r_eci: np.ndarray, s_eci: np.ndarray) -> bool:
        """Check if satellite is in Earth's shadow (eclipse).

        Args:
            t: current time (s)
            r_eci: satellite position in ECI (m)
            s_eci: sun direction unit vector in ECI

        Returns:
            True if in eclipse
        """
        # Check config for forced eclipse
        anomaly_cfg = self.config.get("simulation", {}).get("anomalies", {})
        eclipse_cfg = anomaly_cfg.get("eclipse", {})

        if eclipse_cfg.get("enabled", False):
            eclipse_start = eclipse_cfg.get("start_time", 0.0)
            eclipse_duration = eclipse_cfg.get("duration", 0.0)
            if eclipse_start <= t < eclipse_start + eclipse_duration:
                return True

        # Simple geometric eclipse check: if Earth blocks sun
        # Earth radius ~ 6371 km
        R_earth = 6.371e6  # meters
        r_norm = np.linalg.norm(r_eci)

        # If satellite altitude is low enough and sun is behind Earth
        # Simplified: dot product check
        cos_angle = np.dot(r_eci, s_eci) / r_norm
        if cos_angle < 0 and r_norm < 2 * R_earth:
            # Rough approximation of eclipse
            return True

        return False

    def _check_freeze(self, t: float, sensor_name: str) -> bool:
        """Check if measurement should be frozen at this time.

        Args:
            t: current time (s)
            sensor_name: "magnetometer", "sun", or "star"

        Returns:
            True if this sensor should freeze
        """
        anomaly_cfg = self.config.get("simulation", {}).get("anomalies", {})
        freeze_cfg = anomaly_cfg.get("freeze", {})

        if not freeze_cfg.get("enabled", False):
            return False

        if freeze_cfg.get("sensor", "") != sensor_name:
            return False

        freeze_start = freeze_cfg.get("start_time", 0.0)
        freeze_duration = freeze_cfg.get("duration", 0.0)

        return freeze_start <= t < freeze_start + freeze_duration

    def run(self, cfg: SimulationConfig) -> None:
        """Generate simulation data with potential anomalies."""
        T, dt = self.config["time"]["sim_T"], self.config["time"]["sim_dt"]
        n_steps = int(np.floor(T / dt)) + 1

        # Compute sampling divisors
        gyro_div = self._dt_to_divisor(self.gyro.dt, dt, "gyro")
        mag_div = self._dt_to_divisor(self.mag.dt, dt, "magnetometer")
        sun_div = self._dt_to_divisor(self.sun.dt, dt, "sun_sensor")
        st_div = self._dt_to_divisor(self.star_trk.dt, dt, "star_tracker")

        t_vec = np.linspace(0.0, T, n_steps)
        jd_vec = cfg.start_jd + t_vec / 86400.0

        # Truth logs
        q_true_log = np.zeros((n_steps, 4))
        b_g_true_log = np.zeros((n_steps, 3))
        omega_true_log = np.zeros((n_steps, 3))

        # Environment logs (ECI frame vectors)
        b_eci_log = np.zeros((n_steps, 3))
        s_eci_log = np.zeros((n_steps, 3))

        # Measurement logs
        omega_meas_log = np.full((n_steps, 3), np.nan)
        mag_meas_log = np.full((n_steps, 3), np.nan)
        sun_meas_log = np.full((n_steps, 3), np.nan)
        st_meas_log = np.full((n_steps, 4), np.nan)

        # True states - initialize attitude to point sun sensor (+Z body) toward sun
        # This ensures sun sensor measurements are available from the start
        s_eci_initial = self.env.get_sun_eci(jd_vec[0])
        s_eci_initial = s_eci_initial / np.linalg.norm(s_eci_initial)  # Ensure unit vector

        # Compute rotation that aligns +Z body with sun direction
        z_body = np.array([0.0, 0.0, 1.0])

        # Compute rotation axis and angle using Rodrigues' formula
        axis = np.cross(z_body, s_eci_initial)
        axis_norm = np.linalg.norm(axis)

        if axis_norm > 1e-6:
            # Non-parallel case: compute rotation
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(z_body, s_eci_initial), -1.0, 1.0))
            # Use from_avec which takes angle*axis vector
            q_true = Quaternion.from_avec(axis * angle)
            print(f"  Initial attitude: pointing sun sensor toward sun (angle={np.rad2deg(angle):.1f}°)")
        else:
            # Parallel or anti-parallel case
            if np.dot(z_body, s_eci_initial) > 0:
                # Already aligned
                q_true = Quaternion(1.0, np.zeros(3))
                print(f"  Initial attitude: sun sensor already aligned with sun")
            else:
                # Anti-parallel: rotate 180° around any perpendicular axis
                q_true = Quaternion.from_avec(np.array([np.pi, 0.0, 0.0]))
                print(f"  Initial attitude: sun sensor opposite sun, rotating 180°")

        b_g_true = np.zeros(3)

        # True gyro-bias random walk: derive from RRW in deg/h/√h
        # RRW means after 1 hour, bias has std = RRW [deg/h]
        # Convert to continuous-time density σ_bg [rad/s/√s]:
        #   σ_bg = RRW × (π/180) / 3600 / 60
        RRW_deg = self.config["sensors"]["gyro"]["noise"]["rrw_deg"]
        sigma_bg_true = RRW_deg * (np.pi / 180.0) / 3600.0 / 60.0

        # Storage for frozen measurements
        frozen_mag = None
        frozen_sun = None
        frozen_st = None

        print(f"Generating {n_steps} samples over {T:.1f} seconds...")
        print(f"Config file: {self.config_file}")

        # Check for anomalies
        anomaly_cfg = self.config.get("simulation", {}).get("anomalies", {})
        if anomaly_cfg:
            print("\nConfigured anomalies:")
            if anomaly_cfg.get("eclipse", {}).get("enabled"):
                eclipse = anomaly_cfg["eclipse"]
                print(f"  Eclipse: {eclipse['start_time']:.1f}s - {eclipse['start_time']+eclipse['duration']:.1f}s")
            if anomaly_cfg.get("freeze", {}).get("enabled"):
                freeze = anomaly_cfg["freeze"]
                print(f"  Freeze ({freeze['sensor']}): {freeze['start_time']:.1f}s - {freeze['start_time']+freeze['duration']:.1f}s")
            if anomaly_cfg.get("high_spin", {}).get("enabled"):
                spin = anomaly_cfg["high_spin"]
                print(f"  High spin: {spin['start_time']:.1f}s - {spin['start_time']+spin['duration']:.1f}s")

        # Check for spikes
        has_spikes = (
            self.config["sensors"]["mag"]["spikes"].get("enabled", False) or
            self.config["sensors"]["sun"]["spikes"].get("enabled", False) or
            self.config["sensors"]["star"]["spikes"].get("enabled", False)
        )
        if has_spikes:
            print("\nSpike generation enabled for one or more sensors")

        for k in range(n_steps):
            t = t_vec[k]
            jd = jd_vec[k]

            # Environment: ECI position, B_eci, sun_eci
            r_eci = self.env.get_r_eci(jd)
            B_eci = self.env.get_B_eci(r_eci, jd)
            s_eci = self.env.get_sun_eci(jd)

            # True body rate profile
            omega_true = self._get_omega_profile(t)
            omega_true_log[k] = omega_true

            # Propagate true attitude
            if k > 0:
                q_true = q_true.propagate(omega_true, dt)

            # Propagate true gyro bias (random walk)
            if k > 0:
                w_bg = sigma_bg_true * np.sqrt(dt) * np.random.randn(3)
                b_g_true = b_g_true + w_bg

            # Check eclipse status
            in_eclipse = self._check_eclipse(t, r_eci, s_eci)

            # Sensor measurements with anomaly handling

            # Gyro (always available)
            if k % gyro_div == 0:
                omega_truth_for_sensor = omega_true + b_g_true
                omega_meas = self.gyro.sample(omega_truth_for_sensor)
                omega_meas_log[k] = omega_meas

            # Magnetometer (with freeze support)
            if k % mag_div == 0:
                if self._check_freeze(t, "magnetometer"):
                    # Use frozen measurement
                    if frozen_mag is not None:
                        mag_meas_log[k] = frozen_mag
                else:
                    # Normal measurement
                    mag_meas = self.mag.sample(q_true=q_true, B_n=B_eci)
                    if mag_meas is None:
                        mag_meas = np.full(3, np.nan)
                    else:
                        frozen_mag = mag_meas.copy()  # Store for potential freeze
                    mag_meas_log[k] = mag_meas

            # Sun sensor (with eclipse and freeze support)
            if k % sun_div == 0:
                if self._check_freeze(t, "sun"):
                    # Use frozen measurement
                    if frozen_sun is not None:
                        sun_meas_log[k] = frozen_sun
                else:
                    # Normal measurement (respecting eclipse)
                    sun_meas = self.sun.sample(q_true=q_true, s_n=s_eci, in_eclipse=in_eclipse)
                    if sun_meas is None:
                        sun_meas = np.full(3, np.nan)
                    else:
                        frozen_sun = sun_meas.copy()  # Store for potential freeze
                    sun_meas_log[k] = sun_meas

            # Star tracker (with freeze and high-rate support)
            if k % st_div == 0:
                if self._check_freeze(t, "star"):
                    # Use frozen measurement
                    if frozen_st is not None:
                        st_meas_log[k] = frozen_st
                else:
                    # Normal measurement (pass omega for rate check)
                    q_st_meas = self.star_trk.sample(q_true=q_true, omega_body=omega_true)
                    if q_st_meas is not None:                    
                        st_quat = q_st_meas.as_array()
                        st_meas_log[k] = st_quat
                        frozen_st = st_quat.copy()  # Store for potential freeze
                    else:
                        st_meas_log[k] = np.full(4, np.nan)

            # Log truth
            q_true_log[k] = q_true.as_array()
            b_g_true_log[k] = b_g_true

            # Log environment vectors
            b_eci_log[k] = B_eci
            s_eci_log[k] = s_eci

        print(f"Generation complete!")

        # Store in database
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
            b_eci=b_eci_log,
            s_eci=s_eci_log,
            config=cfg,
        )
        run_id = self.db.insert_run(result=result)
        print(f"Simulation run stored in database with ID: {run_id}")

        # Print statistics
        n_mag = np.sum(~np.isnan(mag_meas_log[:, 0]))
        n_sun = np.sum(~np.isnan(sun_meas_log[:, 0]))
        n_st = np.sum(~np.isnan(st_meas_log[:, 0]))

        print(f"\nMeasurement statistics:")
        print(f"  Gyro:         {n_steps} / {n_steps} (100%)")
        print(f"  Magnetometer: {n_mag} / {n_steps} ({100*n_mag/n_steps:.1f}%)")
        print(f"  Sun sensor:   {n_sun} / {n_steps} ({100*n_sun/n_steps:.1f}%)")
        print(f"  Star tracker: {n_st} / {n_steps} ({100*n_st/n_steps:.1f}%)")

        return run_id
