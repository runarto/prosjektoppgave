from dataclasses import dataclass
from typing import Optional
import numpy as np
from utilities.states import EskfState
from utilities.quaternion import Quaternion
from typing import Literal
from utilities.states import SensorType


@dataclass
class WindowSample:
    t: float
    jd: float
    x_nom: EskfState        # current nominal state from ESKF
    # You can store measurements per sensor as needed:
    omega_meas: np.ndarray
    z_mag: Optional[np.ndarray] = None
    z_sun: Optional[np.ndarray] = None
    z_st: Optional[Quaternion]  = None


class SlidingWindow:
    def __init__(self, duration: float):
        self.duration = duration
        self.samples: list[WindowSample] = []

    def add(self, sample: WindowSample):
        self.samples.append(sample)
        self._trim()

    def _trim(self):
        if not self.samples:
            return
        t_max = self.samples[-1].t
        t_min = t_max - self.duration if (t_max - self.duration) > 0 else 0.0
        # keep only samples with t >= t_min
        self.samples = [s for s in self.samples if s.t >= t_min]

    @property
    def ready(self) -> bool:
        if len(self.samples) < 2:
            return False
        return self.samples[-1].t - self.samples[0].t >= self.duration
    
    
class ModeSupervisor:
    def __init__(self,
                 nis_threshold: float = 20.0,
                 max_violations: int = 5):
        self.nis_threshold = nis_threshold
        self.max_violations = max_violations
        self.violation_count = 0
        self.current_mode: Literal["KF", "FGO"] = "KF"

    def update_nis(self, nis_k: float):
        # Call this from inside your KF update if you compute NIS
        if nis_k > self.nis_threshold:
            self.violation_count += 1
        else:
            self.violation_count = 0

        if self.violation_count >= self.max_violations:
            self.current_mode = "FGO"

    def reset_to_kf(self):
        self.violation_count = 0
        self.current_mode = "KF"


def get_sensor_config(sensor_type: SensorType, config: dict) -> dict:
    if sensor_type == SensorType.MAGNETOMETER:
        return config["sensors"]["mag"]
    elif sensor_type == SensorType.SUN_VECTOR:
        return config["sensors"]["sun"]
    elif sensor_type == SensorType.STAR_TRACKER:
        return config["sensors"]["star"]
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")
    
def error_angle_deg(q_ref: Quaternion, q_est: Quaternion) -> float:
    """Compute attitude error angle in degrees between two quaternions."""
    q_err = q_ref.conjugate().multiply(q_est).normalize()
    angle_rad = 2.0 * np.arccos(np.clip(q_err.mu, -1.0, 1.0))
    # shortest rotation
    if angle_rad > np.pi:
        angle_rad = 2*np.pi - angle_rad
    angle_deg = np.degrees(angle_rad)
    return angle_deg
    
    
def log_hybrid_status(k: int, N: int, t: float, x_est: EskfState, result) -> float:
    """Log attitude error and current mode in hybrid ESKF-FGO."""    
    q_true_arr = result.q_true[k]
    q_est_arr  = x_est.nom.ori.as_array()

    # Enforce same hemisphere to avoid sign flips
    if np.dot(q_true_arr, q_est_arr) < 0.0:
        q_est_arr = -q_est_arr

    q_t = Quaternion.from_array(q_true_arr)
    q_e = Quaternion.from_array(q_est_arr)

    q_err = q_t.conjugate().multiply(q_e).normalize()
    angle_err_deg = 2.0 * np.arccos(np.clip(q_err.mu, -1.0, 1.0)) * (180.0 / np.pi)
    
    return angle_err_deg


def compare_window_kf_fgo(
    kf_samples: list[WindowSample],
    fgo_states: list[EskfState],
) -> tuple[list[float], list[float]]:
    """
    Compare ALL local ESKF states in the sliding window with ALL
    optimized FGO states returned from optimize_local().

    Args:
        kf_samples: list[WindowSample]     # the KF estimates in the window
        fgo_states: list[EskfState]        # the window-size FGO results

    Returns:
        (err_rad_list, err_deg_list)
        Each element corresponds to one timestep in the window.
    """
    assert len(kf_samples) == len(fgo_states), \
        "Mismatch: KF window size != FGO window size"

    err_rad_list = []
    err_deg_list = []

    for sample, x_fgo in zip(kf_samples, fgo_states):
        x_kf = sample.x_nom

        # quaternion comparison
        q_kf  = x_kf.nom.ori
        q_fgo = x_fgo.nom.ori

        q_err = q_kf.conjugate().multiply(q_fgo).normalize()

        mu = np.clip(q_err.mu, -1.0, 1.0)
        angle_rad = 2.0 * np.arccos(mu)

        # shortest rotation
        if angle_rad > np.pi:
            angle_rad = 2*np.pi - angle_rad

        err_rad_list.append(angle_rad)
        err_deg_list.append(np.degrees(angle_rad))

    return err_rad_list, err_deg_list
