from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

@dataclass
class SimulationConfig:
    T: float                 # total time [s]
    dt: float                # step size [s]
    start_jd: float          # start Julian date
    run_name: str = "default"

    # you can add more here: sensor noise scale factors, etc.
    gyro_noise_scale: float = 1.0
    mag_noise_scale: float = 1.0
    sun_noise_scale: float = 1.0

@dataclass
class SimulationResult:
    # time
    t: np.ndarray            # shape (N,)
    jd: np.ndarray           # shape (N,)

    # truth
    q_true: np.ndarray       # shape (N, 4)
    b_g_true: np.ndarray     # shape (N, 3)
    omega_true: np.ndarray   # shape (N, 3)

    # measurements (perturbed)
    omega_meas: np.ndarray   # shape (N, 3)
    mag_meas: np.ndarray     # shape (N, 3)
    sun_meas: np.ndarray     # shape (N, 3)
    st_meas: np.ndarray      # shape (N, 4), NaN rows when no measurement

    # for bookkeeping
    config: SimulationConfig
    
@dataclass
class EstimationResult:
    t: np.ndarray          # (N,)
    jd: np.ndarray         # (N,)
    q_est: np.ndarray      # (N, 4)
    bg_est: np.ndarray     # (N, 3)
     
