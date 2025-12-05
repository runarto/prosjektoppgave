from data.db import SimulationDatabase

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context="paper", style="whitegrid", palette="deep", font_scale=1.1)

def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion array to Euler angles (roll, pitch, yaw) in radians.
    
    Args:
        q: np.ndarray of shape (N, 4) representing quaternions [q0, q1, q2, q3]
    """
    
    N = q.shape[0]
    euler = np.zeros((N, 3))  # roll, pitch, yaw

    for i in range(N):
        q0, q1, q2, q3 = q[i]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1 - 2 * (q1**2 + q2**2)
        euler[i, 0] = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (q0 * q2 - q3 * q1)
        if abs(sinp) >= 1:
            euler[i, 1] = np.sign(sinp) * (np.pi / 2)  # use 90 degrees if out of range
        else:
            euler[i, 1] = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1 - 2 * (q2**2 + q3**2)
        euler[i, 2] = np.arctan2(siny_cosp, cosy_cosp)

    return euler


class DataPlotter:
    def __init__(self):
        self.db = SimulationDatabase("simulations.db")
        
    def get_data(self, sim_id: int, data_type: str):
        data = self.db.load_run(sim_id)
        match data_type:
            case "star":
                return (data.st_meas, data.t)
            case "sun":
                return (data.sun_meas, data.t)
            case "gyro":
                return (data.omega_meas, data.t)
            case "magnetometer":
                return (data.mag_meas, data.t)
            case "quaternion":
                euler = quat_to_euler(data.q_true)
                return (euler, data.t)
            case "bias":
                return (data.b_g_true, data.t)
            case _:
                raise ValueError(f"Unknown data type: {data_type}")

    def plot_data(self, sim_id: int, data_type: str):
        measurements, t = self.get_data(sim_id, data_type)

        plt.figure(figsize=(10, 6))

        # Handle quaternion → Euler shape mismatch if needed
        if measurements.ndim == 1:
            measurements = measurements[:, None]

        for i in range(measurements.shape[1]):
            # Mask valid entries
            valid = ~np.isnan(measurements[:, i])

            if np.any(valid):   # only plot if at least one valid point
                plt.plot(t[valid], measurements[valid, i],
                        label=f"{data_type} axis {i+1}")

        plt.xlabel("Time [s]")
        plt.ylabel(f"{data_type} measurements")
        plt.title(f"{data_type.capitalize()} Measurements over Time (Sim ID: {sim_id})")
        plt.legend()
        plt.grid()
        plt.show()

        
        
if __name__ == "__main__": 
    plotter = DataPlotter()
    sim_id = 4  # Example simulation ID
    for data_type in ["star", "sun", "gyro", "magnetometer", "quaternion", "bias"]:
        plotter.plot_data(sim_id, data_type)