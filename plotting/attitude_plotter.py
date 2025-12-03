import numpy as np
import matplotlib.pyplot as plt

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from data.classes import EstimationResult

import seaborn as sns
sns.set_theme(context="paper", style="whitegrid", palette="deep", font_scale=1.1)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.2,
    "grid.alpha": 0.3,
})


class AttitudePlotter:
    def __init__(self, db_path: str = "simulations.db"):
        self.db = SimulationDatabase(db_path)

    def _load_estimation_run(self, est_run_id: int) -> EstimationResult:
        return self.db.load_estimated_states(est_run_id)

    @staticmethod
    def _quat_seq_to_euler(q_seq: np.ndarray) -> np.ndarray:
        """(N,4) quats -> (N,3) Euler (roll, pitch, yaw) in rad."""
        e_list = []
        for q_arr in q_seq:
            q = Quaternion.from_array(q_arr)
            e_list.append(q.as_euler())
        return np.array(e_list)

    @staticmethod
    def _quat_angle_error(q_true_arr: np.ndarray, q_est_arr: np.ndarray) -> np.ndarray:
        """
        Compute per-step absolute rotation angle between q_true and q_est.
        q_true_arr, q_est_arr: (N,4) arrays [mu, eta_x, eta_y, eta_z].
        Returns:
            angle_err: (N,) angles [rad].
        """
        N = q_true_arr.shape[0]
        angle_err = np.zeros(N)
        for k in range(N):
            q_t = Quaternion.from_array(q_true_arr[k])
            q_e = Quaternion.from_array(q_est_arr[k])
            # ESKF-consistent error: q_err = q_true^{-1} ⊗ q_est
            q_err = q_t.conjugate().multiply(q_e).normalize()
            mu = np.clip(q_err.mu, -1.0, 1.0)
            ang = 2.0 * np.arccos(mu)
            if ang > np.pi:
                ang -= 2.0 * np.pi
            angle_err[k] = abs(ang)
        return angle_err

    # ---------- plots you care about ----------

    def plot_euler_comparison(self, sim_run_id: int, est_run_id: int, fname: str | None = None):
        """
        Compare true vs estimated Euler angles (roll, pitch, yaw) over time, in degrees.
        """
        sim_result = self.db.load_run(sim_run_id)
        est_result = self._load_estimation_run(est_run_id)

        t_true = sim_result.t
        t_est  = est_result.t
        if len(t_true) != len(t_est) or not np.allclose(t_true, t_est):
            raise ValueError("Time grids of simulation and estimation differ; interpolate first.")

        q_true = sim_result.q_true
        q_est  = est_result.q_est

        e_true = self._quat_seq_to_euler(q_true) * 180.0 / np.pi
        e_est  = self._quat_seq_to_euler(q_est) * 180.0 / np.pi

        labels = ["Roll [deg]", "Pitch [deg]", "Yaw [deg]"]

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(5.0, 4.5))
        for i in range(3):
            ax = axes[i]
            ax.plot(t_true, e_true[:, i], label="True")
            ax.plot(t_true, e_est[:, i], "--", label="Estimate")
            ax.set_ylabel(labels[i])
            ax.grid(True)
            if i == 0:
                ax.legend(loc="upper right")

        axes[-1].set_xlabel("Time [s]")
        fig.suptitle("Euler angles: true vs estimate", y=0.98)

        plt.tight_layout()
        if fname is not None:
            fig.savefig(fname + ".pdf", bbox_inches="tight")
            fig.savefig(fname + ".png", dpi=400, bbox_inches="tight")
        plt.show()

    def plot_attitude_error(self, sim_run_id: int, est_run_id: int, fname: str | None = None):
        """
        Plot total attitude error angle over time, in degrees, with RMSE line.
        """
        sim_result = self.db.load_run(sim_run_id)
        est_result = self._load_estimation_run(est_run_id)

        t_true = sim_result.t
        t_est = est_result.t
        if len(t_true) != len(t_est) or not np.allclose(t_true, t_est):
            raise ValueError("Time grids of simulation and estimation differ; interpolate first.")

        q_true = sim_result.q_true
        q_est  = est_result.q_est

        angle_err_rad = self._quat_angle_error(q_true, q_est)
        angle_err_deg = angle_err_rad * 180.0 / np.pi
        rmse = float(np.sqrt(np.mean(angle_err_deg ** 2)))

        fig, ax = plt.subplots(figsize=(5.0, 2.5))
        ax.plot(t_true, angle_err_deg, label="Attitude error")
        ax.plot(t_true, rmse * np.ones_like(t_true),
                "--", label=f"RMSE = {rmse:.3f} deg")
        ax.grid(True)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Attitude error [deg]")
        ax.set_title("Absolute attitude error")
        # put legend outside to avoid covering the curve
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

        plt.tight_layout()
        if fname is not None:
            fig.savefig(fname + ".pdf", bbox_inches="tight")
            fig.savefig(fname + ".png", dpi=400, bbox_inches="tight")
        plt.show()

        
    def plot_euler_kf_vs_fgo(
        self,
        sim_run_id: int,
        eskf_run_id: int,
        fgo_run_id: int,
        eskf_label: str = "ESKF",
        fgo_label: str = "FGO",
    ):
        """
        Compare true vs ESKF vs FGO Euler angles (roll, pitch, yaw) over time.
        """
        sim_result  = self.db.load_run(sim_run_id)
        eskf_result = self._load_estimation_run(eskf_run_id)
        fgo_result  = self._load_estimation_run(fgo_run_id)

        t_true = sim_result.t
        t_eskf = eskf_result.t
        t_fgo  = fgo_result.t

        # For now require identical time grids
        if (len(t_true) != len(t_eskf)
                or not np.allclose(t_true, t_eskf)
                or len(t_true) != len(t_fgo)
                or not np.allclose(t_true, t_fgo)):
            raise ValueError("Time grids of simulation, ESKF, and FGO differ; interpolate first.")

        q_true = sim_result.q_true   # (N,4)
        q_eskf = eskf_result.q_est   # (N,4)
        q_fgo  = fgo_result.q_est    # (N,4)

        e_true = self._quat_seq_to_euler(q_true)  # (N,3)
        e_eskf = self._quat_seq_to_euler(q_eskf)  # (N,3)
        e_fgo  = self._quat_seq_to_euler(q_fgo)   # (N,3)

        labels = ["Roll [rad]", "Pitch [rad]", "Yaw [rad]"]

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        for i in range(3):
            ax = axes[i]
            ax.plot(t_true, e_true[:, i], label="True", linewidth=1.0)
            ax.plot(t_true, e_eskf[:, i], "--", label=eskf_label, linewidth=1.0)
            ax.plot(t_true, e_fgo[:, i], ":", label=fgo_label, linewidth=1.0)
            ax.set_ylabel(labels[i])
            ax.grid(True)
            if i == 0:
                ax.legend()
        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(f"Euler comparison (sim {sim_run_id}, eskf {eskf_run_id}, fgo {fgo_run_id})")
        plt.tight_layout()
        plt.show()

    def plot_attitude_error_kf_vs_fgo(
        self,
        sim_run_id: int,
        eskf_run_id: int,
        fgo_run_id: int,
        eskf_label: str = "ESKF",
        fgo_label: str = "FGO",
    ):
        """
        Plot attitude error angle over time for ESKF vs FGO.
        Error is total rotation angle between q_true and q_est.
        """
        sim_result  = self.db.load_run(sim_run_id)
        eskf_result = self._load_estimation_run(eskf_run_id)
        fgo_result  = self._load_estimation_run(fgo_run_id)

        t_true = sim_result.t
        t_eskf = eskf_result.t
        t_fgo  = fgo_result.t

        if (len(t_true) != len(t_eskf)
                or not np.allclose(t_true, t_eskf)
                or len(t_true) != len(t_fgo)
                or not np.allclose(t_true, t_fgo)):
            raise ValueError("Time grids of simulation, ESKF, and FGO differ; interpolate first.")

        q_true = sim_result.q_true
        q_eskf = eskf_result.q_est
        q_fgo  = fgo_result.q_est

        def angle_err_series(q_est: np.ndarray) -> np.ndarray:
            N = len(t_true)
            err = np.zeros(N)
            for k in range(N):
                q_t = Quaternion.from_array(q_true[k])
                q_e = Quaternion.from_array(q_est[k])
                # error quaternion q_err = q_e ⊗ q_t^{-1}
                q_err = q_e.multiply(q_t.conjugate()).normalize()
                mu = np.clip(q_err.mu, -1.0, 1.0)
                ang = 2.0 * np.arccos(mu)
                if ang > np.pi:
                    ang -= 2.0 * np.pi
                err[k] = np.abs(ang)
            return err

        err_eskf = angle_err_series(q_eskf)
        err_fgo  = angle_err_series(q_fgo)

        plt.figure(figsize=(8, 4))
        plt.plot(t_true, err_eskf, label=eskf_label)
        plt.plot(t_true, err_fgo, label=fgo_label)
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Attitude error [rad]")
        plt.title(f"Attitude error comparison (sim {sim_run_id})")
        plt.legend()
        plt.tight_layout()
        plt.show()

            
if __name__ == "__main__":
    plotter = AttitudePlotter(db_path="simulations.db")

    sim_run_id = 1
    eskf_run_id = 1  # whatever id you used for the KF
    fgo_run_id  = 3  # your FGO run id
    
    #plotter.plot_angular_velocity_true_vs_meas(sim_run_id)
    
    plotter.plot_euler_comparison(sim_run_id, eskf_run_id, fname="euler_comparison_eskf")
    plotter.plot_attitude_error(sim_run_id, eskf_run_id, fname="attitude_error_eskf")

    #plotter.plot_euler_kf_vs_fgo(sim_run_id, eskf_run_id, fgo_run_id)
    #plotter.plot_attitude_error_kf_vs_fgo(sim_run_id, eskf_run_id, fgo_run_id)

