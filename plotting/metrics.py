import numpy as np
import matplotlib.pyplot as plt

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from logging_config import get_logger

import seaborn as sns
sns.set_theme(context="notebook", style="whitegrid", palette="deep", font_scale=1.1)


logger = get_logger(__name__)

class Metrics:
    """
    Convenience class for computing and plotting NEES and attitude metrics.
    """

    def __init__(self, db_path: str = "simulations.db"):
        self.db = SimulationDatabase(db_path)
        est = self.db.load_estimated_states(1)
        print("Loaded P[0]:\n", est.P_est[0])

    # ----------------- internal helpers -----------------

    @staticmethod
    def _check_time_alignment(t_true: np.ndarray, t_est: np.ndarray) -> None:
        if len(t_true) != len(t_est) or not np.allclose(t_true, t_est):
            raise ValueError("Time grids of simulation and estimation differ; interpolate first.")

    @staticmethod
    def _compute_error_state_series(sim_result, est_result) -> np.ndarray:
        """
        Return error state e_k in R^6 for each k:

            e_k = [δθ_k; δb_g_k]

        where δθ_k is the small-angle orientation error and δb_g_k is
        gyro-bias error (estimate minus true).
        """
        t_true = sim_result.t
        t_est = est_result.t
        Metrics._check_time_alignment(t_true, t_est)

        q_true = sim_result.q_true      # (N,4)
        bg_true = sim_result.b_g_true   # (N,3)
        q_est = est_result.q_est        # (N,4)
        bg_est = est_result.bg_est      # (N,3)

        N = len(t_true)
        e = np.zeros((N, 6))

        for k in range(N):
            q_t = Quaternion.from_array(q_true[k])
            q_e = Quaternion.from_array(q_est[k])

            # orientation error in error-state coordinates
            # consistent with NominalState.diff: self.ori.diff_as_avec(other.ori)
            delta_theta = q_e.diff_as_avec(q_t)  # R^3

            delta_bg = bg_est[k] - bg_true[k]    # R^3

            e[k, 0:3] = delta_theta
            e[k, 3:6] = delta_bg

        return e  # shape (N,6)

    # ----------------- NEES / ANEES -----------------

    def compute_nees(self, sim_run_id: int, est_run_id: int):
        """
        Compute NEES_k and ANEES for a given simulation + estimation run.

        Returns:
            t      : (N,) time vector
            nees   : (N,) NEES values per step
            anees  : float, average NEES over all valid steps
        """
        sim_result = self.db.load_run(sim_run_id)
        est_result = self.db.load_estimated_states(est_run_id)

        t = sim_result.t
        e = self._compute_error_state_series(sim_result, est_result)
        P = est_result.P_est               # (N,6,6)
        
        # print("max ||e_k||:", np.max(np.linalg.norm(e, axis=1)))

        # # 2) Typical covariance scale from first few steps
        # for k in range(0, min(5, len(P))):
        #     print(f"k={k}")
        #     print("diag(P_k):", np.diag(P[k]))
        #     print("eig(P_k):", np.linalg.eigvals(P[k]))

        N, n = e.shape
        assert n == 6, "Expected 6D error state for NEES."

        nees = np.full(N, np.nan)
        for k in range(N):
            ek = e[k]
            Pk = P[k]

            # guard against singular or zero covariance
            if not np.all(np.isfinite(Pk)):
                continue
            try:
                nees[k] = float(ek.T @ np.linalg.solve(Pk, ek))
            except np.linalg.LinAlgError:
                # singular covariance; leave NEES as NaN
                continue

        valid = np.isfinite(nees)
        anees = float(np.mean(nees[valid])) if np.any(valid) else np.nan

        return t, nees, anees

    def plot_nees(self, sim_run_id: int, est_run_id: int, title_suffix: str = ""):
        """
        Plot NEES over time, along with the average NEES (ANEES) as a horizontal line.
        """
        t, nees, anees = self.compute_nees(sim_run_id, est_run_id)
        n = 6  # state dimension

        plt.figure(figsize=(8, 4))
        plt.plot(t, nees, label="NEES")
        if np.isfinite(anees):
            plt.axhline(anees, color="C1", linestyle="--", label=f"ANEES = {anees:.2f}")
        # nominal expectation line
        plt.axhline(n, color="C2", linestyle=":", label=f"E[NEES] = {n}")

        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("NEES")
        base_title = f"NEES (sim {sim_run_id}, est {est_run_id})"
        if title_suffix:
            base_title += f" – {title_suffix}"
        plt.title(base_title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ----------------- Attitude error metrics -----------------

    def compute_attitude_error(self, sim_run_id: int, est_run_id: int, in_degrees: bool = True):
        """
        Compute per-step attitude error angle and its RMSE.

        Returns:
            t           : (N,) time vector
            ang_err     : (N,) attitude error angle (rad or deg)
            rmse        : float, root-mean-square angle error
        """
        sim_result = self.db.load_run(sim_run_id)
        est_result = self.db.load_estimated_states(est_run_id)

        t_true = sim_result.t
        t_est = est_result.t
        self._check_time_alignment(t_true, t_est)

        q_true = sim_result.q_true
        q_est = est_result.q_est

        N = len(t_true)
        ang_err = np.zeros(N)

        for k in range(N):
            q_t = Quaternion.from_array(q_true[k])
            q_e = Quaternion.from_array(q_est[k])

            # rotation error q_err = q_e ⊗ q_t^{-1}
            q_err = q_e.multiply(q_t.conjugate()).normalize()
            mu = np.clip(q_err.mu, -1.0, 1.0)
            angle = 2.0 * np.arccos(mu)  # [rad]
            if angle > np.pi:
                angle -= 2.0 * np.pi
            ang_err[k] = np.abs(angle)

        if in_degrees:
            ang_err = np.degrees(ang_err)

        rmse = float(np.sqrt(np.mean(ang_err ** 2)))
        return t_true, ang_err, rmse

    def plot_attitude_error(self, sim_run_id: int, est_run_id: int, in_degrees: bool = True):
        """
        Plot attitude error norm over time (rad or deg) and show RMSE in the title.
        """
        t, ang_err, rmse = self.compute_attitude_error(
            sim_run_id=sim_run_id,
            est_run_id=est_run_id,
            in_degrees=in_degrees,
        )

        unit = "deg" if in_degrees else "rad"
        plt.figure(figsize=(8, 4))
        plt.plot(t, ang_err)
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel(f"Attitude error [{unit}]")
        plt.title(
            f"Attitude error (sim {sim_run_id}, est {est_run_id}) – "
            f"RMSE = {rmse:.3f} {unit}"
        )
        plt.tight_layout()
        plt.show()
        
        
    def plot_covariance_diagonals(
        self,
        sim_run_id: int,
        est_run_id: int,
        log_scale: bool = False,
    ):
        """
        Plot diagonal elements of the 6x6 covariance P_k over time.

        State ordering assumed: [δθ_x, δθ_y, δθ_z, δb_gx, δb_gy, δb_gz].
        """
        sim_result = self.db.load_run(sim_run_id)
        est_result = self.db.load_estimated_states(est_run_id)

        t_true = sim_result.t
        t_est = est_result.t
        self._check_time_alignment(t_true, t_est)

        P = est_result.P_est  # shape (N, 6, 6)
        N, n, _ = P.shape
        assert n == 6, "Expected 6x6 covariance for error state."

        # Extract diagonals: shape (N, 6)
        diagP = np.zeros((N, 6))
        for k in range(N):
            diagP[k, :] = np.diag(P[k])

        labels = [
            r"$P_{\delta\theta_x\delta\theta_x}$",
            r"$P_{\delta\theta_y\delta\theta_y}$",
            r"$P_{\delta\theta_z\delta\theta_z}$",
            r"$P_{b_{gx}b_{gx}}$",
            r"$P_{b_{gy}b_{gy}}$",
            r"$P_{b_{gz}b_{gz}}$",
        ]

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        # Attitude covariance
        for i in range(3):
            axes[0].plot(t_true, diagP[:, i], label=labels[i])
        axes[0].set_ylabel("Attitude cov diag")
        axes[0].grid(True)
        axes[0].legend()

        # Bias covariance
        for i in range(3, 6):
            axes[1].plot(t_true, diagP[:, i], label=labels[i])
        axes[1].set_ylabel("Gyro bias cov diag")
        axes[1].set_xlabel("Time [s]")
        axes[1].grid(True)
        axes[1].legend()

        if log_scale:
            axes[0].set_yscale("log")
            axes[1].set_yscale("log")

        fig.suptitle(f"Covariance diagonals (sim {sim_run_id}, est {est_run_id})")
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    metrics = Metrics(db_path="simulations.db")

    sim_run_id = 1
    est_run_id = 3  # example

    # NEES / ANEES
    metrics.plot_nees(sim_run_id, est_run_id)

    # Attitude error
    metrics.plot_attitude_error(sim_run_id, est_run_id, in_degrees=True)
    # Covariance diagonals
    metrics.plot_covariance_diagonals(sim_run_id, est_run_id, log_scale=True)
