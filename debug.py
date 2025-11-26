import numpy as np
import matplotlib.pyplot as plt

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from data.classes import EstimationResult


class AttitudeDebugger:
    def __init__(self, db_path: str = "simulations.db"):
        self.db = SimulationDatabase(db_path)

    # ---------- helpers ----------

    def _load_estimation_run(self, est_run_id: int) -> EstimationResult:
        """
        Load estimation run from DB.

        Expects table est_states(est_run_id, idx, t, jd, q0..q3, bgx..bgz).
        """
        conn = self.db.conn
        cur = conn.cursor()
        cur.execute(
            """
            SELECT t, jd, q0, q1, q2, q3, bgx, bgy, bgz
            FROM est_states
            WHERE est_run_id = ?
            ORDER BY idx ASC;
            """,
            (est_run_id,),
        )
        rows = cur.fetchall()
        if not rows:
            raise ValueError(f"No est_states found for est_run_id={est_run_id}")

        t = np.array([r[0] for r in rows], float)
        jd = np.array([r[1] for r in rows], float)
        q_est = np.array([r[2:6] for r in rows], float)   # (N,4)
        bg_est = np.array([r[6:9] for r in rows], float)  # (N,3)

        return EstimationResult(t=t, jd=jd, q_est=q_est, bg_est=bg_est)

    @staticmethod
    def _enforce_quat_continuity(q_seq: np.ndarray) -> np.ndarray:
        """
        Ensure quaternion sequence does not jump between q and -q.

        Modifies a copy of q_seq and returns it.
        """
        q = q_seq.copy()
        for k in range(1, len(q)):
            if np.dot(q[k], q[k - 1]) < 0.0:
                q[k] = -q[k]
        return q

    @staticmethod
    def _quat_error_angle(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
        """
        Compute total rotation error angle between two quaternion sequences.

        q_true, q_est: (N,4), unit quaternions.
        Returns angle_err: (N,) in radians.
        """
        assert q_true.shape == q_est.shape
        N = q_true.shape[0]
        angle_err = np.zeros(N)

        for k in range(N):
            qt = Quaternion.from_array(q_true[k])
            qe = Quaternion.from_array(q_est[k])

            # error q_err = q_e * q_t^{-1}
            q_err = qe.multiply(qt.conjugate()).normalize()
            mu = np.clip(q_err.mu, -1.0, 1.0)
            angle = 2.0 * np.arccos(mu)

            # map to [0, pi]
            if angle > np.pi:
                angle -= 2.0 * np.pi
            angle_err[k] = abs(angle)

        return angle_err

    # ---------- public debug API ----------

    def compute_attitude_error(self, sim_run_id: int, est_run_id: int):
        """
        Return (t, angle_err) for convenience, after enforcing quaternion continuity.
        """
        sim_result = self.db.load_run(sim_run_id)
        est_result = self._load_estimation_run(est_run_id)

        t_true = sim_result.t
        t_est = est_result.t
        if len(t_true) != len(t_est) or not np.allclose(t_true, t_est):
            raise ValueError("Time grids of simulation and estimation differ; interpolate first.")

        q_true = self._enforce_quat_continuity(sim_result.q_true)
        q_est = self._enforce_quat_continuity(est_result.q_est)

        angle_err = self._quat_error_angle(q_true, q_est)
        return t_true, angle_err

    def debug_summary(self, sim_run_id: int, est_run_id: int):
        """
        Print a short numeric summary with key statistics.
        """
        sim_result = self.db.load_run(sim_run_id)
        est_result = self._load_estimation_run(est_run_id)

        # time consistency
        t_true = sim_result.t
        t_est = est_result.t
        if len(t_true) != len(t_est) or not np.allclose(t_true, t_est):
            raise ValueError("Time grids of simulation and estimation differ; interpolate first.")

        # quaternion continuity and norms
        q_true_raw = sim_result.q_true
        q_est_raw = est_result.q_est

        q_true = self._enforce_quat_continuity(q_true_raw)
        q_est = self._enforce_quat_continuity(q_est_raw)

        norm_true = np.linalg.norm(q_true_raw, axis=1)
        norm_est = np.linalg.norm(q_est_raw, axis=1)

        max_dev_true = np.max(np.abs(norm_true - 1.0))
        max_dev_est = np.max(np.abs(norm_est - 1.0))

        # sign flips count (before continuity enforcement)
        flips_true = np.sum(np.einsum("ij,ij->i", q_true_raw[1:], q_true_raw[:-1]) < 0.0)
        flips_est = np.sum(np.einsum("ij,ij->i", q_est_raw[1:], q_est_raw[:-1]) < 0.0)

        # attitude error
        _, angle_err = self.compute_attitude_error(sim_run_id, est_run_id)
        angle_deg = np.degrees(angle_err)

        rms_err = np.sqrt(np.mean(angle_err ** 2))
        rms_deg = np.degrees(rms_err)
        max_err_deg = np.max(angle_deg)
        med_err_deg = np.median(angle_deg)
        p95_err_deg = np.percentile(angle_deg, 95)

        print(f"=== Attitude debug summary (sim {sim_run_id}, est {est_run_id}) ===")
        print(f"Samples: {len(t_true)}")
        print("")
        print("Quaternion norms:")
        print(f"  max |‖q_true‖ - 1| = {max_dev_true:.3e}")
        print(f"  max |‖q_est ‖ - 1| = {max_dev_est:.3e}")
        print("")
        print("Sign flips (q vs previous sample):")
        print(f"  true: {flips_true}")
        print(f"  est : {flips_est}")
        print("")
        print("Attitude error (q-space):")
        print(f"  RMS      = {rms_deg:.3f} deg")
        print(f"  median   = {med_err_deg:.3f} deg")
        print(f"  95th pct = {p95_err_deg:.3f} deg")
        print(f"  max      = {max_err_deg:.3f} deg")

        # optional: gyro bias, if available
        if hasattr(sim_result, "bg_true"):
            bg_true = sim_result.bg_true      # (N,3)
            bg_est = est_result.bg_est        # (N,3)
            bg_err = bg_est - bg_true
            bg_rms = np.sqrt(np.mean(bg_err ** 2, axis=0))
            print("")
            print("Gyro bias RMS error [rad/s]:")
            print(f"  x: {bg_rms[0]:.3e}, y: {bg_rms[1]:.3e}, z: {bg_rms[2]:.3e}")

    def plot_attitude_error(self, sim_run_id: int, est_run_id: int):
        """
        Plot total attitude error angle over time.
        """
        t, angle_err = self.compute_attitude_error(sim_run_id, est_run_id)

        plt.figure(figsize=(8, 4))
        plt.plot(t, angle_err)
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Attitude error [rad]")
        plt.title(f"Attitude error angle (sim {sim_run_id}, est {est_run_id})")
        plt.tight_layout()
        plt.show()

    def plot_quaternion_norms(self, sim_run_id: int, est_run_id: int):
        """
        Plot quaternion norms for true and estimated attitude.
        """
        sim_result = self.db.load_run(sim_run_id)
        est_result = self._load_estimation_run(est_run_id)

        t_true = sim_result.t
        t_est = est_result.t
        if len(t_true) != len(t_est) or not np.allclose(t_true, t_est):
            raise ValueError("Time grids of simulation and estimation differ; interpolate first.")

        norm_true = np.linalg.norm(sim_result.q_true, axis=1)
        norm_est = np.linalg.norm(est_result.q_est, axis=1)

        plt.figure(figsize=(8, 4))
        plt.plot(t_true, norm_true, label="‖q_true‖")
        plt.plot(t_true, norm_est, label="‖q_est‖", linestyle="--")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Norm")
        plt.title(f"Quaternion norms (sim {sim_run_id}, est {est_run_id})")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    dbg = AttitudeDebugger(db_path="simulations.db")
    dbg.debug_summary(sim_run_id=1, est_run_id=2)
    dbg.plot_attitude_error(sim_run_id=1, est_run_id=2)
    dbg.plot_quaternion_norms(sim_run_id=1, est_run_id=2)
