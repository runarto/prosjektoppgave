import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from data.classes import EstimationResult


class AttitudePlotter:
    def __init__(self, db_path: str = "simulations.db"):
        self.db = SimulationDatabase(db_path)

    def _body_axes_from_quat(self, q_arr: np.ndarray) -> np.ndarray:
        q = Quaternion.from_array(q_arr)
        return q.as_rotmat()

    def _cube_vertices(self, size=0.3):
        """Return 8 cube vertices centered at origin."""
        s = size / 2
        return np.array([
            [-s, -s, -s],
            [-s, -s,  s],
            [-s,  s, -s],
            [-s,  s,  s],
            [ s, -s, -s],
            [ s, -s,  s],
            [ s,  s, -s],
            [ s,  s,  s],
        ])  # shape (8,3)

    def _cube_edges(self):
        """Return list of vertex index pairs defining edges of the cube."""
        return [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
        ]

    def animate_attitude(
        self,
        run_id: int,
        step: int = 10,
        interval: int = 50,
        cube_size: float = 0.3,
        save_path=None,
    ):
        result = self.db.load_run(run_id)

        q_seq = result.q_true[::step]
        t_seq = result.t[::step]

        # Precompute axes
        axes_seq = np.zeros((q_seq.shape[0], 3, 3))
        for i, q_arr in enumerate(q_seq):
            axes_seq[i] = self._body_axes_from_quat(q_arr)

        # Cube static structure
        cube_verts0 = self._cube_vertices(size=cube_size)  # (8,3)
        cube_edges = self._cube_edges()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Attitude (run {run_id})")

        # Body axes lines
        origin = np.zeros(3)
        bx_line, = ax.plot([], [], [], lw=2, color='r')
        by_line, = ax.plot([], [], [], lw=2, color='g')
        bz_line, = ax.plot([], [], [], lw=2, color='b')

        # Cube edges
        cube_lines = []
        for _ in cube_edges:
            line, = ax.plot([], [], [], lw=1.5, color='k')
            cube_lines.append(line)

        time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

        def init():
            for line in cube_lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return [bx_line, by_line, bz_line, *cube_lines, time_text]

        def update(frame_idx):
            R = axes_seq[frame_idx]  # (3,3)

            # Body axes
            ex = R[:, 0]
            ey = R[:, 1]
            ez = R[:, 2]

            bx_line.set_data([0, ex[0]], [0, ex[1]])
            bx_line.set_3d_properties([0, ex[2]])

            by_line.set_data([0, ey[0]], [0, ey[1]])
            by_line.set_3d_properties([0, ey[2]])

            bz_line.set_data([0, ez[0]], [0, ez[1]])
            bz_line.set_3d_properties([0, ez[2]])

            # Rotate cube vertices
            cube_rot = cube_verts0 @ R.T  # (8,3)

            # Update cube edges
            for line, (i, j) in zip(cube_lines, cube_edges):
                p1, p2 = cube_rot[i], cube_rot[j]
                line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                line.set_3d_properties([p1[2], p2[2]])

            time_text.set_text(f"t = {t_seq[frame_idx]:.2f} s")
            return [bx_line, by_line, bz_line, *cube_lines, time_text]

        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=axes_seq.shape[0],
            interval=interval,
            blit=False,
        )

        if save_path:
            anim.save(save_path, dpi=150)
        else:
            plt.show()

    def _load_estimation_run(self, est_run_id: int) -> EstimationResult:
        """
        Load estimation run from DB.

        Expects table est_states(est_run_id, idx, t, jd, q0..q3, bgx..bgz).
        """
        conn = self.db.conn  # adjust if your DB wraps conn differently
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
    def _quat_seq_to_euler(q_seq: np.ndarray) -> np.ndarray:
        """Convert (N,4) quaternion sequence to (N,3) euler (roll, pitch, yaw)."""
        e_list = []
        for q_arr in q_seq:
            q = Quaternion.from_array(q_arr)
            e_list.append(q.as_euler())
        return np.array(e_list)  # (N,3)

    # ---------- new plots ----------

    def plot_euler_comparison(self, sim_run_id: int, est_run_id: int):
        """
        Compare true vs estimated Euler angles (roll, pitch, yaw) over time.
        """
        sim_result = self.db.load_run(sim_run_id)
        est_result = self._load_estimation_run(est_run_id)

        # assume same time grid; otherwise you would interpolate
        t_true = sim_result.t
        t_est = est_result.t
        if len(t_true) != len(t_est) or not np.allclose(t_true, t_est):
            raise ValueError("Time grids of simulation and estimation differ; interpolate first.")

        q_true = sim_result.q_true        # (N,4)
        q_est  = est_result.q_est         # (N,4)

        e_true = self._quat_seq_to_euler(q_true)  # (N,3)
        e_est  = self._quat_seq_to_euler(q_est)   # (N,3)

        labels = ["Roll [rad]", "Pitch [rad]", "Yaw [rad]"]

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        for i in range(3):
            ax = axes[i]
            ax.plot(t_true, e_true[:, i], label="True")
            ax.plot(t_true, e_est[:, i], label="Estimate", linestyle="--")
            ax.set_ylabel(labels[i])
            ax.grid(True)
            if i == 0:
                ax.legend()
        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(f"Euler angle comparison (sim {sim_run_id}, est {est_run_id})")
        plt.tight_layout()
        plt.show()
        
    def plot_angular_velocity_true_vs_meas(self, sim_run_id: int):
        """
        Plot true vs measured angular velocities for each axis in one figure.
        Creates 3 stacked subplots: wx, wy, wz.
        """
        sim_result = self.db.load_run(sim_run_id)

        t = sim_result.t
        w_true = sim_result.omega_true      # shape (N, 3)
        w_meas = sim_result.omega_meas      # shape (N, 3)

        labels = [r"$\omega_x$ [rad/s]", r"$\omega_y$ [rad/s]", r"$\omega_z$ [rad/s]"]

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        for i in range(3):
            ax = axes[i]
            ax.plot(t, w_true[:, i], label="True", linewidth=1.0)
            ax.plot(t, w_meas[:, i], "--", label="Measured", linewidth=1.0)
            ax.set_ylabel(labels[i])
            ax.grid(True)
            if i == 0:
                ax.legend()

        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(f"Angular velocity: true vs measured (sim {sim_run_id})")
        plt.tight_layout()
        plt.show()


    def plot_attitude_error(self, sim_run_id: int, est_run_id: int):
        """
        Plot total attitude error angle (norm of rotation error) over time.
        """
        sim_result = self.db.load_run(sim_run_id)
        est_result = self._load_estimation_run(est_run_id)

        t_true = sim_result.t
        t_est = est_result.t
        if len(t_true) != len(t_est) or not np.allclose(t_true, t_est):
            raise ValueError("Time grids of simulation and estimation differ; interpolate first.")

        q_true = sim_result.q_true
        q_est  = est_result.q_est

        N = len(t_true)
        angle_err = np.zeros(N)
        for k in range(N):
            q_t = Quaternion.from_array(q_true[k])
            q_e = Quaternion.from_array(q_est[k])

            # rotation error: q_err = q_e ⊗ q_t^{-1}
            q_err = q_e.multiply(q_t.conjugate()).normalize()
            # clamp for numerical safety
            mu = np.clip(q_err.mu, -1.0, 1.0)
            angle = 2.0 * np.arccos(mu)   # total rotation angle
            if angle > np.pi:
                angle -= 2.0 * np.pi
            angle_err[k] = np.abs(angle)

        plt.figure(figsize=(8, 4))
        plt.plot(t_true, angle_err)
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Attitude error [rad]")
        plt.title(f"Attitude error angle (sim {sim_run_id}, est {est_run_id})")
        plt.tight_layout()
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

    sim_run_id = 2
    eskf_run_id = 2  # whatever id you used for the KF
    fgo_run_id  = 3  # your FGO run id
    
    plotter.plot_angular_velocity_true_vs_meas(sim_run_id)
    
    plotter.plot_attitude_error(sim_run_id, eskf_run_id)
    plotter.plot_euler_comparison(sim_run_id, eskf_run_id)

    #plotter.plot_euler_kf_vs_fgo(sim_run_id, eskf_run_id, fgo_run_id)
    #plotter.plot_attitude_error_kf_vs_fgo(sim_run_id, eskf_run_id, fgo_run_id)

