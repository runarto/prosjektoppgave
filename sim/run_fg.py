from sim.fg import FGO
from environment.environment import OrbitEnvironmentModel
from data.db import SimulationDatabase
from logging_config import get_logger
from tqdm import tqdm
import numpy as np

logger = get_logger(__name__)


class FGO_runner:
    
    def __init__(self, env: OrbitEnvironmentModel, fgo: FGO, db_path: str = "simulations.db"):
        self.env = env
        self.fgo = fgo
        self.db = SimulationDatabase(db_path)
    
    
    def run_on_simulation(
        self,
        est_run_id: int,
        sim_run_id: int,
        env: OrbitEnvironmentModel,
        db_path: str = "simulations.db",
        est_name: str = "fgo",
    ) -> int:
        """Sliding-window factor graph on stored simulation."""

        db = SimulationDatabase(db_path)
        result = db.load_run(sim_run_id)
        estimates = db.load_estimated_states(est_run_id)

        t = result.t
        jd = result.jd
        omega_meas_log = result.omega_meas
        mag_meas_log = result.mag_meas
        sun_meas_log = result.sun_meas
        st_meas_log = result.st_meas
        q_estimated = estimates.q_est
        q_true = result.q_true

        N = t.shape[0]
        window_size = 200  # tune for memory/speed

        # full estimated trajectory storage
        est_states: list[EskfState | None] = [None] * N

        # ----- initialize with first state from ESKF -----
        x0 = self.fgo._make_initial_state(q_estimated[0])

        self.states = []
        self.factors = []
        idx0 = self.fgo.add_state(x0)
        assert idx0 == 0

        # prior on first state
        self.fgo.add_prior(
            state_idx=0,
            prior=x0.nom,
            cov=np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6]),
        )

        start_k = 0  # global index of self.states[0]

        pbar = tqdm(range(1, N), desc="FGO Sliding Window", unit="step")
        for k in pbar:
            # 1) extend graph with new time step
            self.fgo._add_step_state_and_factors(
                k=k,
                t=t,
                jd=jd,
                q_estimated=q_estimated,
                omega_meas_log=omega_meas_log,
                mag_meas_log=mag_meas_log,
                sun_meas_log=sun_meas_log,
                st_meas_log=st_meas_log,
                env=env,
            )

            # 2) check window condition
            window_full = len(self.states) >= window_size
            last_step = (k == N - 1)

            if not (window_full or last_step):
                continue

            # 3) pre-optimization attitude error in this window
            init_errors_deg = self.fgo._attitude_errors_deg_for_states(
                states=self.states,
                start_k=start_k,
                q_true=q_true,
            )

            # 4) optimize current window
            optimized_states = self.fgo.optimize()

            # 5) post-optimization attitude error
            window_errors_deg = self.fgo._attitude_errors_deg_for_states(
                states=list(optimized_states),
                start_k=start_k,
                q_true=q_true,
            )

            if window_errors_deg:
                mean_err = float(np.mean(window_errors_deg))
                max_err = float(np.max(window_errors_deg))
                pbar.set_postfix({
                    "mean_err": f"{mean_err:.3f}",
                    "max_err": f"{max_err:.3f}",
                    "start": start_k,
                })

            # 6) store optimized states globally (except last, which is carried)
            self.fgo._commit_window_to_global(
                optimized_states=optimized_states,
                start_k=start_k,
                est_states=est_states,
            )

            # 7) carry last state as prior for next window
            last_state = optimized_states[-1]
            self.states = [last_state]
            self.factors = []
            start_k = k  # new window starts at global index k

            # fresh prior on carried state
            self.fgo.add_prior(
                state_idx=0,
                prior=last_state.nom,
                cov=np.diag([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8]),
            )

        # ensure final entry is filled
        if est_states[-1] is None:
            est_states[-1] = self.states[0]

        # type ignore: we know all entries are filled now
        filled_states: list[EskfState] = [s for s in est_states if s is not None]  # type: ignore

        est_run_id_out = db.insert_estimation_run(
            sim_run_id=sim_run_id,
            t=t,
            jd=jd,
            states=filled_states,
            name=est_name,
        )
        print(f"FGO estimation run stored with est_run_id={est_run_id_out}")
        return est_run_id_out
    