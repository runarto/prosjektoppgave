from data.db import SimulationDatabase
from sim.gtsam_fg import GtsamFGO, WindowSample, SlidingWindow
from utilities.states import EskfState, NominalState
from utilities.quaternion import Quaternion
from environment.environment import OrbitEnvironmentModel
from utilities.gaussian import MultiVarGauss
from sim.temp import error_angle_deg

import numpy as np

def run_on_simulation(db: SimulationDatabase, sim_run_id: int, est_name: str = "gtsam_fgo") -> int:
    # ----- load data -----
    
    window = SlidingWindow(max_len=300)
    fgo = GtsamFGO()
    env = OrbitEnvironmentModel()
    sim = db.load_run(sim_run_id)
    t = sim.t
    jd = sim.jd
    omega_meas_log = sim.omega_meas
    mag_meas_log   = sim.mag_meas
    sun_meas_log   = sim.sun_meas
    st_meas_log    = sim.st_meas

    N = t.shape[0]

    # global output
    est_states: list[NominalState] = []

    # initial nominal (could also take true or KF)
    q0 = Quaternion.from_array(sim.q_true[0])
    b0 = np.zeros(3)
    x_nom_prev = NominalState(ori=q0, gyro_bias=b0)

    # loop over all time steps
    for k in range(N):
        if k == 0:
            dt = 0.02
        else:
            dt = t[k] - t[k-1]

        omega_k = omega_meas_log[k]
        if np.any(np.isnan(omega_k)):
            omega_k = omega_meas_log[k-1]

        # simple nominal propagation using gyro only
        q_pred = x_nom_prev.ori.propagate(omega_k - x_nom_prev.gyro_bias, dt)
        x_nom = NominalState(ori=q_pred, gyro_bias=x_nom_prev.gyro_bias)

        # build WindowSample for this step
        z_mag = None if np.any(np.isnan(mag_meas_log[k])) else mag_meas_log[k]
        z_sun = None if np.any(np.isnan(sun_meas_log[k])) else sun_meas_log[k]
        z_st  = None
        if not np.any(np.isnan(st_meas_log[k])):
            z_st = Quaternion.from_array(st_meas_log[k])

        sample = WindowSample(
            t=float(t[k]),
            jd=float(jd[k]),
            x_nom=x_nom,
            omega_meas=omega_k,
            z_mag=z_mag,
            z_sun=z_sun,
            z_st=z_st,
        )

        window.add(sample)
        x_nom_prev = x_nom  # for next step

        # when window is full, run smoothing
        if window.ready:

            window_samples = list(window.samples)
            window_states = fgo.optimize_window(window_samples, env)
            
            # Compare error to the true state within the window
            for i, sm_state in enumerate(window_states):
                global_k = k - len(window_states) + 1 + i
                q_true = Quaternion.from_array(sim.q_true[global_k])
                err_angle = error_angle_deg(q_true, sm_state.ori)
                print(f"  Step {global_k}/{N-1}: attitude error = {err_angle:.3f} deg")
                est_states.append(sm_state)
                print(f"  Stored estimated states: {len(est_states)}")
            est_states.pop()

            # keep the last smoothed sample as the nominal for next iteration
            last_smoothed = window_states[-1]
            x_nom_prev = last_smoothed

            # rebuild the window with the *times* but updated nominal state at the end
            # simplest: keep the last sample and clear the rest
            last_sample = window_samples[-1]
            window.samples.clear()
            window.add(
                WindowSample(
                    t=last_sample.t,
                    jd=last_sample.jd,
                    x_nom=x_nom_prev,
                    omega_meas=last_sample.omega_meas,
                    z_mag=last_sample.z_mag,
                    z_sun=last_sample.z_sun,
                    z_st=last_sample.z_st,
                )
            )
            
            print(f"Estimated states stored so far: {len(est_states)}")
            

    # handle the very last state (if not already added)
    if len(est_states) < N:
        # last nominal comes from x_nom_prev
        err = MultiVarGauss(mean=np.zeros(6), cov=fgo.process.Q_c.copy())
        est_states.append(EskfState(nom=x_nom_prev, err=err))
        
    q_estimated = [state.nom.ori for state in est_states]
    q_true = [sim.q_true[k] for k in range(len(est_states))]
    for k in range(len(est_states)):
        err_angle = error_angle_deg(
            Quaternion.from_array(q_true[k]),
            q_estimated[k],
        )
        print(f"Step {k}/{N-1}: attitude error = {err_angle:.3f} deg")


db = SimulationDatabase("simulations.db")
sim_run_id = 1  # change as needed
est_run_id = run_on_simulation(db, sim_run_id, est_name="gtsam_fgo")
print(f"Estimation run stored with ID {est_run_id}")