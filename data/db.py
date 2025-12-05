import sqlite3
import json
import numpy as np
from typing import Dict
from dataclasses import asdict
from data.classes import SimulationConfig, SimulationResult, EstimationResult
from logging_config import get_logger

logger = get_logger(__name__)


class SimulationDatabase:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()

        # --- existing tables: runs, samples ---
        cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            config_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            idx INTEGER NOT NULL,
            t REAL NOT NULL,
            jd REAL NOT NULL,

            q0 REAL, q1 REAL, q2 REAL, q3 REAL,
            bgx REAL, bgy REAL, bgz REAL,
            wx_true REAL, wy_true REAL, wz_true REAL,
            wx_meas REAL, wy_meas REAL, wz_meas REAL,
            bx_meas REAL, by_meas REAL, bz_meas REAL,
            sx_meas REAL, sy_meas REAL, sz_meas REAL,
            st_q0 REAL, st_q1 REAL, st_q2 REAL, st_q3 REAL,
            bx_eci REAL, by_eci REAL, bz_eci REAL,
            sx_eci REAL, sy_eci REAL, sz_eci REAL,

            FOREIGN KEY(run_id) REFERENCES runs(id)
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_samples_run ON samples(run_id, idx);")

        # Migrate existing schema if needed (add b_eci and s_eci columns)
        self._migrate_schema_add_eci_vectors(cur)

        # --- NEW: estimation runs ---
        cur.execute("""
        CREATE TABLE IF NOT EXISTS est_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sim_run_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(sim_run_id) REFERENCES runs(id)
        );
        """)

        # --- NEW: estimation states ---
        cur.execute("""
        CREATE TABLE IF NOT EXISTS est_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            est_run_id INTEGER NOT NULL,
            idx INTEGER NOT NULL,
            t REAL NOT NULL,
            jd REAL NOT NULL,

            q0 REAL, q1 REAL, q2 REAL, q3 REAL,
            bgx REAL, bgy REAL, bgz REAL,

            -- 6x6 covariance, row-major flattened: P[row][col] -> P_rc
            P00 REAL, P01 REAL, P02 REAL, P03 REAL, P04 REAL, P05 REAL,
            P10 REAL, P11 REAL, P12 REAL, P13 REAL, P14 REAL, P15 REAL,
            P20 REAL, P21 REAL, P22 REAL, P23 REAL, P24 REAL, P25 REAL,
            P30 REAL, P31 REAL, P32 REAL, P33 REAL, P34 REAL, P35 REAL,
            P40 REAL, P41 REAL, P42 REAL, P43 REAL, P44 REAL, P45 REAL,
            P50 REAL, P51 REAL, P52 REAL, P53 REAL, P54 REAL, P55 REAL,

            FOREIGN KEY(est_run_id) REFERENCES est_runs(id)
        );
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_est_states_run ON est_states(est_run_id, idx);")

        self.conn.commit()

    def _migrate_schema_add_eci_vectors(self, cur):
        """Migrate existing schema to add b_eci and s_eci columns if they don't exist."""
        # Check if columns exist
        cur.execute("PRAGMA table_info(samples);")
        columns = [row[1] for row in cur.fetchall()]

        eci_columns = ['bx_eci', 'by_eci', 'bz_eci', 'sx_eci', 'sy_eci', 'sz_eci']

        for col in eci_columns:
            if col not in columns:
                logger.info(f"Adding column {col} to samples table")
                cur.execute(f"ALTER TABLE samples ADD COLUMN {col} REAL;")

        self.conn.commit()

    def insert_run(self, result: SimulationResult) -> int:
        """Insert a new run and all its samples; return run_id."""
        cur = self.conn.cursor()

        cfg_dict: Dict = asdict(result.config)
        cur.execute(
            "INSERT INTO runs (name, config_json) VALUES (?, ?);",
            (result.config.run_name, json.dumps(cfg_dict))
        )
        run_id = cur.lastrowid

        # Bulk insert samples
        rows = []
        N = result.t.shape[0]
        for k in range(N):
            q = result.q_true[k]
            bg = result.b_g_true[k]
            w_true = result.omega_true[k]
            w_meas = result.omega_meas[k]
            b_meas = result.mag_meas[k]
            s_meas = result.sun_meas[k]
            st = result.st_meas[k]
            b_eci = result.b_eci[k]
            s_eci = result.s_eci[k]

            rows.append((
                run_id, k, float(result.t[k]), float(result.jd[k]),
                float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                float(bg[0]), float(bg[1]), float(bg[2]),
                float(w_true[0]), float(w_true[1]), float(w_true[2]),
                float(w_meas[0]), float(w_meas[1]), float(w_meas[2]),
                float(b_meas[0]), float(b_meas[1]), float(b_meas[2]),
                float(s_meas[0]), float(s_meas[1]), float(s_meas[2]),
                float(st[0]), float(st[1]), float(st[2]), float(st[3]),
                float(b_eci[0]), float(b_eci[1]), float(b_eci[2]),
                float(s_eci[0]), float(s_eci[1]), float(s_eci[2]),
            ))

        cur.executemany("""
            INSERT INTO samples (
                run_id, idx, t, jd,
                q0, q1, q2, q3,
                bgx, bgy, bgz,
                wx_true, wy_true, wz_true,
                wx_meas, wy_meas, wz_meas,
                bx_meas, by_meas, bz_meas,
                sx_meas, sy_meas, sz_meas,
                st_q0, st_q1, st_q2, st_q3,
                bx_eci, by_eci, bz_eci,
                sx_eci, sy_eci, sz_eci
            ) VALUES (?, ?, ?, ?,
                      ?, ?, ?, ?,
                      ?, ?, ?,
                      ?, ?, ?,
                      ?, ?, ?,
                      ?, ?, ?,
                      ?, ?, ?,
                      ?, ?, ?, ?,
                      ?, ?, ?,
                      ?, ?, ?);
        """, rows)

        self.conn.commit()
        return run_id
    
    def insert_eskf_run(self,
                              sim_run_id: int,
                              t: np.ndarray,
                              jd: np.ndarray,
                              states: list,
                              name: str = "eskf") -> int:
        """
        Store ESKF results for one simulation run.

        Args:
            sim_run_id:  id in 'runs' table this estimation corresponds to
            t, jd:       time vectors (same length as states)
            states:      list of EskfState
            name:        label for this estimation run (e.g. 'eskf_baseline')

        Returns:
            est_run_id: primary key in est_runs.
        """
        cur = self.conn.cursor()

        # 1) create est_run row
        cur.execute(
            "INSERT INTO est_runs (sim_run_id, name) VALUES (?, ?);",
            (sim_run_id, name)
        )
        est_run_id = cur.lastrowid

        # 2) bulk insert est_states
        N = len(states)
        assert N == len(t) == len(jd)

        rows = []
        for k in range(N):
            x_est = states[k]
            q = x_est.nom.ori.as_array()
            bg = np.asarray(x_est.nom.gyro_bias, float).reshape(3)

            # 6x6 covariance -> flattened row-major length-36
            P = x_est.err.cov
            P_flat = P.reshape(36)

            rows.append((
                est_run_id,
                k,
                float(t[k]),
                float(jd[k]),
                float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                float(bg[0]), float(bg[1]), float(bg[2]),
                *[float(v) for v in P_flat],
            ))

        cur.executemany("""
            INSERT INTO est_states (
                est_run_id, idx, t, jd,
                q0, q1, q2, q3,
                bgx, bgy, bgz,
                P00, P01, P02, P03, P04, P05,
                P10, P11, P12, P13, P14, P15,
                P20, P21, P22, P23, P24, P25,
                P30, P31, P32, P33, P34, P35,
                P40, P41, P42, P43, P44, P45,
                P50, P51, P52, P53, P54, P55
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            );
        """, rows)

        self.conn.commit()
        return est_run_id
    
    def insert_fgo_run(self,
                              sim_run_id: int,
                              t: np.ndarray,
                              jd: np.ndarray,
                              states: list,
                              name: str = "fgo") -> int:
        """
        Store FGO results for one simulation run.
        Args:
            sim_run_id:  id in 'runs' table this estimation corresponds to
            t, jd:       time vectors (same length as states)
            states:      list of NominalState
            name:        label for this estimation run (e.g. 'fgo_baseline')
        Returns:
            est_run_id: primary key in est_runs.
        """
        
        cur = self.conn.cursor()    
        # 1) create est_run row
        cur.execute(
            "INSERT INTO est_runs (sim_run_id, name) VALUES (?, ?);",
            (sim_run_id, name)
        )
        est_run_id = cur.lastrowid
        # 2) bulk insert est_states
        N = len(states)
        assert N == len(t) == len(jd)
        rows = []
        for k in range(N):
            x_est = states[k]
            q = x_est.ori.as_array()
            bg = np.asarray(x_est.gyro_bias, float).reshape(3)
            # For FGO, we do not have covariance information; insert zeros
            P_flat = np.zeros(36, float)
            rows.append((
                est_run_id,
                k,
                float(t[k]),
                float(jd[k]),
                float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                float(bg[0]), float(bg[1]), float(bg[2]),
                *[float(v) for v in P_flat],
            ))
        cur.executemany("""
            INSERT INTO est_states (
                est_run_id, idx, t, jd,
                q0, q1, q2, q3,
                bgx, bgy, bgz,
                P00, P01, P02, P03, P04, P05,
                P10, P11, P12, P13, P14, P15,
                P20, P21, P22, P23, P24, P25,
                P30, P31, P32, P33, P34, P35,
                P40, P41, P42, P43, P44, P45,
                P50, P51, P52, P53, P54, P55
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            );
        """, rows)  
        self.conn.commit()
        return est_run_id

    def load_run(self, run_id: int) -> SimulationResult:
        """Reconstruct a SimulationResult from the database."""
        cur = self.conn.cursor()

        cur.execute("SELECT name, config_json FROM runs WHERE id=?;", (run_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"No run with id={run_id}")
        name, cfg_json = row
        cfg_dict = json.loads(cfg_json)
        cfg = SimulationConfig(**cfg_dict)

        cur.execute("""
            SELECT idx, t, jd,
                   q0, q1, q2, q3,
                   bgx, bgy, bgz,
                   wx_true, wy_true, wz_true,
                   wx_meas, wy_meas, wz_meas,
                   bx_meas, by_meas, bz_meas,
                   sx_meas, sy_meas, sz_meas,
                   st_q0, st_q1, st_q2, st_q3,
                   bx_eci, by_eci, bz_eci,
                   sx_eci, sy_eci, sz_eci
            FROM samples
            WHERE run_id=?
            ORDER BY idx ASC;
        """, (run_id,))
        rows = cur.fetchall()

        N = len(rows)
        t = np.zeros(N)
        jd = np.zeros(N)
        q_true = np.zeros((N, 4))
        b_g_true = np.zeros((N, 3))
        omega_true = np.zeros((N, 3))
        omega_meas = np.zeros((N, 3))
        mag_meas = np.zeros((N, 3))
        sun_meas = np.zeros((N, 3))
        st_meas = np.zeros((N, 4))
        b_eci = np.zeros((N, 3))
        s_eci = np.zeros((N, 3))

        for i, r in enumerate(rows):
            (_, t_i, jd_i,
             q0,q1,q2,q3,
             bgx,bgy,bgz,
             wxt,wyt,wzt,
             wxm,wym,wzm,
             bx,by,bz,
             sx,sy,sz,
             st0,st1,st2,st3,
             bx_eci_i,by_eci_i,bz_eci_i,
             sx_eci_i,sy_eci_i,sz_eci_i) = r

            t[i] = t_i
            jd[i] = jd_i
            q_true[i] = [q0,q1,q2,q3]
            b_g_true[i] = [bgx,bgy,bgz]
            omega_true[i] = [wxt,wyt,wzt]
            omega_meas[i] = [wxm,wym,wzm]
            mag_meas[i] = [bx,by,bz]
            sun_meas[i] = [sx,sy,sz]
            st_meas[i] = [st0,st1,st2,st3]
            b_eci[i] = [bx_eci_i,by_eci_i,bz_eci_i]
            s_eci[i] = [sx_eci_i,sy_eci_i,sz_eci_i]

        return SimulationResult(
            t=t, jd=jd,
            q_true=q_true,
            b_g_true=b_g_true,
            omega_true=omega_true,
            omega_meas=omega_meas,
            mag_meas=mag_meas,
            sun_meas=sun_meas,
            st_meas=st_meas,
            b_eci=b_eci,
            s_eci=s_eci,
            config=cfg,
        )
        
    def load_estimated_states(self, est_run_id: int) -> EstimationResult:
        """
        Load estimated states for a given estimation run.

        Args:
            est_run_id: id in est_runs.
        Returns:
            EstimationResult with (t, jd, q_est, bg_est, P_est).
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT idx, t, jd,
                   q0, q1, q2, q3,
                   bgx, bgy, bgz,
                   P00, P01, P02, P03, P04, P05,
                   P10, P11, P12, P13, P14, P15,
                   P20, P21, P22, P23, P24, P25,
                   P30, P31, P32, P33, P34, P35,
                   P40, P41, P42, P43, P44, P45,
                   P50, P51, P52, P53, P54, P55
            FROM est_states
            WHERE est_run_id=?
            ORDER BY idx ASC;
        """, (est_run_id,))
        rows = cur.fetchall()

        N = len(rows)
        q_estimated = np.zeros((N, 4))
        b_g_estimated = np.zeros((N, 3))
        P_estimated = np.zeros((N, 6, 6))
        t = np.zeros(N)
        jd = np.zeros(N)

        for i, r in enumerate(rows):
            (_, t_i, jd_i,
            q0, q1, q2, q3,
            bgx, bgy, bgz, *P_flat) = r

            t[i] = t_i
            jd[i] = jd_i
            q_estimated[i] = [q0, q1, q2, q3]
            b_g_estimated[i] = [bgx, bgy, bgz]
            P_estimated[i] = np.array(P_flat, float).reshape(6, 6)



        return EstimationResult(
            t=t,
            jd=jd,
            q_est=q_estimated,
            bg_est=b_g_estimated,
            P_est=P_estimated,
        )
