from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from estimation.gtsam_fg import WindowSample

from utilities.utils import load_yaml
from logging_config import get_logger
from utilities.process_model import ProcessModel
from utilities.quaternion import Quaternion
from utilities.states import NominalState, SensorType
from environment.environment import OrbitEnvironmentModel

logger = get_logger(__name__)

# --- HEPLERS ---

STATE_DIM = 7  # [q0, q1, q2, q3, b0, b1, b2]

def quat_left_product_matrix(q: Quaternion) -> np.ndarray:
    """
    Left quaternion product matrix L(q) such that q1 ⊗ q2 = L(q1) @ q2_array
    For q = [mu, eta_x, eta_y, eta_z]
    """
    mu = q.mu
    x, y, z = q.eta
    return np.array([
        [ mu, -x, -y, -z],
        [  x, mu, -z,  y],
        [  y,  z, mu, -x],
        [  z, -y,  x, mu]
    ])

def quat_right_product_matrix(q: Quaternion) -> np.ndarray:
    """
    Right quaternion product matrix R(q) such that q1 ⊗ q2 = R(q2) @ q1_array
    For q = [mu, eta_x, eta_y, eta_z]
    """
    mu = q.mu
    x, y, z = q.eta
    return np.array([
        [ mu, -x, -y, -z],
        [  x, mu,  z, -y],
        [  y, -z, mu,  x],
        [  z,  y, -x, mu]
    ])

def unpack_states(x: np.ndarray) -> list[NominalState]:
    assert x.size % STATE_DIM == 0
    num_states = x.size // STATE_DIM
    states: list[NominalState] = []
    for i in range(num_states):
        base = i * STATE_DIM
        q_arr = x[base:base+4]
        b_g   = x[base+4:base+7]
        q     = Quaternion.from_array(q_arr)
        states.append(NominalState(ori=q, gyro_bias=b_g))
    return states

def pack_states(states: list[NominalState]) -> np.ndarray:
    x = np.zeros(len(states) * STATE_DIM)
    for i, st in enumerate(states):
        base = i * STATE_DIM
        x[base:base+4] = st.ori.as_array()
        x[base+4:base+7] = st.gyro_bias
    return x

def attitude_error(q_ref: Quaternion, q: Quaternion) -> np.ndarray:
    # Error quaternion: dq = q_ref ⊗ q^{-1}
    dq = q_ref.multiply(q.conjugate()).normalize()
    # Small-angle approximation: δθ ≈ 2 * eta (vector part)
    delta_theta = 2.0 * dq.eta
    return delta_theta


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from 3-vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    
def jacobian_attitude_error_wrt_q(q_ref: Quaternion, q: Quaternion) -> np.ndarray:
    """
    Compute Jacobian of attitude error δθ w.r.t. quaternion q.
    
    Attitude error definition (from your code):
        dq = q_ref ⊗ q^(-1)
        δθ ≈ 2 * dq.eta  (small angle approximation)
    
    Args:
        q_ref: Reference quaternion
        q: Current quaternion estimate
        
    Returns:
        J: 3×4 matrix, ∂(δθ)/∂q where q is [mu, eta_x, eta_y, eta_z]
    """
    # dq = q_ref ⊗ q*
    # ∂(dq)/∂q = L(q_ref) @ ∂(q*)/∂q
    
    # Conjugate: q* = [mu, -eta]
    # ∂(q*)/∂q for q* = [mu, -eta_x, -eta_y, -eta_z]
    D_conj = np.array([
        [ 1,  0,  0,  0],
        [ 0, -1,  0,  0],
        [ 0,  0, -1,  0],
        [ 0,  0,  0, -1]
    ])
    
    # ∂(dq)/∂q = L(q_ref) @ D_conj
    L_qref = quat_left_product_matrix(q_ref)
    J_dq = L_qref @ D_conj  # 4×4
    
    # Extract vector part: δθ = 2 * dq.eta = 2 * dq[1:4]
    J = 2.0 * J_dq[1:4, :]  # 3×4
    
    return J

def jacobian_quat_propagation(q: Quaternion, omega: np.ndarray, dt: float) -> tuple:
    """
    Compute Jacobians of quaternion propagation following YOUR convention:
        q_new = delta_q ⊗ q_old
    where delta_q = exp(omega * dt / 2)
    
    Args:
        q: Current quaternion
        omega: Angular velocity (3,)
        dt: Time step
        
    Returns:
        J_q: 4×4 matrix, ∂q_new/∂q
        J_omega: 4×3 matrix, ∂q_new/∂ω
    """
    omega = np.asarray(omega, dtype=float).reshape(3)
    omega_norm = np.linalg.norm(omega)
    
    # Compute delta_q = exp(omega * dt / 2)
    if omega_norm < 1e-12:
        # Small angle: delta_q ≈ [1, 0.5*dt*omega]
        delta_mu = 1.0
        delta_eta = 0.5 * dt * omega
        
        # ∂(delta_q)/∂ω
        J_delta_omega = np.zeros((4, 3))
        J_delta_omega[0, :] = 0.0  # ∂mu/∂ω ≈ 0
        J_delta_omega[1:4, :] = 0.5 * dt * np.eye(3)  # ∂eta/∂ω = 0.5*dt*I
    else:
        theta = omega_norm * dt
        half_theta = theta / 2.0
        
        delta_mu = np.cos(half_theta)
        delta_eta = np.sin(half_theta) * omega / omega_norm
        
        # ∂(delta_q)/∂ω
        # Let θ = ||ω||*dt, u = ω/||ω||
        # delta_mu = cos(θ/2)
        # delta_eta = sin(θ/2) * u
        
        # ∂θ/∂ω = dt * ω/||ω|| = dt * u
        dtheta_domega = dt * omega / omega_norm  # 3×1
        
        # ∂mu/∂ω = -sin(θ/2) * ∂(θ/2)/∂ω = -sin(θ/2) * dt/2 * u
        J_delta_omega = np.zeros((4, 3))
        J_delta_omega[0, :] = -np.sin(half_theta) * 0.5 * dtheta_domega
        
        # ∂(sin(θ/2)*u)/∂ω = cos(θ/2) * ∂(θ/2)/∂ω * u ⊗ 1 + sin(θ/2) * ∂u/∂ω
        # where ∂u/∂ω = (I - u⊗u^T) / ||ω||
        u = omega / omega_norm
        outer_u = np.outer(u, u)
        du_domega = (np.eye(3) - outer_u) / omega_norm
        
        J_delta_omega[1:4, :] = (np.cos(half_theta) * 0.5 * dt * outer_u + 
                                  np.sin(half_theta) * du_domega)
    
    # Create delta_q as Quaternion for matrix computation
    delta_q = Quaternion(delta_mu, delta_eta)
    
    # q_new = delta_q ⊗ q
    # Using left product matrix: q_new = L(delta_q) @ q
    L_delta = quat_left_product_matrix(delta_q)
    
    # ∂q_new/∂q: Since delta_q is independent of q
    # ∂(L(delta_q) @ q)/∂q = L(delta_q)
    J_q = L_delta  # 4×4
    
    # ∂q_new/∂ω: Use chain rule
    # q_new = L(delta_q) @ q
    # ∂q_new/∂ω = ∂L(delta_q)/∂ω @ q
    # But L(delta_q) @ q can be rewritten as R(q) @ delta_q
    # So: ∂q_new/∂ω = R(q) @ ∂(delta_q)/∂ω
    R_q = quat_right_product_matrix(q)
    J_omega = R_q @ J_delta_omega  # 4×3
    
    return J_q, J_omega

def jacobian_rotmat_times_vector_wrt_q(q: Quaternion, v: np.ndarray) -> np.ndarray:
    """
    Compute Jacobian of R(q)^T @ v w.r.t. quaternion q.
    
    This is for measurements like magnetometer and sun sensor where:
        y_predicted = R(q)^T @ v_inertial  (rotate from inertial to body)
        residual = y_meas - y_predicted
    
    Args:
        q: Quaternion
        v: Vector in inertial frame (3,)
        
    Returns:
        J: 3×4 matrix, ∂(R^T @ v)/∂q where q is [mu, eta_x, eta_y, eta_z]
    """
    v = np.asarray(v, dtype=float).reshape(3)
    mu = q.mu
    x, y, z = q.eta
    
    # Using your as_rotmat() implementation, R^T rotates from inertial to body
    # We need ∂(R^T @ v)/∂q
    
    # Analytical derivatives of R^T @ v w.r.t. quaternion components
    # These come from differentiating your rotation matrix formula
    
    # ∂(R^T @ v)/∂mu
    dRTv_dmu = 2 * np.array([
        -z*v[1] + y*v[2],
         z*v[0] + x*v[2],
        -y*v[0] + x*v[1]
    ])
    
    # ∂(R^T @ v)/∂x  
    dRTv_dx = 2 * np.array([
         y*v[1] + z*v[2],
         y*v[0] - 2*x*v[1] - mu*v[2],
         z*v[0] + mu*v[1] - 2*x*v[2]
    ])
    
    # ∂(R^T @ v)/∂y
    dRTv_dy = 2 * np.array([
         x*v[1] + mu*v[2] - 2*y*v[0],
         x*v[0] + z*v[2],
        -mu*v[0] + z*v[1] - 2*y*v[2]
    ])
    
    # ∂(R^T @ v)/∂z
    dRTv_dz = 2 * np.array([
        -mu*v[1] + x*v[2] - 2*z*v[0],
         mu*v[0] + y*v[2] - 2*z*v[1],
         x*v[0] + y*v[1]
    ])
    
    J = np.column_stack([dRTv_dmu, dRTv_dx, dRTv_dy, dRTv_dz])  # 3×4
    
    return J

# --- MAIN CLASSES ---

@dataclass
class Factor:
    """
    Lightweight container describing a single factor in the graph.
    Attributes:
        index: Unique index of the factor in the graph.
        factor_type: Type of the factor ("prior", "gyro", "star", "magnetometer", or "sun").
        state_indices: Tuple of node index that this factor connects.
        payload: Dictionary containing factor-specific data.
    """

    factor_type: str
    state_indices: Tuple[int, ...]
    payload: dict
    
    def compute_factor_jacobian(self, factor, states: list) -> tuple:
        """
        Compute the Jacobian for a given factor.
        
        Args:
            factor: Factor object
            states: List of NominalState objects
            
        Returns:
            J or (J_i, J_j, ...): Jacobian matrix/matrices for this factor
            state_indices: List of state indices this factor depends on
        """
        match factor.factor_type:
            case "prior":
                idx = factor.state_indices[0]
                st = states[idx]
                prior = factor.payload["prior"]
                cov = factor.payload["cov"]
                
                q_ref = prior.ori  # Quaternion object
                q = st.ori  # Quaternion object
                
                # Jacobian of attitude error w.r.t q: 3×4
                J_att_q = jacobian_attitude_error_wrt_q(q_ref, q)
                
                # Jacobian of bias error w.r.t bias: 3×3 identity
                J_bias_b = np.eye(3)
                
                # Combine: residual is [δθ; δb] (6×1), state is [q; b] (7×1)
                J = np.zeros((6, 7))
                J[0:3, 0:4] = J_att_q
                J[3:6, 4:7] = J_bias_b
                
                # Apply whitening (information matrix square root)
                info = np.linalg.inv(cov)
                S = np.linalg.cholesky(info)
                J = S @ J
                
                return J, [idx]
            
            case "gyro":
                i, j = factor.state_indices
                st_i = states[i]
                st_j = states[j]
                omega_meas = factor.payload["omega"]
                dt = factor.payload["dt"]
                
                q_i = st_i.ori  # Quaternion object
                q_j = st_j.ori  # Quaternion object
                b_i = st_i.gyro_bias
                
                # Corrected angular velocity
                omega_hat = omega_meas - b_i
                
                # Jacobians of propagation: q_pred = delta_q ⊗ q_i
                J_qpred_qi, J_qpred_omega = jacobian_quat_propagation(q_i, omega_hat, dt)
                # J_qpred_qi: 4×4, J_qpred_omega: 4×3
                
                # Compute predicted quaternion for error computation
                q_pred = q_i.propagate(omega_hat, dt)
                
                # Jacobian of attitude error δθ w.r.t q_pred: 3×4
                J_att_qpred = jacobian_attitude_error_wrt_q(q_pred, q_j)
                
                # Chain rule: ∂(δθ)/∂q_i = ∂(δθ)/∂q_pred @ ∂q_pred/∂q_i
                J_att_qi = J_att_qpred @ J_qpred_qi  # 3×4
                
                # ∂(δθ)/∂b_i = ∂(δθ)/∂q_pred @ ∂q_pred/∂ω @ ∂ω/∂b_i
                # where ∂ω_hat/∂b_i = -I (since ω_hat = ω_meas - b_i)
                J_att_bi = J_att_qpred @ J_qpred_omega @ (-np.eye(3))  # 3×3
                
                # Jacobian w.r.t state i: [q_i; b_i]
                J_i = np.zeros((3, 7))
                J_i[:, 0:4] = J_att_qi
                J_i[:, 4:7] = J_att_bi
                
                # Jacobian w.r.t state j: [q_j; b_j]
                # The residual is δθ = 2*(q_pred ⊗ q_j^(-1)).eta
                # ∂(δθ)/∂q_j from the error function
                J_att_qj = jacobian_attitude_error_wrt_q(q_pred, q_j)
                
                J_j = np.zeros((3, 7))
                J_j[:, 0:4] = J_att_qj
                J_j[:, 4:7] = np.zeros((3, 3))  # residual doesn't depend on b_j
                
                return (J_i, J_j), [i, j]
            
            case "star":
                idx = factor.state_indices[0]
                st = states[idx]
                q_meas = factor.payload["y"]  # Quaternion object
                q_est = st.ori  # Quaternion object
                
                # Residual: δθ = 2 * (q_meas ⊗ q_est^(-1)).eta
                J_att_q = jacobian_attitude_error_wrt_q(q_meas, q_est)
                
                J = np.zeros((3, 7))
                J[:, 0:4] = J_att_q
                J[:, 4:7] = np.zeros((3, 3))
                
                return J, [idx]
            
            case "magnetometer":
                idx = factor.state_indices[0]
                st = states[idx]
                B_n = factor.payload["B_n"]  # Inertial frame magnetic field
                q = st.ori  # Quaternion object
                
                # Residual: r = y_meas - R(q)^T @ B_n
                # ∂r/∂q = -∂(R^T @ B_n)/∂q
                J_rot = jacobian_rotmat_times_vector_wrt_q(q, B_n)
                
                J = np.zeros((3, 7))
                J[:, 0:4] = -J_rot  # negative because r = y - R^T@v
                J[:, 4:7] = np.zeros((3, 3))
                
                return J, [idx]
            
            case "sun":
                idx = factor.state_indices[0]
                st = states[idx]
                s_n = factor.payload["s_n"]  # Inertial frame sun vector
                q = st.ori  # Quaternion object
                
                # Residual: r = y_meas - R(q)^T @ s_n
                J_rot = jacobian_rotmat_times_vector_wrt_q(q, s_n)
                
                J = np.zeros((3, 7))
                J[:, 0:4] = -J_rot
                J[:, 4:7] = np.zeros((3, 3))
                
                return J, [idx]
            
            case "bias":
                i, j = factor.state_indices
                Q_d = factor.payload["Q_d"]
                
                # Residual: r = S_b @ (b_j - b_i) where S_b = chol(Q_d^{-1})
                info_b = np.linalg.inv(Q_d)
                S_b = np.linalg.cholesky(info_b)
                
                # ∂r/∂x_i = [0, -S_b] for state [q_i; b_i]
                J_i = np.zeros((3, 7))
                J_i[:, 4:7] = -S_b
                
                # ∂r/∂x_j = [0, S_b] for state [q_j; b_j]
                J_j = np.zeros((3, 7))
                J_j[:, 4:7] = S_b
                
                return (J_i, J_j), [i, j]
            
            case _:
                raise ValueError(f"Unknown factor type: {factor.factor_type}")
        

@dataclass
class Node:
    """
    Lightweight container describing a single node in the graph.
    Attributes:
        index: Unique index of the node in the graph.
        state: NominalState associated with this node.
    """
    index: int
    state: NominalState
    
class FGO:
    """Simple batch factor graph optimizer for attitude + gyro bias."""
    def __init__(self):
        self.process_model = ProcessModel()
        self.states : List[NominalState] = []
        self.factors : List[Factor] = []
        self.env = OrbitEnvironmentModel()
        pass
    
    def reset(self) -> None:
        """Reset the factor graph."""
        self.states = []
        self.factors = []
    
    def add_state(self, state: NominalState) -> int:
        """Add a new state to the graph and return its index."""
        if len(self.states) == 0:
            self.states.append(state)
            return 0
        else:
            self.states.append(state)
            return len(self.states) - 1
    
    def add_prior(self, state_idx: int, prior: NominalState, cov: np.ndarray) -> None:
        """Add a prior factor on the specified state."""
        self.factors.append(
            Factor(
                factor_type="prior",
                state_indices=(state_idx,),
                payload={"prior": prior, "cov": np.asarray(cov, float)},
            )
        )
        
    def add_gyro_factor(self, i: int, j: int, omega_meas: np.ndarray, dt: float) -> None:
        """
        Add a gyro process factor between states i and j. 
        The factor predicts state j from state i using omega_meas and dt.
        """
        self.factors.append(
            Factor(
                factor_type="gyro",
                state_indices=(i, j),
                payload={"omega": np.asarray(omega_meas, float), 
                         "dt": float(dt)},
            )
        )
        
    def add_bias_factor(self, i: int, j: int, dt: float) -> None:
        """
        Add a gyro-bias random walk factor between states i and j.
        The factor models bias evolution from state i to state j over dt.
        Model: b_j ≈ b_i, with Q_b = (sigma_bg_rw^2 * dt) I_3.
        """
        A = -np.eye(3)
        Q_c = self.process_model.Q_c[3:6, 3:6]
        Q_d = Q_c * dt + 0.5 * (A @ Q_c + Q_c @ A.T) * dt**2 + (1.0 / 3.0) * (A @ Q_c @ A.T) * dt**3
        
        self.factors.append(
            Factor(
                factor_type="bias",
                state_indices=(i, j),
                payload={"Q_d": Q_d,},
            )
        )
    
    def add_measurement(self,
                        idx: int,
                        y: np.ndarray | Quaternion,
                        sensor_type: SensorType,
                        B_n: Optional[np.ndarray] = None,
                        s_n: Optional[np.ndarray] = None) -> None:
        """Add a measurement factor for the specified state."""
        match sensor_type:
            case SensorType.MAGNETOMETER:
                self.factors.append(
                    Factor(
                        factor_type="magnetometer",
                        state_indices=(idx,),
                        payload={
                            "y": y,
                            "B_n": np.asarray(B_n, float),
                        },
                    )
                )
            case SensorType.SUN_VECTOR:
                self.factors.append(
                    Factor(
                        factor_type="sun",
                        state_indices=(idx,),
                        payload={
                            "y": y,
                            "s_n": np.asarray(s_n, float),
                        },
                    )
                )
            case SensorType.STAR_TRACKER:
                self.factors.append(
                    Factor(
                        factor_type="star",
                        state_indices=(idx,),
                        payload={
                            "y": y,
                        },
                    )
                )
            case _:
                raise ValueError(f"Unsupported sensor type {sensor_type}")
            
    def build_graph(self, samples: list[WindowSample]):
        """Build the factor graph from a list of WindowSample objects."""
        for i, sample in enumerate(samples):
            idx = self.add_state(sample.x_nom)
            if i == 0:
                # add prior on first state
                prior_cov = np.diag([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8])
                self.add_prior(
                    state_idx=idx,
                    prior=sample.x_nom,
                    cov=prior_cov,
                )
            else:
                # add gyro factor from previous state
                prev_idx = i - 1
                dt = sample.t - samples[prev_idx].t
                self.add_gyro_factor(
                    i=prev_idx,
                    j=idx,
                    omega_meas=sample.omega_meas,
                    dt=dt,
                )
                
                # add bias factor from previous state
                self.add_bias_factor(
                    i=prev_idx,
                    j=idx,
                    dt=dt,
                )
            
            # add measurement factors if available
            r_eci = self.env.get_r_eci(sample.jd)
            if sample.z_mag is not None:
                self.add_measurement(
                    idx=idx,
                    y=sample.z_mag,
                    sensor_type=SensorType.MAGNETOMETER,
                    B_n=self.env.get_B_eci(r_eci, sample.jd),
                )
            if sample.z_sun is not None:
                self.add_measurement(
                    idx=idx,
                    y=sample.z_sun,
                    sensor_type=SensorType.SUN_VECTOR,
                    s_n=self.env.get_sun_eci(sample.jd),
                )
            if sample.z_st is not None:
                self.add_measurement(
                    idx=idx,
                    y=sample.z_st,
                    sensor_type=SensorType.STAR_TRACKER,
                )
        return

    def jac(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the full Jacobian matrix for all factors.
        
        Args:
            x: Flat state vector [q0, b0, q1, b1, ...] where each qi is 4D
            
        Returns:
            J: Jacobian matrix (total_residual_dim × total_state_dim)
        """
        
        states = unpack_states(x)
        total_state_dim = x.size
        
        # Compute residual dimensions for each factor
        residual_sizes = []
        for factor in self.factors:
            match factor.factor_type:
                case "prior":
                    residual_sizes.append(6)
                case "gyro" | "star" | "magnetometer" | "sun" | "bias":
                    residual_sizes.append(3)
                case _:
                    raise ValueError(f"Unknown factor type: {factor.factor_type}")
        
        total_residual_dim = sum(residual_sizes)
        J_full = np.zeros((total_residual_dim, total_state_dim))
        
        # Fill in the Jacobian
        row_offset = 0
        for factor, res_size in zip(self.factors, residual_sizes):
            result = factor.compute_factor_jacobian(factor, states)
            
            if isinstance(result[0], tuple):
                # Multi-state factor (e.g., gyro, bias)
                J_list, state_indices = result
                for J_local, idx in zip(J_list, state_indices):
                    col_offset = idx * STATE_DIM
                    J_full[row_offset:row_offset + res_size,
                        col_offset:col_offset + STATE_DIM] += J_local
            else:
                # Single-state factor
                J_local, state_indices = result
                idx = state_indices[0]
                col_offset = idx * STATE_DIM
                J_full[row_offset:row_offset + res_size,
                    col_offset:col_offset + STATE_DIM] = J_local
            
            row_offset += res_size
        
        return J_full    
            
    def fun(self, x: np.ndarray) -> np.ndarray:
        """Stacked residuals for all factors given flat state vector x."""
        states = unpack_states(x)
        residuals: list[np.ndarray] = []

        for factor in self.factors:
            match factor.factor_type:
                case "prior":
                    idx = factor.state_indices[0]
                    st  = states[idx]
                    prior: NominalState = factor.payload["prior"]
                    cov  = np.asarray(factor.payload["cov"], float)

                    # 6×1 residual: [δθ; δb]
                    r_att = attitude_error(prior.ori, st.ori)         # 3
                    r_bias = st.gyro_bias - prior.gyro_bias      # 3
                    r = np.concatenate([r_att, r_bias])

                    # whitening
                    info = np.linalg.inv(cov)
                    S = np.linalg.cholesky(info)
                    r = S @ r
                    residuals.append(r)

                case "gyro":
                    i, j = factor.state_indices
                    st_i = states[i]
                    st_j = states[j]
                    omega = factor.payload["omega"]
                    dt    = factor.payload["dt"]

                    # predict j from i
                    omega_hat = omega - st_i.gyro_bias
                    q_pred_j  = st_i.ori.propagate(omega_hat, dt)
                    r_att = attitude_error(q_pred_j, st_j.ori)  # 3-d

                    # simple isotropic gyro noise for now
                    # you can parameterize this via payload later
                    residuals.append(r_att)

                case "star":
                    idx = factor.state_indices[0]
                    st  = states[idx]
                    q_meas: Quaternion = factor.payload["y"]

                    r_att = attitude_error(q_meas, st.ori)  # 3
                    print()
                    residuals.append(r_att)

                case "magnetometer":
                    idx = factor.state_indices[0]
                    st  = states[idx]
                    y   = np.asarray(factor.payload["y"], float)   # body-frame measurement
                    B_n = np.asarray(factor.payload["B_n"], float)

                    # predicted mag in body: R(q)^T * B_n
                    R_bn = st.ori.as_rotmat().T
                    y_pred = R_bn @ B_n

                    r = y - y_pred
                    residuals.append(r)

                case "sun":
                    idx = factor.state_indices[0]
                    st  = states[idx]
                    y   = np.asarray(factor.payload["y"], float)   # body-frame sun vector
                    s_n = np.asarray(factor.payload["s_n"], float)

                    R_bn = st.ori.as_rotmat().T
                    y_pred = R_bn @ s_n

                    r = y - y_pred
                    residuals.append(r)
                    
                case "bias":
                    i, j = factor.state_indices
                    st_i = states[i]
                    st_j = states[j]

                    b_i = st_i.gyro_bias
                    b_j = st_j.gyro_bias

                    Q_d = np.asarray(factor.payload["Q_d"], float)
                    info_b = np.linalg.inv(Q_d)
                    S_b = np.linalg.cholesky(info_b)

                    r_b = b_j - b_i              # 3×1
                    r_b = S_b @ r_b              # whitened

                    residuals.append(r_b)

                case _:
                    raise ValueError(f"Unknown factor type {factor.factor_type}")

        if not residuals:
            return np.zeros(0)

        return np.concatenate(residuals)

    def optimize(self, max_nfev: int = 100) -> list[NominalState]:
        x0 = pack_states(self.states)
        res = least_squares(
            self.fun, 
            x0, 
            jac=self.jac,  # Use analytical Jacobian
            method="trf", 
            max_nfev=max_nfev,
            verbose=1
        )
        
        print(f"\\nOptimization complete:")
        print(f"  Status: {res.message}")
        print(f"  Final cost: {res.cost:.6e}")
        print(f"  Function evals: {res.nfev}")
        print(f"  Jacobian evals: {res.njev}")
        
        self.states = unpack_states(res.x)
        return self.states
        



