import numpy as np

class Quaternion:
    """
    Quaternion represented as q = [mu; eta]
    mu  : scalar part (float)
    eta : vector part (3,)
    """

    def __init__(self, mu: float, eta: np.ndarray):
        self.mu = float(mu)
        self.eta = np.asarray(eta, dtype=float).reshape(3)
        

    def canonical(self):
        """Return a copy with sign chosen so that mu >= 0."""
        if self.mu < 0.0:
            return Quaternion(-self.mu, -self.eta)
        return self


    @staticmethod
    def from_array(q):
        q = np.asarray(q, float).reshape(4)
        return Quaternion(q[0], q[1:4])

    def as_array(self):
        return np.concatenate(([self.mu], self.eta))

    def normalize(self, eps: float = 1e-12):
        n = np.linalg.norm(self.as_array())
        if n < eps:
            self.mu = 1.0
            self.eta[:] = 0.0
        else:
            self.mu /= n
            self.eta /= n
        return self

    @staticmethod
    def normalized(q, eps: float = 1e-12):
        q = q.copy()
        return q.normalize(eps)

    def copy(self):
        return Quaternion(self.mu, self.eta.copy())

    def conjugate(self):
        return Quaternion(self.mu, -self.eta)

    def multiply(self, q2):
        """
        Hamilton product q = self ⊗ q2
        """
        w1 = self.mu
        v1 = self.eta
        w2 = q2.mu
        v2 = q2.eta

        w = w1*w2 - v1.dot(v2)
        v = w1*v2 + w2*v1 + np.cross(v1, v2)
        return Quaternion(w, v)

    # optional: rotate a vector
    def rotate(self, v):
        v = np.asarray(v, float).reshape(3)
        qv = Quaternion(0.0, v)
        return self.multiply(qv).multiply(self.conjugate()).eta
    
    def as_rotmat(self):
        """
        Convert quaternion to rotation matrix.
        """
        mu = self.mu
        x, y, z = self.eta

        R = np.array([[1 - 2*(y**2 + z**2),     2*(x*y - z*mu),     2*(x*z + y*mu)],
                      [    2*(x*y + z*mu), 1 - 2*(x**2 + z**2),     2*(y*z - x*mu)],
                      [    2*(x*z - y*mu),     2*(y*z + x*mu), 1 - 2*(x**2 + y**2)]])
        return R
    
    def as_euler(self): 
        """
        Convert quaternion to euler angles (roll, pitch, yaw).
        """
        mu = self.mu
        x, y, z = self.eta

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (mu * x + y * z)
        cosr_cosp = 1 - 2 * (x**2 + y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (mu * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * (np.pi / 2)  # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (mu * z + x * y)
        cosy_cosp = 1 - 2 * (y**2 + z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])
    
    def diff_as_avec(self, other: 'Quaternion') -> np.ndarray:
        """
        Rotation needed to go from 'other' to 'self', with
        q_self ≈ q_other ⊗ δq, δθ in body frame.
        """
        # δq ≈ other^{-1} ⊗ self
        q_diff = other.conjugate().multiply(self).normalize()

        angle = 2 * np.arccos(np.clip(q_diff.mu, -1.0, 1.0))
        if angle > np.pi:
            angle -= 2 * np.pi

        if abs(angle) < 1e-12:
            return np.zeros(3)

        axis = q_diff.eta / np.sin(angle / 2.0)
        avec = axis * angle
        return avec

    
    @staticmethod
    def from_avec(avec: np.ndarray, eps: float = 1e-12) -> 'Quaternion':
        """Build a quaternion from a rotation vector (angle-axis in vector form)."""
        avec = np.asarray(avec, float).reshape(3)
        angle = np.linalg.norm(avec)
        if angle < eps:
            return Quaternion(1.0, np.zeros(3))
        axis = avec / angle
        half = 0.5 * angle
        return Quaternion(np.cos(half), axis * np.sin(half))

    
    def propagate(self, omega: np.ndarray, dt: float):
        """
        Propagate quaternion given the angular velocity over time dt,
        using the expoential map. Uses left-multiplication convention.
        I.e. q_new = delta_q ⊗ q_old, where delta_q is built from omega*dt.
        omega : angular velocity vector (3,)
        dt    : time step (float)
        """
        omega = np.asarray(omega, float).reshape(3)
        norm_omega = np.linalg.norm(omega)
        if norm_omega < 1e-12:
            delta_q = Quaternion(1.0, np.zeros(3))
        else:
            theta = norm_omega * dt
            axis = omega / norm_omega
            delta_q = Quaternion(np.cos(theta/2.0), axis * np.sin(theta/2.0))
        q_new = delta_q.multiply(self)
        return q_new.normalize()
