import numpy as np


class KalmanCA2D:
    """Constant-acceleration 2D Kalman filter.

    State: [x, z, vx, vz, ax, az]^T
    Measurement: [x, z]^T
    """

    def __init__(self, dt: float = 1.0, process_noise: float = 1.0, measurement_noise: float = 4.0):
        self.dt = float(dt)
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.x = np.zeros((6, 1), dtype=float)
        self.P = np.eye(6, dtype=float) * 1e3
        self.F = self._make_F(self.dt)
        self.H = np.zeros((2, 6), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.Q = np.eye(6, dtype=float) * self.q
        self.R = np.eye(2, dtype=float) * self.r
        self.I = np.eye(6, dtype=float)

    @staticmethod
    def _make_F(dt: float) -> np.ndarray:
        F = np.eye(6, dtype=float)
        F[0, 2] = dt
        F[1, 3] = dt
        F[0, 4] = 0.5 * dt * dt
        F[1, 5] = 0.5 * dt * dt
        F[2, 4] = dt
        F[3, 5] = dt
        return F

    def copy(self):
        k = KalmanCA2D(self.dt, self.q, self.r)
        k.x = self.x.copy()
        k.P = self.P.copy()
        return k

    def initialize(self, x: float, z: float, vx: float = 0.0, vz: float = 0.0, ax: float = 0.0, az: float = 0.0):
        self.x[:, 0] = [x, z, vx, vz, ax, az]
        self.P = np.diag([10, 10, 100, 100, 100, 100]).astype(float)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2, 0].copy()

    def update(self, measured_x: float, measured_z: float) -> np.ndarray:
        z = np.array([[measured_x], [measured_z]], dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        return self.x[:2, 0].copy()

    @property
    def position(self):
        return self.x[:2, 0].copy()

    @property
    def velocity(self):
        return self.x[2:4, 0].copy()

    @property
    def acceleration(self):
        return self.x[4:6, 0].copy()
