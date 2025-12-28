import numpy as np

class TrackKalmanFilter:
    def __init__(self, dt=1.0):
        self.dt = dt
        
        # State: [x, y, z, vx, vy, vz]
        self.state_dim = 6
        self.meas_dim = 3
        
        # 1. State Transition Matrix (F)
        # x_new = x + vx*dt
        # v_new = v
        self.F = np.eye(self.state_dim)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt
        
        # 2. Measurement Matrix (H)
        # We measure [x, y, z]
        self.H = np.zeros((self.meas_dim, self.state_dim))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        
        # 3. Covariance Matrices
        # Process Noise (Q): Uncertainty in model (random acceleration / scattering)
        self.Q = np.eye(self.state_dim) * 0.1 
        
        # Measurement Noise (R): Uncertainty in sensor/voxel position
        self.R = np.eye(self.meas_dim) * 0.5
        
        # Estimate Covariance (P)
        self.P = np.eye(self.state_dim) * 1.0
        
        # Init State
        self.x = np.zeros(self.state_dim)

    def initialize(self, start_pos, start_vel_guess=None):
        """
        Initialize filter at the first point.
        """
        self.x[:3] = start_pos
        if start_vel_guess is not None:
            self.x[3:] = start_vel_guess
        else:
            self.x[3:] = [0, 0, 1] # Generic forward guess
            
        # Reset large covariance for velocity if uncertain?
        self.P = np.eye(self.state_dim) * 1.0

    def predict(self):
        # x = Fx
        self.x = self.F @ self.x
        # P = FPF' + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        # y = z - Hx (Residual)
        z = np.array(measurement)
        y = z - self.H @ self.x
        
        # S = HPH' + R (Residual Covariance)
        S = self.H @ self.P @ self.H.T + self.R
        
        # K = PH'S^-1 (Kalman Gain)
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((self.state_dim, self.meas_dim))
        
        # x = x + Ky
        self.x = self.x + K @ y
        
        # P = (I - KH)P
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
        
    def fit(self, points):
        """
        Fits the filter to a sequence of 3D points.
        Points should be SORTED by distance from start ideally, or time.
        Returns:
            initial_direction: Normalized velocity vector at step 0 (retro-fitted or just first estimates).
            smoothed_points: List of filtered positions.
        """
        if len(points) < 2:
            return np.array([0., 0., 0.]), points
        
        # 1. Initialize at first point
        # Estimate initial velocity from first two points
        v_est = (points[1] - points[0]) / self.dt
        self.initialize(points[0], v_est)
        
        smoothed = []
        smoothed.append(self.x[:3].copy())
        
        # 2. Loop
        for i in range(1, len(points)):
            self.predict()
            self.update(points[i])
            smoothed.append(self.x[:3].copy())
            
        # 3. Extract Direction
        # In a simple online KF, the state at step 0 is the initial guess.
        # But we want the BEST estimate of the start direction given the whole track.
        # Ideally we run a Kalman Smoother (Backward pass).
        # For simplicity, let's take the velocity vector estimated after a few steps (stable)
        # or use the average velocity of the first N steps.
        
        # Let's take the state velocity after 3-5 steps (once it settles)
        # Or better: Just normalize the velocity vector of the current state? No that's the end.
        
        # We returned "smoothed", but we didn't store history of X.
        # Let's re-run quickly to get velocity at start? 
        # Actually, standard KF estimates state[k] given meas[0...k].
        # So at step 1 (second point), we have an estimate of velocity.
        
        # Simple heuristic: Fit line to first N smoothed points.
        first_n = np.array(smoothed[:min(len(smoothed), 10)])
        if len(first_n) < 2: return np.array([0,0,1]), smoothed
        
        # Least squares line fit on smoothed start
        # Center
        mean = np.mean(first_n, axis=0)
        uu, dd, vv = np.linalg.svd(first_n - mean)
        direction = vv[0] # Principal component
        
        # DEBUG SVD
        if dd[0] < 1e-3:
            # Degenerate line (all points same?)
            # print(f"DEBUG KF SVD: Warning Degenerate. dd={dd}, Mean={mean}")
            # If degenerate, return velocity guess from KF state?
            # Or just use last - first
            vec = first_n[-1] - first_n[0]
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                direction = vec / norm
            else:
                 # Absolute fallback
                 direction = np.array([0., 0., 1.])
        else:
             direction = vv[0]
        
        # Ensure it points "forward" (dot product with p[last] - p[0] > 0)
        global_vec = first_n[-1] - first_n[0]
        if np.dot(direction, global_vec) < 0:
            direction = -direction
            
        return direction, np.array(smoothed)
