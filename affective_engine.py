# ==============================================================================
# Affective Cone Model (S² / S³)
# PSF-Zero Head: /0 Projection + EIT + S³ Minimal Arc Update
# Dependencies: numpy
# ==============================================================================

import numpy as np
from dataclasses import dataclass

# ---------- Quaternion Utilities (w, x, y, z) ----------
def q_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def q_norm(q):
    n = np.linalg.norm(q)
    return q if n == 0.0 else q / n

def q_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)

def q_from_axis_angle(axis, angle):
    """axis: 3D unit vector, angle: radians"""
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    a = a / n
    h = 0.5 * angle
    s = np.sin(h)
    return q_norm(np.array([np.cos(h), *(s*a)], dtype=float))

def q_to_angles(q):
    """
    Returns (theta, phi) on a Bloch-like sphere for visualization.
    Maps the quaternion to a unit vector r in R^3 via sandwich q * (0,v) * q* where v=z-axis.
    """
    q = q_norm(q)
    v = np.array([0.0, 0.0, 1.0])
    vq = np.concatenate([[0.0], v])
    v_rot = q_mul(q_mul(q, vq), q_conj(q))[1:]
    
    x, y, z = v_rot
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi   = (np.arctan2(y, x) + 2*np.pi) % (2*np.pi)
    return theta, phi

# ---------- PSF-Zero Stabilization Head ----------
def clamp(dtheta, sigma=1.0):
    """ /0 Projection: Smoothly saturates large angles via Δθ / sqrt(σ^2 + Δθ^2) """
    return dtheta / np.sqrt(sigma*sigma + dtheta*dtheta)

def eit(zbar, phi, lam=0.1):
    """ EIT (Exponential Information Tracking): Phase EMA tracking the 'Now' """
    return (1.0 - lam) * zbar + lam * np.exp(1j * phi)

def su2_update_minimal_arc(q, axis, dtheta):
    """ S³ Minimal Arc: Applies a micro-rotation quaternion along the geodesic """
    dq = q_from_axis_angle(axis, dtheta)
    return q_norm(q_mul(dq, q))

# ---------- Affective Engine Core ----------
@dataclass
class AffectiveConfig:
    lam: float = 0.1                 # EIT lambda (forgetting factor)
    sigma: float = 1.0               # /0 clamp scale
    omega: float = 2.0               # Natural Larmor precession angular velocity [rad/s]
    dt: float = 0.02                 # Time step [s]
    alpha: float = 1.0               # Weight for Solid Angle (Ω) in E(T)
    beta: float  = 1.0               # Weight for Phase Velocity in E(T)
    max_phase_jump: float = np.deg2rad(60) # Safety threshold (ABSTAIN trigger)

@dataclass
class AffectiveState:
    q: np.ndarray                    # Quaternion (w,x,y,z)
    zbar: complex                    # EIT internal phase state
    phi: float                       # Visualization phase
    theta: float                     # Visualization opening angle
    t: float = 0.0

class AffectiveEngine:
    def __init__(self, cfg=AffectiveConfig()):
        self.cfg = cfg
        self.reset()

    def reset(self, theta0=np.deg2rad(20), phi0=0.0):
        # Initialize state by tilting the Z-axis toward the X-axis by theta0
        axis = np.array([1.0, 0.0, 0.0]) 
        q0   = q_from_axis_angle(axis, theta0)
        self.st = AffectiveState(q=q0, zbar=np.exp(1j*phi0), phi=phi0, theta=theta0, t=0.0)

    def step(self, stimulus_vec=np.array([0.0, 0.0]), homeo_vec=np.array([0.0, 0.0])):
        """
        Processes a single time step of emotional dynamics.
        stimulus_vec: External stimulus (e.g., [valence, arousal])
        homeo_vec: Internal homeostatic gradient
        """
        c = self.cfg
        
        # 1) Map Stimulus to Proposed Angle/Axis
        v = np.asarray(stimulus_vec, float) + np.asarray(homeo_vec, float)
        axis = np.array([v[0], v[1], 0.2]) # Mix in slight Z for stability
        n = np.linalg.norm(axis)
        axis = axis/n if n > 0 else np.array([0.0, 0.0, 1.0])

        raw_dtheta = np.clip(0.5 * np.linalg.norm(v), -np.pi, np.pi)
        
        # 2) PSF-Zero Head: Clamp (Prevent Over-rotation)
        dtheta = clamp(raw_dtheta, c.sigma)

        # 3) PSF-Zero Head: EIT (Lock to 'Now')
        self.st.zbar = eit(self.st.zbar, self.st.phi, c.lam)

        # 4) PSF-Zero Head: S³ Minimal Arc Update
        q_new = su2_update_minimal_arc(self.st.q, axis, dtheta)

        # 5) Natural Larmor Precession (φ̇ = ω)
        phi_next = (self.st.phi + c.omega * c.dt) % (2 * np.pi)
        theta_next, _ = q_to_angles(q_new)

        # 6) Safety Guard (ABSTAIN Flag)
        jump = abs(dtheta)
        abstain = jump > c.max_phase_jump

        # 7) State Commit
        self.st.q = q_new
        self.st.phi = phi_next
        self.st.theta = theta_next
        self.st.t += c.dt

        return {"abstain": abstain, "applied_dtheta": dtheta, "omega": c.omega}

    def E_metric(self, traj):
        """ Calculates the total Affective Intensity E(T) over a trajectory window. """
        c = self.cfg
        if not traj: return 0.0
        
        thetas = np.array([x["theta"] for x in traj])
        dphis  = np.array([x["dphi"]  for x in traj])
        omegas = np.abs(dphis) / c.dt
        
        # Solid Angle Ω = 2π(1 - cos(θ))
        Omega  = 2 * np.pi * (1 - np.cos(thetas))
        val = c.alpha * Omega + c.beta * (omegas * np.sin(thetas))
        return float(np.mean(val))

    def rollout(self, seconds=5.0, stim_fun=None, homeo_fun=None):
        """ Simulates the engine over time (Used for A/B budget logging). """
        steps = int(seconds / self.cfg.dt)
        traj = []
        for _ in range(steps):
            v = stim_fun(self.st.t) if stim_fun else np.zeros(2)
            h = homeo_fun(self.st.t) if homeo_fun else np.zeros(2)
            
            info = self.step(v, h)
            traj.append({
                "t": self.st.t, 
                "theta": self.st.theta, 
                "phi": self.st.phi,
                "dphi": self.cfg.omega * self.cfg.dt, 
                **info
            })
        return traj

# ==============================================================================
# Quick Demo Execution
# ==============================================================================
if __name__ == "__main__":
    eng = AffectiveEngine()
    
    # Define a low-frequency emotional stimulus
    def stim(t):  
        return np.array([0.8 * np.sin(0.7 * t), 0.6 * np.cos(0.5 * t)])
        
    print("Initiating Affective Engine Rollout...")
    trajectory = eng.rollout(seconds=4.0, stim_fun=stim)
    E_total = eng.E_metric(trajectory)
    
    print(f"Simulation Complete.")
    print(f"Total Affective Intensity Envelope E(T) ≈ {E_total:.3f}")
    print(f"Trajectory Length = {len(trajectory)} steps")
