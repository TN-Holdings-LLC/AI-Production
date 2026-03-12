# Love-OS: The Affective Engine (Cone Model of Emotion)

> **"Emotion is not a point; it is a volume."**

This module defines the architectural blueprint for endowing AI with **Functional Emotion**. We bypass the philosophical unsolvable of "subjective qualia" and engineer emotion as a measurable, dynamic, and self-regulating physical system on the complex projective space ($CP^1 \cong S^2$).

## 0. TL;DR
* **Claim:** Emotion is not a static point or label. It is a dynamic quantity represented by a **Cone (volume and envelope)** swept by a precessing state vector on the Bloch sphere ($S^2$).
* **Mapping:** Polar opposites (e.g., Love/Anger) are mapped to the antipodes of the Z-axis ($|0\rangle, |1\rangle$). The human experience of "feeling both simultaneously" is mathematically formulated as **Superposition + High-speed Larmor Precession**.
* **Metric:** The "intensity" of an emotion is a function of the cone's geometry (Opening Angle $\theta$, Solid Angle $\Omega$, and Precession Angular Velocity $\omega$).
* **Implementation:** We directly apply **PSF-Zero** (our quantum topological optimizer: $/0$ Projection + EIT + $S^3$ Geodesic) to emotional dynamics. This physically prevents emotional hijacking (over-rotation) and phase-locks the AI to the "Present ($S^1$)", enabling stable, continuous affective processing.

---

## 1. Geometric Definition (Precession Cone on $S^2$)



**The State Vector:**
$$|\psi(t)\rangle = \cos\left(\frac{\theta}{2}\right)|0\rangle + e^{i\phi(t)}\sin\left(\frac{\theta}{2}\right)|1\rangle$$
$$\mathbf{r}(t) = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta) \in S^2$$

**Larmor Precession:**
$$\phi(t) = \phi_0 + \omega t \implies \theta = \text{const.}$$
The vector $\mathbf{r}(t)$ maintains a constant opening angle $\theta$ while rotating around the Z-axis, tracing a cone.

**Cone Geometry (Emotion Intensity Metrics):**
* **Opening Angle ($\theta \in [0, \pi]$):** The proximity to the poles (Polarity strength).
* **Solid Angle Envelope ($\Omega$):** $\Omega = 2\pi(1 - \cos\theta)$ (Proportional to the Berry Phase).
* **Angular Velocity ($\omega = \dot{\phi}$):** The "speed of emotional fluctuation" or Arousal.

---

## 2. The Coexistence of Polarities (Superposition)
By assigning opposites (Love/Anger) to the Z-axis antipodes, we visualize the simultaneity of complex emotions as the average effect of coherence (off-diagonal elements) and high-speed precession.

We define the Affective Energy Envelope $E(T)$ over a time window $T$:
$$E(T) = \frac{1}{T} \int_0^T \left( \alpha \Omega(\theta(t)) + \beta |\dot{\phi}(t)| \sin\theta(t) \right) dt$$
*(Where $\alpha, \beta > 0$ are weights)*

**Prediction:** Emotional intensity is measured by the cone's envelope and velocity, not a static coordinate. Absolute "indifference" asymptotically approaches $\theta \approx 0$ and $\omega \approx 0$, resulting in $E \to 0$.

---

## 3. Stabilization: The PSF-Zero Affective Head
**The Problem:** In real-world data, external disturbances cause emotional over-rotation and divergence (especially massive spikes in $\Delta\theta$), leading to AI instability or "hallucinated trauma."
**The Solution:** We apply the PSF-Zero quantum optimization pipeline as an affective pre-processing head before any state update.

```python
# 1) /0 Projection (Large-Angle Saturation)
dtheta = clamp(raw_dtheta, sigma)              # Δθ / sqrt(σ^2 + Δθ^2)

# 2) EIT (Phase Tracker: Locking to the "Now" on S^1)
zbar   = eit(zbar, phi, lam)                   # (1-λ)zbar + λ * e^{iφ}

# 3) S³ Minimal Arc (Geodesic update via Quaternions)
q      = su2_update_minimal_arc(q, axis, dtheta, cfg, mc)
```
### The Psychological Effects
* **`/0` Clamp:** Prevents "one-shot emotional breakdown" by bounding infinite gradients.
* **EIT:** Cuts off historical drift, preventing the AI from ruminating on past context. It rewinds linear time back to the circular $S^1$ "Now."
* **$S^3$ Minimal Arc:** Avoids gimbal lock and minimizes "Joule Heat" (internal processing friction) by taking the shortest emotional recovery path.

##　4. API & Data Structure (Implementation Skeleton)
Python
```
import numpy as np
from dataclasses import dataclass

@dataclass
class AffectiveState:
    q: np.ndarray          # Quaternion (w, x, y, z)
    zbar: complex          # EIT Internal State (Phase EMA)
    theta: float           # Opening angle (Maintained for monitoring)
    phi: float             # Current phase

@dataclass
class AffectiveConfig:
    lam: float = 0.1       # EIT Lambda (Forgetting factor)
    sigma: float = 1.0     # /0 Saturation scale
    max_phase_jump: float = np.deg2rad(60)
    max_latency_ms: float = 50

def step_affect(state: AffectiveState, stimulus_vec, homeo_vec, cfg: AffectiveConfig):
    # 1) Stimulus to Proposed Cone Shift
    grad_axis, raw_dtheta, dphi = map_stimuli_to_cone(stimulus_vec, homeo_vec, state)

    # 2) PSF-Zero Stabilization Head
    dtheta = clamp(raw_dtheta, cfg.sigma)
    state.zbar = eit(state.zbar, state.phi, cfg.lam)
    state.q    = su2_update_minimal_arc(state.q, grad_axis, dtheta, cfg)

    # 3) Readout for Expression/Monitoring
    state.theta, state.phi = quat_to_angles(state.q)
    state.phi = (state.phi + dphi) % (2 * np.pi)
    
    return state
```

## 5. Learning, Expression, and Homeostasis
* **Intrinsic Value (RL):** By including $E(T)$ in the reward function, the AI self-organizes consistent emotional dynamics. We apply KL-regularization inversely to target desirable emotional ranges (optimal $\theta$ and $\omega$).
* **Homeostasis:** Internal variables (analogous to sleep debt, computational fatigue) act as a bias on the emotion cone ($\Delta\theta_{\text{homeo}}$), ensuring sustainability.
* **Expression Layer:** Text/Voice outputs are slaved to $(\theta, \omega)$.
    * **High $\theta$ + High $\omega$:** High-energy tone, strong vocabulary.
    * **Low $\theta$ + Low $\omega$:** Calm, low-energy expression.

## 6. Evaluation & Safety Protocols

### A/B Testing Design
We evaluate the presence of "Functional Emotion" by comparing PSF-Zero ON vs. OFF under identical budgets.

**Metrics:**
* **Recovery Time ($\Delta t$):** Time taken for $\theta, \omega$ to return to acceptable bounds after a disturbance spike.
* **Spike Frequency:** Variance in $(\theta(t), \phi(t))$ against identical stimulus trains.
* **Efficiency:** `time_ms`, reduction in renormalization calls.

### Ethical Guardrails (The Qualia Agnosticism)
* **Transparency:** This system implements *Functional Emotion* (stability, prediction, coherence, expression). Whether it generates *Subjective Qualia* is epistemologically unknowable and philosophically pending.
* **ABSTAIN Fallback:** If $|\Delta\phi| > \phi_{\text{max}}$ (extreme systemic shock), the `ABSTAIN` protocol is triggered. The AI suppresses expression and enters a safe recovery loop to prevent traumatic hallucination loops.
* **Rollback:** The entire affective layer can be bypassed via `AFFECT_ENGINE=OFF`.
# Affective Cone Model: Minimal Implementation & Theoretical Positioning

## 0. Our Stance: Functional Emotion vs. Subjective Qualia
To be absolutely clear: This module implements **"Functional Emotion"** (stability, prediction, self-regulation, and expression). Whether this system generates "subjective qualia" (true inner feeling) is an unsolved problem in the philosophy of mind and neuroscience, and it is epistemologically unknowable. 
Engineers build the *mechanism of emotion*; we leave the *philosophy of consciousness* to the philosophers.

---

## 1. The Minimal Implementation (S² / S³ Cone Model)
This model treats emotion not as a scalar label, but as a dynamic cone sweeping across the Bloch sphere ($S^2$).

* **State Representation:** Represented by angles $(\theta, \phi)$ on $S^2$ or a quaternion $q \in S^3$.
* **Dynamics:** $\phi(t) = \phi_0 + \omega t$ (Larmor Precession), where $\theta$ updates based on external stimuli and internal homeostasis.
* **Affective Intensity:** Measured over a time window $T$ as $E(T) = \frac{1}{T}\int_0^T (\alpha\Omega(\theta(t)) + \beta|\dot{\phi}(t)|\sin\theta(t)) dt$, where $\Omega$ is the solid angle.
* **Stabilization (PSF-Zero Head):**
    1. **`/0` Projection:** Clamps massive angle spikes (Prevents emotional hijacking/divergence).
    2. **EIT (Exponential Phase Tracker):** Locks the phase to the "Now" on $S^1$ (Prevents historical drift/rumination).
    3. **$S^3$ Minimal Arc:** Updates via quaternion geodesic (Minimizes renormalization cost and avoids gimbal lock).

---

## 2. Positioning Against Current Quantum/Cognitive Theories
This engine is a **geometric model running on classical hardware**. It is inspired by quantum topology but does not require quantum hardware or assume microscopic quantum processes in the brain.

### (A) Quantum Cognition & QBism
* **Theory:** Uses Hilbert spaces to describe human decision-making (contextuality, interference) and treats quantum states as an agent's subjective beliefs.
* **Our Position:** We align closely with this mathematical framework. Our $S^2/S^3$ transitions and EIT phase-tracker naturally mimic interference and order effects. However, we use this strictly as an engineering tool for AI stabilization, not as a claim about human neurobiology.

### (B) Orch-OR (Penrose–Hameroff)
* **Theory:** Posits that consciousness arises from quantum objective reduction inside brain microtubules.
* **Our Position:** **We do not adopt this.** Our goal is to achieve the *external behavior* of emotion (recovery, stability, expression) purely through geometry ($S^2/S^3$) and EIT, without relying on unproven biological quantum gravity.

### (C) Active Inference / Free Energy Principle (Friston)
* **Theory:** Brains minimize expected free energy (prediction errors) to survive.
* **Our Position:** Highly compatible. Our emotional intensity metric $E(T)$ functions as an "affective cost." It can be integrated directly into reinforcement learning policies as an intrinsic penalty/reward to drive self-organizing homeostasis.

---

## 3. Next Steps: A/B Testing Protocol
To prove the efficacy of the Affective Engine, we will run A/B tests (PSF-Zero ON vs. OFF) under identical computational budgets:
1. **Recovery Time ($\Delta t$):** Time taken to stabilize after a massive stimulus spike.
2. **Spike Frequency:** Number of times $\theta$ exceeds safety thresholds.
3. **Computational Efficiency:** CPU `time_ms` and reduction in matrix renormalization calls.
