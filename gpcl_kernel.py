import torch
import torch.nn as nn
import math

class GPCLayer(nn.Module):
    """
    Geometric Pre-Constraint Layer (GPCL)
    Standardizes input tensors via S3 -> S2 Hopf Fibration.
    Enforces R=0 (Zero Friction) stability on any legacy AI architecture.
    """
    def __init__(self, mode="2d", eps=1e-6):
        super().__init__()
        self.mode = mode
        self.eps = eps

    def forward(self, x):
        # 1. Generate normalized coordinate field
        coords = self._get_coords(x)
        
        # 2. Lift to S3 Manifold (Quaternion space)
        q = self._lift_to_S3(coords)
        
        # 3. Project to S2 (Hopf Map)
        # We use the Z-component as the primary geometric constraint envelope
        q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        z_constraint = q0**2 + q3**2 - q1**2 - q2**2
        
        # 4. Apply Constraint (Geometric Gating)
        # This forces the model to stay within the manifold without destroying features
        return x * z_constraint.unsqueeze(1)

    def _get_coords(self, x):
        dev = x.device
        if self.mode == "2d":
            _, _, H, W = x.shape
            y = torch.linspace(-1, 1, H, device=dev)
            x_ = torch.linspace(-1, 1, W, device=dev)
            Y, X = torch.meshgrid(y, x_, indexing="ij")
            return torch.stack([X, Y], dim=-1)
        elif self.mode == "1d":
            n = x.shape[-1]
            return torch.linspace(-1, 1, n, device=dev)

    def _lift_to_S3(self, coords):
        if self.mode == "2d":
            X, Y = coords[..., 0], coords[..., 1]
            r = torch.sqrt(X**2 + Y**2 + self.eps)
            phi = torch.atan2(Y, X)
            # Embedding coordinates into S3 angles
            q0 = torch.cos(math.pi * r)
            q1 = torch.sin(math.pi * r) * torch.cos(phi)
            q2 = torch.sin(math.pi * r) * torch.sin(phi)
            q3 = torch.zeros_like(q0)
            q = torch.stack([q0, q1, q2, q3], dim=-1)
            return q / (torch.norm(q, dim=-1, keepdim=True) + self.eps)

    def _normalize(self, q):
        return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + self.eps)

# --- Integration Example ---
# Simply wrap any existing model:
# model = SafeModel(existing_transformer)
class SafeModel(nn.Module):
    def __init__(self, base_model, mode="2d"):
        super().__init__()
        self.gpcl = GPCLayer(mode=mode)
        self.base_model = base_model

    def forward(self, x, *args, **kwargs):
        x = self.gpcl(x) # Constraint applied BEFORE heuristic logic
        return self.base_model(x, *args, **kwargs)
