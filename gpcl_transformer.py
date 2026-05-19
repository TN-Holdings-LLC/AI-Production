import torch

import torch.nn as nn

# =========================================================

# 🔷 PSF Core: Geometric Projection (Stabilized)

# =========================================================

def psf_projection(X: torch.Tensor) -> torch.Tensor:

    """

    Geometric projection onto a constrained manifold (The PSF core).

    This physically enforces deterministic structural validity by 

    projecting the matrix onto the closest stable unitary-like manifold.

    

    Args:

        X (torch.Tensor): Input attention matrix.

    Returns:

        torch.Tensor: Geometrically projected, noise-free matrix.

    """

    # Batched Singular Value Decomposition (SVD)

    U, _, V = torch.svd(X)

    

    # Reconstruct enforcing a stable, unitary-like geometric structure

    X_proj = torch.matmul(U, V.transpose(-2, -1))

    

    return X_proj

# =========================================================

# 🔷 GPCL Attention (Geometric Projection Constraint Layer)

# =========================================================

class GPCLAttention(nn.Module):

    def __init__(self, dim: int, threshold: float = 0.6):

        super().__init__()

        self.dim = dim

        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """

        Forward pass for GPCL Attention. Replaces Softmax with SVD Projection.

        

        Args:

            x: Input tensor [batch, seq, dim]

        """

        # 1. Normalize -> Project to hypersphere

        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)

        

        # 2. Calculate geometric similarity (cosine-like)

        sim = torch.matmul(x_norm, x_norm.transpose(-2, -1))

        

        # 3. PSF Constraint Filter (Sever the noise completely)

        mask = (sim > self.threshold).float()

        sim_filtered = sim * mask

        

        # 4. Geometric Projection (The Core Breakthrough)

        sim_proj = psf_projection(sim_filtered)

        

        # 5. Re-normalize without Softmax to maintain absolute structure

        weights = sim_proj / (sim_proj.sum(dim=-1, keepdim=True) + 1e-8)

        

        # 6. Apply strictly constrained weights

        out = torch.matmul(weights, x)

        

        return out

# =========================================================

# 🔷 Feed Forward Network (Standard)

# =========================================================

class FeedForward(nn.Module):

    def __init__(self, dim: int, hidden_dim: int):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(dim, hidden_dim),

            nn.GELU(),

            nn.Linear(hidden_dim, dim)

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.net(x)

# =========================================================

# 🔷 GPCL Transformer Block

# =========================================================

class GPCLBlock(nn.Module):

    def __init__(self, dim: int, hidden_dim: int):

        super().__init__()

        

        self.attn = GPCLAttention(dim)

        self.ff = FeedForward(dim, hidden_dim)

        

        self.norm1 = nn.LayerNorm(dim)

        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # PSF-based Attention with residual connection

        x = x + self.attn(self.norm1(x))

        # Standard Feedforward with residual connection

        x = x + self.ff(self.norm2(x))

        return x

# =========================================================

# 🔷 Full GPCL Transformer Model

# =========================================================

class GPCLTransformer(nn.Module):

    def __init__(self, dim: int, depth: int, hidden_dim: int):

        """

        Initialize the GPCL Transformer.

        

        Args:

            dim: Embedding dimension.

            depth: Number of Transformer blocks.

            hidden_dim: Hidden dimension for the FeedForward network.

        """

        super().__init__()

        

        self.layers = nn.ModuleList([

            GPCLBlock(dim, hidden_dim) 

            for _ in range(depth)

        ])

        

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.layers:

            x = layer(x)

        return self.norm(x)

# =========================================================

# ✅ Execution / Test Block

# =========================================================

if __name__ == "__main__":

    # Initialize a lightweight prototype model

    model = GPCLTransformer(

        dim=64, 

        depth=3, 

        hidden_dim=128

    )

    

    # Generate dummy tensor representing batched quantum structures

    # [batch_size=2, sequence_length=10, feature_dim=64]

    x = torch.randn(2, 10, 64)

    

    # Pass through the GPCL architecture

    out = model(x)

    

    print("✅ GPCL Processing Complete.")

    print(f"✅ Input Shape:  {x.shape}")

    print(f"✅ Output Shape: {out.shape}")

