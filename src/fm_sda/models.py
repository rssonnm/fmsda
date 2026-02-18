
import math
import torch
import torch.nn as nn
from typing import List, Optional

def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        t: Tensor of shape (batch_size, 1) or (batch_size,) representing time in [0, 1].
        dim: Output dimension of the embedding.
        max_period: Maximum period for the embeddings.
        
    Returns:
        Tensor of shape (batch_size, dim).
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)
        
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
    ).to(t.device)
    
    args = t.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
    return embedding

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.
    Scale and shift are predicted from the embedding (time + class).
    """
    def __init__(self, num_features: int, emb_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_features, elementwise_affine=False)
        self.emb_proj = nn.Linear(emb_dim, num_features * 2) # Predict scale and shift
        
        # Initialize projection to output 0 scale/shift initially -> identity transform
        nn.init.zeros_(self.emb_proj.weight)
        nn.init.zeros_(self.emb_proj.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, features)
        emb: (batch, emb_dim)
        """
        scale_shift = self.emb_proj(emb)
        scale, shift = scale_shift.chunk(2, dim=1)
        # Reshape scale/shift to match x for broadcasting if x has more dims
        # For MLP: x is (B, D), scale is (B, D) -> OK
        # For DiT: x is (B, L, D), scale is (B, D) -> needs (B, 1, D)
        if x.dim() == 3:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
            
        x = self.norm(x) * (1 + scale) + shift
        return x

class ResidualBlock(nn.Module):
    """
    MLP Residual Block with AdaLN.
    """
    def __init__(self, in_features: int, out_features: int, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.lin1 = nn.Linear(in_features, out_features)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
        self.adaln1 = AdaLN(in_features, emb_dim)
        self.adaln2 = AdaLN(out_features, emb_dim)
        
        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # First layer
        h = self.adaln1(x, emb)
        h = self.act(h)
        h = self.lin1(h)
        
        # Second layer
        h = self.adaln2(h, emb)
        h = self.act(h)
        h = self.dropout(h)
        h = self.lin2(h)
        
        return h + self.skip(x)

class VelocityNetwork(nn.Module):
    """
    MLP-based Velocity Field v_theta(x, t, c).
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int] = [512, 512, 512], 
        time_emb_dim: int = 64,
        num_classes: Optional[int] = None,
        class_emb_dim: int = 32,
        use_cfg_token: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_emb_dim = time_emb_dim
        self.num_classes = num_classes
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Class Embedding (supports +1 for null class if using CFG)
        total_emb_dim = time_emb_dim
        if num_classes is not None:
            # Add 1 slot for unconditional embedding (CFG) if requested
            vocab_size = num_classes + 1 if use_cfg_token else num_classes
            self.class_emb = nn.Embedding(vocab_size, class_emb_dim)
            total_emb_dim += class_emb_dim
            
        self.emb_dim = total_emb_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # Residual Blocks
        self.blocks = nn.ModuleList()
        current_dim = hidden_dims[0]
        for h_dim in hidden_dims:
            self.blocks.append(ResidualBlock(current_dim, h_dim, total_emb_dim))
            current_dim = h_dim
            
        # Output projection
        self.output_norm = AdaLN(current_dim, total_emb_dim)
        self.output_proj = nn.Linear(current_dim, input_dim)
        
        # Initialize output projection to 0
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, D)
        t: (B,) or (B, 1)
        c: (B,) class labels (optional)
        """
        # Time Embedding
        t_emb = sinusoidal_embedding(t, self.time_emb_dim).to(x.device)
        t_emb = self.time_mlp(t_emb)
        
        # Class Embedding
        if self.num_classes is not None:
            if c is None:
                # If c is not provided but model expects it, use null token (index = num_classes)
                # Assuming training handles this, but here we enforce it
                c = torch.full((x.shape[0],), self.num_classes, device=x.device, dtype=torch.long)
                
            c_emb = self.class_emb(c)
            emb = torch.cat([t_emb, c_emb], dim=-1)
        else:
            emb = t_emb
            
        # Network
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, emb)
            
        h = self.output_norm(h, emb)
        return self.output_proj(h)


# ─── DiT-1D Implementation ───────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    DiT Block with AdaLN-Zero conditioning.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, emb_dim: int = 0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        
        # AdaLN-Zero modulation for (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        # 6 parameters per channel
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 6 * hidden_size, bias=True)
        )
        
        # Init AdaLN modulation to zero (identity)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        emb: (B, emb_dim)
        """
        # AdaLN modulation
        # (B, 6 * D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(emb).chunk(6, dim=1)
        )
        
        # Add singleton dim for broadcasting to (B, L, D)
        shift_msa, scale_msa, gate_msa = map(lambda t: t.unsqueeze(1), (shift_msa, scale_msa, gate_msa))
        shift_mlp, scale_mlp, gate_mlp = map(lambda t: t.unsqueeze(1), (shift_mlp, scale_mlp, gate_mlp))
        
        # Attention Block
        # Modulate
        x_mod = x * (1 + scale_msa) + shift_msa
        x_mod = self.norm1(x_mod)
        
        # Attention
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        
        # Gate & Skip
        x = x + gate_msa * attn_out
        
        # MLP Block
        # Modulate
        x_mod = x * (1 + scale_mlp) + shift_mlp
        x_mod = self.norm2(x_mod)
        
        # MLP
        mlp_out = self.mlp(x_mod)
        
        # Gate & Skip
        x = x + gate_mlp * mlp_out
        
        return x

class DiT1D(nn.Module):
    """
    Diffusion Transformer for 1D Data (DiT-1D).
    Treats the 1D signal as a sequence of patches (or tokens).
    If patch_size = 1, it's token-per-feature.
    """
    def __init__(
        self,
        input_dim: int,
        patch_size: int = 10, # depends on input_dim, e.g. 150 -> 15 patches
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        time_emb_dim: int = 256,
        num_classes: Optional[int] = None,
        class_emb_dim: int = 256, # usually matches time_emb_dim in DiT
    ):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        assert input_dim % patch_size == 0, f"Input dim {input_dim} must be divisible by patch size {patch_size}"
        self.num_patches = input_dim // patch_size
        
        # 1. Patch Embedding
        # Input: (B, input_dim) -> (B, num_patches, patch_size)
        # Linear projection: (B, num_patches, patch_size) -> (B, num_patches, hidden_size)
        self.patch_embed = nn.Linear(patch_size, hidden_size)
        
        # 2. Positional Embedding (Learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        
        # 3. Time & Class Embedding
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        if num_classes is not None:
            # +1 for null token (CFG)
            self.class_emb = nn.Embedding(num_classes + 1, time_emb_dim) # Add to time_emb_dim space
        
        # 4. DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, emb_dim=time_emb_dim)
            for _ in range(depth)
        ])
        
        # 5. Final Layer (AdaLN + Linear)
        self.final_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * hidden_size, bias=True) # AdaLN parameters for final layer
        )
        nn.init.zeros_(self.final_layer[1].weight)
        nn.init.zeros_(self.final_layer[1].bias)
        
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(hidden_size, patch_size) # Project back to patch pixels
        
        # Init final linear to zero
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, input_dim)
        t: (B,)
        c: (B,)
        """
        batch_size = x.shape[0]
        
        # 1. Prepare Embedding
        t_emb = sinusoidal_embedding(t, self.time_emb_dim).to(x.device)
        t_emb = self.time_mlp(t_emb)
        
        if self.num_classes is not None:
            if c is None:
                c = torch.full((batch_size,), self.num_classes, device=x.device, dtype=torch.long)
            c_emb = self.class_emb(c)
            # Standard DiT combines embeddings by summation
            emb = t_emb + c_emb
        else:
            emb = t_emb
            
        # 2. Patchify
        # (B, input_dim) -> (B, num_patches, patch_size)
        x = x.view(batch_size, self.num_patches, self.patch_size)
        
        # 3. Patch Embedding + Positional Embedding
        # (B, num_patches, hidden_size)
        x = self.patch_embed(x) + self.pos_embed
        
        # 4. Blocks
        for block in self.blocks:
            x = block(x, emb)
            
        # 5. Final Layer
        # Unpatchify logic inverse
        
        # AdaLN for final layer
        shift, scale = self.final_layer(emb).chunk(2, dim=1)
        # Broadcast to sequence length
        shift, scale = shift.unsqueeze(1), scale.unsqueeze(1)
        
        x = self.final_norm(x) * (1 + scale) + shift
        
        # Project back to patch size
        x = self.final_linear(x) # (B, num_patches, patch_size)
        
        # Flatten
        x = x.view(batch_size, self.input_dim)
        
        return x
