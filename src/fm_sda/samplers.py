
import torch
from typing import Callable, Optional

def guided_vector_field(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    c: torch.Tensor,
    cfg_scale: float = 0.0,
    null_class_idx: Optional[int] = None
) -> torch.Tensor:
    """
    Computes guided vector field:
    v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
    """
    if cfg_scale == 0.0 or c is None:
        return model(x, t, c)
    
    # Unconditional input (with null class)
    if null_class_idx is None:
        # Assume valid null class exists in model
        if hasattr(model, 'num_classes') and model.num_classes is not None:
             null_class_idx = model.num_classes
        else:
            raise ValueError("Model does not have num_classes or null_class_idx not provided for CFG.")

    # Create batch with both cond and uncond
    # To save compute, can do 2 forward passes or 1 batch
    # Batching doubles memory but is faster? Let's do 2 passes for simplicity/memory balance
    
    # Conditional
    v_cond = model(x, t, c)
    
    # Unconditional
    c_uncond = torch.full_like(c, null_class_idx)
    v_uncond = model(x, t, c_uncond)
    
    # Guidance
    # v_guided = v_uncond + s * (v_cond - v_uncond)
    return v_uncond + cfg_scale * (v_cond - v_uncond)

def euler_solver(
    model: torch.nn.Module,
    x0: torch.Tensor,
    steps: int = 100,
    c: Optional[torch.Tensor] = None,
    cfg_scale: float = 0.0
) -> torch.Tensor:
    """
    Euler solver with optional CFG.
    """
    dt = 1.0 / steps
    x = x0
    device = x0.device
    
    for i in range(steps):
        t_val = i / steps
        t = torch.full((x.shape[0],), t_val, device=device)
        
        v = guided_vector_field(model, x, t, c, cfg_scale)
        x = x + v * dt
        
    return x

def rk4_solver(
    model: torch.nn.Module,
    x0: torch.Tensor,
    steps: int = 100,
    c: Optional[torch.Tensor] = None,
    cfg_scale: float = 0.0
) -> torch.Tensor:
    """
    RK4 solver with optional CFG.
    """
    dt = 1.0 / steps
    x = x0
    device = x0.device
    
    for i in range(steps):
        t_val = i / steps
        t = torch.full((x.shape[0],), t_val, device=device)
        
        # k1
        k1 = guided_vector_field(model, x, t, c, cfg_scale)
        
        # k2
        t_half_val = t_val + (dt / 2)
        t_half = torch.full((x.shape[0],), t_half_val, device=device)
        x_half_1 = x + k1 * (dt / 2)
        k2 = guided_vector_field(model, x_half_1, t_half, c, cfg_scale)
        
        # k3
        x_half_2 = x + k2 * (dt / 2)
        k3 = guided_vector_field(model, x_half_2, t_half, c, cfg_scale)
        
        # k4
        t_next_val = t_val + dt
        t_next = torch.full((x.shape[0],), t_next_val, device=device)
        x_next_est = x + k3 * dt
        k4 = guided_vector_field(model, x_next_est, t_next, c, cfg_scale)
        
        # Update
        x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        
    return x
