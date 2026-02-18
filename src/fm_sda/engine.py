
import math
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment

class FlowMatchingTrainer:
    def __init__(self, model: nn.Module, device: torch.device, sigma_min: float = 1e-4):
        self.model = model
        self.device = device
        self.sigma_min = sigma_min
        self.model.to(device)

    def ot_coupling(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimal Transport Coupling using Hungarian Algorithm.
        """
        with torch.no_grad():
            x0_flat = x0.flatten(start_dim=1)
            x1_flat = x1.flatten(start_dim=1)
            
            cost_matrix = torch.cdist(x0_flat, x1_flat, p=2) ** 2
            cost_matrix_np = cost_matrix.cpu().numpy()
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
            
            x1_aligned = x1[col_ind]
            
            return x0, x1_aligned

    def compute_loss(self, x1: torch.Tensor, c: torch.Tensor = None, p_uncond: float = 0.0) -> torch.Tensor:
        """
        Compute Conditional Flow Matching Loss with CFG support.
        """
        batch_size = x1.shape[0]
        
        # 1. Sample t ~ Uniform[0, 1]
        t = torch.rand(batch_size, device=self.device)
        
        # 2. Sample x0 ~ N(0, I)
        x0 = torch.randn_like(x1)
        
        # 3. OT Coupling
        x0, x1 = self.ot_coupling(x0, x1)
        
        # 4. Interpolant: x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
        t_b = t.view(batch_size, *([1] * (x1.ndim - 1)))
        x_t = (1 - (1 - self.sigma_min) * t_b) * x0 + t_b * x1
        
        # Conditional Vector Field (Target)
        target_v = x1 - (1 - self.sigma_min) * x0
        
        # 5. Classifier-Free Guidance Training
        # Mask class labels with probability p_uncond
        if c is not None and p_uncond > 0:
            mask = torch.rand(batch_size, device=self.device) < p_uncond
            # Assuming model handles null class as last index
            # We need access to num_classes to set null token
            # But the model knows it. We just need to pass a special value or let model handle masking?
            # Better: Pass c directly, but modify c indices for masked ones.
            # Assuming models.py added +1 to num_classes.
            # The null class index is `model.num_classes`.
            
            # Check if model has num_classes attribute
            if hasattr(self.model, 'num_classes') and self.model.num_classes is not None:
                null_class = self.model.num_classes
                # Create a copy so we don't modify original tensor
                c_train = c.clone()
                c_train[mask] = null_class
            else:
                c_train = c
        else:
            c_train = c
        
        # 6. Predict Vector Field
        pred_v = self.model(x_t, t, c_train)
        
        # 7. Loss: MSE
        loss = torch.mean((pred_v - target_v) ** 2)
        
        return loss

    def train_step(self, optimizer: torch.optim.Optimizer, x1: torch.Tensor, c: torch.Tensor = None, p_uncond: float = 0.1) -> float:
        optimizer.zero_grad()
        loss = self.compute_loss(x1, c, p_uncond=p_uncond)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        return loss.item()
