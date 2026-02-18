
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import pickle

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from fm_sda import models, samplers, data

def generate_reflow_data(args):
    # Device
    device = torch.device(args.device) if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load Model 1
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    input_dim = checkpoint.get('input_dim', 150)
    num_classes = checkpoint.get('num_classes', 5)
    
    # Determine model type (MLP vs DiT)
    # Check args in checkpoint or infer?
    # For now assume matches current code structure (VelocityNetwork is MLP)
    # If we switch to DiT later, we need logic here.
    # Let's try to load as VelocityNetwork (MLP) first.
    
    # FUTURE PROOF: Check if DiT
    is_dit = 'DiT' in str(type(models.DiT1D)) # Just a check, implementation logic below
    # Actually just reuse the model class from checkpoint args if possible?
    # Simpler: Try loading state dict.
    
    # We will assume MLP for now as per previous phase, 
    # unless we restart training with DiT.
    # The user instruction was to implement DiT, but we haven't trained it yet.
    # So this script is for generating reflow data from *whatever model* is in the checkpoint.
    
    model = models.VelocityNetwork(
        input_dim=input_dim,
        hidden_dims=checkpoint['hidden_dims'],
        time_emb_dim=checkpoint['time_dim'],
        num_classes=num_classes,
        class_emb_dim=checkpoint['class_dim'],
        use_cfg_token=False # Old model trained without CFG token
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model 1 loaded.")

    # Generate Data
    print(f"Generating {args.num_samples} pairs...")
    
    z0_list = []
    z1_list = []
    c_list = []
    
    batch_size = args.batch_size
    num_batches = args.num_samples // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches)):
            # Sample z0 ~ N(0, I)
            z0 = torch.randn(batch_size, input_dim, device=device)
            
            # Sample random classes
            c = torch.randint(0, num_classes, (batch_size,), device=device)
            
            # z1 = ODE(z0, 0 -> 1)
            # Use high precision solver (RK4) for ground truth pairing
            z1 = samplers.rk4_solver(model, z0, steps=args.steps, c=c)
            
            z0_list.append(z0.cpu())
            z1_list.append(z1.cpu())
            c_list.append(c.cpu())
            
    # Concatenate
    z0_all = torch.cat(z0_list, dim=0)
    z1_all = torch.cat(z1_list, dim=0)
    c_all = torch.cat(c_list, dim=0)
    
    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "reflow_data.pt")
    
    torch.save({
        'z0': z0_all,
        'z1': z1_all,
        'c': c_all,
        'model1_checkpoint': args.checkpoint
    }, save_path)
    
    print(f"Reflow dataset saved to {save_path}")
    print(f"Shape: {z0_all.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Model 1 checkpoint")
    parser.add_argument("--save_dir", type=str, default="../data/processed", help="Directory to save reflow data")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--steps", type=int, default=100, help="ODE steps for generation")
    parser.add_argument("--device", type=str, default="mps", help="Device")
    
    args = parser.parse_args()
    generate_reflow_data(args)
