
import os
import argparse
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from fm_sda import models, samplers, data

# Plotting settings for Q1 Journal Quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300

def visualize(args):
    # Device
    device = torch.device(args.device) if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load Model Checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = checkpoint.get('args') or args 
    input_dim = checkpoint.get('input_dim', 150) # Fallback if not saved
    num_classes = checkpoint.get('num_classes', 5)
    
    # Check for DiT or MLP
    # Simple check based on keys
    is_dit = 'final_layer.1.weight' in checkpoint['model_state_dict']
    
    if is_dit:
        print("Detected DiT Model.")
        # Need to infer or get DiT args. 
        # For now assume default if not present, but safer if saved.
        # This script might fail if DiT args aren't saved.
        # Let's hope checkpoint has 'args' with DiT params.
        # If not, use defaults or error out.
        velocity_net = models.DiT1D(
            input_dim=input_dim,
            # Use args from checkpoint if available, else CLI args
            patch_size=getattr(model_args, 'patch_size', 10),
            hidden_size=getattr(model_args, 'hidden_size', 384),
            depth=getattr(model_args, 'depth', 12),
            num_heads=getattr(model_args, 'num_heads', 6),
            time_emb_dim=getattr(model_args, 'time_dim', 256),
            num_classes=num_classes,
            class_emb_dim=getattr(model_args, 'time_dim', 256)
        )
    else:
        print("Detected MLP Model.")
        velocity_net = models.VelocityNetwork(
            input_dim=input_dim,
            hidden_dims=checkpoint['hidden_dims'] if 'hidden_dims' in checkpoint else [512, 512, 512],
            time_emb_dim=checkpoint['time_dim'] if 'time_dim' in checkpoint else 64,
            num_classes=num_classes,
            class_emb_dim=checkpoint['class_dim'] if 'class_dim' in checkpoint else 32
        )
        
    velocity_net.load_state_dict(checkpoint['model_state_dict'])
    velocity_net.to(device)
    velocity_net.eval()
    
    print("Model loaded successfully.")

    # Load Scaler
    scaler_path = os.path.join(os.path.dirname(args.checkpoint), "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        print("Warning: scaler.pkl not found. Visualization will be on normalized data.")
        scaler = None

    # Load Real Data for comparison
    print(f"Loading real data from {args.data_path}...")
    dataset = data.SpectralDataset(
        args.data_path, 
        apply_derivative=args.derivative,
        normalize=True
    )
    real_data = dataset.spectral_data.numpy()
    real_labels = dataset.labels.numpy()
    classes = dataset.label_encoder.classes_ if hasattr(dataset, 'label_encoder') else range(num_classes)

    # 1. Standard Generation (Comparison)
    if not args.cfg_sweep:
        print("Generating synthetic data (Standard)...")
        synthetic_data_list = []
        synthetic_labels_list = []
        
        samples_per_class = args.samples_per_class
        
        with torch.no_grad():
            for c in range(num_classes):
                x0 = torch.randn(samples_per_class, input_dim, device=device)
                c_tensor = torch.full((samples_per_class,), c, device=device, dtype=torch.long)
                
                # Use guided solver with cfg = args.cfg_scale
                x1 = samplers.rk4_solver(velocity_net, x0, steps=args.steps, c=c_tensor, cfg_scale=args.cfg_scale)
                
                synthetic_data_list.append(x1.cpu().numpy())
                synthetic_labels_list.append(np.full(samples_per_class, c))
                
        synthetic_data = np.concatenate(synthetic_data_list, axis=0)
        synthetic_labels = np.concatenate(synthetic_labels_list, axis=0)
        
        # Inverse Transform
        if scaler:
            real_data_orig = scaler.inverse_transform(real_data)
            synthetic_data_orig = scaler.inverse_transform(synthetic_data)
        else:
            real_data_orig = real_data
            synthetic_data_orig = synthetic_data
        
        # Plot Spectra
        os.makedirs(args.results_dir, exist_ok=True)
        for i, class_name in enumerate(classes):
            idx_real = real_labels == i
            idx_syn = synthetic_labels == i
            
            if np.sum(idx_real) == 0: continue
            
            real_mean = np.mean(real_data_orig[idx_real], axis=0)
            real_std = np.std(real_data_orig[idx_real], axis=0)
            syn_mean = np.mean(synthetic_data_orig[idx_syn], axis=0)
            syn_std = np.std(synthetic_data_orig[idx_syn], axis=0)
            
            plt.figure(figsize=(10, 6))
            x_axis = np.arange(len(real_mean))
            plt.plot(x_axis, real_mean, 'b-', label='Real', linewidth=2)
            plt.fill_between(x_axis, real_mean - real_std, real_mean + real_std, color='b', alpha=0.1)
            plt.plot(x_axis, syn_mean, 'r--', label='Reflow/CFG', linewidth=2)
            plt.fill_between(x_axis, syn_mean - syn_std, syn_mean + syn_std, color='r', alpha=0.1)
            plt.title(f"Spectral Comparison - {class_name} (CFG={args.cfg_scale})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_dir, f"spectra_{class_name}_cfg{args.cfg_scale}.png"))
            plt.close()

    # 2. CFG Sweep Visualization
    if args.cfg_sweep:
        print("Running CFG Sweep...")
        sweep_scales = [0.0, 1.0, 2.5, 5.0]
        # Pick one class to visualize
        target_class = 0 
        class_name = classes[target_class]
        
        plt.figure(figsize=(12, 8))
        
        # Plot Real Mean
        idx_real = real_labels == target_class
        if scaler:
            real_data_viz = scaler.inverse_transform(real_data[idx_real])
        else:
            real_data_viz = real_data[idx_real]
        real_mean = np.mean(real_data_viz, axis=0)
        plt.plot(real_mean, 'k-', linewidth=3, label='Real Data', alpha=0.6)
        
        with torch.no_grad():
            x0 = torch.randn(100, input_dim, device=device) # Fixed noise for fair comparison
            c_tensor = torch.full((100,), target_class, device=device, dtype=torch.long)
            
            for scale in sweep_scales:
                x1 = samplers.rk4_solver(velocity_net, x0, steps=args.steps, c=c_tensor, cfg_scale=scale)
                x1_np = x1.cpu().numpy()
                if scaler:
                    x1_np = scaler.inverse_transform(x1_np)
                
                mean_signal = np.mean(x1_np, axis=0)
                plt.plot(mean_signal, label=f'CFG Scale {scale}')
                
        plt.title(f"Effect of Classifier-Free Guidance - Class: {class_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.results_dir, "cfg_sweep_analysis.png"))
        plt.close()
        print("CFG Sweep plot saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default="../data/raw/UV_Vis_NuocCam_ChuaDaoHam.csv", help="Path to real data")
    parser.add_argument("--results_dir", type=str, default="../results", help="Directory to save plots")
    parser.add_argument("--samples_per_class", type=int, default=50, help="Number of samples to generate per class")
    parser.add_argument("--steps", type=int, default=50, help="Number of ODE steps")
    parser.add_argument("--device", type=str, default="mps", help="Device to use")
    parser.add_argument("--derivative", action="store_true", default=True, help="Apply derivative preprocessing")
    
    # CFG Args
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="CFG scale (0.0 = standard conditional)")
    parser.add_argument("--cfg_sweep", action="store_true", help="Run CFG sweep visualization")
    
    args = parser.parse_args()
    visualize(args)
