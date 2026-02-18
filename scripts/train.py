
import os
import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Hack to make imports work without installing the package
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from fm_sda import models, engine, data

def train(args):
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = "cpu"
        print("Using CPU device.")
        
    # Data Loading
    print(f"Loading data from {args.data_path}...")
    dataset = data.SpectralDataset(
        args.data_path, 
        apply_derivative=args.derivative,
        normalize=True
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"Data shape: {dataset.input_dim} features, {len(dataset)} samples.")
    # Handle label encoder class access safely
    classes = dataset.label_encoder.classes_ if hasattr(dataset, 'label_encoder') else ['Unknown']
    print(f"Classes: {dataset.num_classes} ({classes})")

    # Model Initialization
    velocity_net = models.VelocityNetwork(
        input_dim=dataset.input_dim,
        hidden_dims=args.hidden_dims,  # type: ignore
        time_emb_dim=args.time_dim,
        num_classes=dataset.num_classes,
        class_emb_dim=args.class_dim
    )
    
    optimizer = torch.optim.AdamW(velocity_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    trainer = engine.FlowMatchingTrainer(
        model=velocity_net,
        device=device,
        sigma_min=args.sigma_min
    )
    
    # Training Loop
    print("Starting training...")
    loss_history = []
    
    velocity_net.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        # Use simple loop if tqdm fails, but usually fine
        # Using enumerate to avoid issues
        batch_count = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward + Backward
            optimizer.zero_grad()
            loss = trainer.compute_loss(batch_x, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(velocity_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
        avg_loss = epoch_loss / batch_count
        loss_history.append(avg_loss)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            
    # Save Model
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "fm_sda_model.pth")
    
    # Save full checkpoint
    checkpoint = {
        'model_state_dict': velocity_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'input_dim': dataset.input_dim,
        'hidden_dims': args.hidden_dims,
        'time_dim': args.time_dim,
        'class_dim': args.class_dim,
        'num_classes': dataset.num_classes,
        'classes': classes
    }
    torch.save(checkpoint, save_path)
    
    # Save scaler separately using pickle for easy loading
    with open(os.path.join(args.save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(dataset.scaler, f)
        
    print(f"Model saved to {save_path}")
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CFM Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.save_dir, "loss_curve.png"))
    print("Loss curve saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flow Matching for Spectral Data")
    parser.add_argument("--data_path", type=str, default="../data/raw/UV_Vis_NuocCam_ChuaDaoHam.csv", help="Path to CSV data")
    parser.add_argument("--save_dir", type=str, default="../checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 512, 512], help="Hidden dimensions")
    parser.add_argument("--time_dim", type=int, default=64, help="Time embedding dimension")
    parser.add_argument("--class_dim", type=int, default=32, help="Class embedding dimension")
    parser.add_argument("--sigma_min", type=float, default=1e-4, help="Sigma min for OT path")
    parser.add_argument("--derivative", action="store_true", default=True, help="Apply derivative preprocessing")
    
    args = parser.parse_args()
    train(args)
