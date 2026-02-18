
import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from fm_sda import models, engine

def train_reflow(args):
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
        
    # Load Reflow Data
    print(f"Loading reflow data from {args.data_path}...")
    data = torch.load(args.data_path)
    z0 = data['z0']
    z1 = data['z1']
    c = data['c']
    
    dataset = TensorDataset(z0, z1, c)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    input_dim = z0.shape[1]
    # Estimate num_classes from data or args
    num_classes = int(c.max().item()) + 1
    
    print(f"Data shape: {input_dim}, Samples: {len(z0)}")

    # Model Initialization (Model 2)
    # Can use MLP or DiT
    if args.use_dit:
        print("Initializing DiT-1D Model...")
        velocity_net = models.DiT1D(
            input_dim=input_dim,
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            time_emb_dim=args.time_dim,
            num_classes=num_classes, # Will add +1 internally
            class_emb_dim=args.time_dim
        )
    else:
        print("Initializing MLP Model...")
        velocity_net = models.VelocityNetwork(
            input_dim=input_dim,
            hidden_dims=args.hidden_dims,
            time_emb_dim=args.time_dim,
            num_classes=num_classes, # Will add +1 internally
            class_emb_dim=args.class_dim
        )
            
    optimizer = torch.optim.AdamW(velocity_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Trainer
    # For Reflow, the target is FIXED: v_target = z1 - z0
    # The path is straight: x_t = (1-t)z0 + t*z1
    # We can reuse FlowMatchingTrainer but need to bypass OT coupling and use (z0, z1) pairs explicitly.
    # Let's write a custom loop here since logic is slightly different (no OT, explicit pairs).
    
    velocity_net.to(device)
    velocity_net.train()
    
    print("Starting Reflow training...")
    loss_history = []
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Reflow Epoch {epoch+1}/{args.epochs}", leave=False)
        
        for batch_z0, batch_z1, batch_c in pbar:
            batch_z0, batch_z1, batch_c = batch_z0.to(device), batch_z1.to(device), batch_c.to(device)
            
            optimizer.zero_grad()
            
            # Reflow Logic
            batch_size = batch_z0.shape[0]
            t = torch.rand(batch_size, device=device)
            t_b = t.view(batch_size, *([1] * (batch_z0.ndim - 1)))
            
            # Straight path
            x_t = (1 - t_b) * batch_z0 + t_b * batch_z1
            target_v = batch_z1 - batch_z0
            
            # CFG Training
            if args.p_uncond > 0:
                mask = torch.rand(batch_size, device=device) < args.p_uncond
                batch_c_train = batch_c.clone()
                # Null class is at index num_classes (since model has num_classes+1 embeddings)
                # But wait, models.DiT1D/VelocityNetwork take 'num_classes' in init and creating 'num_classes + 1' embedding slots.
                # So indices 0..num_classes-1 are valid. Index num_classes is null.
                batch_c_train[mask] = num_classes
            else:
                batch_c_train = batch_c
                
            pred_v = velocity_net(x_t, t, batch_c_train)
            
            loss = torch.mean((pred_v - target_v) ** 2)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(velocity_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss:.6f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f}")
            
    # Save Reflow Model
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "reflow_model.pth")
    
    torch.save({
        'model_state_dict': velocity_net.state_dict(),
        'args': args,
        'num_classes': num_classes,
        'input_dim': input_dim
    }, save_path)
    
    print(f"Reflow model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to generated Reflow data")
    parser.add_argument("--save_dir", type=str, default="../checkpoints", help="Save directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    # Model Args
    parser.add_argument("--use_dit", action="store_true", help="Use DiT instead of MLP")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--patch_size", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--time_dim", type=int, default=256)
    parser.add_argument("--class_dim", type=int, default=32)
    
    # CFG
    parser.add_argument("--p_uncond", type=float, default=0.1, help="Probability of unconditional training")
    
    args = parser.parse_args()
    train_reflow(args)
