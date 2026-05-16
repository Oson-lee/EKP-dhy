import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import our custom modules from the 'src' directory
from src.dataset import EnzymeSubstrateDataset, collate_fn
from src.models import HeteroscedasticEnzymeModel
from src.losses import HeteroscedasticGaussianNLLLoss
from src.metrics import compute_regression_metrics

def main():
    # -------------------------------------------------------------------------
    # 1. Configuration and Hyperparameters
    # -------------------------------------------------------------------------
    EPOCHS = 20
    BATCH_SIZE = 16  # Adjust based on your GPU VRAM (16-32 is usually safe)
    LEARNING_RATE = 1e-3 
    
    # Path setup based on the documented directory architecture
    BASE_DIR = "."
    DATA_MASTER = os.path.join(BASE_DIR, "datasets", "processed", "kcat_max_wt_singleSeqs_wpdbs.csv")
    TRAIN_SPLIT = os.path.join(BASE_DIR, "datasets", "splits", "kcat-random_train.csv")
    VAL_SPLIT   = os.path.join(BASE_DIR, "datasets", "splits", "kcat-random_val.csv")
    
    # Create directory for saving model checkpoints
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Multi-Modal Training Pipeline on: {device}")

    # -------------------------------------------------------------------------
    # 2. Initialize DataLoaders
    # -------------------------------------------------------------------------
    print("[*] Loading Datasets...")
    train_dataset = EnzymeSubstrateDataset(DATA_MASTER, TRAIN_SPLIT)
    val_dataset = EnzymeSubstrateDataset(DATA_MASTER, VAL_SPLIT)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print(f"[-] Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # -------------------------------------------------------------------------
    # 3. Initialize Model, Loss, Optimizer, and Scheduler
    # -------------------------------------------------------------------------
    model = HeteroscedasticEnzymeModel().to(device)
    criterion = HeteroscedasticGaussianNLLLoss()
    
    # Convert to list so it can be reused by both Optimizer and Gradient Clipping
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=1e-4)
    
    # [Claude's Awesome Suggestion]: Reduce LR when validation R2 stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # -------------------------------------------------------------------------
    # 4. Main Training Loop
    # -------------------------------------------------------------------------
    best_val_r2 = -float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*20} Epoch {epoch}/{EPOCHS} {'='*20}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[*] Current Learning Rate: {current_lr}")
        
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_seqs, batch_smiles, batch_targets in train_pbar:
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted_means, predicted_logvars = model(batch_seqs, batch_smiles, device)
            
            # Calculate Loss
            loss = criterion(predicted_means, predicted_logvars, batch_targets)
            
            # Backward pass
            loss.backward()
            
            # [Claude's Awesome Suggestion]: Gradient Clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            # Optimization step
            optimizer.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0.0
        all_val_targets = []
        all_val_preds = []
        
        val_pbar = tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for batch_seqs, batch_smiles, batch_targets in val_pbar:
                batch_targets = batch_targets.to(device)
                
                # Forward pass
                predicted_means, predicted_logvars = model(batch_seqs, batch_smiles, device)
                loss = criterion(predicted_means, predicted_logvars, batch_targets)
                
                total_val_loss += loss.item()
                
                # Collect predictions and targets for metric calculation
                all_val_targets.extend(batch_targets.cpu().numpy())
                all_val_preds.extend(predicted_means.cpu().numpy())
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Calculate human-readable regression metrics
        metrics = compute_regression_metrics(all_val_targets, all_val_preds)
        
        # Print Epoch Summary
        print(f"[Train] NLL Loss: {avg_train_loss:.4f}")
        print(f"[Valid] NLL Loss: {avg_val_loss:.4f} | R2: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.4f} | Pearson: {metrics['Pearson']:.4f}")
        
        # Update Learning Rate Scheduler based on Validation R2 score
        scheduler.step(metrics['R2'])
        
        # --- MODEL CHECKPOINTING ---
        if metrics['R2'] > best_val_r2:
            best_val_r2 = metrics['R2']
            save_path = os.path.join(CHECKPOINT_DIR, "best_multimodal_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"[***] New best model saved! (Validation R2 improved to {best_val_r2:.4f})")

    print("\n[*] Training Complete. Best model weights are stored in the 'checkpoints' directory.")

if __name__ == "__main__":
    main()