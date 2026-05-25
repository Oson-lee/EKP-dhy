import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import EnzymeSubstrateDataset, collate_fn
from src.models import HeteroscedasticEnzymeModel
from src.losses import HeteroscedasticGaussianNLLLoss
from src.metrics import compute_regression_metrics

def main():
    # =========================================================================
    # 🎯 CENTRAL SWITCH PANEL (Toggle configurations across targets and scopes)
    # =========================================================================
    TARGET_TASK = "kcat"       # Target Endpoint Matrix. Options: "kcat", "km", "ki"
    MODEL_MODE  = "seq_only"    # Architecture Strategy. Options: "seq_only", "bimodal", "trimodal"
    # =========================================================================
    
    EPOCHS = 20
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    
    # Automated routing infrastructure verified against physical file records
    DATA_ROUTING = {
        "kcat": {
            "master": "datasets/processed/kcat_max_wt_singleSeqs_wpdbs.csv",
            "train": "datasets/splits/kcat-random_train.csv",
            "val": "datasets/splits/kcat-random_val.csv"
        },
        "km": {
            "master": "datasets/processed/km_mean_wt_singleSeqs_wpdbs.csv", 
            "train": "datasets/splits/km-random_train.csv",
            "val": "datasets/splits/km-random_val.csv"
        },
        "ki": {
            "master": "datasets/processed/ki_mean_wt_singleSeqs_wpdbs.csv",
            "train": "datasets/splits/ki-random_train.csv",
            "val": "datasets/splits/ki-random_val.csv"
        }
    }
    
    cfg = DATA_ROUTING[TARGET_TASK]
    CHECKPOINT_DIR = "./checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'*'*60}\n[*] Training Booted | Target: {TARGET_TASK.upper()} | Strategy: {MODEL_MODE.upper()} | Backend: {device}\n{'*'*60}")

    # Initialize targeted standard loaders
    train_dataset = EnzymeSubstrateDataset(cfg["master"], cfg["train"])
    val_dataset = EnzymeSubstrateDataset(cfg["master"], cfg["val"])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize model structures and standard heteroscedastic criterion
    model = HeteroscedasticEnzymeModel().to(device)
    criterion = HeteroscedasticGaussianNLLLoss()
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_r2 = -float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        
        # --- TRAINING STAGE ---
        model.train()
        total_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc="Training Optimization", leave=False)
        for batch_seqs, batch_smiles, batch_paths, batch_targets in train_pbar:
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            
            means, logvars = model(batch_seqs, batch_smiles, batch_paths, device, modality=MODEL_MODE)
            loss = criterion(means, logvars, batch_targets)
            loss.backward()
            
            # Clip gradients to guarantee numerical boundaries for variance values
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        # --- VALIDATION STAGE ---
        model.eval()
        total_val_loss = 0.0
        all_val_targets, all_val_preds = [], []
        with torch.no_grad():
            for batch_seqs, batch_smiles, batch_paths, batch_targets in val_loader:
                batch_targets = batch_targets.to(device)
                means, logvars = model(batch_seqs, batch_smiles, batch_paths, device, modality=MODEL_MODE)
                loss = criterion(means, logvars, batch_targets)
                total_val_loss += loss.item()
                
                all_val_targets.extend(batch_targets.cpu().numpy())
                all_val_preds.extend(means.cpu().numpy())
                
        metrics = compute_regression_metrics(all_val_targets, all_val_preds)
        print(f"[Valid] Loss: {total_val_loss/len(val_loader):.4f} | R2: {metrics['R2']:.4f} | Pearson: {metrics['Pearson']:.4f}")
        
        scheduler.step(metrics['R2'])
        
        # Unique naming matrix to catalog different models independently
        if metrics['R2'] > best_val_r2:
            best_val_r2 = metrics['R2']
            save_name = f"best_model_{TARGET_TASK}_{MODEL_MODE}.pt"
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, save_name))
            print(f"[***] Extracted Optimal Weights -> Saved as {save_name}")

    print(f"\n[+] Task pipeline completed successfully.")

if __name__ == "__main__":
    main()