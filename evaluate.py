import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import custom modules
from src.dataset import EnzymeSubstrateDataset, collate_fn
from src.models import HeteroscedasticEnzymeModel
from src.metrics import compute_regression_metrics

def evaluate_ood():
    # 1. Configuration
    BATCH_SIZE = 16
    BASE_DIR = "."
    DATA_MASTER = os.path.join(BASE_DIR, "datasets", "processed", "kcat_max_wt_singleSeqs_wpdbs.csv")
    
    # We choose the most rigorous OOD split: 40% sequence identity maximum
    TEST_SPLIT = os.path.join(BASE_DIR, "datasets", "splits", "kcat-seq_test_sequence_40cluster.csv")
    MODEL_WEIGHTS = os.path.join(BASE_DIR, "checkpoints", "best_multimodal_model.pt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Out-of-Distribution (OOD) Evaluation on: {device}")
    print(f"[*] Test Split: {os.path.basename(TEST_SPLIT)}")

    # 2. Load Dataset
    test_dataset = EnzymeSubstrateDataset(DATA_MASTER, TEST_SPLIT)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    print(f"[-] OOD Test samples: {len(test_dataset)}")

    # 3. Initialize Model and Load Best Weights
    model = HeteroscedasticEnzymeModel().to(device)
    
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Cannot find trained weights at {MODEL_WEIGHTS}. Did you run train.py?")
        
    # Load the state dict saved during training
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
    model.eval()
    print("[*] Best model weights loaded successfully!")

    # 4. Inference Loop
    all_targets = []
    all_preds = []
    
    test_pbar = tqdm(test_loader, desc="Evaluating OOD")
    with torch.no_grad():
        for batch_seqs, batch_smiles, batch_targets in test_pbar:
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            predicted_means, predicted_logvars = model(batch_seqs, batch_smiles, device)
            
            all_targets.extend(batch_targets.cpu().numpy())
            all_preds.extend(predicted_means.cpu().numpy())
            
    # 5. Calculate and Output Final Metrics
    print("\n" + "="*40)
    print("🏆 FINAL OOD TEST SET PERFORMANCE 🏆")
    print("="*40)
    metrics = compute_regression_metrics(all_targets, all_preds)
    
    print(f"-> R-Squared (R2)      : {metrics['R2']:.4f}")
    print(f"-> Pearson Correlation : {metrics['Pearson']:.4f}")
    print(f"-> RMSE                : {metrics['RMSE']:.4f}")
    print(f"-> MAE                 : {metrics['MAE']:.4f}")
    print("="*40)
    print("Ready to copy into your course project report!")

if __name__ == "__main__":
    evaluate_ood()