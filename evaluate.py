import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import EnzymeSubstrateDataset, collate_fn
from src.models import HeteroscedasticEnzymeModel
from src.metrics import compute_regression_metrics

def evaluate_pipeline():
    # =========================================================================
    # 🎯 EVALUATION PANEL (Configure to match your target matrix cell)
    # =========================================================================
    TARGET_TASK = "kcat"       # Target choice. Options: "kcat", "km", "ki"
    MODEL_MODE  = "bimodal"    # Modality profile. Options: "seq_only", "bimodal", "trimodal"
    # =========================================================================

    # Explicit routing map targeting high-order sequence-disjoint OOD files
    DATA_ROUTING = {
        "kcat": {
            "master": "datasets/processed/kcat_max_wt_singleSeqs_wpdbs.csv",
            "test": "datasets/splits/kcat-seq_test_sequence_40cluster.csv"
        },
        "km": {
            "master": "datasets/processed/km_mean_wt_singleSeqs_wpdbs.csv",
            "test": "datasets/splits/km-seq_test_sequence_40cluster.csv"
        },
        "ki": {
            "master": "datasets/processed/ki_mean_wt_singleSeqs_wpdbs.csv",
            "test": "datasets/splits/ki-seq_test_sequence_40cluster.csv"
        }
    }
    
    cfg = DATA_ROUTING[TARGET_TASK]
    MODEL_WEIGHTS = f"./checkpoints/best_model_{TARGET_TASK}_{MODEL_MODE}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n[*] Deploying Inference System | Task: {TARGET_TASK.upper()} | Mode: {MODEL_MODE.upper()}")
    
    test_dataset = EnzymeSubstrateDataset(cfg["master"], cfg["test"])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    print(f"[-] OOD Target Size: {len(test_dataset)} elements")
    
    # Initialize blank weights layout and bind matching parameters files
    model = HeteroscedasticEnzymeModel().to(device)
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Missing weights archive at: {MODEL_WEIGHTS}. Verify training completeness first.")
        
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
    model.eval()
    
    all_targets, all_preds = [], []
    with torch.no_grad():
        for batch_seqs, batch_smiles, batch_paths, batch_targets in tqdm(test_loader, desc="OOD Inference Processing"):
            batch_targets = batch_targets.to(device)
            means, _ = model(batch_seqs, batch_smiles, batch_paths, device, modality=MODEL_MODE)
            all_targets.extend(batch_targets.cpu().numpy())
            all_preds.extend(means.cpu().numpy())
            
    # Calculate OOD benchmark properties
    metrics = compute_regression_metrics(all_targets, all_preds)
    print(f"\n{'='*50}\n🏆 FINAL OOD TEST SET PERFORMANCE ({TARGET_TASK.upper()} - {MODEL_MODE.upper()}) \n{'='*50}")
    print(f"-> R-Squared (R2)      : {metrics['R2']:.4f}")
    print(f"-> Pearson Correlation : {metrics['Pearson']:.4f}")
    print(f"-> RMSE                : {metrics['RMSE']:.4f}")
    print(f"-> MAE                 : {metrics['MAE']:.4f}\n{'='*50}")

if __name__ == "__main__":
    evaluate_pipeline()