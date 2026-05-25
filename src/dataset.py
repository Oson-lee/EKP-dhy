import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class EnzymeSubstrateDataset(Dataset):
    def __init__(self, data_master_path, split_mask_path):
        # Programmatically evaluate and adjust environment-relative file targets
        if not os.path.exists(data_master_path):
            data_master_path = os.path.join("..", data_master_path)
        if not os.path.exists(split_mask_path):
            split_mask_path = os.path.join("..", split_mask_path)
            
        self.master_df = pd.read_csv(data_master_path)
        self.mask_df = pd.read_csv(split_mask_path)
        
        # Merge tracking indices via UniProt keys to slice correct data clusters
        self.df = pd.merge(self.master_df, self.mask_df, on='uniprot', how='inner').reset_index(drop=True)
        
        self.sequences = self.df['sequence'].tolist()
        self.smiles = self.df['reactant_smiles'].tolist()
        self.pdb_paths = self.df['pdbpath'].tolist()
        
        # Precise Target Extraction System: prioritizes aggregated endpoints over individual raw tokens
        priority_targets = ['log10kcat_max', 'log10km_mean', 'log10ki_mean']
        matched_target = [c for c in priority_targets if c in self.df.columns]
        
        if matched_target:
            self.targets = self.df[matched_target[0]].values
        else:
            # Secondary fallback logic scanning for generalized logarithmic scales
            backup_log_cols = [c for c in self.df.columns if 'log10' in c]
            if backup_log_cols:
                self.targets = self.df[backup_log_cols[0]].values
            else:
                self.targets = self.df['value'].values 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.smiles[idx],
            self.pdb_paths[idx],
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

def collate_fn(batch):
    sequences = [item[0] for item in batch]
    smiles = [item[1] for item in batch]
    pdb_paths = [item[2] for item in batch]
    targets = torch.stack([item[3] for item in batch])
    return sequences, smiles, pdb_paths, targets