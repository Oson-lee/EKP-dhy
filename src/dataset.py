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
        
        # Determine overlapping columns to use as joint keys to prevent row explosion
        join_keys = ['uniprot', 'sequence', 'reactant_smiles', 'smiles', 'substrate_smiles']
        common_keys = [col for col in join_keys if col in self.master_df.columns and col in self.mask_df.columns]
        
        # Combine joint keys with clear suffixes to protect original master column names
        self.df = pd.merge(
            self.master_df, 
            self.mask_df, 
            on=common_keys, 
            how='inner',
            suffixes=('', '_split')
        ).reset_index(drop=True)
        
        # 1. Extract protein sequence securely
        self.sequences = [str(s) for s in self.df['sequence'].tolist()]
        
        # 2. Adaptive Substrate SMILES Extraction System: scans for all common naming conventions
        smiles_candidates = ['reactant_smiles', 'smiles', 'substrate_smiles', 'SMILES']
        matched_smiles = [c for c in smiles_candidates if c in self.df.columns]
        if matched_smiles:
            self.smiles = [str(s) if pd.notna(s) else "" for s in self.df[matched_smiles[0]].tolist()]
        else:
            self.smiles = [""] * len(self.df) # Safe fallback vector
            
        # 3. Adaptive 3D Structure Path Scanner
        struct_candidates = ['pdbpath', 'pdb_path', 'pdbpath_split']
        matched_struct = [c for c in struct_candidates if c in self.df.columns]
        if matched_struct:
            self.pdb_paths = [str(p) if pd.notna(p) else "" for p in self.df[matched_struct[0]].tolist()]
        else:
            self.pdb_paths = [""] * len(self.df)
        
        # 4. Precise Target Extraction System: prioritizes aggregated endpoints over individual raw tokens
        priority_targets = ['log10kcat_max', 'log10km_mean', 'log10ki_mean', 'log10km_max', 'log10ki_max']
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