import pandas as pd
import torch
from torch.utils.data import Dataset

class EnzymeSubstrateDataset(Dataset):
    def __init__(self, data_path, split_path=None):
        """
        Custom dataset for handling enzyme-substrate data points.
        :param data_path: Path to the main processed CSV file.
        :param split_path: Path to the specific split CSV file (optional).
        """
        # Load the master processed dataframe
        self.df = pd.read_csv(data_path)
        
        # Filter the rows if a homological split file is provided
        if split_path:
            split_df = pd.read_csv(split_path)
            if 'Unnamed: 0' in split_df.columns:
                valid_indices = split_df['Unnamed: 0'].values
                self.df = self.df[self.df['Unnamed: 0'].isin(valid_indices)].reset_index(drop=True)
            else:
                print("Warning: 'Unnamed: 0' index column not found in the split file.")

    def __len__(self):
        """Returns the total number of samples in this dataset split."""
        return len(self.df)

    def __getitem__(self, idx):
        """Fetches a single data sample by index."""
        row = self.df.iloc[idx]
        
        # 1. Extract the primary amino acid sequence
        sequence = str(row['sequence']).upper()
        
        # 2. Extract SMILES (kcat uses 'reaction_smiles', km/ki use 'substrate_smiles')
        if 'reaction_smiles' in row:
            smiles = str(row['reaction_smiles'])
        else:
            smiles = str(row['substrate_smiles'])
            
        # 3. Extract the target log10-transformed kinetic value
        target = torch.tensor(row['log10_value'], dtype=torch.float32)
        
        return sequence, smiles, target

def collate_fn(batch):
    """Custom collate function to batch variable-length string inputs together."""
    sequences, smiles, targets = zip(*batch)
    return list(sequences), list(smiles), torch.stack(targets)