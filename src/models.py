import torch
import torch.nn as nn
from src.encoders import ProteinSequenceEncoder, SubstrateSMILESEncoder, EnzymeStructureEncoder

class HeteroscedasticEnzymeModel(nn.Module):
    def __init__(self, seq_dim=320, smiles_dim=384, struct_dim=128, hidden_dim=256):
        super(HeteroscedasticEnzymeModel, self).__init__()
        
        # Initialize the three specialized modular feature encoders
        self.seq_encoder = ProteinSequenceEncoder()
        self.smiles_encoder = SubstrateSMILESEncoder()
        self.struct_encoder = EnzymeStructureEncoder(output_dim=struct_dim)
        
        # Projection layers to map diverse modalities into a unified hidden space
        self.proj_seq = nn.Sequential(nn.Linear(seq_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.proj_smiles = nn.Sequential(nn.Linear(smiles_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.proj_struct = nn.Sequential(nn.Linear(struct_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        
        # Dual-head regression head supporting heteroscedastic uncertainty estimation
        # Allocates maximum capacity to accommodate concatenated multimodal features (hidden_dim * 3)
        self.regressor_mean = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        self.regressor_logvar = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, sequences, smiles, pdb_paths, device, modality="bimodal"):
        # Extract foundational representations from the frozen sequence encoder
        seq_emb = self.proj_seq(self.seq_encoder(sequences, device))
        
        # Dynamic Feature Routing Layer for structural ablation experiments
        if modality == "seq_only":
            # Deterministically zero-mask the chemical and structural feature pathways
            smiles_emb = torch.zeros_like(seq_emb).to(device)
            struct_emb = torch.zeros_like(seq_emb).to(device)
            
        elif modality == "bimodal":
            # Enable chemical sequences but keep the 3D structural stream masked
            smiles_emb = self.proj_smiles(self.smiles_encoder(smiles, device))
            struct_emb = torch.zeros_like(seq_emb).to(device)
            
        elif modality == "trimodal":
            # Activate all feature streams concurrently including 3D spatial geometry descriptors
            smiles_emb = self.proj_smiles(self.smiles_encoder(smiles, device))
            struct_emb = self.proj_struct(self.struct_encoder(pdb_paths, device))
            
        else:
            raise ValueError(f"Unsupported modality configuration: {modality}")
            
        # Splice active embedding profiles into an aggregate multimodal feature tensor
        fused_representation = torch.cat([seq_emb, smiles_emb, struct_emb], dim=-1)
        
        # Map fused tensor to target values distribution attributes
        predicted_means = self.regressor_mean(fused_representation).squeeze(-1)
        predicted_logvars = self.regressor_logvar(fused_representation).squeeze(-1)
        
        return predicted_means, predicted_logvars