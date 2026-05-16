import torch
import torch.nn as nn
from .encoders import SequenceEncoder, SubstrateEncoder

class HeteroscedasticEnzymeModel(nn.Module):
    def __init__(self, esm_model="facebook/esm2_t6_8M_UR50D", chemberta_model="DeepChem/ChemBERTa-77M-MTR", hidden_dim=256, dropout=0.2):
        """
        Multimodal fusion network returning Mean (μ) and Log-Variance (log σ²)
        for accurate regression paired with localized uncertainty forecasting.
        """
        super().__init__()
        # 1. Initialize the dual encoders
        self.seq_encoder = SequenceEncoder(model_name=esm_model)
        self.sub_encoder = SubstrateEncoder(model_name=chemberta_model)
        
        # 2. Dynamically calculate combined embedding dimension
        # ESM-2 (8M) outputs 320, ChemBERTa outputs 384 -> Combined: 704
        seq_dim = self.seq_encoder.esm_model.config.hidden_size
        sub_dim = self.sub_encoder.sub_model.config.hidden_size
        combined_dim = seq_dim + sub_dim
        
        # 3. Multimodal fusion network trunk (MLP)
        self.fusion_trunk = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )
        
        # 4. Dual heads for probabilistic output
        self.mean_head = nn.Linear(hidden_dim // 2, 1)    # Outputs predicted log10 value
        self.logvar_head = nn.Linear(hidden_dim // 2, 1)  # Outputs log variance for uncertainty

    def forward(self, sequences, smiles_list, device):
        """Forward pass integrating modalities and emitting probabilistic parameters."""
        # Step A: Independent modality extraction
        seq_emb = self.seq_encoder(sequences, device)
        sub_emb = self.sub_encoder(smiles_list, device)
        
        # Step B: Modality concatenation
        fused_features = torch.cat([seq_emb, sub_emb], dim=-1)
        
        # Step C: Nonlinear transformation
        X = self.fusion_trunk(fused_features)
        
        # Step D: Probabilistic regression mapping
        mean = self.mean_head(X).squeeze(-1)
        log_var = self.logvar_head(X).squeeze(-1)
        
        # Stable clamping to avoid exponential explosion during NLL loss calculation
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        
        return mean, log_var