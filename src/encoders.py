import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class SequenceEncoder(nn.Module):
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        """
        Protein Language Model Encoder using pre-trained ESM-2.
        Note: You can upgrade to 'facebook/esm2_t33_650M_UR50D' later for enhanced production accuracy.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 【关键修复】：使用 AutoModel 替代 EsmModel，完美兼容所有 transformers 版本
        self.esm_model = AutoModel.from_pretrained(model_name)
        
        # Freeze encoder weights to retain evolutionary semantics and speed up execution
        for param in self.esm_model.parameters():
            param.requires_grad = False

    def forward(self, sequences, device):
        """Extracts dense fixed-size vector representations for raw protein strings."""
        # Tokenize and pad the sequence inputs to equal tensor lengths
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        outputs = self.esm_model(**inputs)
        
        # Shape: [batch_size, sequence_length, hidden_dimension]
        last_hidden_states = outputs.last_hidden_state
        
        # Masked Mean Pooling: collapse sequence length dimension while ignoring padding tokens
        mask = inputs['attention_mask'].unsqueeze(-1) # [batch_size, sequence_length, 1]
        sum_embeddings = torch.sum(last_hidden_states * mask, dim=1)
        sum_mask = torch.clamp(torch.sum(mask, dim=1), min=1e-9) # Prevent division by zero
        
        return sum_embeddings / sum_mask

class SubstrateEncoder(nn.Module):
    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MTR"):
        """
        Chemical Language Model Encoder using pre-trained ChemBERTa.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sub_model = AutoModel.from_pretrained(model_name)
        
        # Freeze encoder weights to prevent overwriting pre-trained chemical semantics
        for param in self.sub_model.parameters():
            param.requires_grad = False

    def forward(self, smiles, device):
        """Extracts dense embedding representations for chemical SMILES strings."""
        # Tokenize and pad SMILES strings
        inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = self.sub_model(**inputs)
        
        # Perform Mean Pooling over the token length dimension
        last_hidden_states = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        sum_embeddings = torch.sum(last_hidden_states * mask, dim=1)
        sum_mask = torch.clamp(torch.sum(mask, dim=1), min=1e-9)
        
        return sum_embeddings / sum_mask