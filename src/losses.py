import torch
import torch.nn as nn
import numpy as np

class HeteroscedasticGaussianNLLLoss(nn.Module):
    def __init__(self):
        """
        Negative Log-Likelihood Loss for tracking predictive mean and variance simultaneously.
        Enables the model to attenuate loss weights for noisier, sparse data targets.
        """
        super().__init__()

    def forward(self, mean, log_var, target):
        """
        Calculates NLL Loss.
        Math formula: 0.5 * exp(-log_var) * (target - mean)^2 + 0.5 * log_var
        """
        # precision is the inverse of variance: e^(-log_var) = 1 / variance
        precision = torch.exp(-log_var)
        squared_error = torch.square(target - mean)
        
        # Calculate loss (ignoring the constant 0.5 * log(2*pi) as it doesn't affect gradients)
        loss = 0.5 * precision * squared_error + 0.5 * log_var
        
        return torch.mean(loss)