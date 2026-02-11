import torch
import torch.nn as nn
from typing import List, Optional

class MLPClassifier(nn.Module):

    """
    Generic MLP classification head for binary classification.

    Args:
        in_features (int): Feature dimension from backbone
        hidden_dims (List[int]): Hidden layer sizes
        dropout (float): Dropout probability
        activation (str): 'relu', 'gelu', 'silu'
        use_batchnorm (bool): Whether to use BatchNorm
    """

    def __init__(
        self,
        num_layers: int,
        in_features: int,
        expansion_ratio: List[int],
        dropout: float=0.3,
        activation: str = 'relu',
        use_batchnorm: bool = True,
        out_classes: int = 2,
    ):

        super().__init__()

        assert all(isinstance(i, int) for i in expansion_ratio), f"Expansion ratios should contain all integers, got {expansion_ratio}"

        act_layer = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }[activation]

        layers = []
        init_dim = in_features

        # Construct the hidden layers
        for i in range(num_layers):
            out_dim = expansion_ratio[i] * init_dim
            layers.append(nn.Linear(init_dim, out_dim))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))

            layers.append(act_layer())
            layers.append(nn.Dropout(dropout))
            init_dim = out_dim
        
        layers.append(nn.Linear(init_dim, 2))

        self.ffn = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Args: 
            x: (B, F)
        
        Returns:
            logits: (B, 1)
        """
        
        return self.ffn(x)