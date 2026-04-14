import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MLP(nn.Module):
    """Configurable Multi-Layer Perceptron with optional BatchNorm and Dropout."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout: float = 0.3,
        activation: str = "relu",
        batch_norm: bool = True,
    ) -> None:
        """
        Args:
            input_size: Flattened input dimension.
            hidden_sizes: List of hidden layer widths.
            num_classes: Number of output classes.
            dropout: Dropout probability applied after each activation.
            activation: Activation function, either 'relu' or 'gelu'.
            batch_norm: Whether to apply BatchNorm1d before each activation.
        """
        super().__init__()
        layers = []
        in_dim = input_size
        act    = nn.GELU() if activation == "gelu" else nn.ReLU()

        for h in hidden_sizes:
            block = [nn.Linear(in_dim, h)]
            if batch_norm:
                block.append(nn.BatchNorm1d(h))
            block += [act, nn.Dropout(dropout)]
            layers += block
            in_dim = h

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Flatten input and pass through the network."""
        x = x.view(x.size(0), -1)
        return self.net(x)


class MLP2(nn.Module):
    """Minimal MLP using ModuleList, without BatchNorm or Dropout."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256],
        num_classes: int = 10,
    ) -> None:
        """
        Args:
            input_dim: Flattened input dimension.
            hidden_dims: List of hidden layer widths.
            num_classes: Number of output classes.
        """
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer  = nn.Linear(prev_dim, num_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Flatten input, apply ReLU hidden layers, and return logits."""
        x = x.view(x.size(0), -1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)
