import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Minimal one-layer MLP for tabular data.
    You're welcome (and encouraged) to modify this architecture/add new models as you see fit.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, input_dim]
        return self.fc(x)
