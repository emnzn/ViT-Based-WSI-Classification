import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBasedMIL(nn.Module):

    """
    Constructs the gated variant of the ABMIL architecture.

    Parameters
    ----------
    input_dim: int
        The dimension of the input.

    embed_dim: int
        The output dimension of the first fully connected layer.

    hidden_dim: int
        The output dimension of the second fully connected layer.

    num_classes: int
        The number of output classes.

    Returns
    -------
    logits: torch.Tensor
        The output of the model.

    attention_weights: torch.Tensor
        The attention weights applied to each instance.
    """

    def __init__(
        self, 
        input_dim: int,
        embed_dim: int, 
        hidden_dim: int, 
        num_classes: int
        ):
        super(AttentionBasedMIL, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.10)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        
        self.attention = GatedAttention(hidden_dim)

        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        weighted_sum, attention_weights = self.attention(x)
        
        logits = self.head(weighted_sum)

        return logits, attention_weights

class GatedAttention(nn.Module):

    """
    Parameters
    ----------
    hidden_dim: int
        The input dimension to the attention mechanism.

    Returns
    -------
    weighted_sum: torch.Tensor
        The weighted sum of the input and 
    """

    def __init__(self, hidden_dim):
        super(GatedAttention, self).__init__()

        self.hidden_dim = hidden_dim

        self.U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.W = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        U = self.U(x)
        V = self.V(x)
        
        attention_weights = F.softmax(self.W(V * U), dim=1)
        weighted_sum = torch.sum(attention_weights * x, dim=1)

        return weighted_sum, attention_weights