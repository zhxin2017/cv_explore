from torch import nn
import torch
import torch.nn.functional as F


class MLP(nn.Module):
    """The implementation of simple multi-layer perceptron layer
    without dropout and identity connection.

    The feature process order follows `Linear -> ReLU -> Linear -> ReLU -> ...`.

    Args:
        input_dim (int): The input feature dimension.
        hidden_dim (int): The hidden dimension of MLPs.
        output_dim (int): the output feature dimension of MLPs.
        num_layer (int): The number of FC layer used in MLPs.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        """Forward function of `MLP`.

        Args:
            x (torch.Tensor): the input tensor used in `MLP` layers.

        Returns:
            torch.Tensor: the forward results of `MLP` layer
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FFN(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.c_fc = nn.Linear(dim, 4 * dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class FilteredLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.filter_linear = nn.Linear(in_dim, out_dim)
        self.filter_relu = nn.ReLU()
        self.forward_linear = nn.Linear(in_dim, out_dim)
        self.forward_relu = nn.ReLU()
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        filter_ = self.filter_relu(self.filter_linear(x))
        forward = self.forward_relu(self.forward_linear(x))
        x = self.ln(filter_ * forward)
        return x
