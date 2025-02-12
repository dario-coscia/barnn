import torch
import torch.nn as nn
import torch.nn.functional as F

from pina.model.layers import SpectralConvBlock1D
from .layers import (_BaseBARNN, ARDVariationalDropoutConv1d,
                    ConditionalSpectralConvBlock1D, LinearBARNN, Conv1dBARNN)
from .utils import positional_embedding

class AlphaFunction(nn.Module):
    def forward(self, x):
        p = torch.sigmoid(x).as_subclass(torch.Tensor)
        return p / (1.-p)
    
###### Prior and Posterio Encoding #######
class Encoding(nn.Module):
    def __init__(self,
                 input_dimension,
                 number_var_layers,
                 width=16,
                 dimension=1):
        """
        Encoding the input. The input shape shoud
        be [B, hidden_size, ...] and the output dimension will be
        discretization invariant i.e. [number_var_layers, B, output_dimension, ...] 
        """
        super().__init__()
        self.number_var_layers = number_var_layers 
        func = nn.SiLU()
        if dimension == 1:
            conv = nn.Conv1d
        elif dimension == 2:
            conv = nn.Conv2d
        else:
            raise NotImplementedError
        self.pre_layer = conv(input_dimension, width, 1)
        self.feat_transform = nn.Sequential(
            conv(width, width, 1),
            func,
            conv(width, width, 1),
            func,
            conv(width, width, 1),
            func,
            conv(width, self.number_var_layers, 1),
            AlphaFunction()
            )
    def forward(self, input, condition):
        """
        condition.shape = [batch, 1]
        input.shape = [batch, input_dimension, ...]
        """
        batch_dim  = input.shape[0]
        extra_dims = input.shape[2:]
        # Perform embeddings
        input = self.pre_layer(input)
        scale, shift = positional_embedding(condition.float(), 2*input.shape[1]).chunk(2, dim=-1)
        # repeat and apply scale and shift
        dims_to_add = input.dim() - scale.dim()
        scale = scale.view(batch_dim, -1, *([1] * dims_to_add)).expand_as(input)
        shift = shift.view(batch_dim, -1, *([1] * dims_to_add)).expand_as(input)
        # MLP to downsample
        alpha = self.feat_transform(input * scale + shift)
        # alpha = self.feat_transform(aggregated_out)
        alpha = alpha.reshape(batch_dim, self.number_var_layers, -1, *extra_dims)
        alpha = alpha.permute(1, 0, *range(2, alpha.ndim))
        return alpha

###### One Dimensional Fourier Neural Operator #######
class FNO1D(nn.Module):
    _modalities = ['dropout', 'barnn', 'standard', 'ard', 'refiner', 'perturb']
    def __init__(self,
                 time_history = 1,
                 modes = 32,
                 width = 64,
                 num_layers = 4,
                 modality = 'standard',
                 dropout = 0.2,
                 ):
        super(FNO1D, self).__init__()
        f"""
        Initialize the overall FNO network. It contains 5 layers of the Fourier layer.
        The input to the forward pass has the shape [batch, time_history, x].
        The output has the shape [batch, time_future, x].
        Args:
            pde (PDE): PDE at hand
            time_history (int): input timesteps of the trajectory
            modes (int): low frequency Fourier modes considered for multiplication in the Fourier space
            width (int): hidden channel dimension
            num_layers (int): number of FNO layers
            modality (str): must be any of {self._modalities}
            dropout (float): droput rate, only activated if modality = 'dropout'
        """
        self.modes = modes
        self.width = width
        self.time_history = time_history
        self.num_layers = num_layers
        input_dim = self.time_history
        # choose modality
        if modality == 'dropout' or modality == 'perturb':
            fourier_block = SpectralConvBlock1D
            conv = nn.Conv1d
        elif modality == 'barnn':
            fourier_block = SpectralConvBlock1D
            conv = Conv1dBARNN
            self.encoding_alpha = Encoding(
                input_dimension=self.time_history,
                number_var_layers=self.num_layers+1)
        elif modality == 'ard':
            fourier_block = SpectralConvBlock1D
            conv = ARDVariationalDropoutConv1d
        elif modality == 'refiner':
            conv = nn.Conv1d
            fourier_block = ConditionalSpectralConvBlock1D
            input_dim = self.time_history + 1 # extra input for PDE refiner
        else:
            raise NotImplementedError
        # linear embedding
        self.fc0 = nn.Linear(input_dim, self.width)
        if modality == 'barnn':
            self.fc1 = LinearBARNN(self.width, self.time_history)
        else:
            self.fc1 = nn.Linear(self.width, self.time_history)
            if modality == 'dropout':
                self.drop_fc1 = nn.Dropout(dropout)
            else:
                self.drop_fc1 = nn.Identity()
        # build network
        fourier_layers = []
        conv_layers = []
        dropout_layers = []
        for _ in range(num_layers):
            fourier_layers.append(fourier_block(self.width, self.width, self.modes))
            conv_layers.append(conv(self.width, self.width, 1))
            if modality == 'dropout':
                dropout_layers.append(nn.Dropout(dropout))
            else: # no dropout
                dropout_layers.append(nn.Identity())
        self.fourier_layers = nn.ModuleList(fourier_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.dropout_layers = nn.ModuleList(dropout_layers)

    def forward(self, u, emb=None):
        """
        Forward pass of FNO network.
        The input to the forward pass has the shape [batch, time_history, x]. 
        Returns: torch.Tensor: output has the shape [batch, time_history, x]
        """
        # permute input perform linear pass and permute back, treat time as channel
        x = u.permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # perform fourier
        for idx in range(self.num_layers):
            if isinstance(self.fourier_layers[idx], ConditionalSpectralConvBlock1D):
                x1 = self.fourier_layers[idx](x, emb.unsqueeze(-1))
            else:
                x1 = self.fourier_layers[idx](x)
            x = self.dropout_layers[idx](x)                                     # only activated if modality = 'dropout' otherwise Identity layer
            if isinstance(self.conv_layers[idx], _BaseBARNN):
                x2 = self.conv_layers[idx](x, emb[idx])
            else:
                x2 = self.conv_layers[idx](x)
            x = x1 + x2
            x = F.silu(x)
        # permute input perform linear pass and permute back, treat time as channel
        x = x.permute(0, 2, 1)
        if isinstance(self.fc1, _BaseBARNN):
            x = self.fc1(x, emb[-1].permute(0, 2, 1))
        else:
            x = self.drop_fc1(x)
            x = self.fc1(x)
        # permute back to input arguments
        x = x.permute(0, 2, 1)
        return x