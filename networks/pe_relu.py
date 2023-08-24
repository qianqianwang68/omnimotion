import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def positionalEncoding_vec(in_tensor, b):
    original_shape = in_tensor.shape
    in_tensor_flatten = in_tensor.reshape(torch.prod(torch.tensor(original_shape[:-1])), -1)
    proj = torch.einsum('ij, k -> ijk', in_tensor_flatten, b)  # shape (batch, in_tensor.size(1), freqNum)
    mapped_coords = torch.cat((torch.sin(proj), torch.cos(proj)), dim=1)  # shape (batch, 2*in_tensor.size(1), freqNum)
    output = mapped_coords.transpose(2, 1).contiguous().view(mapped_coords.size(0), -1)
    output = output.reshape(original_shape[:-1] + (-1,))
    return output


class MLPf(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=256,
                 skip_layers=[4, 6],
                 num_layers=8,
                 use_pe=False,
                 pe_freq=10,
                 device='cuda',
                 ):
        super(MLPf, self).__init__()
        if use_pe:
            encoding_dimensions = 2 * 2 * pe_freq + input_dim  # only encode the pixel locations not latent codes
            self.b = torch.tensor([(2 ** j) * np.pi for j in range(pe_freq)], requires_grad=False).to(device)
        else:
            encoding_dimensions = input_dim

        self.hidden = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_dims = encoding_dimensions
            elif i in skip_layers:
                input_dims = hidden_dim + encoding_dimensions
            else:
                input_dims = hidden_dim

            if i == num_layers - 1:
                # last layer
                self.hidden.append(nn.Linear(input_dims, output_dim, bias=True))
            else:
                self.hidden.append(nn.Linear(input_dims, hidden_dim, bias=True))

        self.skip_layers = skip_layers
        self.num_layers = num_layers
        self.use_pe = use_pe
        self.pe_freq = pe_freq

    def forward(self, x):
        if self.use_pe:
            coord = x[..., :2]
            pos = positionalEncoding_vec(coord, self.b)
            x = torch.cat([pos, x], dim=-1)

        input = x
        for i, layer in enumerate(self.hidden):
            if i > 0:
                x = F.relu(x)
            if i in self.skip_layers:
                x = torch.cat((x, input), -1)
            x = layer(x)
        return x


class MLPb(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim=3,
                 hidden_dim=256,
                 skip_layers=[4, 6],
                 num_layers=8,
                 use_pe=False,
                 pe_freq=10,
                 device='cuda',
                 ):
        super(MLPb, self).__init__()
        if use_pe:
            encoding_dimensions = 2 * input_dim * pe_freq
            self.b = torch.tensor([(2 ** j) * np.pi for j in range(pe_freq)], requires_grad=False).to(device)
        else:
            encoding_dimensions = input_dim

        self.hidden = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_dims = encoding_dimensions
            elif i in skip_layers:
                input_dims = hidden_dim + encoding_dimensions
            else:
                input_dims = hidden_dim

            if i == num_layers - 1:
                # last layer
                self.hidden.append(nn.Linear(input_dims, output_dim, bias=True))
            else:
                self.hidden.append(nn.Linear(input_dims, hidden_dim, bias=True))

        self.skip_layers = skip_layers
        self.num_layers = num_layers
        self.use_pe = use_pe
        self.pe_freq = pe_freq

    def forward(self, x):
        if self.use_pe:
            pos = positionalEncoding_vec(x, self.b)
            x = pos

        input = x
        for i, layer in enumerate(self.hidden):
            if i > 0:
                x = F.relu(x)
            if i in self.skip_layers:
                x = torch.cat((x, input), -1)
            x = layer(x)
        return x


class GaussianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=256,
                 skip_layers=[4],
                 num_layers=8,
                 act='relu',
                 use_pe=False,
                 pe_freq=10,
                 pe_dims=None,
                 device='cuda',
                 act_trainable=False,
                 **kwargs):
        super(MLP, self).__init__()
        self.pe_dims = pe_dims
        if use_pe:
            if pe_dims == None:
                encoding_dimensions = 2 * input_dim * pe_freq + input_dim
            else:
                encoding_dimensions = 2 * len(pe_dims) * pe_freq + input_dim
            self.b = torch.tensor([(2 ** j) * np.pi for j in range(pe_freq)], requires_grad=False).to(device)
        else:
            encoding_dimensions = input_dim

        self.hidden = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_dims = encoding_dimensions
            elif i in skip_layers:
                input_dims = hidden_dim + encoding_dimensions
            else:
                input_dims = hidden_dim

            if act == 'relu':
                act_ = nn.ReLU(True)
            elif act == 'elu':
                act_ = nn.ELU(True)
            elif act == 'leakyrelu':
                act_ = nn.LeakyReLU(True)
            elif act == 'gaussian':
                act_ = GaussianActivation(a=kwargs['a'], trainable=act_trainable)
            else:
                raise Exception('unknown activation function!')

            if i == num_layers - 1:
                # last layer
                self.hidden.append(nn.Linear(input_dims, output_dim, bias=True))
            else:
                self.hidden.append(nn.Sequential(nn.Linear(input_dims, hidden_dim, bias=True), act_))

        self.skip_layers = skip_layers
        self.num_layers = num_layers
        self.use_pe = use_pe
        self.pe_freq = pe_freq

    def forward(self, x):
        if self.use_pe:
            coord = x[..., self.pe_dims] if self.pe_dims is not None else x
            pos = positionalEncoding_vec(coord, self.b)
            x = torch.cat([pos, x], dim=-1)

        input = x
        for i, layer in enumerate(self.hidden):
            if i in self.skip_layers:
                x = torch.cat((x, input), -1)
            x = layer(x)
        return x