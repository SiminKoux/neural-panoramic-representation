import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def positionalEncoding_vec(in_tensor, b):
    '''
     in_tensor: [H*W, 4]
     b: [positional_dim]
     proj: [in_tensor.size(0), in_tensor.size(1), b.size()]
     mapped_coords: [in_tensor.size(0), in_tensor.size(1)*2, b.size()]
     output: [in_tensor.size(0), in_tensor.size(1) * 2 * b.size()]
        tensor.contiguous() is a method that can be used to 
            create a new tensor with the same data as the original tensor, 
            but with a different memory layout. 
            The dimensions of new tensor rearranged in memory so that they are contiguous)
    '''
    proj = torch.einsum('ij, k -> ijk', in_tensor, b)  # shape (in_tensor.size(0), in_tensor.size(1), freqNum)
    mapped_coords = torch.cat((torch.sin(proj), torch.cos(proj)), dim=1)  # shape (batch, 2*in_tensor.size(1), freqNum)
    output = mapped_coords.transpose(2, 1).contiguous().view(mapped_coords.size(0), -1)
    return output


class PositionalModel(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim = 256,
            use_positional = True,
            positional_dim = 10,
            skip_layers = [4, 7],
            num_layers = 8,
            verbose = True,
            use_tanh = False,
            use_sin = False):
        super(PositionalModel, self).__init__()
        self.omega_0 = 4
        self.verbose = verbose
        self.use_tanh = use_tanh  # default: False -> (-1, 1)
        self.use_sin = use_sin    # default: Fasle -> [-1, 1]
        
        if use_positional:
            encoding_dimensions = 2 * input_dim * positional_dim
            self.b = torch.tensor([(2 ** j) * np.pi for j in range(positional_dim)],requires_grad = False)
        else:
            encoding_dimensions = input_dim

        self.hidden = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # input layer
                input_dims = encoding_dimensions
            elif i in skip_layers:
                input_dims = hidden_dim + encoding_dimensions
            else:
                # hidden layers
                input_dims = hidden_dim

            if self.use_sin:
                if i == 0:
                    # input layer
                    self.init = nn.Linear(input_dims, hidden_dim, bias=True)
                    # with torch.no_grad():
                        # self.init.weight.uniform_(-1 / input_dims, 1 / input_dims) 
                    nn.init.xavier_uniform_(self.init.weight) 
                    self.hidden.append(self.init)
                elif i == num_layers - 1:
                    # last layer
                    self.last = nn.Linear(input_dims, output_dim, bias=True)
                    nn.init.xavier_uniform_(self.last.weight)
                    # with torch.no_grad():
                    #     self.last.weight.uniform_(-np.sqrt(6 / input_dims) / self.omega_0, 
                    #                             np.sqrt(6 / input_dims) / self.omega_0)
                    self.hidden.append(self.last)
                else:
                    # hidden layers
                    self.hid = nn.Linear(input_dims, hidden_dim, bias=True)
                    # with torch.no_grad():
                    #     self.hid.weight.uniform_(-np.sqrt(6 / input_dims) / self.omega_0, 
                    #                             np.sqrt(6 / input_dims) / self.omega_0)
                    nn.init.xavier_uniform_(self.hid.weight)
                    self.hidden.append(self.hid)
            else:
                if i == num_layers - 1:
                    # last layer
                    self.last = nn.Linear(input_dims, output_dim, bias=True)
                    self.hidden.append(self.last)
                else:
                    # hidden layers
                    self.hid = nn.Linear(input_dims, hidden_dim, bias=True)
                    self.hidden.append(self.hid)

        self.skip_layers = skip_layers
        self.num_layers = num_layers

        self.positional_dim = positional_dim
        self.use_positional = use_positional

        if self.verbose:
            print(f'Model has {count_parameters(self)} params')

    def forward(self, x):
        '''
         x: [samples, 4] -> (x, y, z, t)
        '''
        if self.use_positional: # default: True
            if self.b.device!=x.device:
                self.b=self.b.to(x.device)
            pos = positionalEncoding_vec(x, self.b)
            x = pos  # (2 * input_dim * positional_dim)
        
        input = x.detach().clone()
        for i, layer in enumerate(self.hidden):
            if i > 0:
                if self.use_sin:
                    x = torch.sin(self.omega_0 * x)
                else:
                    x = F.relu(x)
            if i in self.skip_layers:
                x = torch.cat((x, input), 1)
            x = layer(x)
        
        if self.use_tanh:
            x = torch.tanh(x)
        else:
            x = torch.sin(x)
        
        return x
    
