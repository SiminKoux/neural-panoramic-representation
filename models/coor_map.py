import torch
import torch.nn as nn
from .backbone_mlp import PositionalModel

def convert_uvw_to_uv(uvw):
    u = uvw[:, 0]  # [samples, 1]
    v = uvw[:, 1]  # [samples, 1]
    w = uvw[:, 2]  # [samples, 1]

    target_uv = torch.zeros((uvw.shape[0], 2), dtype = torch.float32)
    r = torch.sqrt(u**2 + v**2 + w**2)
    # Find where the tensor is zero
    if torch.any(r == 0):
        epsilon = 1.0  # small constant
        # Get float tensor where True is converted to 1.0 and False to 0.0
        zero_indices = (r == 0).float()  
        # Add epsilon to zero elements
        r = r + zero_indices * epsilon

    theta = torch.atan2(v, u)  # [-pi, pi)
    theta[torch.isnan(theta)] = 0
    phi = torch.asin(w/r)      # [-pi/2, pi/2]

    target_uv[:, 0] = theta / torch.pi
    target_uv[:, 1] = phi / (torch.pi/2)
    return target_uv

class CoordinateMapping(nn.Module):
    def __init__(self, args):
        super().__init__()
        print("Initializing Coordinate Mapping Model (MLP)...")
        # M_f's hyper parameters
        self.use_positional_mapping_f = args.use_positional_mapping1
        self.n_positional_mapping_f = args.n_positional_mapping1
        self.n_channels_mapping_f = args.n_channels_mapping1
        self.n_layers_mapping_f = args.n_layers_mapping1
         # M_b's hyper parameters
        self.use_positional_mapping_b = args.use_positional_mapping2
        self.n_positional_mapping_b = args.n_positional_mapping2
        self.n_channels_mapping_b = args.n_channels_mapping2
        self.n_layers_mapping_b = args.n_layers_mapping2
        
        # R^4->R^3: (x,y,z,t) -> (u,v,w)
        self.mapping_uvw_f = PositionalModel(
            input_dim = 4,                                       # x,y,z,t
            output_dim = 3,                                      # u,v,w
            hidden_dim = self.n_channels_mapping_f,              # 256
            use_positional = self.use_positional_mapping_f,      # True
            positional_dim = self.n_positional_mapping_f,        # 6
            num_layers = self.n_layers_mapping_f,                # 8
            skip_layers = [4, 7])
        
        # R^4->R^3: (x,y,z,t) -> (u,v,w)
        self.mapping_uvw_b = PositionalModel(
            input_dim = 4,                                       # x,y,z,t
            output_dim = 3,                                      # u,v,w
            hidden_dim = self.n_channels_mapping_b,              # 256
            use_positional = self.use_positional_mapping_b,      # True
            positional_dim = self.n_positional_mapping_b,        # 6
            num_layers = self.n_layers_mapping_b,                # 8
            skip_layers = [4, 7])

    def forward(self, coords, coords_p):
        # get the implicit coordinates uvw from the mapping network
        if coords_p is not None:
            mapped_coords_f = self.mapping_uvw_f(coords)       # [samples, 3]
            mapped_coords_p_f = self.mapping_uvw_f(coords_p)   # [samples*8, 3]
            mapped_coords_b = self.mapping_uvw_b(coords)       # [samples, 3]
            mapped_coords_p_b = self.mapping_uvw_b(coords_p)   # [samples*8, 3]

            return {"uvw_f": mapped_coords_f,
                    "uvw_p_f": mapped_coords_p_f,
                    "uvw_b": mapped_coords_b,
                    "uvw_p_b": mapped_coords_p_b}
        else:
            mapped_coords_f = self.mapping_uvw_f(coords)  # [samples, 3]
            mapped_coords_b = self.mapping_uvw_b(coords)  # [samples, 3]
            return {"uvw_f": mapped_coords_f,
                    "uvw_b": mapped_coords_b,}
