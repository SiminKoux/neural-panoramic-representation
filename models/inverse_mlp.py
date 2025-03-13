import torch.nn as nn
from .backbone_mlp import PositionalModel

class InverseMapping(nn.Module):
    def __init__(self, args):
        super().__init__()
        print("Initializing Inverse Mapping Model (MLP)...")
        # B_f's hyper parameters
        self.use_positional_mapping_f = args.use_positional_mapping1
        self.n_positional_mapping_f = args.n_positional_mapping1
        self.n_channels_mapping_f = args.n_channels_mapping1
        self.n_layers_mapping_f = args.n_layers_mapping1
         # B_b's hyper parameters
        self.use_positional_mapping_b = args.use_positional_mapping2
        self.n_positional_mapping_b = args.n_positional_mapping2
        self.n_channels_mapping_b = args.n_channels_mapping2
        self.n_layers_mapping_b = args.n_layers_mapping2
        
        # R^4->R^4: (u,v,w,t) -> (x,y,z,t)
        self.inverse_mapping_f = PositionalModel(
            input_dim = 4,                                       # u,v,w,t
            output_dim = 4,                                      # x,y,z,t
            hidden_dim = self.n_channels_mapping_f,              # 256
            use_positional = self.use_positional_mapping_f,      # True
            positional_dim = self.n_positional_mapping_f,        # 6
            num_layers = self.n_layers_mapping_f,                # 8
            skip_layers = [4, 7])
        
        # R^4->R^4: (u,v,w,t) -> (x,y,z,t)
        self.inverse_mapping_b = PositionalModel(
            input_dim = 4,                                       # u,v,w,t
            output_dim = 4,                                      # x,y,z,t
            hidden_dim = self.n_channels_mapping_b,              # 256
            use_positional = self.use_positional_mapping_b,      # True
            positional_dim = self.n_positional_mapping_b,        # 6
            num_layers = self.n_layers_mapping_b,                # 8
            skip_layers = [4, 7])
        

    def forward(self, coords):
        mapped_coords_f = self.inverse_mapping_f(coords)  # [samples, 4]
        mapped_coords_b = self.inverse_mapping_b(coords)  # [samples, 4]
        return {"inverse_xyzt_f": mapped_coords_f, "inverse_xyzt_b": mapped_coords_b}
        
