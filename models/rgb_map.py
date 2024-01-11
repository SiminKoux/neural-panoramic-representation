import torch.nn as nn
from .backbone_mlp import PositionalModel

class RGBMapping(nn.Module):
    def __init__(self, args):
        super().__init__()
        print("Initializing RGB Mapping Model (MLP)...")
        # RGB MLP's hyper parameters
        self.n_positional_atlas = args.n_positional_atlas
        self.n_channels_atlas = args.n_channels_atlas
        self.n_layers_atlas = args.n_layers_atlas   
        
        # R^2->R^3: (u,v) -> (r,g,b)
        self.rgb_mapping_f = PositionalModel(
            input_dim = 3,                                         # u,v
            output_dim = 3,                                        # r,g,b
            hidden_dim = self.n_channels_atlas,                    # 256
            use_positional = True,
            positional_dim = self.n_positional_atlas,              # 10
            num_layers = self.n_layers_atlas,                      # 8
            skip_layers = [4, 7])
        
        self.rgb_mapping_b = PositionalModel(
            input_dim = 3,                                         # u,v
            output_dim = 3,                                        # r,g,b
            hidden_dim = self.n_channels_atlas,                    # 256
            use_positional = True,
            positional_dim = self.n_positional_atlas,              # 10
            num_layers = self.n_layers_atlas,                      # 8
            skip_layers = [4, 7])


    def forward(self, mapped_uvw_f, mapped_uvw_b):
        '''
         Sample position from uv space,
            and then learn the rgb values for each samples position.
          The 'mapped_uv_f' and 'mapped_uv_b' 
            are the position sampled from the implicit space.
          They are in the range of [-1, 1], which is the unit spherical surface.
         Return: rgb values in the range of [0, 1]
        '''
        mapped_rgb_f = (self.rgb_mapping_f(mapped_uvw_f) + 1.0) * 0.5
        mapped_rgb_b = (self.rgb_mapping_b(mapped_uvw_b) + 1.0) * 0.5
        return {"mapped_rgb_f": mapped_rgb_f, "mapped_rgb_b": mapped_rgb_b}
    