import torch.nn as nn
from .backbone_mlp import PositionalModel

class AlphaPredModle(nn.Module):
    def __init__(self, args):
        super().__init__()
        print("Initializing Alpha Prediction Model (MLP)...")
        # M_A's hyper parameters
        self.n_positional_encoding_alpha = args.n_positional_encoding_alpha
        self.n_channels_alpha = args.n_channels_alpha
        self.n_layers_alpha = args.n_layers_alpha
        
        # R^4->R^1: (x,y,z,t) -> (a)
        self.model_alpha = PositionalModel(
            input_dim = 4,
            output_dim = 1,
            hidden_dim = self.n_channels_alpha,                    # 256
            use_positional = True,
            positional_dim = self.n_positional_encoding_alpha,     # 5
            num_layers = self.n_layers_alpha,                      # 8
            skip_layers = [])

    def forward(self, coords):
        '''
         Get the alpha value of each implicit coordinates uvw
         Return: alpha values in the range of [0,1]
        '''
        alpha = self.model_alpha(coords)  # [samples, 1]
        mapped_alpha = 0.5 * (alpha + 1.0) # map output in range of [-1, 1] to [0,1]
        return {"alpha": mapped_alpha}
        
