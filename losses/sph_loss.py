import torch
import torch.nn as nn

class SphericalLoss(nn.Module):
    def __init__(
        self,
        weight,
    ):
        super().__init__()
        self.weight = weight
    
    def forward(self, batch_in, batch_out):
        uvw_f = batch_out["uvw_f"]      # [samples, 2]
        uvw_b = batch_out["uvw_b"]      # [samples, 2]
        if self.weight <= 0:
            return None
        uvw_f_distance = torch.norm(uvw_f, dim=1) - 1.0
        uvw_b_distance = torch.norm(uvw_b, dim=1) - 1.0
        sph_loss = torch.mean(uvw_f_distance ** 2) + torch.mean(uvw_b_distance ** 2)
        output_sph_loss = self.weight * sph_loss
        return output_sph_loss