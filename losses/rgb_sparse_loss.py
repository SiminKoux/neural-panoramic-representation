import torch
import torch.nn as nn

class RGBSparseLoss(nn.Module):
    def __init__(
        self,
        weight,
    ):
        super().__init__()
        self.weight = weight
    
    def forward(self, batch_in, batch_out):
        if self.weight <= 0:
            return None
        recon_conv = batch_out["occ_rgb"]
        rgb_loss_sparsity = (torch.norm(recon_conv, dim=1) ** 2).mean()
        
        output_loss = self.weight * rgb_loss_sparsity
        return output_loss