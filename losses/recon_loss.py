import torch
import torch.nn as nn

class MapReconLoss(nn.Module):
    def __init__(
        self,
        weight,
    ):
        super().__init__()
        self.weight = weight

    def forward(self, batch_in, batch_out):
        rgb = batch_out["rgb_current"]
        if self.weight <= 0:
            return None
        recon = batch_out["mapped_rgb"]
        rgb_loss = (torch.norm(recon - rgb, dim=1) ** 2).mean()
        output_recon_loss = self.weight * rgb_loss
        return output_recon_loss