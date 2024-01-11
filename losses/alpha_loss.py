import torch
import torch.nn as nn

class AlphaLoss(nn.Module):
    def __init__(
        self,
        weight,
    ):
        super().__init__()
        self.weight = weight

    def forward(self, batch_in, batch_out):
        gt_alpha = batch_out["alpha_current"]
        if self.weight <= 0:
            return None
        pred_alpha = batch_out["alpha"]
        pred_alpha = pred_alpha * 0.99
        pred_alpha = pred_alpha + 0.001
        alpha_loss = torch.mean(-gt_alpha * torch.log(pred_alpha) - (1-gt_alpha) * torch.log(1-pred_alpha))
        output_alpha_loss = self.weight * alpha_loss
        return output_alpha_loss