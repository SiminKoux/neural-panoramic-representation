import torch.nn as nn
import torch.nn.functional as F

class InverseLoss(nn.Module):
    def __init__(
        self,
        weight,
    ):
        super().__init__()
        self.weight = weight

    def forward(self, batch_in, batch_out):
        gt_xyzt = batch_out["gt_xyzt"]
        pred_xyzt_f = batch_out["inverse_xyzt_f"]
        pred_xyzt_b = batch_out["inverse_xyzt_b"]
        if self.weight <= 0:
            return None
        inverse_loss_f = F.l1_loss(pred_xyzt_f, gt_xyzt)
        inverse_loss_b = F.l1_loss(pred_xyzt_b, gt_xyzt)
        inverse_loss = inverse_loss_f + inverse_loss_b
        output_inverse_loss = self.weight * inverse_loss
        return output_inverse_loss
