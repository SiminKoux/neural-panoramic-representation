import torch
import torch.nn as nn

class RelativePositionalLoss(nn.Module):
    def __init__(
        self,
        weight,
    ):
        super().__init__()
        self.weight = weight

    def forward(self, batch_in, batch_out):
        uvw_f = batch_out["uvw_f"]      # [samples, 2]
        uvw_b = batch_out["uvw_b"]      # [samples, 2]
        uvw_p_f = batch_out["uvw_p_f"]  # [samples*8, 2]
        uvw_p_b = batch_out["uvw_p_b"]  # [samples*8, 2]
        if "xyz_p" in batch_out: # [samples, 8, 3]
            xyz_p = batch_out["xyz_p"]
        else:
            xyz_p =batch_out["xyz_p_val"]
        
        u_p_f = uvw_p_f[:, 0].view(8, -1)  # [8, samples]
        v_p_f = uvw_p_f[:, 1].view(8, -1)  # [8, samples]
        w_p_f = uvw_p_f[:, 2].view(8, -1)  # [8, samples]
        u_p_b = uvw_p_b[:, 0].view(8, -1)  # [8, samples]
        v_p_b = uvw_p_b[:, 1].view(8, -1)  # [8, samples]
        w_p_b = uvw_p_b[:, 2].view(8, -1)  # [8, samples]

        # The directionality for the current samples uvw points
        u_p_d_f_ = uvw_f[:, 0].unsqueeze(0) - u_p_f  # [8, samples]
        v_p_d_f_ = uvw_f[:, 1].unsqueeze(0) - v_p_f  # [8, samples]
        w_p_d_f_ = uvw_f[:, 2].unsqueeze(0) - w_p_f  # [8, samples]
        u_p_d_b_ = uvw_b[:, 0].unsqueeze(0) - u_p_b  # [8, samples]
        v_p_d_b_ = uvw_b[:, 1].unsqueeze(0) - v_p_b  # [8, samples]
        w_p_d_b_ = uvw_b[:, 2].unsqueeze(0) - w_p_b  # [8, samples]

        u_p_d_f = u_p_d_f_ / torch.norm(u_p_d_f_)
        v_p_d_f = v_p_d_f_ / torch.norm(v_p_d_f_)
        w_p_d_f = w_p_d_f_ / torch.norm(w_p_d_f_)
        u_p_d_b = u_p_d_b_ / torch.norm(u_p_d_b_)
        v_p_d_b = v_p_d_b_ / torch.norm(v_p_d_b_)
        w_p_d_b = w_p_d_b_ / torch.norm(w_p_d_b_)

        x_p_d = xyz_p[:, :, 0].permute(1, 0)
        y_p_d = xyz_p[:, :, 1].permute(1, 0)
        z_p_d = xyz_p[:, :, 2].permute(1, 0)
        
        target_loss_f = torch.mean((u_p_d_f - x_p_d).norm(dim=1) ** 2 +
                                   (v_p_d_f - y_p_d).norm(dim=1) ** 2 +
                                   (w_p_d_f - z_p_d).norm(dim=1) ** 2)
        target_loss_b = torch.mean((u_p_d_b - x_p_d).norm(dim=1) ** 2 +
                                   (v_p_d_b - y_p_d).norm(dim=1) ** 2 +
                                   (w_p_d_b - z_p_d).norm(dim=1) ** 2)
        target_loss = target_loss_f + target_loss_b
        output_rigid_loss = self.weight * target_loss
        return output_rigid_loss