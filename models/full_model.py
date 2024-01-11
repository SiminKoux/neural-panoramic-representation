import torch
import torch.nn as nn
import numpy as np

from .alpha_mlp import AlphaPredModle
from .coor_map import CoordinateMapping
from .rgb_map import RGBMapping
from .inverse_mlp import InverseMapping
import getData
import utils 

DEVICE = torch.device("cuda")

class FullModel(nn.Module):
    """
    Full model
    cfg loaded from configs/model
    """
    def __init__(self, dset, n_layers, cfg):
        super().__init__()

        self.dset = dset
        self.N, self.H, self.W = len(dset), dset.height, dset.width
        self.n_layers = n_layers
        print("The model is initinalizing...")

        # set the hyper-parameters
        args = cfg
        self.samples = args.samples_batch
        self.dev_amount = args.dev_amount
        self.dev_amount_item = args.dev_amount_item
        
        # initialize coordinates mapping
        self.coord_mapping = CoordinateMapping(args)
        optims = [{"params": self.coord_mapping.parameters(), "lr": args.lr}]

        # initialize alpha model
        self.alpha_pred = AlphaPredModle(args)
        optims.append({"params": self.alpha_pred.parameters(), "lr": args.lr})
        
        # initialize RGB mapping
        self.rgb_mapping = RGBMapping(args)
        optims.append({"params": self.rgb_mapping.parameters(), "lr": args.lr})

        # initialize Inverse mapping
        self.inverse_mapping = InverseMapping(args)
        optims.append({"params": self.inverse_mapping.parameters(), "lr": args.lr})
        
        self.optim = torch.optim.Adam(optims)
    
    def forward(
        self,
        batch_in,
        ret_map=True,
        ret_input_idx=False,
        ret_inputs=True,
        ret_evaluation=False,
        global_position=True,
        ret_inverse=False,
    ):
        # Define the output dictionary
        out = {}

        # Self-supervision Information
        idx = len(batch_in["idx"])          # [B]
        gt_rgb = batch_in["rgb"]            # [B, 3, H, W]
        gt_masks = batch_in['gt'][0]        # [B, H, W]
        
        # Canonical Coordinates for planar and spherical format 
        # (2D grid & 3D Cartesian)
        xyt_all = getData.get_xyt(self.N, self.H, self.W)  # [3, H*W*N]
        xyzt_all = getData.get_xyzt(self.N, self.H, self.W) # [4, H*W*N]
 
        # Spread the indices of inputs into outputs
        if ret_input_idx:
            out["idx"] = batch_in["idx"]
        
        # Get the ground-truth RGB values and their gradient for all frames
        if ret_inputs:
            out["gt_rgb"] = gt_rgb  # [B, 3, H, W]
        
        if ret_map:
            # Randomly choose indices from all frames
            inds_train = torch.randint(xyzt_all.shape[1], 
                                       (np.int64(self.samples * 1.0), 1)) # [samples, 1]
            
            # Get the current canonical coordinates batch
            xyzt_current = xyzt_all[:, inds_train]  # [4, samples, 1], coordinates are in [-1,1]
            xyt_current = xyt_all[:, inds_train]  # [3, samples, 1], coordinates've been not normalizaed

            # Get the current ground-truth RGB & RGB_dx & RGB_dy batch
            rgb_current = gt_rgb[xyt_current[2, :], :,
                                 xyt_current[1, :],
                                 xyt_current[0, :]].squeeze(1)  # [samples, 3]
            mask_current = gt_masks[xyt_current[2, :], 
                                    xyt_current[1, :],
                                    xyt_current[0, :]].squeeze(1).unsqueeze(-1).to(DEVICE)  # [samples, 1]
            
            out["rgb_current"] = rgb_current  # [samples, 1, 3]
            out["alpha_current"] = mask_current  # [samples, 1]
            
            # Get the current 3D cartesian coordinates XYZT
            xyzt = torch.cat((xyzt_current[0, :], 
                              xyzt_current[1, :], 
                              xyzt_current[2, :], 
                              xyzt_current[3, :] / (self.N / 2.0) - 1), 
                             dim = 1).to(DEVICE)  # [samples, 4]
            
            # Get the current derivate amount batch
            if global_position:
                dev_amount = self.dev_amount      # 0.1
            else:
                dev_amount = self.dev_amount_item # 0.01
            
            # Create eight offset tensors
            offsets = dev_amount * torch.tensor([
                                [1, 1, 1, 0],
                                [1, 1, -1, 0],
                                [1, -1, 1, 0],
                                [1, -1, -1, 0],
                                [-1, 1, 1, 0],
                                [-1, 1, -1, 0],
                                [-1, -1, 1, 0],
                                [-1, -1, -1, 0]
                            ], dtype=torch.float32).to(DEVICE)  # [8, 4]
            
            # Repeat the tensor along the first dimension for the 'xyzt'
            repeated_samples = xyzt.unsqueeze(1).repeat(1, 8, 1)  # [samples, 8, 4]
            
            # Create the adjacents tensor for 'xyzt'
            adjacent_points_all = torch.zeros((self.samples, 8, 4)).to(DEVICE)
            for i in range(self.samples): 
                adjacent_points_all[i, :, :] = repeated_samples[i, :, :] + offsets
            adjacent_points_all[:, :, 0:3] /= torch.norm(adjacent_points_all[:, :, 0:3], dim=2, keepdim=True)

            dxs_current_adjacent = adjacent_points_all[:, :, 0]  # [samples, 8]
            dys_current_adjacent = adjacent_points_all[:, :, 1]  # [samples, 8]
            dzs_current_adjacent = adjacent_points_all[:, :, 2]  # [samples, 8]
            dts_current = adjacent_points_all[:, :, 3] # [samples, 8]

            xyzt_p = torch.cat((dxs_current_adjacent.reshape(-1).unsqueeze(1), 
                                dys_current_adjacent.reshape(-1).unsqueeze(1), 
                                dzs_current_adjacent.reshape(-1).unsqueeze(1), 
                                dts_current.reshape(-1).unsqueeze(1)), 
                               dim=1).to(DEVICE)  # [samples*8, 4]
            
            # Compute the direction​ for the sampled points and their adjacents
            x_p_d_ = repeated_samples[:, :, 0] - dxs_current_adjacent  # [samples, 8]
            y_p_d_ = repeated_samples[:, :, 1] - dys_current_adjacent  # [samples, 8]
            z_p_d_ = repeated_samples[:, :, 2] - dzs_current_adjacent  # [samples, 8]

            # Normalize the direction to make it back to the unit sphere
            x_p_d = x_p_d_ / torch.norm(x_p_d_)
            y_p_d = y_p_d_ / torch.norm(y_p_d_)
            z_p_d = z_p_d_ / torch.norm(z_p_d_)

            # The directionality for the current samples points
            out["xyz_p"] = torch.cat((x_p_d.unsqueeze(2), 
                                      y_p_d.unsqueeze(2), 
                                      z_p_d.unsqueeze(2)), 
                                     dim=2).to(DEVICE)  # [samples, 8, 3]
            
            coord_map_dict = self.coord_mapping(xyzt, xyzt_p)
            out.update(coord_map_dict)
            uvw_f = out["uvw_f"]
            uvw_b = out["uvw_b"]
            
            alpha_map_dict = self.alpha_pred(xyzt)
            out.update(alpha_map_dict)
            alpha = out["alpha"]

            rgb_map_dict = self.rgb_mapping(uvw_f, uvw_b)
            out.update(rgb_map_dict)
            mapped_rgb_f = out["mapped_rgb_f"]
            mapped_rgb_b = out["mapped_rgb_b"]

            mapped_rgb = mapped_rgb_f * alpha + mapped_rgb_b * (1.0 - alpha)
            occ_rgb = mapped_rgb_f * (1.0 - alpha)
            out["mapped_rgb"] = mapped_rgb
            out["occ_rgb"] = occ_rgb
                
        if ret_inverse:
            with torch.no_grad():
                xyzt_all_temp = xyzt_all.unsqueeze(-1)   # [4, H*W*N, 1]
                xyzt_all_ = torch.cat((xyzt_all_temp[0, :], 
                                       xyzt_all_temp[1, :], 
                                       xyzt_all_temp[2, :], 
                                       xyzt_all_temp[3, :] / (self.N / 2.0) - 1), 
                                      dim = 1).to(DEVICE)  # [H*W*N, 4]
                uvw_all_dict = self.coord_mapping(xyzt_all_, None)
                out.update(uvw_all_dict)
                uvw_all = out["uvw_f"]
                uvw_all = uvw_all.permute(1, 0)  # [3, H*W*N]

            inds_inverse_train = torch.randint(xyzt_all.shape[1], 
                                       (np.int64(self.samples * 1.0), 1)) # [samples, 1]
            # Get the current canonical coordinates batch
            xyzt_all = xyzt_all.to(DEVICE)
            xyzt_inverse_current = xyzt_all[:, inds_inverse_train]
            xyzt_inverse = torch.cat((xyzt_inverse_current[0, :], 
                                      xyzt_inverse_current[1, :], 
                                      xyzt_inverse_current[2, :], 
                                      xyzt_inverse_current[3, :] / (self.N / 2.0) - 1), 
                                     dim = 1).to(DEVICE)  # [samples, 4]
            out["gt_xyzt"] = xyzt_inverse
            uvw_current = uvw_all[:, inds_inverse_train]  # [3, samples, 1]
            uvwt_current = torch.cat((uvw_current[0, :], 
                                      uvw_current[1, :], 
                                      uvw_current[2, :], 
                                      xyzt_inverse_current[3, :] / (self.N / 2.0) - 1), 
                                     dim = 1).to(DEVICE)  # [samples, 4]
            pred_xyzt_dict = self.inverse_mapping(uvwt_current)
            out.update(pred_xyzt_dict)

        if ret_evaluation:
            c_uvw_f_val = torch.zeros((self.N, self.H, self.W, 3))
            c_uvw_b_val = torch.zeros((self.N, self.H, self.W, 3))
            c_rgb_val = torch.zeros(gt_rgb.shape)
            c_alpha = torch.zeros((self.N, self.H, self.W))
            
            inds_order = torch.arange(self.H * self.W).unsqueeze(1) # [0, 1, ..., H*W-1]
            for i in range(idx):
                # Orderly choose indices for the current frame
                batch_idx = int(batch_in["idx"][i])
                inds_val = inds_order + batch_idx * (self.H * self.W)   # [H*W,1]
                
                # Get the current canonical coordinates for all frames
                xyzt_val_current = xyzt_all[:, inds_val]   # [4, H*W, 1]
                
                # Get all points in one frame
                xyzt_val = torch.cat((xyzt_val_current[0, :], 
                                      xyzt_val_current[1, :], 
                                      xyzt_val_current[2, :], 
                                      xyzt_val_current[3, :] / (self.N / 2.0) - 1), 
                                      dim = 1).to(DEVICE)  # [H*W, 4]

                coord_map_dict_val = self.coord_mapping(xyzt_val, None)
                out.update(coord_map_dict_val)

                uvw_f_val = out["uvw_f"]  # [H*W, 3]
                uvw_b_val = out["uvw_b"]  # [H*W, 3]
                c_uvw_f_val[i, :, :, :] = uvw_f_val.reshape(self.H, self.W, 3)
                c_uvw_b_val[i, :, :, :] = uvw_b_val.reshape(self.H, self.W, 3)

                alpha_pred_dict_val = self.alpha_pred(xyzt_val)
                out.update(alpha_pred_dict_val)
                alpha_val = out["alpha"]
                c_alpha[i, :, :] = alpha_val.reshape(self.H, self.W)

                rgb_map_dict_val = self.rgb_mapping(uvw_f_val, uvw_b_val)
                out.update(rgb_map_dict_val)
                mapped_rgb_f_val = out["mapped_rgb_f"]
                mapped_rgb_b_val = out["mapped_rgb_b"]

                mapped_rgb_val = mapped_rgb_f_val * alpha_val + mapped_rgb_b_val * (1.0 - alpha_val)
                c_rgb_val[i, :, :, :] = mapped_rgb_val.reshape(self.H, self.W, 3).permute(2, 0, 1)
                
            out["coords_uvw_f"] = c_uvw_f_val
            out["coords_uvw_b"] = c_uvw_b_val
            out["recons"] = c_rgb_val
            out["alpha_vis"] = c_alpha.unsqueeze(1)
            
            theta_phi = getData.get_theta_phi(self.H, self.W)
            theta_indices = theta_phi[0, 0, :]  # [W]
            phi_indices = theta_phi[1, :, 1]    # [H]

            counter = 0
            texture_f = torch.zeros((self.H, self.W, 3))
            texture_b = torch.zeros((self.H, self.W, 3))
        
            for i in phi_indices:
                uv_f_temp = torch.cat((theta_indices.unsqueeze(1),
                                       i * torch.ones_like(theta_indices.unsqueeze(1))),
                                       dim=1)  # [W, 2]
                uv_b_temp = torch.cat((theta_indices.unsqueeze(1),
                                       i * torch.ones_like(theta_indices.unsqueeze(1))),
                                       dim=1)  # [W, 2]
                uvw_f_temp = utils.uv_to_uvw(uv_f_temp, self.W).to(DEVICE)
                uvw_b_temp = utils.uv_to_uvw(uv_b_temp, self.W).to(DEVICE)
                texture_rgb_dict = self.rgb_mapping(uvw_f_temp, uvw_b_temp)
                out.update(texture_rgb_dict)
                texture_rgb_f = out["mapped_rgb_f"]  # [W, 3]
                texture_rgb_b = out["mapped_rgb_b"]  # [W, 3]
                texture_f[counter, :, :] = texture_rgb_f
                texture_b[counter, :, :] = texture_rgb_b
                counter = counter + 1  # counter: from 0 to H-1
            out["texture_f"] = texture_f
            out["texture_b"] = texture_b

        return out
