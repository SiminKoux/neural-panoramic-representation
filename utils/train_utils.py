import os
import cv2
import glob
import torch
import imageio
from tqdm import tqdm
from PIL import Image
import torch.optim as optim
import torchvision.utils as torch_utils

import utils
from losses import *
from getData.get_data import *
from getData.load_tensor import *
DEVICE = torch.device("cuda")

def warm_start(args, warmstart_model, label, N, H, W, writer, loader=None):
    # set the hyper-parameters for warmstart
    load_checkpoint = args.load_checkpoint # set to true to continue from a checkpoint
    checkpoint_path = args.checkpoint_path
    samples = args.samples_batch
    warmstart_mapping_uvw_f = args.warmstart_mapping1      # True
    warmstart_mapping_uvw_b = args.warmstart_mapping2      # True
    warmstart_alpha_pred = args.warmstart_alpha_pred       # True
    warmstart_iters = args.warmstart_iter_number           # 5000
    
    if not load_checkpoint:
        print("Warmstarting...")
        if label == "coord_map":
            if warmstart_mapping_uvw_f:
                warmstart_label = "uvw_mapping_f"
                print("Warmstart coordinates mapping (UVW_f).")
                warmstart_model.mapping_uvw_f = warmstarting_model(warmstart_label, 
                                                                   samples, N, H, W,
                                                                   warmstart_model.mapping_uvw_f,
                                                                   warmstart_iters = warmstart_iters, 
                                                                   warmstart_writer = writer)
                save_path_1 = os.path.join(checkpoint_path, 'warmstart_uvw_mapping_f.pth')
                torch.save(warmstart_model.mapping_uvw_f.state_dict(), save_path_1)
            if warmstart_mapping_uvw_b:
                warmstart_label = "uvw_mapping_b"
                print("Warmstart coordinates mapping (UVW_b).")
                warmstart_model.mapping_uvw_b = warmstarting_model(warmstart_label, 
                                                                   samples, N, H, W,
                                                                   warmstart_model.mapping_uvw_b, 
                                                                   warmstart_iters = warmstart_iters, 
                                                                   warmstart_writer = writer)
                save_path_2 = os.path.join(checkpoint_path, 'warmstart_uvw_mapping_b.pth')
                torch.save(warmstart_model.mapping_uvw_b.state_dict(), save_path_2)
        if label == "alpha_pred":
            if warmstart_alpha_pred:
                warmstart_label = "alpha_pred"
                print("Warmstart alpha prediction model.")
                warmstart_model.model_alpha = warmstarting_model(warmstart_label, 
                                                                 samples, N, H, W,
                                                                 warmstart_model.model_alpha,
                                                                 warmstart_iters = warmstart_iters,
                                                                 warmstart_writer = writer,
                                                                 loader = loader)
                save_path_3 = os.path.join(checkpoint_path, 'warmstart_alpha_pred.pth')
                torch.save(warmstart_model.model_alpha.state_dict(), save_path_3)
    else:
        if label == "coord_map":
            if warmstart_mapping_uvw_f:
                init_file_mapping1_path = os.path.join(checkpoint_path, "warmstart_uvw_mapping_f.pth")
                uvw_mapping_f = torch.load(init_file_mapping1_path)
                warmstart_model.mapping_uvw_f.load_state_dict(uvw_mapping_f)  
                print("Load warmstarted foreground coordinates mapping model has done!")
            if warmstart_mapping_uvw_b:
                init_file_mapping2_path = os.path.join(checkpoint_path, "warmstart_uvw_mapping_b.pth")
                uvw_mapping_b = torch.load(init_file_mapping2_path)
                warmstart_model.mapping_uvw_b.load_state_dict(uvw_mapping_b)  
                print("Load warmstarted background coordinates mapping model has done!")
        if label == "alpha_pred":
            if warmstart_alpha_pred:
                init_file_alpha_pred_path = os.path.join(checkpoint_path, "warmstart_alpha_pred.pth")
                alpha_pred = torch.load(init_file_alpha_pred_path)
                warmstart_model.model_alpha.load_state_dict(alpha_pred)  
                print("Load warmstarted alpha prediction model has done!")

def warmstarting_model(warmstart_label, samples, N, H, W, warmstart_model, 
                       warmstart_iters = 10000, warmstart_writer = None, loader=None):
    
    optimizer_mapping = optim.Adam(warmstart_model.parameters(), lr = 0.0001)
    
    for i in tqdm(range(warmstart_iters)):
        xyzt_all = get_xyzt(N, H, W)
        xyt_all = get_xyt(N, H, W)
        inds_train = torch.randint(xyzt_all.shape[1], (np.int64(samples * 1.0), 1))
        xyzt_current = xyzt_all[:, inds_train]  # [4, samples, 1]
        xyt_current = xyt_all[:, inds_train]    # [3, samples, 1]
        xyzt = torch.cat((xyzt_current[0, :], 
                          xyzt_current[1, :], 
                          xyzt_current[2, :], 
                          xyzt_current[3, :] / (N / 2.0) - 1), 
                          dim = 1).to(DEVICE)  # [samples, 4]
    
        output = warmstart_model(xyzt)   # [samples, 3] or [samples, 1]
        if warmstart_label == 'alpha_pred':
            # map output from the range [-1,1] of the alpha network to the range (0,1)
            output = 0.5 * (output + 1.0)
            '''prevent a situation of alpha=0, or alpha=1 
            for the BCE loss that uses log(alpha), log(1-alpha)'''
            output = output * 0.99
            output = output + 0.001
        
        warmstart_model.zero_grad()
        if warmstart_label in ["uvw_mapping_f", "uvw_mapping_b"]:
            loss = (xyzt[:, :3] - output).norm(dim = 1).mean()
        if warmstart_label == "alpha_pred":
            for batch_in in loader:
                masks_all = batch_in['gt'][0]   # [N, H, W]
                mask_current = masks_all[xyt_current[2, :], 
                                        xyt_current[1, :],
                                        xyt_current[0, :]].squeeze(1).unsqueeze(-1).to(DEVICE)  # [samples, 1]
                loss = torch.mean(-mask_current * torch.log(output) - (1-mask_current) * torch.log(1-output))
        loss.backward()
        optimizer_mapping.step()
        
        if warmstart_writer is not None:
            if warmstart_label == "uvw_mapping_f":
                warmstart_writer.add_scalar(f"warmstart_loss/uvw_mapping_f", loss.item(), i)
            if warmstart_label == "uvw_mapping_b":
                warmstart_writer.add_scalar(f"warmstart_loss/uvw_mapping_b", loss.item(), i)
            if warmstart_label == "alpha_pred":
                warmstart_writer.add_scalar(f"warmstart_loss/alpha_pred", loss.item(), i)

        # Visualization   
        if i == 0 or (i+1) % 50 == 0:
            xyzt_val = get_xyzt(N, H, W)
            inds = torch.arange(H * W).unsqueeze(1)
            if warmstart_label in ["uvw_mapping_f", "uvw_mapping_b"]:
                uvw_list = []
                with torch.no_grad():
                    for f in range(N): 
                        inds_val = inds + f * (H*W)   # [H*W,1]
                        xyzt_val_current = xyzt_val[:, inds_val]
                        val_xyzt = torch.cat((xyzt_val_current[0, :], 
                                              xyzt_val_current[1, :], 
                                              xyzt_val_current[2, :], 
                                              xyzt_val_current[3, :] / (N / 2.0) - 1), 
                                              dim = 1).to(DEVICE)  # [H*W, 4]  
                        
                        uvw_vis_temp = warmstart_model(val_xyzt) # [H*W, 3]
                        uvw_mp4_vis = uvw_vis_temp.reshape(H, W, 3)
                        uvw_list.append(uvw_mp4_vis)
                        
                        each_uvw_vis = uvw_vis_temp.reshape(H, W, 3).permute(2, 0, 1)
                        each_uvw_vis = (each_uvw_vis + 1) / 2
                        
                        if warmstart_label == 'uvw_mapping_f':
                            filename = f'/local/scratch2/SiminKou/Codes/IEEE_VR/outputs/pretrain/uvw/uvw_f/png/iter{i+1}_{f+1}.png'
                        if warmstart_label == 'uvw_mapping_b':
                            filename = f'/local/scratch2/SiminKou/Codes/IEEE_VR/outputs/pretrain/uvw/uvw_b/png/iter{i+1}_{f+1}.png'
                        torch_utils.save_image(each_uvw_vis, filename)   
                        
                    if warmstart_label == 'uvw_mapping_f':
                        out_name = f"/local/scratch2/SiminKou/Codes/IEEE_VR/outputs/pretrain/uvw/uvw_f/mp4"
                        uvw_path = os.path.join(out_name, f"uvw_f_{i+1}.mp4")
                    if warmstart_label == 'uvw_mapping_b':
                        out_name = f"/local/scratch2/SiminKou/Codes/IEEE_VR/outputs/pretrain/uvw/uvw_b/mp4"
                        uvw_path = os.path.join(out_name, f"uvw_b_{i+1}.mp4")
                    writer_uvw = imageio.get_writer(uvw_path, fps=10) 
                    for l in range(len(uvw_list)):
                        uvw_vis = ((uvw_list[l] + 1)/2)*255
                        writer_uvw.append_data(uvw_vis.detach().cpu().numpy().astype(np.uint8))
                    writer_uvw.close()
            
            if warmstart_label == "alpha_pred":
                with torch.no_grad():
                    for f in range(N): 
                        inds_val = inds + f * (H*W)   # [H*W,1]
                        xyzt_val_current = xyzt_val[:, inds_val]
                        val_xyzt = torch.cat((xyzt_val_current[0, :], 
                                              xyzt_val_current[1, :], 
                                              xyzt_val_current[2, :], 
                                              xyzt_val_current[3, :] / (N / 2.0) - 1), 
                                              dim = 1).to(DEVICE)  # [H*W, 4]  
                        
                        alpha_temp = warmstart_model(val_xyzt) # [H*W, 1]
                        alpha_temp = 0.5 * (alpha_temp + 1.0)
                        alpha_vis = alpha_temp.reshape(H, W, 1)
                        # Set a threshold
                        threshold = 0.5
                        # Create a binary mask
                        binary_mask = np.where(alpha_vis.detach().cpu().numpy() > threshold, 255, 0).astype(np.uint8)
                        output_dir = '/local/scratch2/SiminKou/Codes/IEEE_VR/outputs/pretrain/alpha/'
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        mask_path = os.path.join(output_dir, f'iter{i+1}_{f+1}.png')
                        # Save or use the binary mask
                        cv2.imwrite(mask_path, binary_mask)
    
    return warmstart_model

def load_ckpt(args, coord_model, alpha_model, rgb_model, inverse_model):
    # The path of the best forward model
    checkpoint_path = args.checkpoint_path
    
    # Load forward best ckpt
    model_path = os.path.join(checkpoint_path, "best_ckpt.pth")
    model = torch.load(model_path)
    
    # Filter out unwanted keys
    coord_f_dict = {k.replace('coord_mapping.mapping_uvw_f.', ''): v for k, v in model.items() if 'coord_mapping.mapping_uvw_f.' in k}
    coord_b_dict = {k.replace('coord_mapping.mapping_uvw_b.', ''): v for k, v in model.items() if 'coord_mapping.mapping_uvw_b.' in k}
    alpha_dict = {k.replace('alpha_pred.model_alpha.', ''): v for k, v in model.items() if 'alpha_pred.model_alpha.' in k}
    rgb_f_dict = {k.replace('rgb_mapping.rgb_mapping_f.', ''): v for k, v in model.items() if 'rgb_mapping.rgb_mapping_f.' in k}
    rgb_b_dict = {k.replace('rgb_mapping.rgb_mapping_b.', ''): v for k, v in model.items() if 'rgb_mapping.rgb_mapping_b.' in k}
    inverse_f_dict = {k.replace('inverse_mapping.inverse_mapping_f.', ''): v for k, v in model.items() if 'inverse_mapping.inverse_mapping_f.' in k}
    inverse_b_dict = {k.replace('inverse_mapping.inverse_mapping_b.', ''): v for k, v in model.items() if 'inverse_mapping.inverse_mapping_b.' in k}
    
    # Load the filtered state_dict into the model
    coord_model.mapping_uvw_f.load_state_dict(coord_f_dict, strict=False)
    coord_model.mapping_uvw_b.load_state_dict(coord_b_dict, strict=False)
    alpha_model.model_alpha.load_state_dict(alpha_dict)
    rgb_model.rgb_mapping_f.load_state_dict(rgb_f_dict, strict=False)
    rgb_model.rgb_mapping_b.load_state_dict(rgb_b_dict, strict=False)
    inverse_model.inverse_mapping_f.load_state_dict(inverse_f_dict, strict=False)
    inverse_model.inverse_mapping_b.load_state_dict(inverse_b_dict, strict=False)

    print("Load Full Model has done!")

def load_forward_ckpt(args, coord_model, alpha_model, rgb_model):
    # The path of the best forward model
    checkpoint_path = args.checkpoint_path
    
    # Load forward best ckpt
    forward_mapping_path = os.path.join(checkpoint_path, "best_ckpt.pth")
    forward_mapping = torch.load(forward_mapping_path)
    
    # Filter out unwanted keys
    coord_f_dict = {k.replace('coord_mapping.mapping_uvw_f.', ''): v for k, v in forward_mapping.items() if 'coord_mapping.mapping_uvw_f.' in k}
    coord_b_dict = {k.replace('coord_mapping.mapping_uvw_b.', ''): v for k, v in forward_mapping.items() if 'coord_mapping.mapping_uvw_b.' in k}
    alpha_dict = {k.replace('alpha_pred.model_alpha.', ''): v for k, v in forward_mapping.items() if 'alpha_pred.model_alpha.' in k}
    rgb_f_dict = {k.replace('rgb_mapping.rgb_mapping_f.', ''): v for k, v in forward_mapping.items() if 'rgb_mapping.rgb_mapping_f.' in k}
    rgb_b_dict = {k.replace('rgb_mapping.rgb_mapping_b.', ''): v for k, v in forward_mapping.items() if 'rgb_mapping.rgb_mapping_b.' in k}
    
    # Load the filtered state_dict into the model
    coord_model.mapping_uvw_f.load_state_dict(coord_f_dict, strict=False)
    coord_model.mapping_uvw_b.load_state_dict(coord_b_dict, strict=False)
    alpha_model.model_alpha.load_state_dict(alpha_dict)
    rgb_model.rgb_mapping_f.load_state_dict(rgb_f_dict, strict=False)
    rgb_model.rgb_mapping_b.load_state_dict(rgb_b_dict, strict=False)

    print("Load deformation model has done!")

def update_config(cfg, loader):
    """
    we provide a min number of iterations for each phase,
        need to update the config to reflect this
    """
    N = len(loader) * cfg.batch_size
    print("There are {} groups of inputs in the training stage!".format(N))
    
    for phase, epochs in cfg.epochs_per_phase.items():
        n_iters = cfg.iters_per_phase[phase]
        cfg.epochs_per_phase[phase] = max(n_iters, epochs)

    # also update the vis and val frequency in iterations
    cfg.vis_every = max(cfg.vis_every, cfg.vis_epochs * N)
    cfg.val_every = max(cfg.val_every, cfg.val_epochs * N)
    print("Train_epochs", cfg.epochs_per_phase["train"])
    print("vis_every", cfg.vis_every)
    print("val_every", cfg.val_every)
    return cfg


def optimize_model(
    n_epochs,
    loader,
    loss_fncs,
    model,
    model_kwargs={},
    start=0,
    label="forward_train",
    train_writer=None,
    **kwargs,
):
    print("This is the training step...")
    step_ct = start

    # Set initial best loss
    best_loss = float('inf')
    best_model_wts = model.state_dict().copy()
    if label == "backward_train":
        model_kwargs = dict(ret_map=False, global_position=False, ret_inverse=True)
        print("model_kwargs:", model_kwargs)
    
    # optimize for 'n_epochs'
    for _ in tqdm(range(n_epochs)):
        # optimize for each batch in the dataloader
        total_combined_loss = 0.0
        for batch_in in loader:
            model.optim.zero_grad()
            
            if step_ct >= 5000:
                model_kwargs = dict(global_position = False)
            
            batch_in = utils.move_to(batch_in, DEVICE)  # put 'batch' onto the cuda
            # with torch.autograd.detect_anomaly(): # Alert when Nan occur
            out_dict = model(batch_in, **model_kwargs)  # get the output dictionary from model
            loss_dict = compute_losses(loss_fncs, batch_in, out_dict)
            step_ct += len(loader)
            if len(loss_dict) < 1:
                continue
            combined_loss = sum(loss_dict.values())
            rgb_loss = loss_dict["reconstruct"].item()
            
            # If NaN is detected in the outputs or Loss, stop the training
            outputs_check = any(torch.isnan(val).any() for val in out_dict.values())
            if outputs_check or torch.isnan(combined_loss):
                model.load_state_dict(best_model_wts)
                exit("NaN detected! Reverting to best model and breaking...")

            combined_loss.backward() 
            model.optim.step()
            total_combined_loss += combined_loss.item()
            
            if train_writer is not None:
                for name, loss in loss_dict.items():
                    train_writer.add_scalar(f"loss/{name}", loss.item(), step_ct)
        
        avg_combined_loss = total_combined_loss / len(loader)
        # Save if it's the best model so far
        if avg_combined_loss < best_loss:
            best_loss = avg_combined_loss
            best_model_wts = model.state_dict().copy()
            torch.save(best_model_wts, 'best_ckpt.pth')
        
    # At the end of training, load the best model weights
    model.load_state_dict(best_model_wts)     
    return step_ct


def infer_model(
    step_ct,
    val,
    loader,
    model,
    model_kwargs={},
    val_writer=None,
    skip_keys=[],
    **kwargs
):
    """
    Inference Process
        Run the model on all data points
    """
    print("This is the inference step...")
    out_name = f"{step_ct:08d}_val"
    print("Inference step {:08d} saving to {}".format(step_ct, out_name))
    
    if len(model_kwargs) == 0:
        model_kwargs = dict(ret_map = False, ret_evaluation=True)
    
    out_dicts = []
    for batch_in in loader:
        batch_in = utils.move_to(batch_in, DEVICE)
        with torch.no_grad():
            out_dict = get_vis_batch(batch_in, model, model_kwargs)
            if 'recons' in out_dict:
                out_dict = utils.compute_psnr(batch_in, out_dict)
                # out_dict = utils.compute_ssim(batch_in, out_dict)

        out_dicts.append(
            {k: v.detach().cpu() for k, v in out_dict.items() if k not in skip_keys}
        )

    out_dict = utils.cat_tensor_dicts(out_dicts)
    
    if out_name is not None:
        utils.save_vis_dict(out_name, out_dict)
        if "texture_f" and "texture_b" in out_dict:
            texture_f = out_dict["texture_f"] # [H, W, 3]
            texture_f = (texture_f.numpy() *255).astype(np.uint8)
            texture_f_path = os.path.join(out_name, f"texture_f.png")
            imageio.imwrite(texture_f_path, texture_f)
            texture_b = out_dict["texture_b"] # [H, W, 3]
            texture_b = (texture_b.numpy() *255).astype(np.uint8)
            texture_b_path = os.path.join(out_name, f"texture_b.png")
            imageio.imwrite(texture_b_path, texture_b)
        if "coords_uvw_f" and "coords_uvw_b" in out_dict:
            torch.save(out_dict["coords_uvw_f"], f"{out_name}/coords_uvw_f.pth")
            torch.save(out_dict["coords_uvw_b"], f"{out_name}/coords_uvw_b.pth")
            coords_uvw_f = out_dict["coords_uvw_f"]  # [N, H, W, 3]
            uvw_f_path = os.path.join(out_name, f"uvw_f.mp4")
            writer_uvw_f = imageio.get_writer(uvw_f_path, fps=10)
            coords_uvw_b = out_dict["coords_uvw_b"]  # [N, H, W, 3]
            uvw_b_path = os.path.join(out_name, f"uvw_b.mp4")
            writer_uvw_b = imageio.get_writer(uvw_b_path, fps=10)
            for i in range(coords_uvw_f.shape[0]):
                writer_uvw_f.append_data((((coords_uvw_f[i, ...]+1)/2)*(255)).numpy().astype(np.uint8))
                writer_uvw_b.append_data((((coords_uvw_b[i, ...]+1)/2)*(255)).numpy().astype(np.uint8))
            writer_uvw_f.close()
            writer_uvw_b.close()
        if "recons" in out_dict:
            torch.save(out_dict["recons"], f"{out_name}/recons_rgb.pth")
            torch.save(out_dict["gt_rgb"], f"{out_name}/gt_rgb.pth")
        if "psnr" in out_dict:
            utils.save_metric(out_name, out_dict, val_writer, step_ct, name = "psnr")
    return out_dict, out_name


def opt_infer_step(
    n_epochs,
    train_loader,
    val_loader,
    loss_fncs,
    model,
    model_kwargs={},
    start=0,
    val_every=0,
    batch_size=8,
    label="forward_train",
    ckpt=None,
    save_grid=False,
    **kwargs,
):
    """
    optimizes model for n_epochs, 
        then saves a checkpoint and evaluation visualizations
    """
    if label == "backward_train":
        print("This is the backward training stage!")
        for param in model.coord_mapping.parameters():
            param.requires_grad = False
            print("Learned coordinates mapping module has been freezen!")
        for param in model.rgb_mapping.parameters():
            param.requires_grad = False
            print("Learned rgb mapping module has been freezen!")
        for param in model.alpha_pred.parameters():
            param.requires_grad = False
            print("Learned alpha prediction module has been freezen!")
    
    if ckpt is None:
        ckpt = "latest_ckpt.pth"

    # Set Epochs
    step = start
    steps_total = n_epochs
    # steps_total = n_epochs * len(train_loader) * batch_size
    val_epochs = max(1, steps_total // val_every)
    n_epochs_per_val = max(1, n_epochs // val_epochs)
    print(f"running {val_epochs} train/val steps with {n_epochs_per_val} epochs each.")

    for val in range(val_epochs):
        # Training step
        step = optimize_model(
            n_epochs_per_val,
            train_loader,
            loss_fncs,
            model,
            model_kwargs,
            start=step,
            label=label,
            **kwargs,
        )

        # Save the current ckpt
        print("Iter {:6d} saving checkpoint to {}".format(step, ckpt))
        latest_model_wts = model.state_dict().copy()
        torch.save(latest_model_wts, ckpt)
        
        # Evaluation step in training process
        val_dict, val_out_dir = infer_model(step, val, val_loader, model, model_kwargs, **kwargs)
        
        if save_grid:
            utils.save_grid_vis(val_out_dir, val_dict)

    return step, val_dict


def get_vis_batch(batch_in, model, model_kwargs={}, 
                  loss_fncs={}, vis_grad=False, **kwargs):
    # batch_in = utils.move_to(batch_in, DEVICE)
    out_dict = model(batch_in, **model_kwargs)
    # save mask gradients if loss functions
    if vis_grad and len(loss_fncs) > 0:
        grad_img, grad_max = get_loss_grad(batch_in, out_dict, loss_fncs, "pred")
        out_dict["pred_grad"] = grad_img
    return out_dict


