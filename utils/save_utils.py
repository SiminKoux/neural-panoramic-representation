import os
import torch
import imageio
import numpy as np
from torchvision.transforms import functional as TF

from .flow_viz import flow_to_image
DEVICE = torch.device("cuda")


def save_checkpoint(path, i, **kwargs):
    print("Iter {:6d} saving checkpoint to {}".format(i, path))
    save_dict = {name: module.state_dict() for name, module in kwargs.items()}
    save_dict["i"] = i
    torch.save(save_dict, path)


def save_pretrain_alpha(path, i, model, loader, model_kwargs={}): 
    pre_alpha_dicts = []
    skip_keys = []

    for batch_in in loader:
        batch_in = move_to(batch_in, DEVICE)
        with torch.no_grad():
            out_dict = model(batch_in, **model_kwargs)
            
        pre_alpha_dicts.append(
        {k: v.detach().cpu() for k, v in out_dict.items() if k not in skip_keys}
    )
    pre_alpha_dict = cat_tensor_dicts(pre_alpha_dicts)

    # Save the dictionary to a file
    print("Iter {:6d} saving 'Pre-trained alpha dictionary' to {}".format(i, path))
    torch.save(pre_alpha_dict, path)


def save_vis_dict(out_dir, vis_dict, save_keys=[], skip_keys=[], overwrite=False):
    """
    :param out_dir
    :param vis_dict dict of 4+D tensors
    :param skip_keys (optional) list of keys to skip
    :return the paths each tensor is saved to
    """
    if os.path.isdir(out_dir) and not overwrite:
        print("{} exists already, skipping".format(out_dir))
        return

    if len(vis_dict) < 1:
        return []

    os.makedirs(out_dir, exist_ok=True)
    vis_dict = {k: v.detach().cpu() for k, v in vis_dict.items()}

    if len(save_keys) < 1:
        save_keys = vis_dict.keys()
    save_keys = set(save_keys) - set(skip_keys)

    out_paths = {}
    for name, vis_batch in vis_dict.items():
        if name not in save_keys:
            continue
        if vis_batch is None:
            continue
        out_paths[name] = save_vis_batch(out_dir, name, vis_batch)
    return out_paths


def save_vis_batch(out_dir, name, vis_batch, rescale=False, save_dir=False):
    """
    :param out_dir
    :param name
    :param vis_batch (B, *, C, H, W) first dimension is time dimension
    """
    if len(vis_batch.shape) < 4:
        return None

    C = vis_batch.shape[-3]
    if C > 3:
        return

    if C == 2:  # is a flow map
        vis_batch = flow_to_image(vis_batch)

    if rescale:
        vmax = vis_batch.amax(dim=(-1, -2), keepdim=True)
        vmax = torch.clamp_min(vmax, 1)
        vis_batch = vis_batch / vmax

    return save_batch_imgs(os.path.join(out_dir, name), vis_batch, save_dir)


def save_batch_imgs(name, vis_batch, save_dir):
    """
    Saves a 4+D tensor of (B, *, 3, H, W) in separate image dirs of B files.
    :param out_dir_pre prefix of output image directories
    :param vis_batch (B, *, 3, H, W)
    """
    vis_batch = vis_batch.detach().cpu()
    B, *dims, C, H, W = vis_batch.shape
    vis_batch = vis_batch.view(B, -1, C, H, W)
    vis_batch = (255 * vis_batch.permute(0, 1, 3, 4, 2)).byte()
    M = vis_batch.shape[1]

    paths = []
    for m in range(M):
        if B == 1:  # save single image
            path = f"{name}_{m}.png"
            imageio.imwrite(path, vis_batch[0, m])
        elif save_dir:  # save directory of images
            path = f"{name}_{m}"
            save_img_dir(path, vis_batch[:, m])
        else:  # save gif
            path = f"{name}_{m}.mp4"
            imageio.mimwrite(path, vis_batch[:, m], fps=10)

        paths.append(path)
    return paths


def save_img_dir(out, vis_batch):
    os.makedirs(out, exist_ok=True)
    for i in range(len(vis_batch)):
        path = f"{out}/{i:05d}.png"
        imageio.imwrite(path, vis_batch[i])


def save_metric(out_dir, out_dict, val_writer, step_ct, name="psnr"):
    os.makedirs(out_dir, exist_ok=True)
    if name not in out_dict:
        return

    vec = out_dict[name].detach().cpu()
    if len(vec.shape) > 2:
        return

    ok = (vec >= 0).all(dim=-1)
    vec = vec[ok]
    # if name == "psnr":
    writer_psnr = vec.mean(dim=1)
    if val_writer is not None and writer_psnr.nelement() > 0:
        val_writer.add_scalar(f"psnr", writer_psnr.item(), step_ct)
    # elif name == "ssim":
    #     writer_ssim = vec.mean(dim=1)
    #     if val_writer is not None and writer_ssim.nelement() > 0:
    #         val_writer.add_scalar(f"ssim", writer_ssim.item(), step_ct)
    
    np.savetxt(os.path.join(out_dir, f"frame_{name}.txt"), vec, delimiter='\n', fmt='%.8f')
    np.savetxt(os.path.join(out_dir, f"mean_{name}.txt"), vec.mean(dim=1), fmt='%.8f')


def pad_cat_groups_vert(tensor, pad=4):
    """
    :param tensor (B, M, 3, h, w)
    :param pad (int)
    """
    padded = TF.pad(tensor, pad, fill=1)  # (B, M, 3, h+2*pad, w+2*pad)
    B, M, C, H, W = padded.shape
    catted = padded.transpose(1, 2).reshape(B, C, -1, W)
    return catted[..., pad:-pad, :]  # remove top-most and bottom-most padding


def make_grid_vis(vis_dict, pad=4):
    """
    make panel vis with input, layers, view_vis, textures, and recon
    :param rgb (B, 3, H, W)
    :param recons (B, 3, H, W)
    :param layers (B, M, 3, H, W)
    :param texs (1, M, 3, H, W)
    :param view_vis (B, M, 3, H, W)
    """
    required = ["rgb", "recons", "layers", "texs", "view_vis"]
    if not all(x in vis_dict for x in required):
        print(f"not all keys in vis_dict, cannot make grid vis")
        return

    rgb = vis_dict["rgb"]
    N, _, h, w = rgb.shape
    texs_rs = TF.resize(
        vis_dict["texs"][0], size=(h, w), antialias=True
    )  # (M, 3, h, w)
    texs_rs = texs_rs[None].repeat(N, 1, 1, 1, 1)  # (N, M, 3, h, w)

    texs_vert = pad_cat_groups_vert(texs_rs, pad=pad)
    layers_vert = pad_cat_groups_vert(vis_dict["layers"], pad=pad)
    tforms_vert = pad_cat_groups_vert(vis_dict["view_vis"], pad=pad)

    N, _, H, _ = texs_vert.shape
    diff = (H - h) // 2
    rgb_pad = TF.pad(rgb, (0, diff, pad, H - h - diff), fill=1)
    recon_pad = TF.pad(vis_dict["recons"], (pad, diff, 0, H - h - diff), fill=1)

    final = torch.cat([rgb_pad, texs_vert, tforms_vert, layers_vert, recon_pad], dim=-1)
    return final

def save_vid(path, vis_batch):
    """
    :param vis_batch (B, 3, H, W)
    """
    vis_batch = vis_batch.detach().cpu()
    save = (255 * vis_batch.permute(0, 2, 3, 1)).byte()
    imageio.mimwrite(path, save)

def save_grid_vis(out_dir, vis_dict, pad=4):
    grid_keys = ["rgb", "recons", "layers", "texs", "view_vis"]
    if not all(x in vis_dict for x in grid_keys):
        print(f"not all keys in vis_dict, cannot save to {out_dir}")
        return

    vis_dict = {k: v.detach().cpu() for k, v in vis_dict.items()}
    os.makedirs(out_dir, exist_ok=True)
    grid = make_grid_vis(vis_dict, pad=pad)
    grid_path = os.path.join(out_dir, "grid_vis.mp4")
    save_vid(grid_path, grid)


def save_res_img_dirs(out_dir, vis_dict, save_keys):
    for save in save_keys:
        save_dir = os.path.join(out_dir, save)
        save_batch_imgs(save_dir, vis_dict[save], True)


def cat_tensor_dicts(dict_list, dim=0):
    if len(dict_list) < 1:
        return {}
    keys = dict_list[0].keys()
    return {k: torch.cat([d[k] for d in dict_list], dim=dim) for k in keys}


def move_to(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, dict):
        return dict([(k, move_to(v, device)) for k, v in item.items()])
    if isinstance(item, (tuple, list)):
        return [move_to(x, device) for x in item]
    print(type(item))
    raise NotImplementedError