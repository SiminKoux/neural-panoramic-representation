import torch
import numpy as np
from torch.utils.data import(
    DataLoader,
    BatchSampler,
    SubsetRandomSampler,
)

from getData.rgbDataset import RGBDataset
from getData.maskDataset import MaskDataset
from getData.comDataset import CompositeDataset

from getData.get_path import get_data_dirs


def get_dataset(args):
    print("------------------------------------")
    print("Getting the customized Dataset...")
    rgb_dir, gt_dir = get_data_dirs(args.root, args.name)
    print("rgb_dir is:", rgb_dir)
    print("gt_dir is:", gt_dir)
    required_dirs = [rgb_dir, gt_dir]
    assert all(d is not None for d in required_dirs), required_dirs

    rgb_dset = RGBDataset(rgb_dir, scale=args.scale, ext = ".png")
    mask_dset = MaskDataset(gt_dir, rgb_dset=rgb_dset)
    dsets = {"rgb": rgb_dset, "gt": mask_dset}

    return CompositeDataset(dsets)


def get_theta_phi(H, W):
    '''
    params: H, W are the height and width of the frames
    return: Tensor [2, H, W]
        The obtained canonical spherical coordinates:
        (1)'theta' is in range of [-pi, pi];
        (2)'phi' is in range of [-pi/2, pi/2].
    '''
    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')  # [H, W]
    # theta = 2 * torch.pi * (u / (W-1)) - torch.pi # [-pi, pi] without zero
    theta = 2 * torch.pi * (u / W) - torch.pi # [-pi, pi) with zero (but no pi)
    phi = torch.pi * (v / H) - (torch.pi / 2)  # [-pi/2, pi/2) with zero (but no North pole)
    
    theta_phi = torch.zeros((2, H, W), dtype = torch.float32)
    theta_phi[0, :, :] = theta
    theta_phi[1, :, :] = phi
    
    return theta_phi

def get_sph(H, W):
    '''
    params: H, W are the height and width of the frames
    return: Tensor [3, H, W]
        The obtained canonical coordinates are all in the range [-1,1].
    '''
    theta_phi = get_theta_phi(H, W)
    theta = theta_phi[0, ...]   # [H, W]
    phi = theta_phi[1, ...]     # [H, W]

    # more common for computer graphics
    x = torch.cos(phi) * torch.cos(theta)
    y = torch.cos(phi) * torch.sin(theta)
    z = torch.sin(phi)
    
    spherical_coords = torch.zeros((3, H, W), dtype = torch.float32)
    spherical_coords[0, :, :] = x
    spherical_coords[1, :, :] = y
    spherical_coords[2, :, :] = z
    
    return spherical_coords

def get_xyzt(N, H, W):
    '''
    params: N (int), the number of input video frames
    params: H, W (int), the height and width of each input video frame
    return: xyzt_all (Tensor), [4, H*W*N]
    '''
    sph = get_sph(H, W)
    input_sph = sph.reshape(3, H*W)
    xyzt_all = []
    for t in range(N):
        relxs = input_sph[0,:]  # cos(phi) * cos(theta) 
        relxs = torch.clamp(relxs, min=relxs.min()+(1e-2),
                            max=relxs.max()-(1e-2))  # Add disturbance term
        relys = input_sph[1,:]  # cos(phi) * sin(theta)
        relys = torch.clamp(relys, min=relys.min()+(1e-2),
                            max=relys.max()-(1e-2))  # Add disturbance term
        relzs = input_sph[2,:]  # sin(phi)
        relzs = torch.clamp(relzs, min=relzs.min()+(1e-2),
                            max=relzs.max()-(1e-2))  # Add disturbance term
        xyzt_all.append(torch.stack(((relxs, relys, relzs, t * torch.ones_like(relxs)))))
    xyzt_patch = torch.cat(xyzt_all, dim = 1)  # [4, H*W*N]
    return xyzt_patch

def get_xy(H, W):
    x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    xy_coords = torch.zeros((2, H, W), dtype = torch.int64)  # type: int
    xy_coords[0, :, :] = x
    xy_coords[1, :, :] = y
    return xy_coords

def get_xyt(N, H, W):
    x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    # xy_coords = torch.zeros((2, H, W), dtype = torch.float32)
    xy_coords = torch.zeros((2, H, W), dtype = torch.int64)
    xy_coords[0, :, :] = x
    xy_coords[1, :, :] = y
    input_xy = xy_coords.reshape(2, H*W)
    xyt_all = []
    for t in range(N):
        relxs = input_xy[0,:]
        relys = input_xy[1,:]
        xyt_all.append(torch.stack(((relxs, relys, t * torch.ones_like(relxs)))))
    xyt_patch = torch.cat(xyt_all, dim=1) # [3, H*W*N]
    return xyt_patch


def convert_uv_to_uvw(uv, H, W):
    u = uv[:, 0] # [resolution, 1], range: (-1, 1)
    v = uv[:, 1] # [resolution, 1], all elements are the same
    u = (u + 1) * 0.5 * (W-1)
    v = (v + 1) * 0.5 * (H-1)
    
    theta = 2 * torch.pi * (u / (W - 1)) - torch.pi  # [H, W] -> [-pi, pi]
    phi = torch.pi * (v / (H - 1)) - torch.pi / 2    # [H, W] -> [-pi/2, pi/2]
    
    x = torch.cos(phi) * torch.sin(theta)   # (-1, 1)
    y = torch.sin(phi)
    z = torch.cos(phi) * torch.cos(theta)   # (-1, 1)

    target_uvw = torch.zeros((uv.shape[0], 3), dtype = torch.float32)
    target_uvw[:, 0] = x
    target_uvw[:, 1] = y
    target_uvw[:, 2] = z

    return target_uvw
    
def convert_uvw_to_uv(uvw, hight, width):
    u = uvw[:, 0]  # [samples, 1]
    v = uvw[:, 1]  # [samples, 1]
    w = uvw[:, 2]  # [samples, 1]

    target_uv = torch.zeros((uvw.shape[0], 2), dtype = torch.float32)
    r = torch.sqrt(u**2 + v**2 + w**2)
    theta = torch.atan2(v, u)  # [-pi, pi]
    phi = torch.asin(w/r)      # [-pi/2, pi/2]

    target_uv[:, 0] = theta / torch.pi
    target_uv[:, 1] = phi / (torch.pi/2)

    return target_uv

def convert_xyt_to_xyzt_current(xyt_current, N, H, W):
    u = xyt_current[:, 0]  # [samples, 1]
    v = xyt_current[:, 1]  # [samples, 1]
    t = xyt_current[:, 2]  # [samples, 1]
    
    # Check if there are any zeros in the tensor
    if not torch.any(u == 0):
        # If there are no zeros, replace the maximum element with zero
        u[u == torch.max(u)] = 0
    if not torch.any(v == 0):
        # If there are no zeros, replace the maximum element with zero
        v[v == torch.max(v)] = 0
    
    # print("con_u:", u)
    # print("con_v:", v)

    theta = 2 * torch.pi * (u / W) - torch.pi # [-pi, pi) with zero (but no pi)
    phi = torch.pi * (v / H) - (torch.pi / 2)  # [-pi/2, pi/2) with zero (but no North pole)
    # print("con_theta:", theta)
    # print("con_phi:", phi)

    x = torch.cos(phi) * torch.cos(theta)
    y = torch.cos(phi) * torch.sin(theta)
    z = torch.sin(phi)

    xyz_current = torch.zeros((3, xyt_current.shape[0]), dtype = torch.float32)
    xyz_current[0, :] = x
    xyz_current[1, :] = y
    xyz_current[2, :] = z

    # xyz_current = torch.clamp(target_tensor, 
    #                           min=target_tensor.min()+(1e-2), 
    #                           max=target_tensor.max()-(1e-2))
    
    xyzt_current = torch.cat((xyz_current[0].unsqueeze(-1), 
                              xyz_current[1].unsqueeze(-1), 
                              xyz_current[2].unsqueeze(-1), 
                              t.unsqueeze(-1) / (N / 2.0) - 1), dim=1)  # [batch, 4]
    
    return xyzt_current


def add_disturbance(tensor):
    if torch.any(tensor == -1.0):
        # Find the indices where tensor equals the specific value (-1.0)
        indices = torch.where(tensor == (-1.0))
        # Change the values at -1.0+(1e-2)
        tensor[indices] = tensor[indices] + (1e-2)

    if torch.any(tensor == 1.0):
        # Find the indices where tensor equals the specific value (1.0)
        indices = torch.where(tensor == 1.0)
        # Change the values at 1.0+(1e-2)
        tensor[indices] = tensor[indices] - (1e-2)

    return tensor

def get_dx_dy(xyt_current, xplus1yt_current, xyplus1t_current, H, W):
    x = xyt_current[0, :].squeeze(1)  # [samples]
    y = xyt_current[1, :].squeeze(1)  # [samples]

    xplus1 = xplus1yt_current[:, 0]   # [samples]
    yplus1 = xyplus1t_current[:, 1]   # [samples]
    x_yplus1 = xyplus1t_current[:, 0] # [samples]

    # There is a pixel out of right side!
    if torch.any(x == (W-1)):
        target_x_indices = (x == (W-1)).nonzero().squeeze()
        xplus1[target_x_indices] = 0

    # There is a pixel out of bottom side!
    if torch.any(y == (H-1)):
        target_y_indices = (y == (H-1)).nonzero().squeeze()
        yplus1[target_y_indices] = (H-1)
        x_yplus1[target_y_indices] += (W//2)
        # Mask of elements greater than (W-1)
        mask_H = x_yplus1[target_y_indices] > (W-1)
        # Prepare the values to be scattered
        value_H = x_yplus1[target_y_indices][mask_H] - W
        # Get a mask for the entire x_ tensor
        full_mask_H = torch.zeros_like(x_yplus1, dtype=torch.bool)
        full_mask_H[target_y_indices[mask_H]] = True
        # Use masked_scatter_ to replace the values in x_
        x_yplus1.masked_scatter_(full_mask_H, value_H)
    
    return xplus1, yplus1, x_yplus1

def convert_xygrid_to_cartesian_xyz(grid_x, grid_y, H, W):
    '''
        grid_x: Tensor of shape [samples]
        grid_y: Tensor of shape [samples]
        Return: Tensor of shape [3, samples]
    '''
    # theta = 2 * torch.pi * (grid_x / (W-1)) - torch.pi # [-pi, pi] without zero
    theta = 2 * torch.pi * (grid_x / W) - torch.pi # [-pi, pi) with zero (but no pi)
    phi = torch.pi * (grid_y / H) - (torch.pi / 2)  # [-pi/2, pi/2) with zero (but no North pole)

    # more common for computer graphics
    x = torch.cos(phi) * torch.cos(theta)
    y = torch.cos(phi) * torch.sin(theta)
    z = torch.sin(phi)

    xyz = torch.zeros((3, grid_x.shape[0]), dtype = torch.float32)
    xyz[0, :] = x
    xyz[1, :] = y
    xyz[2, :] = z

    return xyz

def get_xyz_dx_dy(xyt_current, xplus1yt_current, xyplus1t_current, N, H, W):
    '''
     Get the gradient of the xyt_cuurent: (x+1, y, t) and (x, y+1, t)
      (1) xyt_current: Tensor of shape [3, samples, 1]
      (2) xplus1yt_current: Tensor of shape [samples, 3]
      (3) xyplus1t_current: Tensor of shape [samples, 3]
      Return: 
      'xyzt_dx' and 'xyzt_dy', Tensors of shape [samples, 4]
    '''
    y = xyt_current[1, :].squeeze(1)  # [samples]
    t = xyt_current[2, :]  # [samples]

    xplus1, yplus1, x_yplus1 = get_dx_dy(xyt_current, 
                                         xplus1yt_current, 
                                         xyplus1t_current, 
                                         H, W)

    # get xyzt_dx
    xyz_dx_current = convert_xygrid_to_cartesian_xyz(xplus1, y, H, W)   # [3, samples]
    xyzt_dx = torch.cat((add_disturbance(xyz_dx_current[0, :]).unsqueeze(-1), 
                         add_disturbance(xyz_dx_current[1, :]).unsqueeze(-1), 
                         add_disturbance(xyz_dx_current[2, :]).unsqueeze(-1), 
                         t / (N / 2.0) - 1), dim=1)  # [samples, 4]
   
    # get xyzt_dy
    xyz_dy_current = convert_xygrid_to_cartesian_xyz(x_yplus1, yplus1, H, W)  # [3, samples]
    xyzt_dy = torch.cat((add_disturbance(xyz_dy_current[0, :]).unsqueeze(-1), 
                         add_disturbance(xyz_dy_current[1, :]).unsqueeze(-1), 
                         add_disturbance(xyz_dy_current[2, :]).unsqueeze(-1), 
                         t / (N / 2.0) - 1), dim=1)  # [samples, 4]

    return xyzt_dx, xyzt_dy

def convert_xygrid_to_theta_phi(grid_x, grid_y, H, W):
    '''
        grid_x: Tensor of shape [samples]
        grid_y: Tensor of shape [samples]
        Return: Tensor of shape [2, samples]
    '''
    theta = 2 * torch.pi * (grid_x / W) - torch.pi # [-pi, pi) with zero (but no pi)
    phi = torch.pi * (grid_y / H) - (torch.pi / 2)  # [-pi/2, pi/2) with zero (but no North pole)


    theta_phi = torch.zeros((2, grid_x.shape[0]), dtype = torch.float32)
    theta_phi[0, :] = theta / torch.pi
    theta_phi[1, :] = phi / (torch.pi / 2)

    return theta_phi

def get_theta_phi_dx_dy(xyt_current, xplus1yt_current, xyplus1t_current, H, W):
    '''
     Get the gradient of the xyt_cuurent: (x+1, y, t) and (x, y+1, t)
      (1) xyt_current: Tensor of shape [3, samples, 1]
      (2) xplus1yt_current: Tensor of shape [samples, 3]
      (3) xyplus1t_current: Tensor of shape [samples, 3]
      Return: 
      'theta_phi_dx' and 'theta_phi_dy', Tensors of shape [samples, 3]
    '''
    y = xyt_current[1, :].squeeze(1)  # [samples]
    
    xplus1, yplus1, x_yplus1 = get_dx_dy(xyt_current, 
                                         xplus1yt_current, 
                                         xyplus1t_current, 
                                         H, W)

    # get theta_phi_t_dx
    theta_phi_dx = convert_xygrid_to_theta_phi(xplus1, y, H, W)  # [2, samples]
    theta_offset = torch.cat((theta_phi_dx[0, :].unsqueeze(-1),
                              theta_phi_dx[1, :].unsqueeze(-1)), 
                              dim=1)   # [samples, 2]
    
    # get theta_phi_t_dy
    theta_phi_dy = convert_xygrid_to_theta_phi(x_yplus1, yplus1, H, W)  # [2, samples]
    phi_offset = torch.cat((theta_phi_dy[0, :].unsqueeze(-1),
                            theta_phi_dy[1, :].unsqueeze(-1)), 
                            dim=1)   # [samples, 2]

    return theta_offset, phi_offset


def get_ordered_loader(dset, batch_size, preloaded):
    print("Get the evaluation dataloader...")
    num_workers = batch_size if not preloaded else 0
    persistent_workers = True if num_workers > 0 else False  # True
    return DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=persistent_workers,
        shuffle=False,
        persistent_workers=persistent_workers,
    )

def get_random_ordered_batch_loader(dset, batch_size, preloaded, min_batch_size=None):
    print("Get the training dataloader...")
    total_size = len(dset)  # numbers of video frames
    if min_batch_size is None:
        # min_batch_size = batch_size // 2
        min_batch_size = batch_size - 1
    
    # create a list of indices for the dataset
    idcs = list(range(total_size - min_batch_size)) # [0,1,...,(total_size-min_batch_size)-1]

    # The parameters of the DataLoader class/object
    sampler = SubsetRandomSampler(idcs)  # sample randomly without replacement
    batch_sampler = OrderedBatchSampler(sampler, total_size, batch_size)
    num_workers = batch_size if not preloaded else 0
    persistent_workers = True if num_workers > 0 else False   # True
    
    return DataLoader(
        dset,
        num_workers=1,
        batch_sampler=batch_sampler,
        pin_memory=persistent_workers,
        persistent_workers=persistent_workers,
    )

class OrderedBatchSampler(BatchSampler):
    """
    For any base sampler, 
    make a batch with the ordered elements after the base sampled index
    Sampler -> i -> OrderedBatchSampler -> [i, i+1, ..., i+batch_size-1]
    """

    def __init__(self, sampler, total_size, batch_size):
        super().__init__(sampler, batch_size, drop_last=False)
        self.total_size = total_size
        self.batch_size = batch_size

    def __iter__(self):
        """
        returns an iterator returning batch indices
        """
        for idx in self.sampler:
            n_batch = min(self.batch_size, self.total_size - idx)
            # returns a list of indices that represents a single batch
            yield [idx + i for i in range(n_batch)] 

    def __len__(self):
        return len(self.sampler)