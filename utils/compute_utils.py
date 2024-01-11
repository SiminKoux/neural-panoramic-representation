import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import getData
import numpy as np

def compute_psnr(batch_in, batch_out):
    '''
    :param mapped_rgb (B, 3, H, W)
    :param ori_rgb (B, 3, H, W)
    :returns psnr (B) chooses the best psnr for each frame
    '''
    if "rgb" not in batch_in:
        return batch_out
    
    gt_rgb_total = batch_in["rgb"]
    _, _, H, W = gt_rgb_total.shape
    if len(batch_in["idx"]) < 1:
        return batch_out
    
    psnr = []
    with torch.no_grad():
        for i in range(len(batch_in["idx"])):
            gt_rgb = batch_in["rgb"][i].permute(1, 2, 0)
            recon_rgb = batch_out["recons"][i].permute(1, 2, 0)
            psnr.append(peak_signal_noise_ratio(gt_rgb.cpu().numpy(), recon_rgb.cpu().numpy()))
    
    psnr_tensor = torch.tensor(psnr)
    batch_out["psnr"] = psnr_tensor
    return batch_out

def compute_ssim(batch_in, batch_out):
    '''
    :param mapped_rgb (B, 3, H, W)
    :param ori_rgb (B, 3, H, W)
    :returns ssim (B) chooses the best ssim for each frame
    '''
    if "rgb" not in batch_in:
        return batch_out
    
    gt_rgb_total = batch_in["rgb"]
    _, _, H, W = gt_rgb_total.shape
    if len(batch_in["idx"]) < 1:
        return batch_out
    
    ssim = []
    with torch.no_grad():
        for i in range(len(batch_in["idx"])):
            gt_rgb = batch_in["rgb"][i].permute(1, 2, 0)
            recon_rgb = batch_out["recons"][i].permute(1, 2, 0)
            ssim.append(structural_similarity(gt_rgb.cpu().numpy(), recon_rgb.cpu().numpy(), multichannel=True))
    
    ssim_tensor = torch.tensor(ssim)
    batch_out["ssim"] = ssim_tensor
    return batch_out

def compute_multiple_iou(batch_in, batch_out):
    """
    :param masks (B, M, *, H, W)
    :param gt (B, C, H, W)
    :returns iou (B, M) chooses the best iou for each mask
    """
    if "gt" not in batch_in:
        return batch_out

    gt, ok = batch_in["gt"]
    print("gt_masks:", gt.shape)
    if ok.sum() < 1:
        return batch_out

    with torch.no_grad():
        masks = batch_out["masks"]
        print("masks:", masks.shape)

        B, C, H, W = gt.shape
        masks_bin = masks.view(B, -1, 1, H, W) > 0.5
        gt_bin = gt.view(B, 1, C, H, W) > 0.5
        ious = compute_iou(masks_bin, gt_bin, dim=(-1, -2))  # (B, M, C)
        ious = ious.amax(dim=-1)  # (B, M)
        ious[~ok] = -1

    batch_out["ious"] = ious
    return batch_out


def compute_iou(pred, target, dim=None):
    """
    Compute region similarity as the Jaccard Index.
    :param pred (binary tensor) prediction
    :param target (binary tensor) ground truth
    :param dim (optional, int) the dimension to reduce across
    :returns jaccard (float) region similarity
    """
    intersect = pred & target
    union = pred | target
    if dim is None:
        intersect = intersect.sum()
        union = union.sum()
    else:
        intersect = intersect.sum(dim)
        union = union.sum(dim)
    return (intersect + 1e-6) / (union + 1e-6)


def get_uv_grid(H, W, homo=False, align_corners=False, device=None):
    """
    Get uv grid renormalized from -1 to 1
    :returns (H, W, 2) tensor
    """
    if device is None:
        device = torch.device("cpu")
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    if align_corners:
        xx = 2 * xx / (W - 1) - 1
        yy = 2 * yy / (H - 1) - 1
    else:
        xx = 2 * (xx + 0.5) / W - 1
        yy = 2 * (yy + 0.5) / H - 1
    if homo:
        return torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    return torch.stack([xx, yy], dim=-1)


def get_flow_coords(flow, align_corners=False):
    """
    :param flow (*, H, W, 2) normalized flow vectors
    :returns (*, H, W, 2)
    """
    device = flow.device
    *dims, H, W, _ = flow.shape
    uv = get_uv_grid(H, W, homo=False, align_corners=align_corners, device=device)
    uv = uv.view(*(1,) * len(dims), H, W, 2)
    return uv + flow


def inverse_flow_warp(I2, F_12, O_12=None):
    """
    Given image I2 and the flow field from I1 to I2, sample I1 from I2,
    except at points that are disoccluded
    :param I2 (B, C, H, W)
    :param F_12 flow field from I1 to I2 in uv coords (B, H, W, 2)
    :param O_12 (optional) mask of disocclusions (B, 1, H, W)
    """
    C_12 = get_flow_coords(F_12, align_corners=False)
    I1 = F.grid_sample(I2, C_12, align_corners=False)
    if O_12 is not None:
        mask = ~(O_12 == 1)
        I1 = mask * I1
    return I1


# In case of 2D/planar optical flow with one layer
def get_matching_points(xyt_current,
                        flows, 
                        number_of_frames,
                        height,
                        width,
                        is_forward):
    
    xyt_current_flow = xyt_current[:, :, 0]   # [3, sample_batch]
    # print("xyt_current_flow:",xyt_current_flow.shape )
    frames_amount = torch.ones(xyt_current_flow.size(1), dtype=torch.int64)  # [1, 1, ..., 1] shape: [sample_batch]
    
    flows_cpu = flows.cpu()
    flows_for_loss = flows_cpu[xyt_current_flow[2], :, xyt_current_flow[1], xyt_current_flow[0]] # [2, sample_batch]
    # print("flows_for_loss:", flows_for_loss.shape)
    if is_forward:
        xyt_current_should_match = torch.stack(
            (xyt_current_flow[0] + flows_for_loss[:, 0],
             xyt_current_flow[1] + flows_for_loss[:, 1],
             xyt_current_flow[2] + frames_amount))
    else:
        xyt_current_should_match = torch.stack(
            (xyt_current_flow[0] + flows_for_loss[:, 0],
             xyt_current_flow[1] + flows_for_loss[:, 1],
             xyt_current_flow[2] - frames_amount))
    
    xyt_current_should_match = xyt_current_should_match.permute(1, 0)
    # print("xyt_current_should_match.shape:", xyt_current_should_match.shape)
    xyzt_current_should_match = getData.convert_xyt_to_xyzt_current(xyt_current_should_match, 
                                                            number_of_frames,
                                                            height, 
                                                            width)
    return xyzt_current_should_match


def compute_occlusion_locs(fwd, bck, gap, method="brox", thresh=1.5, ret_locs=False):
    """
    compute the locations of the occluding pixels using round-trip flow
    :param fwd (N, 2, H, W) flow from 1->2, 2->3, etc
    :param bck (N, 2, H, W) flow from 2->1, 3->2, etc
    :param method (str) brox implementation taken from
        https://github.com/google-research/google-research/blob/master/uflow/uflow_utils.py#L312
        otherwise use a threshold on the forward backward distance
    :param thresh (float) if not using the brox method, the fb distance threshold to use
    :return occ_map (N, 1, H, W) bool binary mask of pixels that get occluded
            occ_locs (N, H, W, 2) O[i,j] location of the pixel that occludes the pixel at i, j
    """
    N, _, H, W = fwd.shape

    ## get the backward flow at the points the forward flow maps points to
    fwd_vec = fwd.permute(0, 2, 3, 1)
    inv_flo = inverse_flow_warp(bck, fwd_vec)  # (N, 2, H, W)
    fb_sq_diff = torch.square(fwd + inv_flo).sum(dim=1, keepdim=True)

    sq_thresh = (thresh / H) ** 2
    if method == "brox":
        fb_sum_sq = (fwd ** 2 + inv_flo ** 2).sum(dim=1, keepdim=True)
        occ_map = fb_sq_diff > (0.01 * fb_sum_sq + sq_thresh)
    else:  # use a simple threshold
        occ_map = fb_sq_diff > sq_thresh

    # get the mask of points that don't go out of frame
    uv_fwd = get_flow_coords(fwd_vec, align_corners=False)  # (N, H, W, 2)
    valid = ((uv_fwd < 0.99) & (uv_fwd > -0.99)).all(dim=-1)  # (N, H, W)
    occ_map = valid[:, None] & occ_map

    out = [occ_map]

    if ret_locs:
        # the inverse warped locs in the original image
        occ_locs = uv_fwd + inv_flo.permute(0, 2, 3, 1)
        out.append(occ_locs)
    return out

def compute_sampson_error(x1, x2, F):
    """
    :param x1 (*, N, 2)
    :param x2 (*, N, 2)
    :param F (*, 3, 3)
    """
    h1 = torch.cat([x1, torch.ones_like(x1[..., :1])], dim=-1)
    h2 = torch.cat([x2, torch.ones_like(x2[..., :1])], dim=-1)
    d1 = torch.matmul(h1, F.transpose(-1, -2))  # (B, N, 3)
    d2 = torch.matmul(h2, F)  # (B, N, 3)
    z = (h2 * d1).sum(dim=-1)  # (B, N)
    err = z ** 2 / (
        d1[..., 0] ** 2 + d1[..., 1] ** 2 + d2[..., 0] ** 2 + d2[..., 1] ** 2
    )
    return err


def get_mapping_area(uvw, H, W):
    minx = 1
    miny = 1
    maxx = -1
    maxy = -1

    uv = getData.convert_uvw_to_uv(uvw, H, W)  # range: [-1, 1]
    
    curminx = torch.min(uv[:, 0])
    curminy = torch.min(uv[:, 1])
    curmaxx = torch.max(uv[:, 0])
    curmaxy = torch.max(uv[:, 1])
    
    minx = torch.min(torch.tensor([curminx, minx]))
    miny = torch.min(torch.tensor([curminy, miny]))
    maxx = torch.max(torch.tensor([curmaxx, maxx]))
    maxy = torch.max(torch.tensor([curmaxy, maxy]))

    minx = np.maximum(minx, -1)  # tensor(-1.0000, dtype=torch.float64)
    miny = np.maximum(miny, -1)  # tensor(-0.9974, dtype=torch.float64)
    maxx = np.minimum(maxx, 1)   # tensor(0.9999, dtype=torch.float64)
    maxy = np.minimum(maxy, 1)   # tensor(0.9979, dtype=torch.float64)

    return maxx, minx, maxy, miny

def get_uv_area(uv):
    minu = 1
    minv = 1
    maxu = -1
    maxv = -1

    curminu = torch.min(uv[:, 0])
    curminv = torch.min(uv[:, 1])
    curmaxu = torch.max(uv[:, 0])
    curmaxv = torch.max(uv[:, 1])
    
    minu = torch.min(torch.tensor([curminu, minu]))
    minv = torch.min(torch.tensor([curminv, minv]))
    maxu = torch.max(torch.tensor([curmaxu, maxu]))
    maxv = torch.max(torch.tensor([curmaxv, maxv]))

    minu = np.maximum(minu, -1)  # tensor(-1.0000, dtype=torch.float64)
    minv = np.maximum(minv, -1)  # tensor(-0.9974, dtype=torch.float64)
    maxu = np.minimum(maxu, 1)   # tensor(0.9999, dtype=torch.float64)
    maxv = np.minimum(maxv, 1)   # tensor(0.9979, dtype=torch.float64)

    return minu, maxu, minv, maxv

def get_uvw_area(uvw):
    minu = 1
    minv = 1
    minw = 1
    maxu = -1
    maxv = -1
    maxw = -1

    curminu = torch.min(uvw[:, 0])
    curminv = torch.min(uvw[:, 1])
    curminw = torch.min(uvw[:, 2])
    curmaxu = torch.max(uvw[:, 0])
    curmaxv = torch.max(uvw[:, 1])
    curmaxw = torch.min(uvw[:, 2])
    
    minu = torch.min(torch.tensor([curminu, minu]))
    minv = torch.min(torch.tensor([curminv, minv]))
    minw = torch.min(torch.tensor([curminw, minw]))
    maxu = torch.max(torch.tensor([curmaxu, maxu]))
    maxv = torch.max(torch.tensor([curmaxv, maxv]))
    maxw = torch.max(torch.tensor([curmaxw, maxw]))

    minu = np.maximum(minu, -1)  # tensor(-1.0000, dtype=torch.float64)
    minv = np.maximum(minv, -1)  # tensor(-0.9974, dtype=torch.float64)
    minw = np.maximum(minw, -1)  # tensor(-0.9974, dtype=torch.float64)
    maxu = np.minimum(maxu, 1)   # tensor(0.9999, dtype=torch.float64)
    maxv = np.minimum(maxv, 1)   # tensor(0.9979, dtype=torch.float64)
    maxv = np.minimum(maxv, 1)   # tensor(0.9979, dtype=torch.float64)

    return minu, maxu, minv, maxv, minw, maxw

# Given uv points in the range (-1,1) and an image (with a given "resolution") 
# that represents a crop (defined by "minx", "maxx", "miny", "maxy")
# Change uv points to pixel coordinates, and sample points from the image
def get_colors(resolution, minx, maxx, miny, maxy, pointx, pointy):
    pixel_size = resolution / (maxx - minx)
    # Change uv to pixel coordinates of the discretized image
    pointx2 = ((pointx - minx) * pixel_size).numpy()
    pointy2 = ((pointy - miny) * pixel_size).numpy()

    # Relevant pixel locations should be positive
    pos_logicaly = np.logical_and(np.ceil(pointy2) >= 0, np.floor(pointy2) >= 0)
    pos_logicalx = np.logical_and(np.ceil(pointx2) >= 0, np.floor(pointx2) >= 0)
    pos_logical = np.logical_and(pos_logicaly, pos_logicalx)

    # Relevant pixel locations should be inside the image borders
    mx_logicaly = np.logical_and(np.ceil(pointy2) < resolution, np.floor(pointy2) < resolution)
    mx_logicaxlx = np.logical_and(np.ceil(pointx2) < resolution, np.floor(pointx2) < resolution)
    mx_logical = np.logical_and(mx_logicaly, mx_logicaxlx)

    # Relevant should satisfy both conditions
    relevant = np.logical_and(pos_logical, mx_logical)
    return pointx2[relevant], pointy2[relevant]

def get_relevant_pixel_coordinates(H, W, minx, maxx, miny, maxy, pointx, pointy):
    pixel_size_x = W / (maxx-minx)
    pixel_size_y = H / (maxy-miny)

    # convert uv to pixel coordinates of the discretized image
    pixel_x = ((pointx - minx) * pixel_size_x).numpy()
    pixel_y = ((pointy - miny) * pixel_size_y).numpy()

    # conditions for pixel locations to be inside image boundaries
    is_positive_x = np.logical_and(np.ceil(pixel_x) >= 0, np.floor(pixel_x) >= 0)
    is_positive_y = np.logical_and(np.ceil(pixel_y) >= 0, np.floor(pixel_y) >= 0)

    within_boundaries_x = np.logical_and(np.ceil(pixel_x)< W, np.floor(pixel_x) < W)
    within_boundaries_y = np.logical_and(np.ceil(pixel_y)< H, np.floor(pixel_y) < H)

    is_relevant = np.logical_and(np.logical_and(is_positive_x, is_positive_y), np.logical_and(within_boundaries_x, within_boundaries_y))

    return pixel_x[is_relevant], pixel_y[is_relevant]

def uv_to_uvw(uv, W):
    theta = uv[:, 0]
    phi = uv[:, 1]
    x = torch.cos(phi) * torch.cos(theta)
    y = torch.cos(phi) * torch.sin(theta)
    z = torch.sin(phi)
    
    spherical_coords = torch.zeros((W, 3), dtype = torch.float32)
    spherical_coords[:, 0] = x
    spherical_coords[:, 1] = y
    spherical_coords[:, 2] = z
    
    return spherical_coords