import os
import torch
import numpy as np
from PIL import Image
import sys
import cv2

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]

def is_image(path):
    path = path.lower()
    ext = os.path.splitext(path)[-1]
    return ext == ".png" or ext == ".jpg" or ext == ".bmp"

def read_img(path, scale=1):
    im = Image.open(path)
    if scale != 1:
        W, H = im.size
        w, h = int(scale * W), int(scale * H)
        im = im.resize((w, h), Image.ANTIALIAS)
    return im

def load_img_tensor(path, scale=1):
    """
    Load image, rescale to [0., 1.]
    Returns (C, H, W) float32
    """
    im = read_img(path, scale)
    tensor = torch.from_numpy(np.array(im))
    if tensor.ndim < 3:
        tensor = tensor[..., None]
    return tensor.permute(2, 0, 1) / 255.0 
    # return tensor.permute(0, 1, 2) / 255.0   # tensor.permute(0,1,2) is to get (H,W,C)

def load_mask_tensor(path, H, W):
    """
    Load mask, 
        Returns (H, W) 
    """
    # mask_im = read_img(path, scale)
    mask_im = Image.open(path)  # [4096x2048]
    mask_tensor = np.array(mask_im).astype(np.float64) / 255.
    mask = cv2.resize(mask_tensor, (W, H), cv2.INTER_NEAREST)
    mask_frame = torch.from_numpy(mask)
    return mask_frame

TAG_FLOAT = 202021.25

def read_flo(filename):
    """
    returns (H, W, 2) numpy array flow field
    """
    assert type(filename) is str, "filename is not str %r" % str(filename)
    assert os.path.isfile(filename) is True, "file does not exist %r" % str(filename)
    assert filename[-4:] == ".flo", "file ending is not .flo %r" % filename[-4:]
    
    f = open(filename, "rb")
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, (
        "Flow number %r incorrect. Invalid .flo file" % flo_number
    )
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
    
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow

def write_flo(filename, uv):
    """
    According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
    Contact: dqsun@cs.brown.edu
    Contact: schar@middlebury.edu
    """
    TAG_STRING = np.array(TAG_FLOAT, dtype=np.float32)
    if uv.shape[2] != 2:
        sys.exit("write_flo: flow must have two bands!")
    H = np.array(uv.shape[0], dtype=np.int32)
    W = np.array(uv.shape[1], dtype=np.int32)
    with open(filename, "wb") as f:
        f.write(TAG_STRING.tobytes())
        f.write(W.tobytes())
        f.write(H.tobytes())
        f.write(uv.tobytes())

def load_flow_tensor(path, scale=1, normalize=True, align_corners=True):
    """
    Load flow, scale the pixel values according to the resized scale.
    If normalize is true, return rescaled in normalized pixel coordinates
    where pixel coordinates are in range [-1, 1].
    NOTE: RAFT USES ALIGN_CORNERS=TRUE SO WE NEED TO ACCOUNT FOR THIS
    Returns (2, H, W) float32
    """
    flow = read_flo(path).astype(np.float32)
    H, W, _ = flow.shape
    u, v = flow[..., 0], flow[..., 1]
    if normalize:
        if align_corners:
            u = 2.0 * u / (W - 1)
            v = 2.0 * v / (H - 1)
        else:
            u = 2.0 * u / W
            v = 2.0 * v / H
    else:
        u = scale * u
        v = scale * v

    if scale != 1:
        h, w = int(scale * H), int(scale * W)
        u = Image.fromarray(u).resize((w, h), Image.ANTIALIAS)
        v = Image.fromarray(v).resize((w, h), Image.ANTIALIAS)
        u, v = np.array(u), np.array(v)
    return torch.from_numpy(np.stack([u, v], axis=0))
