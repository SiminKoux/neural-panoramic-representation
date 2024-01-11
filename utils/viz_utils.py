import cv2
import torch
import numpy as np

def get_sign_image(tensor):
    """
    :param tensor (*, 1, H, W) image-like single-channel tensor
    """
    vmax = torch.abs(tensor).amax(dim=(-1, -2), keepdim=True)  # (*, 1, 1, 1)
    pos = torch.zeros_like(tensor)
    pos[tensor > 0] = tensor[tensor > 0]
    neg = torch.zeros_like(tensor)
    neg[tensor < 0] = -tensor[tensor < 0]
    sign_im = (
        torch.cat([pos, neg, torch.zeros_like(tensor)], dim=-3) / vmax
    )  # (*, 3, H, W)
    return sign_im, vmax

def composite_rgba_checkers(masks, rgbs, n_rows=24, fac=0.2):
    """
    :param masks (*, 1, H, W)
    :param rgbs (*, 3, H, W)
    """
    *dims, _, H, W = masks.shape
    checkers = get_gray_checkerboard(H, W, n_rows, fac, device=masks.device)
    checkers = checkers.view(*(1,) * len(dims), 3, H, W)
    return masks * rgbs + (1 - masks) * checkers

def get_gray_checkerboard(H, W, n_rows, fac=0.2, shade=0.7, device=None):
    '''
    generates a gray checkerboard pattern and returns it as a tensor
    param H, W: int, the height and width of the checkerboard pattern in pixels
    param n_rows: int, the number of rows in the checkerboard pattern
    param fac: float, the size of each square in the checkerboard pattern relative to the size of the input images
    param shade: the shade of gray to use for the checkerboard squares
    Return: Tensor (3, H, W)
    '''
    if device is None:
        device = torch.device("cpu")
    checkers = get_checkerboard(H, W, n_rows, device=device)
    bg = torch.ones(3, H, W, device=device, dtype=torch.float32) * shade
    return fac * checkers + (1 - fac) * bg
 
def get_checkerboard(H, W, n_rows, device=None):
    '''
    generate a checkerboard pattern and return it as a tensor
    param H, W: int, the height and width of the checkerboard pattern in pixels
    param n_rows: int, the number of rows in the checkerboard pattern
    Return: Tensor (3, H, W)
    '''
    if device is None:
        device = torch.device("cpu")
    stride = H // n_rows
    n_cols = W // stride
    checkers = np.indices((n_rows, n_cols)).sum(axis=0) % 2  # contains alternating 0s and 1s
    checkers = cv2.resize(checkers, (W, H), interpolation=cv2.INTER_NEAREST)
    checkers = torch.tensor(checkers, device=device, dtype=torch.float32)
    return checkers[None].repeat(3, 1, 1)

# if __name__ == "__main__":
#     from PIL import Image
#     import numpy as np
#     checkers = get_gray_checkerboard(240, 480, 24)
#     array = (checkers.cpu().numpy() * 255).astype(np.uint8)
#     image = Image.fromarray(np.transpose(array, (1,2,0)))
#     image.save('./gray_checkboard.png')