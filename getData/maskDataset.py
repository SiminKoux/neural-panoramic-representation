import os
import glob
import torch
from torch.utils.data import Dataset
from getData.get_data import *
from getData.get_path import get_path_name
from getData.load_tensor import load_img_tensor, load_mask_tensor

class MaskDataset(Dataset):
    def __init__(self, gt_dir, scale=1, rgb_dset=None):
        super().__init__()
        self.gt_dir = gt_dir

        # segtrack and fbms clean labels are organized by object number
        self.subdirs = sorted(glob.glob("{}/**/".format(gt_dir)))
        if len(self.subdirs) < 1:  # just the original input directory
            self.subdirs = [gt_dir]
        subd = self.subdirs[0]
        self.subd = subd
        self.n_channels = len(self.subdirs) + 1

        # find the examples that are labeled (usually all of them, except fbms)
        if rgb_dset:
            self.names = rgb_dset.names
            self.scale, self.height, self.width = (rgb_dset.scale, rgb_dset.height, rgb_dset.width)
            print("MASK Scale {}, Shape (1, {}, {})".format(scale, self.height, self.width))
        else:
            files = sorted(glob.glob("{}/*.png".format(subd)))
            self.names = [get_path_name(f) for f in files]
            print("FOUND {} files in {}".format(len(files), subd))

            self.scale = scale
            test = load_img_tensor(files[0], scale)
            self.height, self.width = test.shape[-2:]
            print("MASK Scale {}, Shape {}".format(scale, test.numpy().shape))

        paths = [os.path.join(subd, "{}.png".format(name)) for name in self.names]
        self.is_valid = [os.path.isfile(path) for path in paths]
        self.val_idcs = [i for i, path in enumerate(paths) if os.path.isfile(path)]
        print("FOUND {} matching masks in {}".format(len(self.val_idcs), gt_dir))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.is_valid[idx]:
            name = self.names[idx]
            path = os.path.join(self.subd, "{}.png".format(name))
            img = load_mask_tensor(path, self.height, self.width)
            # print("img:", img.shape)
            return img, torch.tensor(1, dtype=bool)

        ## return an empty mask
        img = torch.zeros(self.n_channels, self.height, self.width, dtype=torch.float32)
        return img, torch.tensor(0, dtype=bool)
