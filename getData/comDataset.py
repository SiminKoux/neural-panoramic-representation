import numpy as np
from torch.utils.data import Dataset

from utils import *
from getData.get_data import *
from getData.check_data import check_dims_dsets, check_names_dsets


class CompositeDataset(Dataset):
    def __init__(self, dsets: dict, idcs=None):
        super().__init__()

        self.height, self.width = check_dims_dsets(dsets.values())
        # print("Data's size is {}x{}".format(self.width, self.height))
        self.names = check_names_dsets(dsets.values())
        self.dsets = dsets

        # print("Dataset lengths:", {k: len(v) for k, v in self.dsets.items()})
        size = min([len(d) for d in self.dsets.values()])

        if idcs is None:
            idcs = np.arange(size)
        assert all(i < size and i >= 0 for i in idcs), "invalid indices {}".format(idcs)
        self.idcs = idcs
        self.cache = [None for _ in self.idcs]
        self.device = None

    def set_device(self, device):
        self.device = device
        print("Setting dataset device to {}".format(device))

    def __len__(self):
        return len(self.idcs)

    def has_set(self, name):
        return name in self.dsets

    def get_set(self, name):
        return self.dsets[name]

    def compute_item(self, idx):
        out = {name: dset[idx] for name, dset in self.dsets.items()}
        out["idx"] = torch.tensor(idx)
        return out

    def __getitem__(self, i):
        if self.cache[i] is None:
            idx = self.idcs[i]
            self.cache[i] = self.compute_item(idx)
            if self.device is not None:
                self.cache[i] = move_to(self.cache[i], self.device)
        return self.cache[i]