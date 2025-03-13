import glob
from torch.utils.data import Dataset
from getData.get_path import get_path_name
from getData.load_tensor import load_img_tensor, is_image

class RGBDataset(Dataset):
    def __init__(self, src_dir, scale=1, start=0, end=-1, ext=""):
        super().__init__()
        self.src_dir = src_dir
        files = sorted(filter(is_image, glob.glob(f"{src_dir}/*{ext}")))
        if len(files) < 1:
            raise NotImplementedError
        names = [get_path_name(p) for p in files]

        if end < 0:  # (-1 -> all, -2 -> all but last)
            end += len(files) + 1
        
        self.start = start
        self.end = end
        self.names = names[start:end]
        self.files = files[start:end]
        self.scale = scale
        print("FOUND {} RGB video frames in '{}'".format(len(files), src_dir))
        
        test = load_img_tensor(self.files[0], scale)
        self.height, self.width = test.shape[-2:]
        print("RGB Scale {}, shape {}".format(scale, test.numpy().shape))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return load_img_tensor(self.files[idx], self.scale)
