import os

def get_custom_dir(root, subd):
    return os.path.join(root, subd).rstrip("/")


def get_data_dirs(root, name):
    data_path = os.path.join(root, name)
    subds = ["frames","masks"]
    subdirs = [get_custom_dir(data_path, sd) for sd in subds]
    return subdirs


def get_path_name(path):
    return os.path.splitext(os.path.basename(path))[0]

