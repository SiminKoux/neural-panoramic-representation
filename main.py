import os
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter

import hydra
from omegaconf import DictConfig, OmegaConf

import models
from losses import *
from utils.train_utils import *
from utils.compute_utils import *
from getData.get_data import *

DEVICE = torch.device("cuda")
ORIGINAL_WORKING_DIR = os.getcwd()

@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    print("Let's begin!")
    print(OmegaConf.to_yaml(cfg))

    # -------- Dataset -------- #
    dataset_root = cfg.data.root  # Get dataset path from config
    dataset_name = cfg.data.name
    dataset_scale = cfg.data.scale
    # Ensure dataset path remains absolute and unchanged
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.abspath(os.path.join(ORIGINAL_WORKING_DIR, dataset_root))
    print(f"Using dataset from: {dataset_root}")
    dset = get_dataset(dataset_root, dataset_name, dataset_scale)
    
    N, H, W = len(dset), dset.height, dset.width
    print("There are {} frames in the given video.".format(N))
    
    can_preload = N < 200 and cfg.data.scale < 0.5   # False
    preloaded = cfg.preload and can_preload          # False
    data_set = dset.dsets
    print("The customized Dataset includes {}".format(data_set.keys()))
    print("Get the customized Dataset done!")
    print("------------------------------------")

    # -------- DataLoader -------- #
    print("Getting DataLoaders...")
    num_workers = cfg.batch_size if not preloaded else 0  # batch_size
    print("Numbers of workers:", num_workers)
    
    train_loader = get_random_ordered_batch_loader(dset, cfg.batch_size, preloaded)
    print("There are {} batches in training loader.".format(len(train_loader)))
    val_loader = get_ordered_loader(dset, cfg.batch_size, preloaded)
    print("There are {} batches in evaluation loader.".format(len(val_loader)))
    print("------------------------------------")
         
    # -------- Set Model -------- #
    model = models.FullModel(dset, cfg.n_layers, cfg.model)
    model.to(DEVICE)
    print("Model Initialization Done!")
    print("------------------------------------")

    # -------- Set Log -------- #
    # determines logging dir in hydra config
    log_dir = os.getcwd()
    train_log_dir = os.path.join(log_dir,"train")
    val_log_dir = os.path.join(log_dir,"val")
    train_writer = SummaryWriter(log_dir=train_log_dir)
    val_writer = SummaryWriter(log_dir=val_log_dir)
    print("saving output to:", log_dir)
    
    # -------- Set Train -------- #
    cfg = update_config(cfg, train_loader)
    args = cfg.model
    save_args = dict(
        train_writer=train_writer,
        val_writer=val_writer,
        vis_every=cfg.vis_every,
        val_every=cfg.val_every,
        batch_size=cfg.batch_size
    )
     
    if args.inverse:
        loss_fncs = {"inverse_loss": InverseLoss(weight=cfg.w_inverse),}
    else:
        loss_fncs = {
            "reconstruct": MapReconLoss(weight = cfg.w_recon),
            "spherical_regularization": SphericalLoss(weight=cfg.w_sph),
            "positional_loss": RelativePositionalLoss(weight=cfg.w_pos),
            "sparse_loss": RGBSparseLoss(weight=cfg.w_sparse),
            "alpha_loss": AlphaLoss(weight=cfg.w_alpha),
        }
    print("The loss is {}".format(loss_fncs))
    
    opt_infer_helper = partial(
        opt_infer_step,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_fncs=loss_fncs,
        **save_args,
    )
    
    # -------- Train -------- #
    print("------------------------------------")
    print("Start Training...")
    coord_map_model = model.coord_mapping
    alpha_pred_model = model.alpha_pred
    rgb_map_model = model.rgb_mapping
    
    if args.warmstart_mapping1 and args.warmstart_mapping2:
        print("Warmstart the coordinates mapping...")
        label_1 = "coord_map"
        warm_start(args, coord_map_model, label_1, N, H, W, train_writer)
    
    if args.warmstart_alpha_pred:
        print("Warmstart the alpha prediction...")
        label_2 = "alpha_pred"
        warm_start(args, alpha_pred_model, label_2, N, H, W, train_writer, train_loader)
    
    print("------------------------------------")
    
    if args.inverse:
        label = "backward_train"
        load_forward_ckpt(coord_map_model, alpha_pred_model, rgb_map_model)
    else:
        label = "forward_train"

    if args.main_train:
        # n_epochs = 10 # for debugging
        n_epochs = cfg.epochs_per_phase["train"]
        step_ct, val_dict = opt_infer_helper(n_epochs, label=label)
        print("Training Step:", step_ct)
        print("Training is done!")
        print("Outputs:", val_dict.keys())
    
    if args.inference:
        print("Start Inference...")
        n_epochs = 1
        load_ckpt(coord_map_model, alpha_pred_model, rgb_map_model)
        step_ct, val_dict = opt_infer_helper(n_epochs, label="inference")
        print("Inference Step:", step_ct)
        print("Inference is done!")
        print("Outputs:", val_dict.keys())


if __name__ == "__main__":
    main()
