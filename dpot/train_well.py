import os

import argparse
import torch
import numpy as np
import torch.nn as nn

from pathlib import Path

import yaml

import wandb

from accelerate import Accelerator
from timeit import default_timer
from torch.optim.lr_scheduler import (
    OneCycleLR,
)
from dpot.utils.optimizer import Adam
from dpot.utils.utilities import count_parameters, load_model_from_checkpoint
from dpot.well_ds import get_dataset
from dpot.models.dpot import DPOTNet
from dpot.models.dpot_res import CDPOTNet
from dpot.loss_fns import RNMSELoss, RVMSELoss, NMSELoss


################################################################
# configs
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --num_processes 8 --multi_gpu --main_process_port 5005 train_temporal_parallel.py
################################################################


def get_args():
    parser = argparse.ArgumentParser(
        description="Training or pretraining for the same data type"
    )

    parser.add_argument("--model", type=str, default="FNO")
    parser.add_argument("--dataset", type=str, default="ns2d")

    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument(
        "--train_paths", nargs="+", type=str, default=["ns2d_pdb_M1_eta1e-1_zeta1e-1"]
    )
    parser.add_argument(
        "--test_paths", nargs="+", type=str, default=["ns2d_pdb_M1_eta1e-1_zeta1e-1"]
    )
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--ntrain_list", nargs="+", type=int, default=[9000])
    parser.add_argument("--data_weights", nargs="+", type=int, default=[1])
    parser.add_argument("--use_writer", action="store_true", default=False)

    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--noise_scale", type=float, default=0.0)
    # parser.add_argument('--n_channels',type=int,default=-1)

    ### shared params
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--act", type=str, default="gelu")

    ### FNO params
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--use_ln", type=int, default=1)
    parser.add_argument("--normalize", type=int, default=0)

    ### AFNO
    parser.add_argument("--patch_size", type=int, default=1)
    parser.add_argument("--n_blocks", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=int, default=1)
    parser.add_argument("--out_layer_dim", type=int, default=32)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--opt", type=str, default="adam", choices=["adam", "lamb"])
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--lr_method", type=str, default="step")
    parser.add_argument("--grad_clip", type=float, default=10000.0)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--step_gamma", type=float, default=0.5)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--S", type=int, default=64)
    parser.add_argument("--T_in", type=int, default=10)
    parser.add_argument("--T_ar", type=int, default=1)
    # parser.add_argument('--T_ar_test', type=int, default=10)
    parser.add_argument("--T_bundle", type=int, default=1)
    # parser.add_argument('--T', type=int, default=20)
    # parser.add_argument('--step', type=int, default=1)
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()
    return args


def log_msg(msg: str):
    if os.environ.get("RANK", "0") == "0":
        print(msg)


def init_wandb(config):
    log_path = config["log_path"]
    log_path = Path(log_path)
    name = log_path.name

    run = wandb.init(
        project="Large-Physics-Foundation-Model",
        config=config,
        name=f"{name}",
        dir=str(log_path),
    )
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training or pretraining for the same data type"
    )
    parser.add_argument(
        "--config_file", type=str, default="./configs/pretrain_medium.yaml"
    )
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config_file, "r"), Loader=yaml.FullLoader)
    config["data_path"] = args.data_path
    config["log_path"] = args.checkpoint_path

    # Enable unused parameter detection for DDP
    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    ################################################################
    # load data and dataloader
    ################################################################

    train_dataset = get_dataset(
        path=config["data_path"],
        split_name="train",
        datasets=config["datasets"],
        num_channels=config["num_channels"],
        min_stride=config["min_stride"],
        max_stride=config["max_stride"],
        T_in=config["T_in"],
        T_out=config["T_bundle"],
        use_normalization=config["normalize"],
        full_trajectory_mode=False,
    )
    test_dataset = get_dataset(
        path=config["data_path"],
        split_name="valid",
        datasets=config["datasets"],
        num_channels=config["num_channels"],
        min_stride=config["min_stride"],
        max_stride=config["max_stride"],
        T_in=config["T_in"],
        T_out=config["T_bundle"],
        use_normalization=config["normalize"],
        full_trajectory_mode=False,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        drop_last=True,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    ################################################################
    # load model
    ################################################################
    if config["model"] == "DPOT":
        model = DPOTNet(
            img_size=config["res"],
            patch_size=config["patch_size"],
            in_channels=config["num_channels"],
            in_timesteps=config["T_in"],
            out_timesteps=config["T_bundle"],
            out_channels=config["num_channels"],
            normalize=config["normalize"],
            embed_dim=config["width"],
            depth=config["n_layers"],
            n_blocks=config["n_blocks"],
            mlp_ratio=config["mlp_ratio"],
            out_layer_dim=config["out_layer_dim"],
            act=config["act"],
            n_cls=len(config["datasets"]),
        ).to(device)
    elif config["model"] == "CDPOT":
        model = CDPOTNet(
            img_size=config["res"],
            patch_size=config["patch_size"],
            in_channels=config["num_channels"],
            in_timesteps=config["T_in"],
            out_timesteps=config["T_bundle"],
            out_channels=config["num_channels"],
            normalize=config["normalize"],
            embed_dim=config["width"],
            modes=config["modes"],
            depth=config["n_layers"],
            n_blocks=config["n_blocks"],
            mlp_ratio=config["mlp_ratio"],
            out_layer_dim=config["out_layer_dim"],
            act=config["act"],
            n_cls=len(config["datasets"]),
        ).to(device)
    else:
        raise NotImplementedError

    resume_path = config.get("resume_path", "")
    if resume_path is not None and resume_path != "":
        log_msg("Loading models and fine tune from {}".format(config["resume_path"]))
        # model.load_state_dict(torch.load(config['resume_path'],map_location='cuda:{}'.format(config.gpu))['model'])
        load_model_from_checkpoint(
            model, torch.load(resume_path, map_location="cpu")["model"]
        )

    #### set optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=1e-6,
    )

    log_msg("Using cycle learning rate schedule")
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        div_factor=1e4,
        pct_start=(config["warmup_epochs"] / config["epochs"]),
        final_div_factor=1e4,
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"],
    )

    log_path = config["log_path"]
    os.makedirs(log_path, exist_ok=True)
    ckpt_save_epochs = 50

    count_parameters(model)

    if os.environ.get("RANK", "0") == "0":
        run = init_wandb(config)
    else:
        run = None

    ##multi-gpu
    model, optimizer, scheduler, train_loader, test_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, test_loader
    )
    ################################################################
    # Main function for pretraining
    ################################################################
    criterion = NMSELoss(dims=(1, 2, 3))
    rnmse = RNMSELoss(dims=(1, 2, 3))
    rvmse = RVMSELoss(dims=(1, 2, 3))
    iter = 0
    for ep in range(config["epochs"]):
        log_msg(f"Epoch {ep} ---------------------")
        model.train()

        t1 = t_1 = default_timer()
        t_load, t_train = 0.0, 0.0
        train_l2_full = 0
        loss_previous = np.inf

        torch.cuda.empty_cache()

        train_steps = 0
        for xx, yy, _, _ in train_loader:
            train_steps += 1
            t_load += default_timer() - t_1
            t_1 = default_timer()

            xx = xx.to(device)  ## B, n, n, T_in, C
            yy = yy.to(device)  ## B, n, n, T_ar, C

            # log_msg(f"mean xx: {xx.mean().item()}, mean yy: {yy.mean().item()}")
            # log_msg(f"std xx: {xx.std().item()}, std yy: {yy.std().item()}")

            ### auto-regressive training
            xx = xx + config["noise_scale"] * torch.sum(
                xx**2, dim=(1, 2, 3), keepdim=True
            ) ** 0.5 * torch.randn_like(xx)

            # log_msg(f"xx shape: {xx.shape}, yy shape: {yy.shape}")
            im, _ = model(xx)

            loss = criterion(im, yy)

            optimizer.zero_grad()
            accelerator.backward(loss)
            # total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()

            iter += 1

            t_train += default_timer() - t_1
            t_1 = default_timer()
            if iter % 100 == 0:
                log_msg(f"epoch {ep} iter {iter} step loss {loss.item()}")

            if iter % 1000 == 0 and os.environ.get("RANK", "0") == "0":
                path = log_path + f"/model_{ep}.pth"
                torch.save(
                    {
                        "config": config,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    path,
                )

            with torch.no_grad():
                nmse_loss = criterion(im.detach(), yy)
                rnmse_loss = rnmse(im.detach(), yy)
                rvmse_loss = rvmse(im.detach(), yy)
                # sync between processes
                nmse_loss = accelerator.gather_for_metrics(nmse_loss)
                nmse_loss = nmse_loss.mean().item()
                rnmse_loss = accelerator.gather_for_metrics(rnmse_loss)
                rnmse_loss = rnmse_loss.mean().item()
                rvmse_loss = accelerator.gather_for_metrics(rvmse_loss)
                rvmse_loss = rvmse_loss.mean().item()
            if run is not None:
                run.log(
                    {
                        "train/nmse": nmse_loss,
                        "train/rnmse": rnmse_loss,
                        "train/rvmse": rvmse_loss,
                    }
                )

            if config["epoch_length"] < train_steps:
                break

        log_msg("start eval")
        nmse_loss = torch.tensor(0.0).to(device)
        rnmse_loss = torch.tensor(0.0).to(device)
        rvmse_loss = torch.tensor(0.0).to(device)
        model.eval()

        eval_steps = 0
        with torch.no_grad():
            for xx, yy, _, _ in test_loader:
                eval_steps += 1

                xx = xx.to(device)
                yy = yy.to(device)

                im, _ = model(xx)
                nmse_loss += criterion(im, yy)
                rnmse_loss += rnmse(im, yy)
                rvmse_loss += rvmse(im, yy)

            if eval_steps < config["eval_length"]:
                break

            nmse_loss = nmse_loss / len(test_loader)
            rnmse_loss = rnmse_loss / len(test_loader)
            rvmse_loss = rvmse_loss / len(test_loader)

            # sync between processes
            nmse_loss = accelerator.gather_for_metrics(nmse_loss)
            nmse_loss = nmse_loss.mean().item()
            rnmse_loss = accelerator.gather_for_metrics(rnmse_loss)
            rnmse_loss = rnmse_loss.mean().item()
            rvmse_loss = accelerator.gather_for_metrics(rvmse_loss)
            rvmse_loss = rvmse_loss.mean().item()

            if run is not None:
                run.log(
                    {
                        "test/nmse": nmse_loss,
                        "test/rnmse": rnmse_loss,
                        "test/rvmse": rvmse_loss,
                    }
                )

        if os.environ.get("RANK", "0") == "0":
            path = log_path + f"/model_{ep}.pth"
            torch.save(
                {
                    "config": config,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                path,
            )

        t_test = default_timer() - t_1
        t2 = t_1 = default_timer()
        lr = optimizer.param_groups[0]["lr"]
        log_msg(f"epoch {ep}, time {t2 - t1:.5f}, lr {lr:.2e}, nmse: {nmse_loss:.5f}")
    log_msg("Training done.")
    if run is not None:
        run.finish()
