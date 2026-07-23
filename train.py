import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, random_split
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, find_latest_checkpoint
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import os
import numpy as np
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import Subset

# python train.py

def normalize_wandb_mode(value) -> str:
    mode = str(value).strip().lower()
    aliases = {
        "0": "disabled",
        "false": "disabled",
        "no": "disabled",
        "off": "disabled",
        "disable": "disabled",
        "none": "disabled",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError(f"Unsupported wandb_mode={value!r}; expected online, offline, or disabled")
    return mode


def read_global_step(ckpt_path: str) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "global_step" in ckpt:
        return int(ckpt["global_step"])
    loops = ckpt.get("loops", {})
    fit = loops.get("fit_loop", {}) or loops.get("FitLoop", {})
    for k in ["epoch_loop", "EpochLoop", "epoch_loop.state_dict", "epoch_loop._batches_that_stepped"]:
        node = fit.get(k, {})
        for path in [
            ("batch_progress","total","completed"),
            ("batch_progress","total","ready"),
            ("state_dict","batch_progress","total","completed"),
        ]:
            cur = node
            ok = True
            for p in path:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    ok = False
                    break
            if ok and isinstance(cur, (int, float)):
                return int(cur)
    # Fall back to 0 if the step count cannot be found.
    return 0

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method, cfg.data)

    model = build_model(cfg)

    train_set = build_dataset(cfg, val=False)
    val_set = build_dataset(cfg, val=True)

    train_batch_size = cfg.method['train_batch_size'] 
    eval_batch_size = cfg.method['eval_batch_size']

    log_root = os.path.join('CTFM', cfg.exp_name)
    checkpoint_dir = os.path.join(log_root, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Replace with your validation metric
        dirpath=checkpoint_dir,
        filename=f'{cfg.model_name}-' + '{epoch}-{val_loss:.6f}',
        save_top_k=10,
        mode='min',  # 'min' for loss/error, 'max' for accuracy
        save_last=True,
    )

    call_backs.append(checkpoint_callback)
    call_backs.append(LearningRateMonitor(logging_interval='step'))

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, shuffle=True, drop_last=False,
        collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False, pin_memory=True,
        collate_fn=train_set.collate_fn)
    

    # Prefer Hydra config/CLI overrides over the shell environment so
    # `wandb_mode=offline` works even inside containers with WANDB_MODE set.
    cfg_wandb_mode = getattr(cfg, "wandb_mode", None)
    wandb_mode = normalize_wandb_mode(
        cfg_wandb_mode if cfg_wandb_mode is not None else os.environ.get("WANDB_MODE", "offline")
    )
    os.environ["WANDB_MODE"] = wandb_mode
    wandb_dir = os.environ.get("WANDB_DIR") or os.path.join(log_root, "wandb")
    os.makedirs(wandb_dir, exist_ok=True)

    run_name = f"{cfg.exp_name}-seed{cfg.seed}"
    csv_logger = CSVLogger(save_dir=log_root, name="csv_logs")
    loggers = [csv_logger]
    if wandb_mode != "disabled":
        wandb_project = str(getattr(cfg, "wandb_project", "Trajworld"))
        wandb_kwargs = {
            "project": wandb_project,
            "name": run_name,
            "save_dir": wandb_dir,
            "offline": (wandb_mode == "offline"),
        }
        wandb_run_id = getattr(cfg, "wandb_run_id", None)
        if wandb_run_id:
            wandb_kwargs["id"] = str(wandb_run_id)
            wandb_kwargs["resume"] = str(getattr(cfg, "wandb_resume", "allow"))
        wandb_logger = WandbLogger(**wandb_kwargs)
        loggers.insert(0, wandb_logger)

    # automatically resume training
    if cfg.ckpt_path is None:
        cfg.ckpt_path = find_latest_checkpoint(checkpoint_dir)
    
    # Compute the target absolute training step count when resuming.
    extra_steps = int(getattr(cfg.method, "resume_extra_steps", 100))  # Train for 100 additional steps
    prev_steps = 0
    if cfg.ckpt_path is not None:
        prev_steps = read_global_step(cfg.ckpt_path)
    target_steps = prev_steps + extra_steps if bool(getattr(cfg.method, "resume_fixed_lr_mode", False)) or bool(getattr(cfg.method, "finetune_few_params", False)) else cfg.method.total_steps
    val_check_interval = 1.0 if bool(getattr(cfg.method, "resume_fixed_lr_mode", False)) or bool(getattr(cfg.method, "finetune_few_params", False)) else cfg.method.val_every_n_steps
    max_epochs = -1 if bool(getattr(cfg.method, "resume_fixed_lr_mode", False)) or bool(getattr(cfg.method, "finetune_few_params", False)) else cfg.method.epochs

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_steps=target_steps,
        val_check_interval=val_check_interval,  # e.g. 1000 means running validation every 1000 optimizer steps
        logger=loggers,
        default_root_dir=log_root,
        devices=cfg.devices,
        accelerator="gpu",
        profiler="simple",
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=call_backs,
        log_every_n_steps=int(getattr(cfg, "log_every_n_steps", 1)),
        gradient_clip_val=0.25,                
        gradient_clip_algorithm="norm", 
        accumulate_grad_batches=cfg.method.get('accumulate_grad_batches', 4),  # Number of gradient accumulation steps
        precision="16-mixed"  # Use mixed-precision training (16-bit floating point) for memory efficiency
    )
    print(f"[logging] wandb mode: {wandb_mode}")
    print(f"[logging] wandb dir: {wandb_dir}")
    print(f"[logging] csv metrics: {os.path.join(csv_logger.log_dir, 'metrics.csv')}")
    print(f"[checkpoint] dir: {checkpoint_dir}")
    if cfg.ckpt_path is not None:
        state = torch.load(cfg.ckpt_path, map_location="cpu")
        sd = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("[load ckpt] missing:", missing)
        print("[load ckpt] unexpected:", unexpected)
    ckpt_path = None if bool(getattr(cfg.method, "finetune_few_params", False)) else cfg.ckpt_path
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    train()
