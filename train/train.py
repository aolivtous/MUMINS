import sys
import os
import hydra
import atexit
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
try:
    import wandb
except ImportError:
    wandb = None
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from ddpm.diffusion import Unet3D, Trainer, GaussianDiffusion_Nolatent
from omegaconf import DictConfig
from dataset.NGP_Dataset import NGPPairDataset
from dataset.OASIS_Dataset import BrainPairDataset
from ddpm.unet import UNet
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cfg_get(node, key, default=None):
    try:
        value = node.get(key, default)
    except Exception:
        return default
    return default if value == "???" else value

def setup_distributed():
    if 'RANK' not in os.environ and 'SLURM_PROCID' in os.environ:
        os.environ['RANK'] = os.environ['SLURM_PROCID']
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
        os.environ['LOCAL_RANK'] = os.environ.get('SLURM_LOCALID', os.environ['SLURM_PROCID'])

    if ('RANK' in os.environ and 'WORLD_SIZE' in os.environ
            and int(os.environ['WORLD_SIZE']) > 1):
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))

        if not torch.cuda.is_available():
            raise RuntimeError("FSDP training requires CUDA when RANK/WORLD_SIZE are set")

        n_visible = torch.cuda.device_count()
        if n_visible == 1:
            local_rank = 0
        elif local_rank >= n_visible:
            raise RuntimeError(
                f"local_rank={local_rank} but only {n_visible} GPU(s) visible to this process"
            )

        # Write the corrected value back so any other code (e.g. Trainer)
        # that reads os.environ['LOCAL_RANK'] directly gets the right GPU index.
        os.environ['LOCAL_RANK'] = str(local_rank)

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
        )
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    set_seed(1)

    rank, world_size, local_rank, use_fsdp = setup_distributed()
    is_main = (rank == 0)
    completed = False

    if is_main:
        os.makedirs(cfg.model.results_folder, exist_ok=True)
        print(OmegaConf.to_container(cfg, resolve=True))
        print(f"[distributed] rank={rank} world_size={world_size} "
              f"local_rank={local_rank} use_fsdp={use_fsdp}")

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    use_wandb = cfg_get(cfg.model, 'use_wandb', True)
    wandb_run = None
    if use_wandb and is_main and wandb is not None:
        wandb_run = wandb.init(
            project="nodule-diffusion",
            name=f"diffAtlas_fsdp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(
                _disable_stats=False,
                _disable_meta=False,
            ),
        )

    writer = None
    if is_main:
        log_dir = os.path.join(
            cfg.model.results_folder, "logs_tensorboard",
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        writer = SummaryWriter(log_dir=log_dir)

    conditioning_dim = 256 if cfg.model.use_time_interval else 0
    denoising_fn = cfg.model.denoising_fn

    if use_fsdp and denoising_fn != 'Unet3D':
        raise ValueError("FSDP mode currently expects cfg.model.denoising_fn='Unet3D'")

    if denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            cond_dim=conditioning_dim,
            learned_variance=cfg.model.learned_variance,
            use_checkpoint=cfg_get(cfg.model, 'use_checkpoint', False),
        )
    elif denoising_fn == 'UNet':
        model = UNet(
            in_ch=cfg.model.diffusion_num_channels,
            out_ch=cfg.model.diffusion_num_channels,
            spatial_dims=3
        )
    else:
        raise ValueError(f"Model {denoising_fn} doesn't exist")

    if not use_fsdp and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total:     {total_params/1e6:.2f} M")
    print(f"Trainable: {trainable_params/1e6:.2f} M") 
    
    diffusion = GaussianDiffusion_Nolatent(
        model,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
        learned_variance=cfg.model.learned_variance,
        device=device,
    )
    if not use_fsdp:
        diffusion = diffusion.to(device)

    if cfg.dataset.name == 'NGP':
        train_dataset = NGPPairDataset(
            root_dir=cfg.dataset.root_dir,
            mode=cfg.dataset.mode,
            dim=cfg.model.diffusion_img_size,
            diff=cfg.dataset.diff,
        )
    elif cfg.dataset.name == 'OASIS':
        train_dataset = BrainPairDataset(
            root_dir=cfg.dataset.root_dir,
            mode=cfg.dataset.mode,
            dim=cfg.model.diffusion_img_size,
            diff=cfg.dataset.diff,
        )
    else:
        raise ValueError("No Such Dataset")

    try:
        trainer = Trainer(
            diffusion,
            cfg=cfg,
            dataset=train_dataset,
            train_batch_size=cfg.model.batch_size,
            save_and_sample_every=cfg.model.save_and_sample_every,
            train_lr=cfg.model.train_lr,
            train_num_steps=cfg.model.train_num_steps,
            gradient_accumulate_every=cfg.model.gradient_accumulate_every,
            ema_decay=cfg.model.ema_decay,
            amp=cfg.model.amp,
            results_folder=cfg.model.results_folder,
            num_workers=cfg.model.num_workers,
            diff=cfg.dataset.diff,
            device=device,
            writer=writer,
            use_fsdp=use_fsdp,
            fsdp_mixed_precision=cfg_get(cfg.model, 'fsdp_mixed_precision', True),
            max_grad_norm=cfg_get(cfg.model, 'max_grad_norm', 1.0),
        )

        if cfg.model.load_milestone:
            trainer.load(cfg.model.load_milestone)

        trainer.train()
        completed = True
    finally:
        if writer is not None:
            writer.close()
        if wandb_run is not None:
            wandb.finish()
        if use_fsdp and dist.is_initialized():
            if completed:
                dist.barrier()
            dist.destroy_process_group()

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            if not f.closed:
                f.write(obj)
                f.flush()

    def flush(self):
        for f in self.files:
            if not f.closed:
                f.flush()


if __name__ == '__main__':
    # Only rank 0 writes a tee'd log file to avoid 4 processes clobbering it.
    rank = int(os.environ.get('RANK', '0'))
    if rank == 0:
        log_dir = os.environ.get("TRAIN_LOG_DIR", "./log_train")
        os.makedirs(log_dir, exist_ok=True)
        filename = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")
        log_file = open(filename, 'w', encoding='utf-8')
        sys.stdout = Tee(sys.stdout, log_file)
        atexit.register(lambda: log_file.close())
    run()