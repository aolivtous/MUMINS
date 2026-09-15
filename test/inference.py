import csv
import os
import io
import blobfile as bf
import torch as th
import sys
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import atexit
import time
import matplotlib.pyplot as plt
import random

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import binary_dilation
from omegaconf import DictConfig, OmegaConf
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from ddpm.diffusion import Unet3D, GaussianDiffusion_Nolatent
from dataset.NGP_Dataset import NGPPairDataset
from dataset.OASIS_Dataset import BrainPairDataset


# ============================================================================
# diff_blur handling: 4 modes, replacing the old boolean `zero_diff_blur`.
#
#   'gt'     -- Delta b computed from the TRUE follow-up (blur1 - blur0).
#               This is the ORACLE / leakage setting -- report as an upper
#               bound only, never as the primary deployable result.
#   'zero'   -- Delta b = 0 (assume unchanged acquisition protocol). No
#               leakage, but only a valid default if 0 is actually close to
#               the population -- check via blur_check.py's printed
#               "|0 - mean| / std" before treating this as representative.
#   'mean'   -- Delta b = fixed TRAIN-set per-axis mean (see
#               DIFF_BLUR_CONSTANTS below). No leakage: this constant is
#               computed once from train only, then applied uniformly.
#   'median' -- same as 'mean' but the TRAIN-set per-axis median.
#
# Fill in the constants below from blur_check.py's / blur_check_oasis.py's
# printed "db_mean_per_axis" / "db_median_per_axis" lines -- TRAIN split
# ONLY. Do not compute these from val/test.
# ============================================================================
DIFF_BLUR_CONSTANTS = {
    'NGP': {
        'mean':   [-0.172, -0.280, -0.390],   
        'median': [-0.125, -0.174, -0.195],   
    },
    'OASIS': {
        'mean':   [-0.183, -0.153, -0.133],  
        'median': [-0.032, -0.038, -0.027],   
    },
}

def get_diff_blur(mode, dataset_name, blur0, blur1, device):
    """
    Returns a [B, 3] tensor to concatenate into `cond`, per `mode`:
        'gt' | 'zero' | 'mean' | 'median'
    """
    b = blur0.shape[0]
    if mode == 'gt':
        return (blur1 - blur0).to(device)
    if mode == 'zero':
        return torch.zeros_like(blur0, device=device)
    if mode in ('mean', 'median'):
        vals = DIFF_BLUR_CONSTANTS.get(dataset_name, {}).get(mode)
        if vals is None or any(v is None for v in vals):
            raise ValueError(
                f"DIFF_BLUR_CONSTANTS['{dataset_name}']['{mode}'] is not filled "
                f"in. Run blur_check.py (NGP) / blur_check_oasis.py (OASIS) on "
                f"the TRAIN split and paste its printed db_{mode}_per_axis "
                f"values into DIFF_BLUR_CONSTANTS at the top of this file."
            )
        const = torch.tensor(vals, dtype=blur0.dtype, device=device)
        return const.unsqueeze(0).repeat(b, 1)
    raise ValueError(f"Unknown diff_blur_mode: {mode!r} "
                     f"(expected one of 'gt', 'zero', 'mean', 'median')")


def dev(device):
    if device is None:
        if th.cuda.is_available():
            return th.device(f"cuda")
        return th.device("cpu")
    return th.device(device)

def load_state_dict(path, backend=None, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            if not f.closed: # Check if file is open
                f.write(obj)
                f.flush()
    def flush(self):
        for f in self.files:
            if not f.closed:
                f.flush()


def dilate_mask(mask, iterations=3):
    """Dilate binary mask to expand nodule region."""
    if mask is None:
        return None
    if torch.is_tensor(mask):
        mask_np = mask.squeeze().cpu().numpy()  # Remove all singleton dimensions
    else:
        mask_np = np.squeeze(mask)  # Remove all singleton dimensions
    dilated = binary_dilation(mask_np, iterations=iterations)
    return dilated

def compute_metrics_with_mask(pred, target, mask=None):
    """
    Compute metrics optionally within a masked region.
    Returns dict with metric values.
    """
    if mask is not None:
        # Apply mask
        pred_masked = pred[mask > 0]
        target_masked = target[mask > 0]
        
        if len(pred_masked) == 0:
            return None
        
        mse = np.mean((pred_masked - target_masked) ** 2)
        mae = np.mean(np.abs(pred_masked - target_masked))
    else:
        mse = np.mean((pred - target) ** 2)
        mae = np.mean(np.abs(pred - target))
    
    return {"mse": mse, "mae": mae}


def compute_psnr_custom(img1, img2, mask=None):
    """Compute PSNR using consistent method"""
    if mask is not None:
        img1 = img1[mask > 0]
        img2 = img2[mask > 0]
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    return psnr


def compute_ssim_custom(img1, img2, mask=None):
    """Compute SSIM using skimage for consistency"""
    # Ensure inputs are numpy arrays
    if torch.is_tensor(img1):
        img1 = img1.squeeze().cpu().numpy()
    else:
        img1 = np.squeeze(img1)
    
    if torch.is_tensor(img2):
        img2 = img2.squeeze().cpu().numpy()
    else:
        img2 = np.squeeze(img2)
    
    if mask is not None:
        if torch.is_tensor(mask):
            mask_np = mask.squeeze().cpu().numpy()
        else:
            mask_np = np.squeeze(mask)
        
        # Crop to bounding box
        coords = np.argwhere(mask_np > 0)
        if len(coords) == 0:
            return 0.0
        
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        
        # Check if region is large enough
        crop_d = z_max - z_min + 1
        crop_h = y_max - y_min + 1
        crop_w = x_max - x_min + 1
        
        if crop_d < 7 or crop_h < 7 or crop_w < 7:
            return 0.0
        
        img1_crop = img1[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        img2_crop = img2[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        
        return ssim(img1_crop, img2_crop, data_range=1.0)
    
    return ssim(img1, img2, data_range=1.0)

def calculate_img_metrics(real, pred):
    """
    Calculates MAE, PSNR, SSIM.
    Expects tensors in range [-1, 1].
    """
    # Convert to numpy and move to 0..1 range
    real_np = (real.cpu().numpy().squeeze() + 1) / 2
    pred_np = (pred.cpu().numpy().squeeze() + 1) / 2
    
    # Clip to valid range
    real_np = np.clip(real_np, 0, 1)
    pred_np = np.clip(pred_np, 0, 1)
    
    # MAE
    mae = np.mean(np.abs(real_np - pred_np))
    mse = np.mean((real_np - pred_np) ** 2)
    
    
    # PSNR
    if mse == 0:
        p_val = float('inf')
    else:
        p_val = 20 * np.log10(1.0) - 10 * np.log10(mse)
    
    # SSIM
    s_val = ssim(real_np, pred_np, data_range=1.0)
    
    return mae, p_val, s_val


def to_slice(t):
    if t is None: return None
    # t shape is likely [1, C, D, H, W] or [C, D, H, W]
    arr = t.detach().cpu().numpy()
    
    # Remove batch dimension if it exists
    if arr.ndim == 5: 
        arr = arr[0] # Now [C, D, H, W]
    
    # If we have multiple channels (like learned variance), 
    # we usually want the second channel [1] for the variance map
    if arr.ndim == 4:
        # Check if this is the variance map (2 channels) 
        # or just a single channel image/mask
        if arr.shape[0] > 1:
            arr = arr[1] # Take the variance channel
        else:
            arr = arr[0] # Take the only channel available

    # Now arr should be [D, H, W]. Get the middle slice.
    if arr.ndim == 3:
        mid = arr.shape[0] // 2
        return arr[mid, :, :]
    
    return arr

def seed_worker(worker_id):
    worker_seed = th.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@hydra.main(config_path='confs', config_name='infer', version_base=None)
def main(conf: DictConfig):
    print(OmegaConf.to_container(conf, resolve=True))

    seed = conf.get('seed', 42)  
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    print(f"Fixed seed: {seed}")

    # --- diff_blur mode: 'gt' | 'zero' | 'mean' | 'median' ---
    diff_blur_mode = conf.get('diff_blur_mode', None)
    if diff_blur_mode is None:
        # backward-compat with the old boolean flag
        if conf.get('zero_diff_blur', False):
            diff_blur_mode = 'zero'
            print("[deprecated] `zero_diff_blur: true` -> treating as diff_blur_mode='zero'")
        else:
            diff_blur_mode = 'gt'
    assert diff_blur_mode in ('gt', 'zero', 'mean', 'median'), \
        f"diff_blur_mode must be one of gt/zero/mean/median, got {diff_blur_mode!r}"
    print(f"\n{'='*60}\n  diff_blur_mode = {diff_blur_mode!r}"
          f"{'  <-- ORACLE / leakage setting' if diff_blur_mode == 'gt' else ''}"
          f"\n{'='*60}\n")

    # --- SETUP DIRS ---
    os.makedirs(os.path.join(conf.out_dir), exist_ok=True)
    filename = os.path.join(conf.out_dir, str(conf.model_num) + ".log")
    log_file = open(filename, 'w', encoding='utf-8')  
    sys.stdout = Tee(sys.stdout, log_file)
    atexit.register(lambda: log_file.close())

    device = dev(conf.get('device'))

    # --- MODEL SETUP ---
    use_time = conf.get('use_time_interval', True) # Default to True for progression
    cond_dim = 256 if use_time else 0 

    model = Unet3D(
        dim=conf.diffusion_img_size,
        dim_mults=conf.dim_mults,
        channels=conf.diffusion_num_channels,
        learned_variance=conf.learned_variance,
        cond_dim=cond_dim,
    )

    diffusion = GaussianDiffusion_Nolatent(
        model,
        image_size=conf.diffusion_img_size,
        num_frames=conf.diffusion_depth_size,
        channels=conf.diffusion_num_channels,
        timesteps=conf.timesteps,
        loss_type=conf.loss_type,
        learned_variance=conf.learned_variance,
        device=device
    )
    diffusion.to(device)

    print(f"trained with fsdp is {conf.trained_with_fsdp}")

    if conf.trained_with_fsdp == False:
        print("Loading model NOT trained with FSDP. Adjusting state dict keys accordingly.")
        # --- LOAD WEIGHTS ---
        model_path = os.path.join(conf.model_path, f"model-{conf.model_num}.pt")
        loaded_state = load_state_dict(os.path.expanduser(model_path), map_location="cpu")
        state_to_load = loaded_state['ema'] if 'ema' in loaded_state else loaded_state['model']

        weights_dict = {}
        for k, v in state_to_load.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v

        diffusion.load_state_dict(weights_dict, strict=False)
        model.eval()
    
    else:

        # --- LOAD WEIGHTS ---
        model_path = os.path.join(conf.model_path, f"model-{conf.model_num}.pt")
        loaded_state = load_state_dict(os.path.expanduser(model_path), map_location="cpu")

        # Prefer EMA weights for inference (cleaner samples for diffusion).
        state_to_load = loaded_state['ema'] if 'ema' in loaded_state else loaded_state['model']

        # Normalize keys so they match a bare Unet3D, regardless of which training
        # script produced the checkpoint:
        #   - DataParallel run: keys look like 'module.denoise_fn.init_conv.weight'
        #   - FSDP run:         keys look like 'init_conv.weight'
        def _normalize_key(k: str) -> str:
            if k.startswith('module.'):
                k = k[len('module.'):]
            if k.startswith('denoise_fn.'):
                k = k[len('denoise_fn.'):]
            return k

        weights_dict = {_normalize_key(k): v for k, v in state_to_load.items()}

        # Load into the BARE Unet3D (not the diffusion wrapper).
        missing, unexpected = model.load_state_dict(weights_dict, strict=False)
        if missing:
            print(f"[load] missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[load] unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        assert not missing, "Some Unet3D weights were not loaded — checkpoint key mismatch."

        model.eval()
        diffusion.eval()
        print(f"[load] Loaded {len(weights_dict)} tensors from {model_path}")


    # --- DATASET ---

    if conf.dataset == 'NGP':
        dataset = NGPPairDataset(
            root_dir=conf.root_dir, 
            mode=conf.mode,
            dim=conf.diffusion_img_size,
            diff=conf.diff
        )
    elif conf.dataset == 'OASIS':
        dataset = BrainPairDataset(
            root_dir=conf.root_dir,
            mode=conf.mode,
            dim=conf.diffusion_img_size,
            diff=conf.diff,
        )
       
    else:
        raise ValueError("No Such Dataset")


    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,  # Should be False for reproducibility
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=th.Generator().manual_seed(seed)
    )
    
    idx = 0
    os.makedirs(os.path.join(conf.out_dir, 'samples'), exist_ok=True)
    
    num_seeds =  conf.get('num_seeds', 1)  # for full uncertainty analysis, set to 10 or more. For quick testing, keep at 1.
    base_seed = conf.get('base_seed', 42)
    variance_start_step = int(diffusion.num_timesteps * conf.get('variance_start', 0.05))
    
    # 1. Get the total number of batches for the progress log
    total_batches = len(dataloader)
    
    print(f"Starting inference on {total_batches} samples.")
    print(f"Doing {num_seeds} seeds per sample for later uncertainty analysis...\n")

    # Dictionary to keep track of results per seed cleanly
    results_per_seed = {base_seed + i: [] for i in range(num_seeds)}
    sampler = conf.get('sampler', 'ddpm')
    eta=conf.get('eta', 0)
    skip_interval = conf.get('skip_interval', 10)
    print(f"Using sampler: {sampler} with skip_interval={skip_interval} (only for DDIM)")
    
    for batch in dataloader:
        idx += 1
        
        imgage_0 = batch['img0']
        imgage_1 = batch['img1']
        dt = batch['dt'] if 'dt' in batch else None
        pair = batch['pair'] if 'pair' in batch else None
        msk1 = batch.get('mask1', None)
        blur0 = batch['blur0']
        blur1 = batch['blur1']
        age0 = batch['age0'] 

        diff_blur = get_diff_blur(diff_blur_mode, conf.dataset, blur0, blur1, device=dt.device)

        cond = torch.cat([dt, diff_blur, age0], dim=1)

        if msk1 is not None and msk1.min() < 0:
            msk1 = (msk1 + 1) / 2  

        def safe_val(v):
            return v.item() if torch.is_tensor(v) else v    

        gt_name = f"pred_{batch['PatientId'][0]}_{safe_val(batch['NoduleId'][0])}_{safe_val(batch['t0'][0])}_{safe_val(batch['t1'][0])}"

        print(f"[{idx}/{total_batches}] Processing {gt_name}...")

        real_img_1 = imgage_1.cpu()
        if conf.diff:
            real_img_1 = (real_img_1 * 2.0) + imgage_0.cpu()

        for i in range(num_seeds):
            current_seed = base_seed + i
            print(f"  -> Running seed {i+1}/{num_seeds} (Seed value: {current_seed})")
            
            seed_dir = os.path.join(conf.out_dir, 'samples', f"seed_{current_seed}")
            os.makedirs(seed_dir, exist_ok=True)

            if sampler == 'ddim':
                th.cuda.synchronize()          # ensure GPU is idle before timing
                t_start = time.perf_counter()
                result, variance_map = diffusion.p_sample_loop_pair_ddim_A(
                    shape_img0=imgage_0.size(),
                    shape_img1=imgage_1.size(),
                    cond=cond,
                    device=device,
                    image=imgage_0.to(device),
                    seed=current_seed,
                    skip_interval=skip_interval,
                    eta=eta,  # You can adjust eta for more stochasticity in DDIM
                    variance_start_step=variance_start_step
                )
                th.cuda.synchronize()          # wait for all GPU work to finish
                elapsed = time.perf_counter() - t_start
                print(f"  -> DDIM sampling took {elapsed:.2f}s")

            else:
                result, variance_map = diffusion.p_sample_loop_pair(
                    shape_img0=imgage_0.size(),
                    shape_img1=imgage_1.size(),
                    cond=cond,
                    device=device,
                    image=imgage_0.to(device),
                    seed=current_seed,
                )

            gen_image_1 = result[:, 1:(result.size()[1]), :, :, :].cpu()
            
            if conf.diff:
                gen_image_1 = (gen_image_1 * 2.0) + imgage_0.cpu()  

            np.save(os.path.join(seed_dir, f"{gt_name}.npy"), gen_image_1.numpy())

            if variance_map is not None:
                var_np = variance_map.cpu().numpy() if torch.is_tensor(variance_map) else variance_map
                np.save(os.path.join(seed_dir, f"{gt_name}_var.npy"), var_np)

            mae, p_val, s_val = calculate_img_metrics(real_img_1, gen_image_1)
            
            if msk1 is not None:
                dilated_mask = dilate_mask(msk1.squeeze(), iterations=3)
                real_np = np.clip((real_img_1.numpy().squeeze() + 1) / 2, 0, 1)
                pred_np = np.clip((gen_image_1.numpy().squeeze() + 1) / 2, 0, 1)
                
                nodule_metrics = compute_metrics_with_mask(pred_np, real_np, mask=dilated_mask)
                if nodule_metrics is not None:
                    mae_nodule = nodule_metrics["mae"]
                    psnr_nodule = compute_psnr_custom(pred_np, real_np, mask=dilated_mask)
                    ssim_nodule = compute_ssim_custom(pred_np, real_np, mask=dilated_mask)
                else:
                    mae_nodule = psnr_nodule = ssim_nodule = 0.0
            else:
                mae_nodule = psnr_nodule = ssim_nodule = None
            
            dice_score = 0
            
            # Append results for THIS seed
            results_per_seed[current_seed].append({
                "sample": gt_name,
                "pair": pair[0] if pair is not None else 'N/A',
                "diff_blur_mode": diff_blur_mode,
                "diff_blur_used": diff_blur.squeeze().tolist() if torch.is_tensor(diff_blur) else diff_blur,
                "blur0": blur0.squeeze().tolist() if torch.is_tensor(blur0) else blur0,
                "blur1": blur1.squeeze().tolist() if torch.is_tensor(blur1) else blur1,
                "age0": age0.squeeze().tolist() if torch.is_tensor(age0) else age0,
                "mae": mae,
                "psnr": p_val,
                "ssim": s_val,
                "dice": dice_score, 
                "mae_nodule": mae_nodule if mae_nodule is not None else 'N/A',
                "psnr_nodule": psnr_nodule if psnr_nodule is not None else 'N/A',
                "ssim_nodule": ssim_nodule if ssim_nodule is not None else 'N/A'        
            })


    # --- PRINTING AND SAVING LOGIC ---
    for current_seed, seed_results in results_per_seed.items():
        print(f"\n--- Seed {current_seed} Metrics (diff_blur_mode={diff_blur_mode}) ---")
        if len(seed_results) > 0:
            avg_mae = sum(r['mae'] for r in seed_results) / len(seed_results)
            avg_psnr = sum(r['psnr'] for r in seed_results) / len(seed_results)
            avg_ssim = sum(r['ssim'] for r in seed_results) / len(seed_results)
            avg_dice = sum(r['dice'] for r in seed_results) / len(seed_results)
            print(f"Average MAE:  {avg_mae:.4f}")
            print(f"Average PSNR: {avg_psnr:.4f}")
            print(f"Average SSIM: {avg_ssim:.4f}")
            print(f"Average DICE: {avg_dice:.4f}")
            
            nodule_metrics = [r for r in seed_results if r['mae_nodule'] != 'N/A']
            if nodule_metrics:
                avg_mae_nodule = sum(r['mae_nodule'] for r in nodule_metrics) / len(nodule_metrics)
                avg_psnr_nodule = sum(r['psnr_nodule'] for r in nodule_metrics) / len(nodule_metrics)
                avg_ssim_nodule = sum(r['ssim_nodule'] for r in nodule_metrics) / len(nodule_metrics)
                print(f"\nNodule Region Metrics (dilated, n={len(nodule_metrics)}):")
                print(f"Average MAE:  {avg_mae_nodule:.4f}")
                print(f"Average PSNR: {avg_psnr_nodule:.4f}")
                print(f"Average SSIM: {avg_ssim_nodule:.4f}")
        else:
            print("No samples found for this seed.")

        csv_path = os.path.join(
            conf.out_dir, f"results_{conf.model_num}_{diff_blur_mode}_seed_{current_seed}.csv"
        )
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ["sample", "pair", "diff_blur_mode", "diff_blur_used",
                        "blur0", "blur1", "age0", "mae", "psnr", "ssim", "dice",
                        "mae_nodule", "psnr_nodule", "ssim_nodule"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in seed_results:
                writer.writerow(row)
        print(f"Results saved to {csv_path}")
        
    print("\n--- Inference Complete ---")

if __name__ == "__main__":
    main()