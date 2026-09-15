"""
brain_pair_dataset.py
=====================
Longitudinal brain MRI pair dataset for OASIS-3 (or any BrLP-format dataset).

Reads pairs from `our_B.csv` (the nodule-dataloader-format B file generated
by make_csv_B.py) and serves (img0, img1) pairs with the same interface as
the original NGPPairDataset, but using BrLP-style image-space preprocessing
instead of the CT-style preprocessing.

Differences from NGPPairDataset
-------------------------------
- Reads pairs directly from a single CSV (no JSON, no triplet expansion).
- Splits come from the `split` column ('train' / 'valid' / 'test'),
  consistent with BrLP's convention.
- Intensity scaling matches turboprep's output range (the brain MRIs are
  already intensity-normalized to roughly [0, 1]); we only rescale to
  [-1, 1] so it matches your existing model's expected range.
- Mask handling: turboprep's segm.nii.gz has multiple integer SynthSeg
  labels. We default to thresholding (m > 0) -> 1 to get a brain mask.
  Set `mask_mode='multilabel'` if your model expects the raw labels.
"""

import os
import re
import torch
import numpy as np
import pandas as pd
from scipy import ndimage
from torch.utils.data import Dataset
from monai import transforms as monai_transforms
from monai.data.image_reader import NibabelReader


class FillOutsideMaskD(monai_transforms.MapTransform):
    """
    Set voxels outside the (optionally dilated, hole-filled) brain mask to
    `fill_value`, BUT only if their current intensity looks like background
    (<= max_fill_intensity). High-intensity voxels are left alone even if
    they're outside the mask — they're probably mis-labeled brain.

    Args:
        image_key:          key for the image to modify
        mask_key:           key for the SynthSeg mask
        fill_value:         value to write at filled voxels
        dilation:           # voxels to dilate the binary mask before use
        fill_holes:         fill enclosed 3D holes in the mask before dilation
        max_fill_intensity: only overwrite voxels with intensity <= this value.
                            None means no intensity guard (fill everything
                            outside the mask).
        verify:             assert that no voxel we promised not to touch
                            actually changed
    """
    def __init__(self, image_key, mask_key, fill_value,
                 dilation=2, fill_holes=True,
                 max_fill_intensity=None, verify=False):
        super().__init__(keys=[image_key])
        self.image_key          = image_key
        self.mask_key           = mask_key
        self.fill_value         = float(fill_value)
        self.dilation           = int(dilation)
        self.fill_holes         = bool(fill_holes)
        self.max_fill_intensity = (None if max_fill_intensity is None
                                   else float(max_fill_intensity))
        self.verify             = bool(verify)

    def _build_keep_region(self, msk_np):
        keep = msk_np > 0
        if self.fill_holes:
            if keep.ndim == 4 and keep.shape[0] == 1:
                keep = ndimage.binary_fill_holes(keep[0])[None]
            else:
                keep = ndimage.binary_fill_holes(keep)
        if self.dilation > 0:
            if keep.ndim == 4 and keep.shape[0] == 1:
                keep = ndimage.binary_dilation(keep[0], iterations=self.dilation)[None]
            else:
                keep = ndimage.binary_dilation(keep, iterations=self.dilation)
        return keep

    def __call__(self, data):
        d = dict(data)
        img = d[self.image_key]
        msk = d[self.mask_key]

        # Mask -> numpy for morphology
        if isinstance(msk, torch.Tensor):
            msk_np = msk.detach().cpu().numpy()
        else:
            msk_np = np.asarray(msk)
        keep_np = self._build_keep_region(msk_np)   # True = leave alone

        if isinstance(img, torch.Tensor):
            keep_t = torch.from_numpy(keep_np).to(img.device)
            # voxels to fill: outside keep region AND (no intensity guard OR
            # intensity is below threshold)
            fill_t = ~keep_t
            if self.max_fill_intensity is not None:
                fill_t = fill_t & (img <= self.max_fill_intensity)
            new_img = torch.where(fill_t,
                                  torch.full_like(img, self.fill_value),
                                  img)
            if self.verify:
                protected = ~fill_t
                assert torch.equal(new_img[protected], img[protected]), \
                    "FillOutsideMaskD modified a voxel it shouldn't have!"
        else:
            img_np = np.asarray(img)
            fill = ~keep_np
            if self.max_fill_intensity is not None:
                fill = fill & (img_np <= self.max_fill_intensity)
            new_img = np.where(fill, self.fill_value, img_np)
            if self.verify:
                protected = ~fill
                assert np.array_equal(new_img[protected], img_np[protected]), \
                    "FillOutsideMaskD modified a voxel it shouldn't have!"

        d[self.image_key] = new_img
        return d
    
def _gradient(volume):
    gx = ndimage.sobel(volume, axis=0)
    gy = ndimage.sobel(volume, axis=1)
    gz = ndimage.sobel(volume, axis=2)
    return gx, gy, gz


def tenengrad(volume):
    gx, gy, gz = _gradient(volume)
    return float(np.mean(gx**2 + gy**2 + gz**2))


def tenengrad_per_axis(volume):
    gx, gy, gz = _gradient(volume)
    return float(np.mean(gx**2)), float(np.mean(gy**2)), float(np.mean(gz**2))



class BrainPairDataset(Dataset):
    """
    Longitudinal brain MRI pair dataset.

    Args:
        root_dir:       Path to our_B.csv (or BrLP_B.csv if you use BrLP names).
        mode:           'train' / 'valid' / 'test' — selects from the `split` column.
        dim:            Target spatial dim. int (cube) or tuple/list of 3 ints.
        diff:           If True, returns (img1 - img0)/2 in `img1` (matching your
                        existing NGP convention).
        mask_mode:      'binary' (m > 0 -> 1, default) or 'multilabel' (keep raw
                        SynthSeg labels).
        cache_meta:     If True (default), MONAI transforms keep meta-tensors;
                        set False to strip metadata for slightly faster loading.

    Output dict keys (matching NGPPairDataset):
        baseline_img:   raw starting image
        img0, img1:     starting & followup (or difference, if diff=True)
        mask0, mask1:   brain masks rescaled to [-1, 1]
        dt:             time gap (months — same unit as your NGP dataloader)
        age0:           starting age in years
        gender:         sex (0=M, 1=F per BrLP convention)
        pair:           always 0 for brain (no T1/T2/T3 categories)
        blur0, blur1:   per-axis tenengrad (3 values each)
        PatientId:      subject_id
        NoduleId:       subject_id
        t0, t1:         days-from-entry for starting/followup scans
    """

    def __init__(
        self,
        root_dir: str,
        mode: str,
        dim,
        diff: bool = False,
        mask_mode: str = 'binary',
        intensity_scope=(-88, 25),     # turboprep's normalized output after WhiteStripe (we checked all dataset min max values and got that approx this range covers all scans)
        intensity_range=(-1.0, 1.0),    # rescale target
    ):
        super().__init__()
        assert mode in ('train', 'valid', 'test'), \
            f"mode must be train/valid/test, got {mode!r}"
        assert mask_mode in ('binary', 'multilabel'), \
            f"mask_mode must be binary or multilabel, got {mask_mode!r}"

        self.root_dir = root_dir
        self.mode = mode
        self.dim = (dim, dim, dim) if isinstance(dim, int) else tuple(dim)
        self.diff = diff
        self.mask_mode = mask_mode
        self.intensity_scope = intensity_scope
        self.intensity_range = intensity_range

        # ---- Load and filter pairs ----
        df = pd.read_csv(self.root_dir)
        df = df[df['split'] == mode].copy().reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(
                f"No rows found for split={mode!r} in {self.root_dir}. "
                f"Available splits: {pd.read_csv(self.root_dir)['split'].unique()}"
            )
        self.df = df
        print(f"[{mode}] Loaded {len(df)} pairs / "
              f"{df['PatientId'].nunique()} subjects from {self.root_dir}")
        # ---- Build the MONAI transform pipeline ----
        # We load image+segm separately (different dtypes), apply
        # spatial-padding, intensity scaling, then convert to tensors.
        is_train = (mode == 'train')

        common_steps = [
            monai_transforms.LoadImageD(keys=['im0', 'im1', 'msk0', 'msk1'], reader=NibabelReader()),
            monai_transforms.EnsureChannelFirstD(keys=['im0', 'im1', 'msk0', 'msk1']),

            # NEW: unify the background. Brain voxels (msk > 0) untouched.
            FillOutsideMaskD('im0', 'msk0', fill_value=intensity_scope[0], dilation=4),  # -57
            FillOutsideMaskD('im1', 'msk1', fill_value=intensity_scope[0], dilation=4),  # -57

            # Pad with the same background value — no discontinuity possible.
            # Downsample with a SINGLE uniform scale factor (longest edge -> dim),
            # so no axis is distorted relative to another. Nothing is cropped.
            monai_transforms.ResizeD(
                keys=['im0', 'im1'],
                spatial_size=max(self.dim),
                size_mode='longest',
                mode='trilinear',
                align_corners=False,
            ),
            monai_transforms.ResizeD(
                keys=['msk0', 'msk1'],
                spatial_size=max(self.dim),
                size_mode='longest',
                mode='nearest',          # preserve integer SynthSeg labels
            ),
            # Letterbox the short axes back out to the full cube with the
            # background value (image) / label 0 (mask).
            monai_transforms.SpatialPadD(
                keys=['im0', 'im1'],
                spatial_size=self.dim,
                mode='constant',
                constant_values=intensity_scope[0],
            ),
            monai_transforms.SpatialPadD(
                keys=['msk0', 'msk1'],
                spatial_size=self.dim,
                mode='constant',          # defaults to 0 = outside-brain label
            ),
            # Rescale intensity from turboprep's range to [-1, 1].
            monai_transforms.ScaleIntensityRangeD(
                keys=['im0', 'im1'],
                a_min=intensity_scope[0], a_max=intensity_scope[1],
                b_min=intensity_range[0], b_max=intensity_range[1],
                clip=True,
            ),
        ]

        train_aug = [
            monai_transforms.RandFlipD(keys=['im0', 'im1', 'msk0', 'msk1'],
                                       prob=0.2, spatial_axis=0),
            monai_transforms.RandFlipD(keys=['im0', 'im1', 'msk0', 'msk1'],
                                       prob=0.2, spatial_axis=1),
            monai_transforms.RandFlipD(keys=['im0', 'im1', 'msk0', 'msk1'],
                                       prob=0.2, spatial_axis=2),
            monai_transforms.RandScaleIntensityD(
                keys=['im0', 'im1'], factors=0.1, prob=0.1),
            monai_transforms.RandShiftIntensityD(
                keys=['im0', 'im1'],
                offsets=0.1 * (intensity_range[1] - intensity_range[0]),
                prob=0.1),

            monai_transforms.Lambdad(
                keys=['im0', 'im1'],
                func=lambda x: x.clip(min=intensity_range[0], max=intensity_range[1])
            )
        ]

        finalize = [
            monai_transforms.ToTensorD(
                #keys=['im0', 'im1', 'msk0', 'msk1'], track_meta=False), 
                keys=['im0', 'im1', 'msk0', 'msk1']),
        ]

        steps = common_steps + (train_aug if is_train else []) + finalize
        self.transforms = monai_transforms.Compose(steps)

        if not is_train:
            print(f"[{mode}] Using deterministic transforms.")

    def __len__(self):
        return len(self.df)

    def _binarize_mask(self, m: torch.Tensor) -> torch.Tensor:
        """SynthSeg labels (0..N) -> binary brain mask (0/1)."""
        if self.mask_mode == 'binary':
            return (m > 0).float()
        return m.float()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1) Apply MONAI transforms (load, pad, crop, scale).
        sample_in = {
            'im0':  row['starting_image_path'],
            'im1':  row['followup_image_path'],
            'msk0': row['starting_segm_path'],
            'msk1': row['followup_segm_path'],
        }
        out = self.transforms(sample_in)

        i0 = out['im0'].float()
        i1 = out['im1'].float()
        seg1 = out['msk1'].float()   
        m0 = self._binarize_mask(out['msk0'])
        m1 = self._binarize_mask(out['msk1'])

        # 2) Per-axis Tenengrad blur on the raw intensity images.
        blur0_x, blur0_y, blur0_z = tenengrad_per_axis(i0.squeeze().numpy())
        blur1_x, blur1_y, blur1_z = tenengrad_per_axis(i1.squeeze().numpy())

        # 3) Rescale masks from {0, 1} to {-1, 1} to match image range.
        if self.mask_mode == 'binary':
            m0 = m0 * 2.0 - 1.0
            m1 = m1 * 2.0 - 1.0

        # 4) Save the raw baseline image before any differencing.
        baseline_img = i0.clone()

        # 5) Difference mode (matches your NGP convention).
        if self.diff:
            img1_diff = (i1 - i0) / 2.0          # difference normalized to [-1, 1]
            if not torch.allclose(img1_diff * 2.0 + i0, i1, atol=1e-5):
                print(f"⚠️ Difference reconstruction failed at idx={idx}")
            i1 = img1_diff

        # 6) Build the output dict.
        return {
            'baseline_img': baseline_img,
            'img0':         i0,
            'img1':         i1,
            'mask0':        m0,
            'mask1':        m1,
            'seg1':         seg1,         
            'dt':           torch.tensor([row['dt_months']], dtype=torch.float32),
            'age0':         torch.tensor([row['age0']],      dtype=torch.float32),
            'gender':       torch.tensor([row['gender']],    dtype=torch.float32),
            'pair':         torch.tensor([row['pair']],      dtype=torch.int64),
            'blur0':        torch.tensor([blur0_x, blur0_y, blur0_z],
                                          dtype=torch.float32),
            'blur1':        torch.tensor([blur1_x, blur1_y, blur1_z],
                                          dtype=torch.float32),
            'PatientId':    row['PatientId'],
            'NoduleId':     row['NoduleId'],
            't0':           int(row['t0']) if not pd.isna(row['t0']) else -1,
            't1':           int(row['t1']) if not pd.isna(row['t1']) else -1,
        }