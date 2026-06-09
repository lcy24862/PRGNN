"""
Data utilities for PET brain scan classification.
Supports multiple tracers (FDG, FBB, AV45, AV1451) and classification tasks.

Data directory structure:
    data/
    ├── 18F-FDG/
    │   ├── all.csv          # Master file list with columns: filename, mask, DX
    │   ├── AD/              # .nii.gz files for Alzheimer's Disease
    │   ├── HC/              # .nii.gz files for Healthy Controls
    │   ├── MCI/             # .nii.gz files for Mild Cognitive Impairment
    │   ├── EMCI/            # .nii.gz files for Early MCI
    │   ├── LMCI/            # .nii.gz files for Late MCI
    │   ├── AD_HC/           # 5-fold CV splits for AD vs HC (binary)
    │   ├── HC_MCI/          # 5-fold CV splits for HC vs MCI
    │   ├── EMCI_LMCI/       # 5-fold CV splits for EMCI vs LMCI
    │   └── HC_ALL_MCI/      # 5-fold CV splits for HC vs all MCI subtypes
    ├── 18F-FBB/
    │   ├── all.csv
    │   ├── HC/              MCI/
    │   └── HC_MCI/
    ├── 18F-AV45/
    │   ├── all.csv
    │   ├── AD/  HC/  MCI/  EMCI/  LMCI/
    │   ├── AD_HC/  HC_MCI/  EMCI_LMCI/  HC_ALL_MCI/
    └── 18F-AV1451/
        ├── all.csv
        ├── AD/  HC/  MCI/
        ├── AD_HC/  HC_MCI/

CSV format (fold CSVs and all.csv):
    filename,mask,DX
    ../ADNI/18F-FDG/AD/PET_003_S_4136.nii.gz,../ADNI/18F-FDG_mask/AD/PET_003_S_4136.nii.gz,AD

The 'filename' column contains a relative path; we extract the basename and locate the
actual file under data/{tracer}/{DX}/.
"""

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from monai.transforms import (
    Compose,
    Resize,
    RandFlip, RandRotate, RandZoom, RandAffine, RandGaussianNoise,
    ToTensor,
)
import numpy as np
import pandas as pd
import nibabel as nib
import os
import argparse


# ---------------------------------------------------------------------------
# Label mappings for each tracer & task
# ---------------------------------------------------------------------------
# Multiclass label dicts (all.csv)
ALL_LABEL_DICT = {
    'HC': 0,
    'AD': 1,
    'MCI': 2,
    'EMCI': 3,
    'LMCI': 4,
}

# Per-task binary labels – which classes are included
TASK_CLASSES = {
    'AD_HC':      ['AD', 'HC'],
    'HC_MCI':     ['HC', 'MCI'],
    'EMCI_LMCI':  ['EMCI', 'LMCI'],
    'HC_ALL_MCI': ['HC', 'MCI'],       # MCI here means all MCI subtypes (EMCI, LMCI, MCI)
    'all':        ['HC', 'AD', 'MCI', 'EMCI', 'LMCI'],
}

# Available tasks per tracer (populated at runtime)
TRACER_TASKS = {
    '18F-FDG':     ['AD_HC', 'HC_MCI', 'EMCI_LMCI', 'HC_ALL_MCI', 'all'],
    '18F-FBB':     ['HC_MCI', 'all'],
    '18F-AV45':    ['AD_HC', 'HC_MCI', 'EMCI_LMCI', 'HC_ALL_MCI', 'all'],
    '18F-AV1451':  ['AD_HC', 'HC_MCI', 'all'],
    'Amyloid':     ['HC_MCI', 'HC_ALL_MCI'],          # merged AV45 + FBB
}


# ---------------------------------------------------------------------------
# Data-level index -> feature-map shape helper
# ---------------------------------------------------------------------------
def compute_feature_map_shapes(input_shape):
    """
    Given an input 3D shape (D, H, W), return the expected feature-map spatial
    sizes after the ResNetFeatures backbone (used by PRGNN)::

        stage0 – after conv1 (kernel=7, stride=2, pad=3)
        stage1 – after maxpool + layer1 (no spatial change from layer1)
        stage2 – after layer2 (stride=2 on first block)
        stage3 – after layer3 (stride=2 on first block)

    Also returns the ViG stem output shape (4× stride-2 conv3d).
    """
    def conv3d_out(in_size, kernel=7, stride=2, pad=3):
        return (in_size + 2 * pad - kernel) // stride + 1

    def pool3d_out(in_size, kernel=3, stride=2, pad=1):
        return (in_size + 2 * pad - kernel) // stride + 1

    D, H, W = input_shape

    # PRGNN / ResNetFeatures stages
    s0 = (conv3d_out(D), conv3d_out(H), conv3d_out(W))          # after conv1
    mp = (pool3d_out(s0[0]), pool3d_out(s0[1]), pool3d_out(s0[2]))  # after maxpool
    s1 = mp                                                      # layer1 (stride=1)
    # layer2 first block uses stride-2 conv
    s2 = (conv3d_out(s1[0], 3, 2, 1), conv3d_out(s1[1], 3, 2, 1), conv3d_out(s1[2], 3, 2, 1))
    s3 = (conv3d_out(s2[0], 3, 2, 1), conv3d_out(s2[1], 3, 2, 1), conv3d_out(s2[2], 3, 2, 1))

    # ViG stem (4 × stride-2)
    vig = input_shape
    for _ in range(4):
        vig = (conv3d_out(vig[0], 3, 2, 1), conv3d_out(vig[1], 3, 2, 1), conv3d_out(vig[2], 3, 2, 1))

    return {
        'input': input_shape,
        'stage0': s0,
        'stage1': s1,
        'stage2': s2,
        'stage3': s3,
        'vig_stem': vig,
        'vig_flatten': vig[0] * vig[1] * vig[2],
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PETDataset(Dataset):
    """
    PET brain scan dataset that loads .nii.gz files from a flat class directory.

    Args:
        df (pd.DataFrame): DataFrame with columns 'filename' (str), 'DX' (str).
        data_root (str): Root of the data directory (e.g. 'data/18F-FDG').
        label_dict (dict): Mapping from DX string → integer label.
        train (bool): Whether to apply training augmentations.
        target_size (tuple | None): If given, resize volumes to this (D, H, W).
    """

    def __init__(self, df, data_root, label_dict, train=True, target_size=None):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.label_dict = label_dict
        self.train = train
        self.target_size = target_size

        # Build augmentation pipelines
        if target_size is not None:
            resize = Resize(spatial_size=target_size, mode='trilinear', align_corners=False)
        else:
            resize = Compose([])   # no-op

        self.train_transform = Compose([
            resize,
            RandFlip(prob=0.2, spatial_axis=0),
            RandRotate(range_x=15, range_y=15, range_z=15, prob=0.3),
            ToTensor(),
        ])

        self.val_transform = Compose([
            resize,
            ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def _locate_file(self, row):
        """
        Given a row with 'filename' and 'DX', find the actual file on disk.
        The CSV 'filename' is a relative path like '../ADNI/18F-FDG/AD/PET_xxx.nii.gz'.
        We extract the basename, try both .nii and .nii.gz extensions, and look
        under data_root / DX / basename.
        """
        csv_path = row['filename']
        basename = os.path.basename(csv_path)

        # Try multiple possible extensions (.nii.gz ↔ .nii)
        if basename.endswith('.nii.gz'):
            candidates_ext = [basename, basename[:-7] + '.nii']
        elif basename.endswith('.nii'):
            candidates_ext = [basename, basename + '.gz']
        else:
            candidates_ext = [basename + '.nii.gz', basename + '.nii']

        dx = row['DX']

        for fname in candidates_ext:
            candidate = os.path.join(self.data_root, dx, fname)
            if os.path.isfile(candidate):
                return candidate

        # Fallback: walk data_root for any matching extension
        for fname in candidates_ext:
            for root, dirs, files in os.walk(self.data_root):
                if fname in files:
                    return os.path.join(root, fname)

        raise FileNotFoundError(
            f"Cannot locate {basename} (tried {candidates_ext}) under {self.data_root}"
        )

    def load_data(self, path):
        data = nib.load(path).get_fdata()
        data = data[np.newaxis, ...]          # (1, D, H, W)
        data = np.nan_to_num(data)
        data = np.clip(data, a_min=0, a_max=1)
        return data.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = self._locate_file(row)
        data = self.load_data(file_path)
        label = self.label_dict[row['DX']]

        if self.train:
            data = self.train_transform(data)
        else:
            data = self.val_transform(data)

        return data, label


def collate_fn(batch):
    data = []
    labels = []
    for img, lbl in batch:
        data.append(img)
        labels.append(lbl)
    labels = torch.tensor(labels, dtype=torch.int64)
    return torch.stack(data).type(torch.float), labels


# ---------------------------------------------------------------------------
# Data-loader builder
# ---------------------------------------------------------------------------
def get_loader(args, fold=0, num_classes=2, batch_size=32, num_workers=0):
    """
    Build train/val/test DataLoaders.

    Parameters
    ----------
    args : argparse.Namespace or similar
        Must have: data_dir (str), tracer (str), task (str),
        target_size (tuple|None), num_folds (int).
    fold : int
        0-based fold index (0–4).
    num_classes : int
        Number of classes in the task.
    batch_size : int
    num_workers : int

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    tracer_dir = os.path.join(args.data_dir, args.tracer)

    # Determine label dict for this task
    task_classes = TASK_CLASSES.get(args.task, TASK_CLASSES['all'])
    if args.task == 'all':
        label_dict = ALL_LABEL_DICT
    else:
        label_dict = {cls_name: i for i, cls_name in enumerate(task_classes)}

    if args.task == 'all':
        # Multiclass: use all.csv, split on the fly
        all_csv = os.path.join(tracer_dir, 'all.csv')
        all_df = pd.read_csv(all_csv)

        # Filter to task_classes if needed
        all_df = all_df[all_df['DX'].isin(task_classes)].reset_index(drop=True)

        # Stratified 5-fold split (fixed seed for reproducibility)
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)
        fold_indices = list(skf.split(np.zeros(len(all_df)), all_df['DX']))

        # fold_indices[fold] = (trainval_idx, test_idx)
        trainval_idx, test_idx = fold_indices[fold]

        # Further split trainval into train/val (80/20 of trainval, by class)
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=0.15, random_state=42 + fold,
            stratify=all_df.iloc[trainval_idx]['DX'],
        )

        train_df = all_df.iloc[train_idx].reset_index(drop=True)
        val_df = all_df.iloc[val_idx].reset_index(drop=True)
        test_df = all_df.iloc[test_idx].reset_index(drop=True)

    else:
        # Binary task: read pre-made fold CSVs
        task_dir = os.path.join(tracer_dir, args.task)
        train_csv = os.path.join(task_dir, f'train_fold{fold}.csv')
        val_csv = os.path.join(task_dir, f'val_fold{fold}.csv')
        test_csv = os.path.join(task_dir, f'test_fold{fold}.csv')

        if not os.path.isfile(train_csv):
            raise FileNotFoundError(
                f"Fold CSV not found: {train_csv}. "
                f"Available tasks for {args.tracer}: {TRACER_TASKS.get(args.tracer, [])}"
            )

        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)

    # Build datasets
    target_size = getattr(args, 'target_size', None)

    train_ds = PETDataset(train_df, tracer_dir, label_dict, train=True, target_size=target_size)
    val_ds = PETDataset(val_df, tracer_dir, label_dict, train=False, target_size=target_size)
    test_ds = PETDataset(test_df, tracer_dir, label_dict, train=False, target_size=target_size)

    # WeightedRandomSampler for training
    value_counts = train_df['DX'].value_counts()
    num_samples = len(train_df)
    class_weights = [num_samples / value_counts.get(cls, 1) for cls in task_classes]
    weights = [class_weights[label_dict[row['DX']]] for _, row in train_df.iterrows()]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        sampler=sampler, drop_last=True, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, drop_last=False, collate_fn=collate_fn, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, drop_last=False, collate_fn=collate_fn, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Argument helpers (to be merged into train.py / test.py)
# ---------------------------------------------------------------------------
def add_data_args(parser: argparse.ArgumentParser):
    """Add data-related arguments to an existing parser."""
    parser.add_argument('--data_dir', default='data', type=str,
                        help='Root directory of the dataset')
    parser.add_argument('--tracer', default='18F-FDG', type=str,
                        choices=['18F-FDG', '18F-FBB', '18F-AV45', '18F-AV1451', 'Amyloid'],
                        help='PET tracer type')
    parser.add_argument('--task', default='AD_HC', type=str,
                        help='Classification task: AD_HC, HC_MCI, EMCI_LMCI, HC_ALL_MCI, or all')
    parser.add_argument('--num_folds', default=5, type=int,
                        help='Number of folds for cross-validation')
    parser.add_argument('--target_size', default=None, type=int, nargs=3,
                        help='Resize volumes to (D, H, W), e.g. --target_size 96 96 96')
    return parser


# ---------------------------------------------------------------------------
# Quick dimension check utility
# ---------------------------------------------------------------------------
def print_dimension_info(args):
    """
    Print information about data dimensions and model compatibility.
    Call this once at startup to verify settings.
    """
    tracer_dir = os.path.join(args.data_dir, args.tracer)

    # Find a sample file
    for root, dirs, files in os.walk(tracer_dir):
        nii_files = [f for f in files if f.endswith('.nii.gz') or f.endswith('.nii')]
        if nii_files:
            sample = os.path.join(root, nii_files[0])
            break
    else:
        print("[WARN] No .nii/.nii.gz files found – cannot check dimensions.")
        return

    img = nib.load(sample)
    native_shape = img.shape  # (D, H, W)

    target = getattr(args, 'target_size', None)
    if target is not None:
        target = tuple(target)
    effective_shape = target if target else native_shape

    shapes = compute_feature_map_shapes(effective_shape)

    print(f"[Data dims] Native: {native_shape}, Effective (after resize): {effective_shape}")
    print(f"  PRGNN stage0 feature map: {shapes['stage0']}")
    print(f"  PRGNN stage1 feature map: {shapes['stage1']}")
    print(f"  PRGNN stage2 feature map: {shapes['stage2']}")
    print(f"  PRGNN stage3 feature map: {shapes['stage3']}")
    print(f"  ViG stem output:          {shapes['vig_stem']} (flatten={shapes['vig_flatten']})")

    # Check against PRGNN hardcoded sizes
    hardcoded = {
        'stage0': (46, 55, 46),
        'stage1': (23, 28, 23),
        'stage2': (12, 14, 12),
        'stage3': (6, 7, 6),
    }
    print(f"\n[Model compatibility]")
    for stage, hc in hardcoded.items():
        actual = shapes[stage]
        match = "OK" if actual == hc else f"MISMATCH (expected {hc})"
        print(f"  {stage}: {actual}  {match}")

    print(f"\n  NOTE: PRGNN contribution maps and ViG pos_embed are hardcoded for")
    print(f"  input ~(91, 109, 91). With native data (121, 145, 121), you must either:")
    print(f"    1. Use --target_size 91 109 91 (or 96 96 96 with model fixes), or")
    print(f"    2. Modify prgnn.py/vig.py to compute sizes dynamically.")

    return shapes
