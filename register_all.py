"""
Batch PET image registration to FDG template space.

Registers all .nii.gz PET images across all tracers to the FDG PET template
(TEMPLATE_FDGPET_100.nii, shape 91×109×91) using ANTsPy.

Usage:
    python register_all.py                        # dry-run: report counts only
    python register_all.py --run                 # run registration (affine, fast)
    python register_all.py --run --method SyN    # run with SyN (slower, more accurate)
    python register_all.py --run --workers 8     # parallel with 8 processes
    python register_all.py --run --tracer 18F-FDG  # only one tracer

Output:
    data_registered/
    ├── 18F-FDG/
    │   ├── AD/
    │   │   └── PET_xxx.nii       # registered .nii files
    │   ├── HC/
    │   └── ...
    └── ...
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from multiprocessing import Pool, cpu_count

import ants
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEMPLATE_PATH = 'template/TEMPLATE_FDGPET_100.nii'
DATA_ROOT = 'data'
OUTPUT_ROOT = 'data_registered'

# Tracer subdirectories that contain .nii.gz files (skip fold-csv dirs)
SKIP_DIRS = {'AD_HC', 'HC_MCI', 'EMCI_LMCI', 'HC_ALL_MCI'}


def collect_files(data_root, tracer_filter=None):
    """
    Walk data/ and collect all .nii.gz files, building output path mapping.

    Returns: list of (input_path, output_path)
    """
    files = []
    for tracer in sorted(os.listdir(data_root)):
        if tracer_filter and tracer != tracer_filter:
            continue
        tracer_path = os.path.join(data_root, tracer)
        if not os.path.isdir(tracer_path):
            continue

        for cls_name in sorted(os.listdir(tracer_path)):
            cls_path = os.path.join(tracer_path, cls_name)
            if not os.path.isdir(cls_path) or cls_name in SKIP_DIRS:
                continue

            for fname in sorted(os.listdir(cls_path)):
                if not fname.endswith('.nii.gz'):
                    continue
                in_path = os.path.join(cls_path, fname)
                # Output: data_registered/tracer/class/file.nii
                out_name = fname.replace('.nii.gz', '.nii')
                out_dir = os.path.join(OUTPUT_ROOT, tracer, cls_name)
                out_path = os.path.join(out_dir, out_name)
                files.append((in_path, out_path))

    return files


def register_one(args_tuple):
    """
    Register a single image. Called by multiprocessing workers.

    Args: (input_path, output_path, template_path, method)
    Returns: (status, input_path, output_path, elapsed_sec, error_msg)
    """
    in_path, out_path, template_path, method = args_tuple

    try:
        # Skip if already done
        if os.path.exists(out_path):
            return ('skipped', in_path, out_path, 0.0, None)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        t0 = time.time()
        fixed = ants.image_read(template_path)
        moving = ants.image_read(in_path)

        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=method,
            random_seed=42,
            verbose=False,
        )
        warped = reg['warpedmovout']
        ants.image_write(warped, out_path)
        elapsed = time.time() - t0

        return ('ok', in_path, out_path, elapsed, None)

    except Exception as e:
        elapsed = time.time() - t0 if 't0' in dir() else 0.0
        return ('error', in_path, out_path, elapsed, str(e))


def _sync_csv_files(tracer_filter=None):
    """
    Copy fold CSV files and all.csv from data/ to data_registered/ so the
    registered output directory is self-contained and can be passed directly
    to --data_dir.
    """
    import shutil
    tracers = [tracer_filter] if tracer_filter else sorted(os.listdir(DATA_ROOT))
    for tracer in tracers:
        tracer_src = os.path.join(DATA_ROOT, tracer)
        tracer_dst = os.path.join(OUTPUT_ROOT, tracer)
        if not os.path.isdir(tracer_src):
            continue

        # Copy all.csv
        all_csv = os.path.join(tracer_src, 'all.csv')
        if os.path.isfile(all_csv):
            os.makedirs(tracer_dst, exist_ok=True)
            shutil.copy2(all_csv, os.path.join(tracer_dst, 'all.csv'))

        # Copy task fold directories (AD_HC, HC_MCI, etc.)
        for item in sorted(os.listdir(tracer_src)):
            item_path = os.path.join(tracer_src, item)
            if not os.path.isdir(item_path):
                continue
            if item not in SKIP_DIRS:
                continue
            dst_dir = os.path.join(tracer_dst, item)
            if os.path.exists(dst_dir):
                continue
            shutil.copytree(item_path, dst_dir)
            print(f'  Copied CSV task: {tracer}/{item}')


def main():
    parser = argparse.ArgumentParser(description='Batch PET registration to FDG template')
    parser.add_argument('--run', action='store_true',
                        help='Actually run registration (default: dry-run)')
    parser.add_argument('--method', default='Affine',
                        choices=['Affine', 'SyN', 'SyNOnly', 'Rigid', 'TRSAA', 'SyNCC'],
                        help='ANTs registration type (default: Affine)')
    parser.add_argument('--workers', type=int, default=4,
                        help=f'Number of parallel workers (default: 4, max CPU: {cpu_count()})')
    parser.add_argument('--tracer', default=None,
                        choices=['18F-FDG', '18F-FBB', '18F-AV45', '18F-AV1451'],
                        help='Process only one tracer')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of images (for testing)')
    args = parser.parse_args()

    # Validate template
    if not os.path.exists(TEMPLATE_PATH):
        print(f'ERROR: Template not found: {TEMPLATE_PATH}')
        sys.exit(1)

    # Collect files
    files = collect_files(DATA_ROOT, args.tracer)
    if args.limit > 0:
        files = files[:args.limit]

    n_total = len(files)
    n_done = sum(1 for _, out in files if os.path.exists(out))

    print(f'{"="*60}')
    print(f'Template: {TEMPLATE_PATH}')
    print(f'Method:   {args.method}')
    print(f'Tracer:   {args.tracer or "all"}')
    print(f'Total:    {n_total} images')
    print(f'Already:  {n_done} done')
    print(f'To do:    {n_total - n_done}')
    print(f'Workers:  {args.workers} (parallel processes)')
    print(f'Output:   {OUTPUT_ROOT}/')
    print(f'{"="*60}')

    if not args.run:
        # Dry-run: show sample
        print('\n[Dry-run mode] Sample files that would be processed:')
        for in_p, out_p in files[:10]:
            status = ' [EXISTS]' if os.path.exists(out_p) else ''
            print(f'  {in_p}')
            print(f'    -> {out_p}{status}')
        if n_total > 10:
            print(f'  ... and {n_total - 10} more')
        print(f'\nAdd --run to execute registration.')
        return

    # Copy CSV files from data/ to data_registered/ so the output is self-contained
    _sync_csv_files(args.tracer)

    if n_total - n_done == 0:
        print('All files already registered!')
        return

    # Prepare work list (skip already done)
    work = [(in_p, out_p, TEMPLATE_PATH, args.method)
            for in_p, out_p in files if not os.path.exists(out_p)]

    print(f'\nProcessing {len(work)} images with {args.workers} workers...\n')

    t_start = time.time()
    ok_count = 0
    err_count = 0
    skip_count = n_done

    with Pool(processes=args.workers) as pool:
        results = pool.imap_unordered(register_one, work)

        for i, (status, in_p, out_p, elapsed, err) in enumerate(results):
            if status == 'ok':
                ok_count += 1
            elif status == 'error':
                err_count += 1
                print(f'  ERROR [{in_p}]: {err}')
            elif status == 'skipped':
                skip_count += 1

            # Progress every 50 images
            done_so_far = ok_count + err_count
            if done_so_far % 50 == 0 and done_so_far > 0:
                elapsed_total = time.time() - t_start
                rate = done_so_far / elapsed_total
                remaining = (len(work) - done_so_far) / rate if rate > 0 else 0
                print(f'  [{done_so_far}/{len(work)}] '
                      f'OK={ok_count} ERR={err_count} '
                      f'Rate={rate:.1f} img/s '
                      f'ETA={remaining/60:.1f} min')

    elapsed_total = time.time() - t_start
    print(f'\n{"="*60}')
    print(f'Complete!')
    print(f'  OK:      {ok_count}')
    print(f'  Errors:  {err_count}')
    print(f'  Skipped: {skip_count}')
    print(f'  Time:    {elapsed_total/60:.1f} min ({elapsed_total:.1f}s)')
    print(f'  Output:  {OUTPUT_ROOT}/')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
