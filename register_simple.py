"""
Simple PET-to-template registration using only nibabel + scipy (NO ANTsPy required).

Performs center-crop or resize to match the template shape (91, 109, 91).
Since ADNI PET data is already coarsely aligned to MNI space, this provides
a reasonable approximation when ANTsPy is unavailable.

Usage:
    python register_simple.py                  # dry-run
    python register_simple.py --run --workers 8
"""

import os
import sys
import time
import argparse
import traceback
from multiprocessing import Pool, cpu_count
from scipy.ndimage import zoom, center_of_mass
import nibabel as nib
import numpy as np

TEMPLATE_PATH = 'template/TEMPLATE_FDGPET_100.nii'
DATA_ROOT = 'data'
OUTPUT_ROOT = 'data_registered'
SKIP_DIRS = {'AD_HC', 'HC_MCI', 'EMCI_LMCI', 'HC_ALL_MCI'}


def collect_files(data_root, tracer_filter=None):
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
                out_name = fname.replace('.nii.gz', '.nii')
                out_dir = os.path.join(OUTPUT_ROOT, tracer, cls_name)
                out_path = os.path.join(out_dir, out_name)
                files.append((in_path, out_path))
    return files


def register_one_simple(args_tuple):
    """Center-crop and resize to target shape."""
    in_path, out_path, target_shape = args_tuple
    try:
        if os.path.exists(out_path):
            return ('skipped', in_path, out_path, 0.0, None)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        t0 = time.time()

        img = nib.load(in_path)
        data = img.get_fdata().astype(np.float32)
        in_shape = np.array(data.shape)
        target = np.array(target_shape)

        # Step 1: Center-crop to match aspect ratio if needed
        crop_start = (in_shape - target) // 2
        crop_start = np.maximum(crop_start, 0)
        crop_end = crop_start + target

        # If input is larger in all dimensions, crop
        if np.all(in_shape >= target):
            cropped = data[crop_start[0]:crop_end[0],
                           crop_start[1]:crop_end[1],
                           crop_start[2]:crop_end[2]]
        else:
            # Otherwise, center and pad
            pad_total = np.maximum(0, target - in_shape)
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            cropped = np.pad(data, list(zip(pad_before, pad_after)),
                             mode='constant', constant_values=0)
            # Re-crop to exact target size after padding
            crop_start = np.zeros(3, dtype=int)
            new_shape = np.array(cropped.shape)
            excess = new_shape - target
            excess_start = np.maximum(excess // 2, 0)
            cropped = cropped[excess_start[0]:excess_start[0] + target[0],
                              excess_start[1]:excess_start[1] + target[1],
                              excess_start[2]:excess_start[2] + target[2]]

        # Step 2: Simple intensity normalization
        cropped = np.nan_to_num(cropped)
        if cropped.max() > 0:
            cropped = np.clip(cropped / cropped.max(), 0, 1)

        # Save with identity-like affine (template space)
        new_img = nib.Nifti1Image(cropped.astype(np.float32), np.eye(4))
        nib.save(new_img, out_path)

        elapsed = time.time() - t0
        return ('ok', in_path, out_path, elapsed, None)

    except Exception as e:
        return ('error', in_path, out_path, 0.0, str(e))


def _sync_csv_files(data_root, output_root, tracer_filter=None):
    import shutil
    tracers = [tracer_filter] if tracer_filter else sorted(os.listdir(data_root))
    for tracer in tracers:
        tracer_src = os.path.join(data_root, tracer)
        tracer_dst = os.path.join(output_root, tracer)
        if not os.path.isdir(tracer_src):
            continue
        all_csv = os.path.join(tracer_src, 'all.csv')
        if os.path.isfile(all_csv):
            os.makedirs(tracer_dst, exist_ok=True)
            shutil.copy2(all_csv, os.path.join(tracer_dst, 'all.csv'))
        for item in sorted(os.listdir(tracer_src)):
            item_path = os.path.join(tracer_src, item)
            if not os.path.isdir(item_path) or item not in SKIP_DIRS:
                continue
            dst_dir = os.path.join(tracer_dst, item)
            if not os.path.exists(dst_dir):
                shutil.copytree(item_path, dst_dir)
                print(f'  Synced CSV: {tracer}/{item}')


def main():
    parser = argparse.ArgumentParser(description='Simple PET registration (no ANTsPy)')
    parser.add_argument('--run', action='store_true', help='Actually run')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--tracer', default=None,
                        choices=['18F-FDG', '18F-FBB', '18F-AV45', '18F-AV1451'])
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    # Load template to get target shape
    if not os.path.exists(TEMPLATE_PATH):
        print(f'ERROR: Template not found: {TEMPLATE_PATH}')
        sys.exit(1)
    tpl = nib.load(TEMPLATE_PATH)
    target_shape = tpl.shape
    print(f'Target shape (from template): {target_shape}')

    files = collect_files(DATA_ROOT, args.tracer)
    if args.limit > 0:
        files = files[:args.limit]

    n_total = len(files)
    n_done = sum(1 for _, out in files if os.path.exists(out))

    print(f'{"="*60}')
    print(f'Method:   center-crop + resize (no ANTsPy)')
    print(f'Tracer:   {args.tracer or "all"}')
    print(f'Total:    {n_total} images')
    print(f'Already:  {n_done} done')
    print(f'To do:    {n_total - n_done}')
    print(f'Workers:  {args.workers}')
    print(f'{"="*60}')

    if not args.run:
        print('\n[Dry-run] Add --run to execute.')
        for in_p, out_p in files[:5]:
            print(f'  {in_p}  ->  {out_p}')
        return

    if n_total - n_done == 0:
        print('All done!')
        return

    _sync_csv_files(DATA_ROOT, OUTPUT_ROOT, args.tracer)

    work = [(in_p, out_p, target_shape)
            for in_p, out_p in files if not os.path.exists(out_p)]

    print(f'\nProcessing {len(work)} images with {args.workers} workers...\n')

    t_start = time.time()
    ok = err = skip = n_done

    with Pool(processes=args.workers) as pool:
        for i, (status, in_p, out_p, elapsed, error) in enumerate(pool.imap_unordered(register_one_simple, work)):
            if status == 'ok':
                ok += 1
            elif status == 'error':
                err += 1
                print(f'  ERROR: {in_p}: {error}')
            else:
                skip += 1
            if (ok + err) % 100 == 0 and (ok + err) > 0:
                elapsed_t = time.time() - t_start
                rate = (ok + err) / elapsed_t
                eta = (len(work) - ok - err) / rate if rate > 0 else 0
                print(f'  [{ok + err}/{len(work)}] OK={ok} ERR={err} '
                      f'Rate={rate:.1f} img/s ETA={eta/60:.1f}min')

    elapsed_t = time.time() - t_start
    print(f'\nComplete: OK={ok} ERR={err} SKIP={skip} Time={elapsed_t/60:.1f}min')


if __name__ == '__main__':
    main()
