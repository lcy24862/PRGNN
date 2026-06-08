"""
Collect all training results into a single summary for local analysis.
Supports multi-tracer directory structure: models/{tracer}/{task}/

Usage (on server):
    python collect_results.py                    # creates results_summary/
    python collect_results.py --output my_results
    python collect_results.py --tracer 18F-FDG   # single tracer only

Output:
    results_summary/
    ├── summary.csv              # all tracers/tasks/folds metrics
    ├── summary_per_fold.csv     # per-fold breakdown
    ├── per_sample_predictions/  # per-task prediction CSVs
    ├── training_curves/         # per-fold metrics JSON
    ├── result_texts/            # raw result files
    └── results_summary.zip      # ready to download
"""

import os, sys, json, shutil, argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

MODELS_DIR = 'models'
RESULTS_DIR = 'results'


def collect_results(output_dir='results_summary', tracer_filter=None):
    os.makedirs(output_dir, exist_ok=True)

    # Discover all tracer/task/model combos
    if not os.path.exists(MODELS_DIR):
        print(f'ERROR: {MODELS_DIR}/ directory not found.')
        return

    rows = []

    # models/{tracer}/{task}/  or  models/{task}/  (legacy)
    for entry1 in sorted(os.listdir(MODELS_DIR)):
        entry1_path = os.path.join(MODELS_DIR, entry1)
        if not os.path.isdir(entry1_path):
            continue

        # Determine if this is a tracer dir (new style) or task dir (old style)
        task_dirs = []
        tracer = None

        # Check if entry1 contains subdirectories → new style (tracer)
        sub_items = [d for d in os.listdir(entry1_path)
                     if os.path.isdir(os.path.join(entry1_path, d))]
        # Check if any sub-item looks like a task dir (has fold CSVs or model files)
        is_tracer_dir = any(
            any(f.endswith('.pth') or f.endswith('.json') or f.endswith('.csv')
                for f in os.listdir(os.path.join(entry1_path, sub)))
            for sub in sub_items
        ) if sub_items else False

        if is_tracer_dir:
            tracer = entry1
            if tracer_filter and tracer != tracer_filter:
                continue
            for task in sorted(sub_items):
                task_dir = os.path.join(entry1_path, task)
                if os.path.isdir(task_dir):
                    task_dirs.append((tracer, task, task_dir))
        else:
            # Old style: models/{task}/
            task = entry1
            task_dir = entry1_path
            task_dirs.append(('unknown', task, task_dir))

        # ---- Process each task dir ----
        for tracer, task, task_dir in task_dirs:
            print(f'Collecting: {tracer}/{task}')

            # ---- Per-fold metrics ----
            for fold in range(5):
                metrics_file = os.path.join(task_dir, f'metrics_fold{fold}.json')
                if os.path.exists(metrics_file):
                    with open(metrics_file) as f:
                        m = json.load(f)

                    # Save metrics curves
                    dest_label = f'{tracer}_{task}' if tracer != 'unknown' else task
                    curves_dir = os.path.join(output_dir, 'training_curves', dest_label)
                    os.makedirs(curves_dir, exist_ok=True)
                    shutil.copy(metrics_file,
                                os.path.join(curves_dir, f'fold{fold}_metrics.json'))

                    if m.get('val_XENT'):
                        best_xent = min(m['val_XENT'])
                        best_epoch = int(np.argmin(m['val_XENT'])) + 1
                        rows.append({
                            'tracer': tracer if tracer != 'unknown' else '-',
                            'task': task,
                            'fold': fold,
                            'best_val_XENT': round(best_xent, 4),
                            'best_epoch': best_epoch,
                            'train_time_min': m.get('training_time', 0),
                        })

            # ---- Per-sample predictions ----
            pred_file = os.path.join(task_dir, 'test_predictions.csv')
            if os.path.exists(pred_file):
                pred_dir = os.path.join(output_dir, 'per_sample_predictions')
                os.makedirs(pred_dir, exist_ok=True)
                dest_name = f'{tracer}_{task}_predictions.csv' if tracer != 'unknown' else f'{task}_predictions.csv'
                shutil.copy(pred_file, os.path.join(pred_dir, dest_name))

            # ---- Copy result text ----
            result_file = os.path.join(task_dir, 'test_results.txt')
            if os.path.exists(result_file):
                txt_dir = os.path.join(output_dir, 'result_texts')
                os.makedirs(txt_dir, exist_ok=True)
                dest_name = f'{tracer}_{task}_results.txt' if tracer != 'unknown' else f'{task}_results.txt'
                shutil.copy(result_file, os.path.join(txt_dir, dest_name))

    # ---- Also copy results/ directory if exists ----
    if os.path.exists(RESULTS_DIR):
        for tracer_dir in sorted(os.listdir(RESULTS_DIR)):
            tracer_path = os.path.join(RESULTS_DIR, tracer_dir)
            if not os.path.isdir(tracer_path):
                continue
            for res_file in sorted(os.listdir(tracer_path)):
                if res_file.startswith('results_') and res_file.endswith('.txt'):
                    txt_dir = os.path.join(output_dir, 'result_texts')
                    os.makedirs(txt_dir, exist_ok=True)
                    shutil.copy(os.path.join(tracer_path, res_file),
                                os.path.join(txt_dir, f'{tracer_dir}_{res_file}'))

    # ---- Build summary CSV ----
    if rows:
        df = pd.DataFrame(rows)

        # Per-task (and per-tracer) aggregation
        agg = df.groupby(['tracer', 'task']).agg(
            folds=('fold', 'count'),
            avg_val_XENT=('best_val_XENT', 'mean'),
            min_val_XENT=('best_val_XENT', 'min'),
            total_time_min=('train_time_min', 'sum'),
        ).reset_index()

        # Try to extract test accuracy from result files
        test_acc_rows = []
        for _, row in agg.iterrows():
            tracer_v = row['tracer']
            task_v = row['task']
            prefix = f'{tracer_v}_{task_v}' if tracer_v != '-' else task_v
            rf = os.path.join(output_dir, 'result_texts', f'{prefix}_results.txt')
            if os.path.exists(rf):
                with open(rf) as f:
                    for line in f:
                        if 'Accuracy:' in line:
                            v = float(line.split(':')[1].strip())
                            test_acc_rows.append({
                                'tracer': tracer_v, 'task': task_v,
                                'test_accuracy': round(v, 4),
                            })
                            break

        if test_acc_rows:
            acc_df = pd.DataFrame(test_acc_rows)
            agg = agg.merge(acc_df, on=['tracer', 'task'], how='left')

        # Save
        per_fold_path = os.path.join(output_dir, 'summary_per_fold.csv')
        summary_path = os.path.join(output_dir, 'summary.csv')

        df.to_csv(per_fold_path, index=False)
        agg.to_csv(summary_path, index=False)

        print(f'\n{"="*55}')
        print(f'Summary by tracer & task:')
        print(agg.to_string(index=False))
        print(f'{"="*55}')
        print(f'Saved: {summary_path}')
        print(f'Saved: {per_fold_path}')
    else:
        print('WARNING: No metrics files found.')

    # Zip
    zip_name = f'{output_dir}.zip'
    shutil.make_archive(output_dir, 'zip', output_dir)
    print(f'\nDownload: {zip_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results_summary')
    parser.add_argument('--tracer', default=None, help='Only collect results for a specific tracer')
    args = parser.parse_args()
    collect_results(args.output, args.tracer)
