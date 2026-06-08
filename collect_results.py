"""
Collect all training results into a single summary for local analysis.

Usage (on server):
    python collect_results.py                    # creates results_summary/
    python collect_results.py --output my_results

Output:
    results_summary/
    ├── summary.csv              # all tasks/folds metrics in one table
    ├── per_sample_predictions/   # per-task prediction CSVs
    ├── training_curves/          # per-fold metrics JSON
    └── collect_results.zip       # ready to download
"""

import os, sys, json, shutil, argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

MODELS_DIR = 'models'
RESULTS_DIR = 'results'


def collect_results(output_dir='results_summary'):
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    tasks = sorted(d for d in os.listdir(MODELS_DIR)
                   if os.path.isdir(os.path.join(MODELS_DIR, d)))

    for task in tasks:
        task_dir = os.path.join(MODELS_DIR, task)
        print(f'Collecting: {task}')

        # ---- Per-fold metrics ----
        best_xents = []
        test_accs = []

        for fold in range(5):
            metrics_file = os.path.join(task_dir, f'metrics_fold{fold}.json')
            if os.path.exists(metrics_file):
                with open(metrics_file) as f:
                    m = json.load(f)
                # Save metrics curves
                curves_dir = os.path.join(output_dir, 'training_curves', task)
                os.makedirs(curves_dir, exist_ok=True)
                shutil.copy(metrics_file, os.path.join(curves_dir, f'fold{fold}_metrics.json'))

                if m.get('val_XENT'):
                    best_xent = min(m['val_XENT'])
                    best_epoch = np.argmin(m['val_XENT']) + 1
                    best_xents.append(best_xent)
                    rows.append({
                        'task': task,
                        'fold': fold,
                        'best_val_XENT': round(best_xent, 4),
                        'best_epoch': best_epoch,
                        'train_time_min': m.get('training_time', 0),
                    })

        # ---- Test results ----
        result_file = os.path.join(task_dir, 'test_results.txt')
        if os.path.exists(result_file):
            with open(result_file) as f:
                content = f.read()
            # Parse key metrics
            for line in content.split('\n'):
                if line.startswith('Accuracy:'):
                    test_accs.append(float(line.split(':')[1].strip()))
                elif line.startswith('F1'):
                    pass  # captured per-task below

        # ---- Per-sample predictions ----
        pred_file = os.path.join(task_dir, 'test_predictions.csv')
        if os.path.exists(pred_file):
            pred_dir = os.path.join(output_dir, 'per_sample_predictions')
            os.makedirs(pred_dir, exist_ok=True)
            shutil.copy(pred_file, os.path.join(pred_dir, f'{task}_predictions.csv'))

        # ---- Copy full result text ----
        if os.path.exists(result_file):
            txt_dir = os.path.join(output_dir, 'result_texts')
            os.makedirs(txt_dir, exist_ok=True)
            shutil.copy(result_file, os.path.join(txt_dir, f'{task}_results.txt'))

        # ---- Copy from results/ if exists ----
        for tracer_dir in os.listdir(RESULTS_DIR) if os.path.exists(RESULTS_DIR) else []:
            tracer_path = os.path.join(RESULTS_DIR, tracer_dir)
            if not os.path.isdir(tracer_path):
                continue
            res_file = os.path.join(tracer_path, f'results_{task}.txt')
            if os.path.exists(res_file):
                txt_dir = os.path.join(output_dir, 'result_texts')
                os.makedirs(txt_dir, exist_ok=True)
                shutil.copy(res_file, os.path.join(txt_dir, f'{task}_results.txt'))

    # ---- Build summary CSV ----
    if rows:
        df = pd.DataFrame(rows)

        # Per-task aggregation
        agg = df.groupby('task').agg(
            folds=('fold', 'count'),
            avg_val_XENT=('best_val_XENT', 'mean'),
            min_val_XENT=('best_val_XENT', 'min'),
            total_time_min=('train_time_min', 'sum'),
        ).reset_index()

        # Add test accuracy from result files
        test_acc_rows = []
        for task in tasks:
            rf = os.path.join(MODELS_DIR, task, 'test_results.txt')
            if os.path.exists(rf):
                with open(rf) as f:
                    for line in f:
                        if 'Accuracy:' in line:
                            v = float(line.split(':')[1].strip())
                            test_acc_rows.append({'task': task, 'test_accuracy': round(v, 4)})
                            break

        if test_acc_rows:
            acc_df = pd.DataFrame(test_acc_rows)
            agg = agg.merge(acc_df, on='task', how='left')

        # Save
        per_fold_path = os.path.join(output_dir, 'summary_per_fold.csv')
        summary_path = os.path.join(output_dir, 'summary.csv')

        df.to_csv(per_fold_path, index=False)
        agg.to_csv(summary_path, index=False)

        print(f'\n{"="*50}')
        print(f'Summary by task:')
        print(agg.to_string(index=False))
        print(f'{"="*50}')
        print(f'Saved: {summary_path}')
        print(f'Saved: {per_fold_path}')

    # Zip
    zip_name = f'{output_dir}.zip'
    shutil.make_archive(output_dir, 'zip', output_dir)
    print(f'\nDownload: {zip_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results_summary')
    args = parser.parse_args()
    collect_results(args.output)
