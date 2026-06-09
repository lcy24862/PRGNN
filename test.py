import torch
from data_utils import PETDataset, get_loader, TASK_CLASSES
from model import get_model

import argparse
import os
from setproctitle import setproctitle
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def get_args_parser():
    parser = argparse.ArgumentParser('PRGNN Test', add_help=False)

    parser.add_argument('--which_model', default='best', type=str, choices=['best', 'last'])
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--model', default='PRGNN_ti', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--dataparallel', action='store_true')

    # Data
    parser.add_argument('--data_dir', default='data_registered', type=str)
    parser.add_argument('--tracer', default='18F-FDG', type=str,
                        choices=['18F-FDG', '18F-FBB', '18F-AV45', '18F-AV1451', 'Amyloid'])
    parser.add_argument('--task', default='AD_HC', type=str,
                        help='Classification task: AD_HC, HC_MCI, EMCI_LMCI, HC_ALL_MCI, all')
    parser.add_argument('--num_folds', default=5, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    # PRGNN hyperparams (must match training)
    parser.add_argument('--roi_mask', default='template/AAL_reduced_mask.nii', type=str)
    parser.add_argument('--drop_path_rate', default=0, type=float)
    parser.add_argument('--k', default=9, type=int)
    parser.add_argument('--act', default='gelu', type=str)
    parser.add_argument('--pool', default='avgpool', type=str)
    parser.add_argument('--relative_pos', action='store_true')

    return parser


def main(args):
    setproctitle('PRGNN Test')
    device = f'cuda:{args.gpu}' if not args.dataparallel else 'cuda'

    # Determine num_classes from task
    task_classes = TASK_CLASSES.get(args.task, TASK_CLASSES['AD_HC'])
    args.num_classes = len(task_classes)
    pred_to_label = {i: name for i, name in enumerate(task_classes)}

    print(f'Testing: model={args.model} tracer={args.tracer} task={args.task} '
          f'classes={args.num_classes} data={args.data_dir}')

    # Accumulators across all folds
    all_true_list, all_preds_list, all_probs_list, all_test_dfs = [], [], [], []

    for fold in range(args.num_folds):
        print(f'\n--- Fold {fold} ---')

        # Load model
        model = get_model(args)
        if args.dataparallel:
            model = torch.nn.DataParallel(model)
        model = model.to(device)

        suffix = 'best' if args.which_model == 'best' else 'last'
        model_path = f"models/{args.tracer}/{args.task}/{args.model}_fold{fold}_{suffix}.pth"

        if not os.path.exists(model_path):
            # Try old naming convention (no tracer subdir)
            model_path = f"models/{args.task}/{args.model}_fold{fold}_{suffix}.pth"
        if not os.path.exists(model_path):
            model_path = f"models/{args.task}/{args.model}_fold{fold}.pth"
        if not os.path.exists(model_path):
            print(f'  SKIP: model not found at {model_path}')
            continue

        model_state = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state)
        model.eval()

        # Load test data
        _, _, test_loader = get_loader(
            args, fold=fold, num_classes=args.num_classes,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )

        # Read test CSV for saving predictions
        tracer_dir = os.path.join(args.data_dir, args.tracer)
        if args.task == 'all':
            # Reconstruct test split
            from sklearn.model_selection import StratifiedKFold, train_test_split
            all_df = pd.read_csv(os.path.join(tracer_dir, 'all.csv'))
            all_df = all_df[all_df['DX'].isin(task_classes)].reset_index(drop=True)
            skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)
            fold_indices = list(skf.split(np.zeros(len(all_df)), all_df['DX']))
            trainval_idx, test_idx = fold_indices[fold]
            train_idx, val_idx = train_test_split(
                trainval_idx, test_size=0.15, random_state=42 + fold,
                stratify=all_df.iloc[trainval_idx]['DX'],
            )
            test_df = all_df.iloc[test_idx].reset_index(drop=True)
        else:
            test_csv = os.path.join(tracer_dir, args.task, f'test_fold{fold}.csv')
            test_df = pd.read_csv(test_csv)

        fold_true_list, fold_preds_list, fold_probs_list = [], [], []

        with torch.no_grad():
            for batched in test_loader:
                data, labels = batched[0].to(device), batched[1].to(device)
                with torch.amp.autocast('cuda'):
                    output = model(data)

                probs = torch.softmax(output, dim=1).detach().cpu().numpy()
                preds = output.argmax(dim=1).detach().cpu().numpy()
                trues = labels.detach().cpu().numpy()

                fold_preds_list.extend(preds.tolist())
                fold_true_list.extend(trues.tolist())
                fold_probs_list.extend(probs.tolist())

        fold_acc = accuracy_score(fold_true_list, fold_preds_list)
        print(f'  Fold {fold} Accuracy: {fold_acc:.4f}')

        all_true_list.extend(fold_true_list)
        all_preds_list.extend(fold_preds_list)
        all_probs_list.extend(fold_probs_list)

        # Save per-fold predictions
        test_df = test_df.copy()
        test_df['pred'] = [pred_to_label.get(p, p) for p in fold_preds_list]
        test_df['correct'] = test_df['pred'] == test_df['DX']
        test_df['fold'] = fold
        all_test_dfs.append(test_df)

    if not all_true_list:
        print('ERROR: No folds evaluated. Check model paths.')
        return

    # Overall metrics
    accuracy = accuracy_score(all_true_list, all_preds_list)
    f1 = f1_score(all_true_list, all_preds_list, average='weighted')

    if args.num_classes == 2:
        probs_pos = [p[1] for p in all_probs_list]
        auc = roc_auc_score(all_true_list, probs_pos)
        tn, fp, fn, tp = confusion_matrix(all_true_list, all_preds_list).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0
    else:
        all_probs_array = np.array(all_probs_list)
        all_probs_array = all_probs_array / all_probs_array.sum(axis=1, keepdims=True)
        auc = roc_auc_score(all_true_list, all_probs_array, multi_class='ovr')
        sensitivity = specificity = None

    print(f'\n{"="*50}')
    print(f'Overall Results: {args.model} / {args.tracer} / {args.task}')
    print(f'  Accuracy:    {accuracy:.4f}')
    print(f'  F1 (weighted): {f1:.4f}')
    print(f'  AUC:         {auc:.4f}')
    if sensitivity is not None:
        print(f'  Sensitivity: {sensitivity:.4f}')
        print(f'  Specificity: {specificity:.4f}')
    print(f'{"="*50}')

    # Save
    save_dir = f'models/{args.tracer}/{args.task}'
    os.makedirs(save_dir, exist_ok=True)
    final_df = pd.concat(all_test_dfs, ignore_index=True)
    final_df.to_csv(f'{save_dir}/test_predictions.csv', index=False)

    with open(f'{save_dir}/test_results.txt', 'w') as f:
        f.write(f'Model: {args.model}\n')
        f.write(f'Tracer: {args.tracer}\n')
        f.write(f'Task: {args.task}\n')
        f.write(f'Classes: {task_classes}\n')
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'F1 (weighted): {f1:.4f}\n')
        f.write(f'AUC: {auc:.4f}\n')
        if sensitivity is not None:
            f.write(f'Sensitivity: {sensitivity:.4f}\n')
            f.write(f'Specificity: {specificity:.4f}\n')

    print(f'Results saved to {save_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PRGNN Test', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    main(args)
