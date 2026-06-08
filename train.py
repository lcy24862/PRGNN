import torch
import torch.nn as nn
from data_utils import PETDataset, get_loader, add_data_args, print_dimension_info, TASK_CLASSES
from datetime import datetime
import numpy as np

import argparse
import os, shutil
from utils import CosineAnnealingWarmUpRestarts, AverageMeter, train_or_val
import json
from setproctitle import setproctitle
import time
from model import get_model

def get_args_parser():
    parser = argparse.ArgumentParser('PET Classification', add_help=False)

    ################ Critical arguments ################
    parser.add_argument('--roi_mask', default='template/AAL_reduced_mask.nii', type=str)

    ####################################################

    parser.add_argument('--batch_size', default=8, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model', default='PRGNN_ti', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--fold', default=0, type=int,
                        help='Fold index (0-based, 0-4)')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--task', default='AD_HC', type=str,
                        help='Classification task: AD_HC, HC_MCI, EMCI_LMCI, HC_ALL_MCI, or all')
    parser.add_argument('--dataparallel', action='store_true')

    # Data arguments (see data_utils.add_data_args)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--tracer', default='18F-FDG', type=str,
                        choices=['18F-FDG', '18F-FBB', '18F-AV45', '18F-AV1451'])
    parser.add_argument('--target_size', default=None, type=int, nargs=3,
                        help='Resize volumes, e.g. --target_size 96 96 96')
    parser.add_argument('--num_folds', default=5, type=int)

    # Hyperparameters for model tuning
    parser.add_argument('--drop_path_rate', default=0, type=float)
    parser.add_argument('--n_filters', default=64, type=int)
    parser.add_argument('--n_blocks', default=3, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lambda_XENT', default=1, type=float, help='Xent loss')
    parser.add_argument('--pool', default='avgpool', type=str, choices=['avgpool', 'maxpool', 'attention'])
    parser.add_argument('--act', default='gelu', type=str, choices=['gelu', 'relu', 'leakyrelu'])
    parser.add_argument('--k', default=9, type=int)
    parser.add_argument('--model_type', default='tiny', type=str)
    parser.add_argument('--stage', default='stage3', type=str)
    parser.add_argument('--relative_pos', action='store_true')
    parser.add_argument('--use_backbone_only', action='store_true')


    return parser

def main(args):

    setproctitle('[DS] MICCAI')

    # Determine num_classes from task
    task_classes = TASK_CLASSES.get(args.task, TASK_CLASSES['AD_HC'])
    args.num_classes = len(task_classes)
    print(f'Task: {args.task}, Classes: {task_classes} ({args.num_classes}-class)')

    # Print dimension compatibility info
    print_dimension_info(args)
    if args.dataparallel:
        device = 'cuda'
    else:
        device = f'cuda:{args.gpu}'

    model_dir = f'models/{args.task}'
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = f'{model_dir}/{args.model}_fold{args.fold}_checkpoint.pt'

    num_classes = args.num_classes

    # ---- Backup model source files (once) ----
    print('Copying model files...')
    shutil.copy('vig.py', f'{model_dir}/vig.py')
    shutil.copy('model.py', f'{model_dir}/model.py')
    shutil.copy('train.py', f'{model_dir}/train.py')
    def ignore_pycache(dirname, filenames):
        return [name for name in filenames if name == '__pycache__']
    shutil.copytree('gcn_lib', f'{model_dir}/gcn_lib', ignore=ignore_pycache, dirs_exist_ok=True)

    # ---- Data loaders ----
    train_loader, val_loader, test_loader = get_loader(
        args, fold=args.fold, num_classes=num_classes,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # ---- Model / optimizer / scheduler ----
    model = get_model(args)
    if args.dataparallel:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion_CN = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=2, eta_max=args.lr, T_up=10, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda')

    # ---- State ----
    start_epoch = 0
    best_metric = 999.0
    best_metric_epoch = 0
    metrics = {'train_loss': [], 'train_acc': [], 'train_XENT': [],
               'val_loss': [], 'val_acc': [], 'val_XENT': [],
               'train_time': 0}
    total_train_time = 0.0

    # ---- Resume from checkpoint if available ----
    if os.path.exists(ckpt_path):
        print(f'[Resume] Found checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        scaler.load_state_dict(ckpt['scaler_state'])

        start_epoch = ckpt['epoch'] + 1   # resume from next epoch
        best_metric = ckpt.get('best_metric', 999.0)
        best_metric_epoch = ckpt.get('best_metric_epoch', 0)
        metrics = ckpt.get('metrics', metrics)
        total_train_time = ckpt.get('train_time', 0.0)

        print(f'[Resume] Epoch {start_epoch}/{args.epochs}  '
              f'best_val_loss={best_metric:.4f} (epoch {best_metric_epoch})')

        # If already finished, skip training
        if start_epoch >= args.epochs:
            print(f'[Resume] Training already completed ({args.epochs} epochs). '
                  f'Jumping to final test.')
            # fall through to test section below
    else:
        print(f'Training fold {args.fold}, model {args.model} (from scratch)')

    # ---- Training loop ----
    wall_start = time.time()
    for epoch in range(start_epoch, args.epochs):

        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.epochs}")

        model.train()
        epoch_loss = train_or_val(
            loader=train_loader, device=device, epoch=epoch, args=args,
            optimizer=optimizer, model=model, criterion_CN=criterion_CN,
            scaler=scaler, scheduler=scheduler, isTrain=True,
        )

        loss_statement = ' '.join([f'{key}: {epoch_loss[key]:.4f}' for key in epoch_loss.keys()])
        print(f"[Train] {loss_statement}")
        for key in epoch_loss.keys():
            metrics[f'train_{key}'].append(epoch_loss[key])

        model.eval()
        epoch_loss = train_or_val(
            loader=val_loader, device=device, epoch=epoch, args=args,
            optimizer=optimizer, model=model, criterion_CN=criterion_CN,
            scaler=scaler, scheduler=scheduler, isTrain=False,
        )

        loss_statement = ' '.join([f'{key}: {epoch_loss[key]:.4f}' for key in epoch_loss.keys()])
        print(f"[Val] {loss_statement}")
        for key in epoch_loss.keys():
            metrics[f'val_{key}'].append(epoch_loss[key])

        if epoch_loss['XENT'] < best_metric:
            best_metric = epoch_loss['XENT']
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), f'{model_dir}/{args.model}_fold{args.fold}_best.pth')
            print("saved new best model (lowest val loss)")

        print("epoch {} | acc: {:.4f} | val_loss: {:.4f} | best_val_loss: {:.4f} (epoch {})".format(
            epoch + 1, epoch_loss['acc'], epoch_loss['XENT'], best_metric, best_metric_epoch,
        ))

        torch.save(model.state_dict(), f'{model_dir}/{args.model}_fold{args.fold}_last.pth')

        # ---- Save full checkpoint for resume ----
        elapsed = round((time.time() - wall_start) / 60, 3)
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'best_metric': best_metric,
            'best_metric_epoch': best_metric_epoch,
            'metrics': metrics,
            'train_time': total_train_time + elapsed,
        }, ckpt_path)

    # ---- Save final metrics ----
    total_train_time = round((time.time() - wall_start) / 60, 3) + total_train_time
    metrics['training_time'] = total_train_time
    with open(f'{model_dir}/metrics_fold{args.fold}.json', 'w') as f:
        json.dump(metrics, f)

    # ---- Final test on best model ----
    model = get_model(args)
    if args.dataparallel:
        model = nn.DataParallel(model)
    model = model.to(device)

    best_path = f'{model_dir}/{args.model}_fold{args.fold}_best.pth'
    model_state = torch.load(best_path, map_location=device)
    model.load_state_dict(model_state)
    model.eval()

    test_loss = train_or_val(
        loader=test_loader, device=device, epoch=args.epochs, args=args,
        optimizer=optimizer, model=model, criterion_CN=criterion_CN,
        scaler=scaler, scheduler=scheduler, isTrain=False,
    )

    with open(f'{model_dir}/result.txt', 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write(f'Best val XENT for fold {args.fold}: {best_metric:.4f} at epoch {best_metric_epoch}\n')
        f.write(f'Test accuracy for fold {args.fold}: {test_loss["acc"]:.4f}\n')
        f.write(f'Total training time: {total_train_time} min\n\n')

    print(f'Fold {args.fold} done. Test acc: {test_loss["acc"]:.4f}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Custom U-Net', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    main(args)
