import torch
import torch.nn as nn
from data_utils import FDG, get_loader
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
    parser = argparse.ArgumentParser('FDG Classification', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model', default='RGNN', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--task', default='0217_RGNN_baseline', type=str)
    parser.add_argument('--dataparallel', action='store_true')
    parser.add_argument('--dataset', default='FDG', type=str, choices=['FDG', 'FBB'])
    parser.add_argument('--use_testloader', action='store_true')
    
    # Hyperparameters for model tuning
    parser.add_argument('--drop_path_rate', default=0, type=float)
    parser.add_argument('--n_filters', default=64, type=int)
    parser.add_argument('--n_blocks', default=3, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lambda_XENT', default=1, type=float, help='Xent loss')
    parser.add_argument('--lambda_AC', default=0, type=float, help='Attention-consistency loss on final node embedding')
    parser.add_argument('--lambda_ES', default=0, type=float, help='Entropy-based sparsity loss on fc layer')
    parser.add_argument('--pool', default='avgpool', type=str, choices=['avgpool', 'maxpool', 'attention'])
    parser.add_argument('--act', default='gelu', type=str, choices=['gelu', 'relu', 'leakyrelu'])
    parser.add_argument('--k', default=9, type=int)
    parser.add_argument('--model_type', default='tiny', type=str)
    parser.add_argument('--stage', default='stage3', type=str)
    parser.add_argument('--relative_pos', action='store_true')
    parser.add_argument('--use_backbone_only', action='store_true')
    parser.add_argument('--which_backbone_stage', default=3, type=int)
    

    return parser

def main(args):
    
    setproctitle('[DS] MICCAI')
    if args.dataparallel:
        device = 'cuda'
    else:
        device = f'cuda:{args.gpu}'
        
    if args.dataset == 'FBB':
        args.num_classes = 2
    else:
        args.num_classes = 4
    start = time.time()

    num_classes = args.num_classes 
    print(f'Training with fold {args.fold}, model {args.model}')

    print(f'Copying model files...')
    if not os.path.exists(f'models/{args.task}/'):
        os.makedirs(f'models/{args.task}/')
    shutil.copy('vig.py', f'models/{args.task}/vig.py')
    shutil.copy('rgnn.py', f'models/{args.task}/rgnn.py')
    shutil.copy('model.py', f'models/{args.task}/model.py')
    shutil.copy('train.py', f'models/{args.task}/train.py')
    def ignore_pycache(dirname, filenames):
        return [name for name in filenames if name == '__pycache__']
    shutil.copytree('gcn_lib', f'models/{args.task}/gcn_lib', ignore=ignore_pycache, dirs_exist_ok=True)
    
    train_loader, val_loader, test_loader = get_loader(args, fold=args.fold, num_classes=num_classes,
                                                       batch_size=args.batch_size, 
                                                       num_workers=args.num_workers)
        
    if args.use_testloader:
        val_loader = test_loader
    model = get_model(args)
    if args.dataparallel:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion_CN = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=2, eta_max=args.lr, T_up=10, gamma=0.5)

    xent_losses = AverageMeter()
    xent_loss = torch.tensor(0.0)
    ac_loss = torch.tensor(0.0)
    es_loss = torch.tensor(0.0)
    metrics = {'train_loss': [],'train_acc': [],'train_XENT': [],'train_AC': [],'train_ES': [],
               'val_loss': [],'val_acc': [],'val_XENT': [],'val_AC': [],'val_ES': [],
               'train_time': 0 }
    
    start = time.time()
    best_metric, best_metric_epoch = (999, 0)
        
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.epochs):
        
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.epochs}")
        
        model.train()
        epoch_loss = train_or_val(loader=train_loader, device=device,
                                                  epoch=epoch, args=args, optimizer=optimizer,
                                                  model=model, criterion_CN=criterion_CN,
                                                  scaler=scaler, scheduler=scheduler,
                                                  isTrain=True)
        
        loss_statement = ' '.join([f'{key}: {epoch_loss[key]:.4f}' for key in epoch_loss.keys()])
        print(f"[Train] {loss_statement}")
        for key in epoch_loss.keys():
            metrics[f'train_{key}'].append(epoch_loss[key])

        model.eval()
        epoch_loss = train_or_val(loader=val_loader, device=device,
                                                  epoch=epoch, args=args, optimizer=optimizer,
                                                  model=model, criterion_CN=criterion_CN,
                                                  scaler=scaler, scheduler=scheduler,
                                                  isTrain=False)
        
        loss_statement = ' '.join([f'{key}: {epoch_loss[key]:.4f}' for key in epoch_loss.keys()])
        print(f"[Val] {loss_statement}")
        for key in epoch_loss.keys():
            metrics[f'val_{key}'].append(epoch_loss[key])

        if epoch_loss['XENT'] < best_metric:
            best_metric = epoch_loss['XENT']
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), f"models/{args.task}/{args.model}_fold{args.fold}.pth")
            print("saved new best metric model")
        print(
            "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                epoch + 1, epoch_loss['acc'], best_metric, best_metric_epoch
            )
        )
        torch.save(model.state_dict(), f"models/{args.task}/{args.model}_fold{args.fold}_last.pth")

    end = time.time()
    metrics['training_time'] = round((end-start)/60, 3)
    with open(f'models/{args.task}/metrics_fold{args.fold}.json', 'w') as f:
        json.dump(metrics, f)

    # FINAL TEST
    model = get_model(args)
    if args.dataparallel:
        model = nn.DataParallel(model)
    model = model.to(device)
    model_path = f"models/{args.task}/{args.model}_fold{args.fold}.pth"
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    model.eval()

    epoch_loss = train_or_val(loader=test_loader, device=device,
                                            epoch=epoch, args=args, optimizer=optimizer,
                                            model=model, criterion_CN=criterion_CN,
                                            scaler=scaler, scheduler=scheduler,
                                            isTrain=False)
        
    with open(f'models/{args.task}/result.txt', 'a') as f:

        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        f.write('\n')
        f.write(f'Best accuracy for fold {args.fold}: {best_metric:.4f} at epoch {best_metric_epoch}')
        f.write('\n')
        f.write(f'Test accuracy for fold {args.fold}: {epoch_loss["acc"]:.4f}')
        f.write('\n')
        f.write(f'Total training time: {round((end-start)/60, 3)}')
        f.write('\n\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Custom U-Net', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    main(args)
