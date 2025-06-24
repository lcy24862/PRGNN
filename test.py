import torch
from data_utils import FDG, get_loader

import argparse
import json
from setproctitle import setproctitle
import time
from model import get_model
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def get_args_parser():
    
    parser = argparse.ArgumentParser('FDG Classification', add_help=False)
    parser.add_argument('--which_model', default='best', type=str, choices=['best', 'last'])
    
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--model', default='PViG_ti', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--stage', default='stage3', type=str)
    
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--task', default='PViG_ti', type=str)
    parser.add_argument('--dataparallel', default=False, type=bool)
    parser.add_argument('--dataset', default='FDG', type=str, choices=['FDG', 'FBB'])
    
    # Hyperparameters
    parser.add_argument('--drop_path_rate', default=0, type=float)
    parser.add_argument('--n_filters', default=128, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lambda_XENT', default=1, type=float, help='Xent loss')
    parser.add_argument('--lambda_AC', default=0, type=float, help='Attention-consistency loss on final node embedding')
    parser.add_argument('--lambda_ES', default=0, type=float, help='Entropy-based sparsity loss on fc layer')
    parser.add_argument('--pool', default='avgpool', type=str, choices=['avgpool', 'maxpool', 'attention'])
    parser.add_argument('--act', default='gelu', type=str, choices=['gelu', 'relu', 'leakyrelu'])
    parser.add_argument('--k', default=9, type=int)
    parser.add_argument('--relative_pos', action='store_true')
    return parser

    
def main(args):
    
    setproctitle('MICCAI Letsgogo')
    device = f'cuda'
    start = time.time()

    if args.dataset == 'FBB':
        args.num_classes = 2
    else:
        args.num_classes = 4
    num_classes = args.num_classes # or 4    
    if args.dataset == 'FBB':
        pred_to_label = {0: 0, 1: 1}
    else:
        pred_to_label = {0: 'NC', 1: 'AD', 2: 'LBD', 3: 'PSP'}

    # Global lists to accumulate results across folds
    all_true_list = []
    all_preds_list = []
    all_probs_list = []
    all_folds_list = []
    all_test_dfs = []  # To store each fold's DataFrame with predictions

    for fold in range(1, 6):
        
        model = get_model(args)
        if args.dataparallel:
            model = torch.nn.DataParallel(model)
        model = model.to(device)
        
        if args.which_model == 'best':
            model_path = f"models/{args.task}/{args.model}_fold{fold}.pth"
        else:
            model_path = f"models/{args.task}/{args.model}_fold{fold}_last.pth"
        model_state = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state)
        model.eval()
        
        # Lists for current fold
        fold_true_list = []
        fold_preds_list = []
        fold_probs_list = []
        
        test_df = pd.read_csv(f'/media/storage2/Daesung/MICCAI/folds_{args.dataset}/fold_{fold}_test.csv')
        test_ds = FDG(args, test_df, train=False, num_classes=num_classes)
        
        for idx in range(len(test_ds)):
            with torch.amp.autocast('cuda'):
                data, label = test_ds[idx]
                # Ensure label is a scalar integer
                true_label = label.item() if torch.is_tensor(label) else label
                
                # if 'PRGNN' in args.model:
                #     output, final_embedding = model(data.unsqueeze(0).to(device).float())
                # else:
                output = model(data.unsqueeze(0).to(device).float())
            
            # Compute probabilities using softmax (assuming output are logits)
            probabilities = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
            predicted_label = int(output.argmax(dim=1).item())
            
            fold_preds_list.append(predicted_label)
            fold_true_list.append(true_label)
            fold_probs_list.append(probabilities)
            all_folds_list.append(fold)
        
        # Append current fold's data to global lists
        all_true_list.extend(fold_true_list)
        all_preds_list.extend(fold_preds_list)
        all_probs_list.extend(fold_probs_list)
        
        # Add predictions to the DataFrame for this fold
        test_df['preds'] = [pred_to_label[pred] for pred in fold_preds_list]
        test_df['is_correct'] = test_df['preds'] == test_df['label']
        test_df['fold'] = fold
        print(f'Fold {fold} accuracy: {sum(test_df["is_correct"])/len(fold_true_list):.4f}')
        all_test_dfs.append(test_df)

    # Combine DataFrames from all folds into a single DataFrame
    final_df = pd.concat(all_test_dfs)
    final_df.to_excel(f'/media/storage2/Daesung/MICCAI/models/{args.task}/test_preds.xlsx', index=False)

    # Calculate overall metrics across all folds
    accuracy = accuracy_score(all_true_list, all_preds_list)
    f1 = f1_score(all_true_list, all_preds_list, average='weighted')

    # Calculate AUC:
    # For binary classification, use the probability of the positive class.
    # For multiclass classification, pass the full probability matrix with multi_class='ovr'.
    if num_classes == 2:
        probs_positive = [p[1] for p in all_probs_list]
        auc = roc_auc_score(all_true_list, probs_positive)
        ################ calculate sensitivity and specificity here
        tn, fp, fn, tp = confusion_matrix(all_true_list, all_preds_list).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # True Positive Rate
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  # True Negative Rate
        ################
    else:
        all_probs_array = np.array(all_probs_list, dtype=np.float64) # shape: (n_samples, n_classes)
        print(all_probs_array)
        sums = all_probs_array.sum(axis=1, keepdims=True)
        all_probs_array = all_probs_array / sums
        all_probs_array = np.array([adjust_probabilities(p.copy()) for p in all_probs_array])
        # print(all_probs_array.sum(axis=1, keepdims=True))
        auc = roc_auc_score(all_true_list, all_probs_array, multi_class='ovr')

    print(f'Overall Accuracy: {accuracy:.4f}')
    print(f'Overall F1 Score: {f1:.4f}')
    print(f'Overall AUC: {auc:.4f}')

    with open(f'models/{args.task}/result.txt', 'a') as f:

        f.write('\n')
        f.write(f'Overall test accuracy ({args.which_model}): {accuracy:.4f} \n')
        f.write(f'Overall test F1 score: {f1:.4f} \n')
        f.write(f'Overall test AUC: {auc:.4f} \n')
        if num_classes == 2:
            f.write(f'Overall test sensitivity: {sensitivity:.4f} \n')
            f.write(f'Overall test specificity: {specificity:.4f} \n')  
        f.write('\n')
        
        
def adjust_probabilities(probs, tol=1e-6):
    """
    Adjust a probability vector so that it sums exactly to 1.
    
    If the sum deviates from 1 by more than tol, re-normalize.
    Otherwise, adjust the last element by the small difference.
    """
    total = np.sum(probs)
    diff = 1.0 - total

    # Small discrepancy: adjust the last element
    if total != 1:
        # print('????????????? why', total, diff)
        probs[np.argmax(probs)] += diff

    return probs
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Custom U-Net', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    main(args)
