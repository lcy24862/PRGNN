# %load_ext autoreload
# %autoreload 2
from monai.networks.nets import resnet18
import torch.nn as nn
import torch
from data_utils import get_loader
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('FDG Classification', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--model', default='PRGNN_m', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--task', default='0217_RGNN_baseline', type=str)
    parser.add_argument('--dataparallel', action='store_true')
    parser.add_argument('--dataset', default='FDG', type=str, choices=['FDG', 'FBB'])
    
    # Hyperparameters for model tuning
    parser.add_argument('--drop_path_rate', default=0, type=float)
    parser.add_argument('--n_filters', default=64, type=int)
    parser.add_argument('--n_blocks', default=3, type=int)
    parser.add_argument('--pool', default='avgpool', type=str, choices=['avgpool', 'maxpool', 'attention'])
    parser.add_argument('--act', default='gelu', type=str, choices=['gelu', 'relu', 'leakyrelu'])
    parser.add_argument('--k', default=9, type=int)
    parser.add_argument('--model_type', default='tiny', type=str)
    parser.add_argument('--relative_pos', action='store_true')

    return parser

class FeatureAndPredModel(nn.Module):
    def __init__(self, model):
        """
        Wraps the given ResNet model to output both the features (after avgpool)
        and the final predictions.

        Args:
            model (nn.Module): The trained ResNet model.
        """
        super().__init__()
        # Extract all modules except the final fully connected layer.
        # This includes conv1, bn1, act, maxpool, layer1-layer4, and avgpool.
        self.features_extractor = nn.Sequential(*list(model.children())[:-1])
        # The final fully connected layer remains unchanged.
        self.fc = model.fc

    def forward(self, x):
        # Obtain features from the model (after avgpool)
        features = self.features_extractor(x)
        # Flatten the features (the original model does x.view(x.size(0), -1) here)
        features_flat = features.view(features.size(0), -1)
        # Compute final predictions using the fully connected layer
        preds = self.fc(features_flat)
        return features_flat, preds
    
parser = argparse.ArgumentParser('Custom U-Net', parents=[get_args_parser()])
args = parser.parse_args([])
          
node_embeddings_list = []
preds_list = []
labels_list = []
folds_list = []
device = 'cuda:1'

for fold in range(1, 6):
        
    model = resnet18(n_input_channels=1, num_classes=4)
    
    model = model.to(device)
    pth = torch.load(f'models/0220_ResNet18/resnet18_fold{fold}.pth')
    model.load_state_dict(pth)
    model.eval()
    
    new_model = FeatureAndPredModel(model)

    train_loader, val_loader, test_loader = get_loader(args, fold=fold, num_classes=4, batch_size=8)

    with torch.no_grad():
        for batched in test_loader:

            data, labels = batched[0].to(device), batched[1].to(device)
            features, preds = new_model(data)  # [batch, n_classes]

            labels_list.append(labels.detach().cpu())
            preds_list.append(preds.detach().cpu().argmax(dim=1))
            node_embeddings_list.append(features)
            folds_list.extend([fold] * features.shape[0])
            
    del model

all_node_embeddings = torch.cat(node_embeddings_list, dim=0)
all_labels = torch.cat(labels_list, dim=0)
all_preds = torch.cat(preds_list, dim=0)

df_embeddings = pd.DataFrame(all_node_embeddings.squeeze().detach().cpu())
df_embeddings['labels'] = all_labels
df_embeddings['preds'] = all_preds
df_embeddings['folds'] = folds_list
df_embeddings.to_csv('embeddings_resnet.csv')