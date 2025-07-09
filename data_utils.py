from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from monai.transforms import (
    Compose, OneOf,
    CropForeground,
    CenterSpatialCrop,
    ToTensor,
    RandRotate, RandFlip, RandSpatialCrop, RandZoom, RandAffine
)
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn as nn

class FDG(Dataset):
    
    """Loads FDG of NC, AD, LBD, FTLD
    Resample dataset using torchio with voxel spacing 2, 2, 1.5 because input shape = 969696
    """

    def __init__(self, args, df, train, num_classes=4):
        """
        Args:
            df (string): patientID 와 label 이 들어있는 파일의 csv 경로
            train: 훈련용 데이터셋인지
            num_classes: 분류할 클래스 수
                4: all
        """
        
        self.df = df
        self.labeldict = {'NC': 0, 'AD':1, 'LBD':2, 'PSP':3}
        self.base = '/media/nvme1/Daesung_FDG/FDG/3_spatial_aal/'
        self.train = train
        
        self.transform = Compose(
        [
            RandFlip(prob=0.2, spatial_axis=0),
            ToTensor()
        ])

        self.valtransform = Compose(
        [
            ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def load_data(self, path):

        data = nib.load(path).get_fdata()
        data = data[np.newaxis,:]
        data = np.clip(data, a_min=0, a_max=1)
        data = np.nan_to_num(data)

        return data

    def __getitem__(self, idx):
        
        data = self.load_data(self.base + 'wc' + self.df.loc[idx, 'imageID'] + '.nii')
        label = self.labeldict[self.df.loc[idx, 'label']]

        if self.train:
            data = self.transform(data)
        else:
            data = self.valtransform(data)

        return data, label
    
def collate_fn(batch):

    data = []
    labels = []

    for img, label in batch:
        data.append(img)
        labels.append(label)

    labels = torch.Tensor(labels).type(torch.int64)
    return torch.stack(data).type(torch.float), labels


def get_loader(args, fold, num_classes, batch_size=32, num_workers=0):

    train_df = pd.read_csv(f'/media/storage2/Daesung/deprecated/MICCAI/folds_FDG/fold_{fold}_train.csv')
    val_df = pd.read_csv(f'/media/storage2/Daesung/deprecated/MICCAI/folds_FDG/fold_{fold}_val.csv')
    test_df = pd.read_csv(f'/media/storage2/Daesung/deprecated/MICCAI/folds_FDG/fold_{fold}_test.csv')

    train_ds = FDG(args, train_df, train=True, num_classes=num_classes)  
    val_ds = FDG(args, val_df, train=False, num_classes=num_classes)
    test_ds = FDG(args, test_df, train=False, num_classes=num_classes)

    ###############################################
    #         Define WeightedRandomSampler        #
    ###############################################

    value_counts = train_df.value_counts('label')
    num_samples = len(train_df)
    class_weights = []

    for label in train_ds.labeldict.keys():
        class_weights.append(num_samples / value_counts[label])

    # now translate this class weights per data
    weights = [class_weights[train_ds.labeldict[row['label']]] for _, row in train_df.iterrows()]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, sampler=sampler, drop_last=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, collate_fn=collate_fn, pin_memory=True)

    return train_loader, val_loader, test_loader
