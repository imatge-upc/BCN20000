from __future__ import print_function

import pandas 
import numpy as np

from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score


def obtain_transform(args : dict):
    if 'eff' in args.model_name:
        if 'b0' in args.model_name:
            size = 224
        elif 'b1' in args.model_name:
            size = 240
        elif 'b2' in args.model_name:
            size = 260
    else:
        size = 224

    train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(
       size=size, scale=(0.8, 1.2), ratio=(0.9, 1.1)),
    transforms.ColorJitter(
        brightness=0.2, saturation=0.20, contrast=0.2, hue=0.20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

        
    data_transform = {
    "train": train_transforms,
    "val": transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
	transforms.Normalize(mean=[0.485,0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    }

    return data_transform


def make_weights_for_balanced_classes(labels : pandas.Series):
    nclasses = len(np.unique(labels))
    if nclasses == 1:
        return [1], [1]

    count = [0] * nclasses
    for label in labels:
        count[label] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(labels)
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[val]
    
    return weight, weight_per_class

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

def acc_class(true : list, pred :list):
    ''' Performs calculating balanced accuracy score
    :param pred: (np.ndarray or torch.Tensor) model prediction
    :param true: (np.ndarray or torch.Tensor) true label
    :return BACC: balanced accuracy score of the model
    '''
    return balanced_accuracy_score(true, pred)
