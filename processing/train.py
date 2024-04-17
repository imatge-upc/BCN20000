import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from processing.utils import make_weights_for_balanced_classes
from processing.train_utils import supervised_training 
from typing import Tuple
 
def train(model : torch.nn.Module, datasets : list, split : str, args : dict, batch_sizes : Tuple):
    # Check which device we are working on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloaders
    dataloaders = obtain_dataloaders(datasets, batch_sizes, args)

    # Loss function, optimizer and scheduler
    loss_fn, optimizer, scheduler = obtain_training_tools(model, args)
    train_tools = [loss_fn, optimizer, scheduler]
     
    supervised_training(dataloaders, model, device, train_tools,
                    args, num_epochs=250, split=split)

def obtain_dataloaders(datasets : list , batch_sizes : Tuple, args : dict):
    # Separate datasets from tuple
    dataset_train = datasets[0]
    dataset_val = datasets[1]
    dataset_test = datasets[2]

    # Separate batch sizes from tuple
    batch_size_train = batch_sizes[0]
    batch_size_test = batch_sizes[1]
        
    # Obtain weights for balance classes
    weights_train, _ = make_weights_for_balanced_classes(
        dataset_train.df.label)
    weights_train = torch.FloatTensor(weights_train)
    sampler_train = WeightedRandomSampler(
        weights_train, len(weights_train), replacement=True)

    # Instantiate loaders
    loader_train = DataLoader(
        dataset_train, batch_size=batch_size_train,
        sampler=sampler_train, num_workers=32, drop_last=False)

    # and for validation (without sampler)
    loader_val = DataLoader(
        dataset_val, batch_size=batch_size_test, num_workers=32, shuffle=False, drop_last=False)

    loader_test = DataLoader(
        dataset_test, batch_size=batch_size_test, num_workers=32, shuffle=False, drop_last=False)

    return {'train': loader_train, 'val': loader_val, 'test': loader_test}

def obtain_training_tools(model : torch.nn.Module , args : dict):
    loss_fn = nn.CrossEntropyLoss()
    print('We are using {} as loss function'.format(loss_fn))
    optimizer = Adam(model.parameters(),
                    lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
       optimizer, 20, T_mult=1)
    return loss_fn, optimizer, scheduler
