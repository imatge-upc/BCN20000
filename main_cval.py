#import yaml
import argparse
import random
import numpy as np
import torch

from argparse import ArgumentParser

from torch import nn
from modelling.model import  CEEffNet, CEResNet

from preparation import dataset_functions as dpr
from preparation.datasets import BCN20k_Dataset
from processing.train import train
from processing.utils import obtain_transform


separator = '-'*40
args_path = 'utils/settings.yaml'



def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)       # Python's built-in random library
    np.random.seed(seed)    # NumPy library
    torch.manual_seed(seed) # PyTorch

    # If using CUDA (PyTorch with GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        # Below ensures that CUDA operations are deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Make sure that seeds are set prior to any other operations
set_seeds(42)




def train_main(args):

    print(separator)
    diff_model_names = ['res50'] 

    for pre_pro in ['uncropped', 'cropped']:
        #if pre_pro=='uncropped':
        #    df['filename'] = ['/home/carlos.hernandez/datasets/images/BCN_backup/'+x.split('/')[-1] for x in df['filename']]
        #    print('Uncropping mode')


        ### TODO
        # 1. Smth is wrong with the training, we are not getting the right results
        # 3. Run experiments to obtain new results
        # 4. VERY IMPORTANT: Update .zip files in Figshare
        if args.paper_splits:
            splits = dpr.load_splits_from_file(args)
        else:
            df = dpr.get_data(args)
            splits = dpr.create_stratified_splits(df)
        for i, split in enumerate(splits):
            for model_name in diff_model_names:
                args.model_name = model_name
                data_transform = obtain_transform(args)
                save_path = args.model_name+'_' +pre_pro+'_'+ str(i)

                datasets = obtain_datasets(split, data_transform, args)
                num_classes = len(split[0].label.unique())
                model = choose_model(args, num_classes)

                print('{}\nEntering training of model {} for split {}\n{}'.format(separator,model_name, i,separator))
                print(data_transform['train'])
                print('The model type is: {}'.format(type(model)))
                print('{}\n Training split: {} ------ Validation split: {}\nTraining will begin with {} # of classes\n{}'.format(separator,
                                                                                                                                 split[0].shape,split[1].shape, num_classes,separator))

                train(model, datasets, save_path, args, batch_sizes=(256, 256))

def obtain_datasets(split, data_transform, args):
    df_train = split[0]
    df_val = split[1]
    df_test = split[2]

    train_transform = data_transform['train']
    val_transform   = data_transform['val']

    dataset_train = BCN20k_Dataset(
        df_train, args, transform=train_transform)

    dataset_val = BCN20k_Dataset(
        df_val, args, transform=val_transform)

    dataset_test = BCN20k_Dataset(
        df_test, args, transform=val_transform)

    datasets = (dataset_train, dataset_val, dataset_test)
    return datasets

def choose_model(args : argparse.ArgumentParser, num_classes : int=2) -> nn.Module:
    """
    Choose the model depending on the
    yaml file
    """
    if 'res' in args.model_name:
        model = CEResNet(num_classes= num_classes, model_name=args.model_name)
    elif 'eff' in args.model_name:
        model = CEEffNet(num_classes = num_classes, model_name=args.model_name)
    else:
        raise Exception('Ey! Specify a model from this list: effb0, effb1, effb2, res18, res34, res50')

    return model 

if __name__== "__main__":

    argparser = argparse.ArgumentParser(description='Train the model')
    argparser.add_argument('--train_csv', type=str, help='Path to the csv path', default='./bcn_20k_train.csv')
    argparser.add_argument('--data_dir', type=str, help='Path to the data', default='/home/carlos.hernandez/datasets/images/BCN_20k_/new_train/')
    argparser.add_argument('--model_name', type=str, help='Model name', default='effb0', choices=['effb0', 'effb1', 'effb2', 'res18', 'res34', 'res50'])
    # add learning rate and weight decay
    argparser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.00030)
    argparser.add_argument('--weight_decay', type=float, help='Weight decay', default=1e-5)
    # Used in combination to reproduce the paper results
    argparser.add_argument('--paper_splits', action='store_true', help='Use the paper splits')
    argparser.add_argument('--master_split_file', type=str, help='Path to the master split file', default='./master_split_file.csv')
    args = argparser.parse_args()

    train_main(args)
