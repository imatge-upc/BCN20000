import argparse
import random
import numpy as np
import torch

from torch import nn
from modelling.model import  CEEffNet, CEResNet

from preparation import dataset_functions as dpr
from preparation.datasets import BCN20k_Dataset
from processing.train import train
from processing.utils import obtain_transform

# Make wandb offline if needed
#os.environ['WANDB_MODE'] = 'offline'


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


def get_batch_size(model_name):
    if model_name in ['res18', 'res34', 'effb0']:
        return 256
    else:
        return 64

def train_main(args):

    print(separator)

    if args.model_name == 'all':
        diff_model_names = ['res18', 'res34', 'res50', 'effb0', 'effb1', 'effb2']
    else:
        diff_model_names = [args.model_name]

    
    if args.paper_splits:
        print(separator)
        print('Using paper splits')
        print(separator)
        splits = dpr.load_splits_from_file(args)
    else:
        df = dpr.get_data(args)
        splits = dpr.create_stratified_splits(df)

    for i, split in enumerate(splits):
        for model_name in diff_model_names:
            args.split = i
            # Only do the first split for each model
            args.model_name = model_name
            # Ensuring that the same hyperparameters are used as in the paper
            if args.paper_splits:
                hyperparameters = get_hyperparameters(args.model_name)
                # Use the same hyperparameters as in the paper
                args.learning_rate = hyperparameters['learning_rate']
                args.weight_decay = hyperparameters['weight_decay']
                print('Using hyperparameters from the paper')
                print('Learning rate: {}'.format(args.learning_rate))
                print('Weight decay: {}'.format(args.weight_decay))
            

            data_transform = obtain_transform(args)
            save_path = args.model_name+'_split_'+ str(i)
            
            datasets = obtain_datasets(split, data_transform, args)
            num_classes = len(split[0].label.unique())
            model = choose_model(args, num_classes)

            print('{}\nEntering training of model {} for split {}\n{}'.format(separator,model_name, i,separator))
            print(data_transform['train'])
            print('The model type is: {}'.format(type(model)))
            print('{}\n Training split: {} ------ Validation split: {}\nTraining will begin with {} # of classes\n{}'.format(separator,
                                                                                                                             split[0].shape,split[1].shape, num_classes,separator))
            # If model resnet0 then bs=400
            train(model, datasets, save_path, args, batch_sizes=(get_batch_size(model_name), 512))

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
        model = CEResNet(num_classes= num_classes, model_name=args.model_name, dropout=args.dropout)
    elif 'eff' in args.model_name:
        model = CEEffNet(num_classes = num_classes, model_name=args.model_name, dropout=args.dropout)
    else:
        raise Exception('Ey! Specify a model from this list: effb0, effb1, effb2, res18, res34, res50')

    return model 


def get_hyperparameters(model_name):
    """
    Return the hyperparameters based on the model name.

    Parameters:
    model_name (str): The name of the model.

    Returns:
    dict: A dictionary containing the learning_rate, weight_decay, and dropout values.
    """
    if model_name == 'res18':
        return {
            'learning_rate': 0.0001,
            'weight_decay': 0.0001,
        }
    elif model_name == 'effb0':
        return {
            'learning_rate': 1e-4,
            'weight_decay': 0.001,
        }
    elif model_name == 'effb2' or model_name == 'effb1':
        return {
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
        }
    elif model_name.startswith('res'):
        # For ResNets
        return {
            'learning_rate': 0.0001,
            'weight_decay': 0.01,
        }
    else:
        raise

if __name__== "__main__":

    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--train_csv', type=str, help='Path to the csv path')
    parser.add_argument('--data_dir', type=str, help='Path to the images')

    parser.add_argument('--model_name', type=str, help='Model name', default='effb1', choices=['effb0', 'effb1', 'effb2', 'res18', 'res34', 'res50', 'all'])
    # add patience
    parser.add_argument('--patience', type=int, help='Patience for early stopping', default=20)
    # add learning rate and weight decay
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.0010)
    parser.add_argument('--weight_decay', type=float, help='Weight decay', default=1e-5)
    parser.add_argument('--dropout', type=float, help='Dropout', default=0.4)
    # Used in combination to reproduce the paper results
    parser.add_argument('--paper_splits', action='store_false', help='Use the paper splits')
    parser.add_argument('--master_split_file', type=str, help='Path to the master split file', default='./master_split_file.csv')
    args = parser.parse_args()

    train_main(args)
