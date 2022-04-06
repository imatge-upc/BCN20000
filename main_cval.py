import yaml

from torch import nn
from modelling.model import  CEEffNet, CEResNet

from preparation import dataset_functions as dpr
from preparation.datasets import BCN20k_Dataset
from processing.train import train
from processing.utils import obtain_transform

separator = '-'*40
settings_path = 'utils/settings.yaml'

def train_main(settings):

# data_transform= obtain_transform(settings)

  print(separator)
  diff_model_names = ['effb2'] 

  for pre_pro in ['cropped', 'uncropped']:
      df = dpr.get_data(settings)
      if pre_pro=='uncropped':
          df['filename'] = ['/home/carlos.hernandez/datasets/images/BCN_backup/'+x.split('/')[-1] for x in df['filename']]
          print('Uncropping mode')
      splits = dpr.get_data_cval(df)
      for i, split in enumerate(splits):
          for model_name in diff_model_names:
              settings['model_name'] = model_name
              data_transform = obtain_transform(settings)
              save_path =settings['model_name']+'_' +pre_pro+'_'+ str(i)
              

              datasets = obtain_datasets(split, data_transform, settings)
              num_classes = len(split[0].label.unique())
              model = choose_model(settings, num_classes)

              print('{}\nEntering training of model {} split {}\n{}'.format(separator,model_name, i,separator))
              print(data_transform['train'])
              print('The model type is: {} with batch size of'.format(type(model)))
              print('{}\n Training split: {} ------ Validation split: {}\nTraining will begin with {} # of classes\n{}'.format(separator,
                                            split[0].shape,split[1].shape,num_classes,separator))
              train(model, datasets, save_path, settings, batch_sizes=(64, 32))
        
def obtain_datasets(split, data_transform, settings):
    df_train = split[0]
    df_val = split[1]
    train_transform = data_transform['train']
    val_transform   = data_transform['val']

    dataset_train = BCN20k_Dataset(
        df_train, settings, transform=train_transform)

    dataset_val = BCN20k_Dataset(
        df_val, settings, transform=val_transform)

    datasets = (dataset_train, dataset_val)
    return datasets

def choose_model(settings : dict, num_classes : int=2) -> nn.Module:
    """
    Choose the model depending on the
    yaml file
    """
    if 'res' in settings['model_name']:
        model = CEResNet(num_classes= num_classes, model_name=settings['model_name'])
    elif 'eff' in settings['model_name']:
        model = CEEffNet(num_classes = num_classes, model_name=settings['model_name'])
    else:
        raise Exception('Ey! Specify a model in the settings.yaml')
        
    return model 

def load_settings(settings_path):
    with open(settings_path, "r") as file_descriptor:
        settings = yaml.load(file_descriptor, Loader=yaml.Loader)

    return settings

if __name__== "__main__":
    
    settings = load_settings(settings_path)
    train_main(settings)
