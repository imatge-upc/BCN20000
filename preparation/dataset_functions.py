import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


path_bcn_20k_csv ='~/datasets/csvs/data_bcn_20k.csv'

replace_dict = {'NV':0, 'MEL':1, 'SCC':2, 'BKL':3, 'BCC':4, 'AK':5, 'DF':6, 'VASC':7}

def get_data(settings : dict) -> pd.DataFrame:
    # Read data file
    df = obtain_dataframe(settings)
    # Here we choose the experiment label
    df['label'] = [replace_dict[x] for x in df['diagnosis']]
    df = df[['filename', 'label', 'id']]
    return df

def obtain_dataframe(settings : dict) -> pd.DataFrame:
    """
    Loads the dataframe according to the settings

    Arguments:
    ---------------------
    settings : settings file : dict
    ---------------------
    Outputs:
    ---------------------
    df       : dataframe w experiment info: pd.DataFrame
    ---------------------
    """
    df = pd.read_csv(path_bcn_20k_csv)
    df = df[df['split']=='train']
    return df

def get_data_cval(df: pd.DataFrame) -> list:
    """
    Divide the data into K folds the dataframe according to the settings

    Arguments:
    ---------------------
    df       : dataframe w experiment info: pd.DataFrame
    ---------------------
    Outputs:
    ---------------------
    splits    : List of tuples containing the datasets of the different splits
    ---------------------
    """
    id_list = df['id'].unique()

    splits = []
    # Stratified Kfold for binary classification
    skf = StratifiedKFold(n_splits=4, random_state=0, shuffle=True)
    for train_index, test_index in skf.split(id_list, df['label']):
        train_df = df[df['id'].isin(id_list[train_index])]
        test_df = df[df['id'].isin(id_list[test_index])]

        # Do proper splitting of the dataset
        val_df, test_df = train_test_split(test_df, test_size = 0.40, stratify=test_df['label'], random_state=0)

        splits.append((train_df, val_df, test_df))

    return splits
