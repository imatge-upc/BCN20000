from argparse import ArgumentParser
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split

replace_dict = {'NV':0, 'MEL':1, 'SCC':2, 'BKL':3, 'BCC':4, 'AK':5, 'DF':6, 'VASC':7}

def get_states(random_state, low, high, size):
    rs = np.random.RandomState(random_state)
    states = rs.randint(low=low, high=high, size=size)
    return states

# Change the type hint to update to an argparser object
def get_data(args : ArgumentParser) -> pd.DataFrame:
    # Read data file
    df = pd.read_csv(args.train_csv)
    # Here we choose the experiment label
    df['label'] = [replace_dict[x] for x in df['diagnosis']]
    df['filename'] = [args.data_dir + x for x in df['bcn_filename']]
    df = df[['filename', 'label', 'lesion_id']]
    return df

def create_stratified_splits(df: pd.DataFrame, n_splits: int = 5, train_size: float = 0.75, 
                             val_size: float = 0.05, random_state: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Splits a DataFrame into stratified training, validation, and testing datasets across specified number of folds.
    Each split ensures that the proportion of categories in the original data is preserved as much as possible
    across each train, validation, and test set.

    Args:
    - df (pd.DataFrame): The input DataFrame to be split.
    - n_splits (int): The number of stratified folds to create.
    - train_size (float): The proportion of the dataset to allocate to the training set.
    - val_size (float): The proportion of the dataset to allocate to the validation set.
    - test_size (float): The proportion of the dataset to allocate to the testing set.
    - random_state (int): A seed value to ensure the reproducibility of the splits.

    Returns:
    - List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]: A list of tuples, where each tuple contains the
      training, validation, and testing datasets for a fold. The datasets are DataFrames.
    """
    # Get a list of random states for each fold
    random_states = get_states(random_state, 2, 28347, size=n_splits)
    
    id_list = df['lesion_id'].unique()
    splits = []
    # Stratified Kfold for binary classification using ids and labels
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_states[0])
    
    for i, (train_index, test_index) in enumerate(skf.split(id_list, df.groupby('lesion_id').first()['label'])):
        train_ids = id_list[train_index]
        test_ids = id_list[test_index]
        # Select training and test DataFrames
        train_df = df[df['lesion_id'].isin(train_ids)]
        test_df = df[df['lesion_id'].isin(test_ids)]
        
        # Split the original training set to create a smaller training set and a validation set
        train_df, val_df = train_test_split(train_df, test_size=val_size/(train_size+val_size), 
                                            stratify=train_df['label'], random_state=random_states[i])
        
        splits.append((train_df, val_df, test_df))

    return splits



def load_splits_from_file(args : ArgumentParser) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Loads and reconstructs dataset splits from a CSV file where each image is represented multiple times,
    indicating its participation in different folds and as part of train, validation, or test sets.

    Args:
    - file_path (str): The file path to the master split CSV file.

    Returns:
    - List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]: A list of tuples, where each tuple contains 
      three DataFrames corresponding to the training, validation, and test datasets for each fold.
    """
    # Read the master split file
    df = pd.read_csv(args.master_split_file)

    df['label'] = [replace_dict[x] for x in df['diagnosis']] 
    df['filename'] = [args.data_dir + x for x in df['bcn_filename']]
    # Determine the number of folds from the file
    num_folds = df['fold_number'].max()
    
    splits = []
    # Iterate over each fold and extract the corresponding train, validation, and test sets
    for fold in range(1, num_folds + 1):
        fold_df = df[df['fold_number'] == fold]
        
        train_df = fold_df[fold_df['split_type'] == 'train']
        val_df = fold_df[fold_df['split_type'] == 'validation']
        test_df = fold_df[fold_df['split_type'] == 'test']
        
        # Append the tuple of DataFrames to the list of splits
        splits.append((train_df, val_df, test_df))
    
    return splits

def create_stratified_splits_master_file(df: pd.DataFrame, n_splits: int = 5, train_size: float = 0.75, 
                       val_size: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    """
    Creates a master file DataFrame with expanded rows representing multiple fold splits of the dataset.
    Each image appears in multiple folds with designated roles (train, validation, test) based on stratified sampling.

    Args:
    - df (pd.DataFrame): DataFrame containing the original dataset with 'lesion_id' and 'diagnosis' columns.
    - n_splits (int): Number of folds for the stratified K-fold split.
    - train_size (float): Proportion of the dataset to be used as training data in each fold.
    - val_size (float): Proportion of the dataset to be used as validation data in each fold.
    - test_size (float): Proportion of the dataset to be used as test data in each fold.
    - random_state (int): Seed for the random number generator to ensure reproducibility.

    Returns:
    - pd.DataFrame: A master DataFrame with additional 'split_type' and 'fold_number' columns indicating
                    the split type (train, validation, test) and the fold number for each row.
    """
    # Initialize random states for each fold
    random_states = get_states(random_state, 2, 28347, size=n_splits + 1)

    id_list = df['lesion_id'].unique()
    master_list = []  # List to hold the expanded DataFrame rows

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_states[0])

    for i, (train_index, test_index) in enumerate(skf.split(id_list, df.groupby('lesion_id').first()['diagnosis'])):
        train_ids = id_list[train_index]
        test_ids = id_list[test_index]

        fold_df = df.copy()
        fold_df['split_type'] = None
        fold_df['fold_number'] = i + 1  # Start fold numbering at 1

        # Select and label the train, validation, and test data
        train_df = fold_df[fold_df['lesion_id'].isin(train_ids)]
        test_df = fold_df[fold_df['lesion_id'].isin(test_ids)]

        # Further split the train data into new train and validation sets
        train_df, val_df = train_test_split(train_df, test_size=val_size/(train_size+val_size), 
                                            stratify=train_df['diagnosis'], random_state=random_states[i + 1])
        
        # Assign split types
        fold_df.loc[train_df.index, 'split_type'] = 'train'
        fold_df.loc[val_df.index, 'split_type'] = 'validation'
        fold_df.loc[test_df.index, 'split_type'] = 'test'
        # Append the modified DataFrame to the master list
        master_list.append(fold_df)

    # Concatenate all fold DataFrames into one master DataFrame
    master_df = pd.concat(master_list, ignore_index=True)
    return master_df
