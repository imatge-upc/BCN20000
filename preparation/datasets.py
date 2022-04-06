# For image loading
from PIL import Image

from torch.utils.data import Dataset

import numpy as np

class BCN20k_Dataset(Dataset):
    def __init__(self, df, settings, transform=None):
        # Initialize the dataset from a dataframe and by default no transformations
        self.df = df
        self.transform = transform
        self.settings = settings        

    def __len__(self):
        # return the length of the df as the length of the dataset class
        return len(self.df)

    def __getitem__(self, idx):
        # Obtain the values of a patient 
        row = self.df.iloc[idx]
    
        try:
            sample = Image.open(row["filename"])
            sample = sample.convert('RGB')
            
            # Transform if there are any transformations to do
            if self.transform:
                sample = self.transform(sample)
        except Exception:
            import traceback
            traceback.print_exc()
            print(row['filename'])
            quit()

        return sample, row["label"]
