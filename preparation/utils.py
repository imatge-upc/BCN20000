import torch
#from torch._six import int_classes as _int_classes

from torch.utils.data.sampler import Sampler

from processing.utils import make_weights_for_balanced_classes
import pandas as pd


class ConditionalSampler(Sampler):
    r"""
    Sampler that ingests data based on two conditions
    """

    def __init__(self, df, replacement=True, batch_size=4, val=False):
        '''
        df = Dataframe containing the dataset to be used
        replacement = With replacement=True, each sample can be picked in each draw again.
        batch_size = int which chooses the size the size of the batch
        '''

        # self.num_samples = len(df)
        self.replacement = replacement
        self.df = df

        self.batch_size = batch_size

        def has_label(df):
            has_label = 1 in df['label'].unique()
            row = df.iloc[0]
            row['has_label'] = int(has_label)
            del row['label'], row['filename']
            return row

        self.img_info = df.groupby('id').apply(has_label).reset_index(drop=True)

        self.val = val

        '''
        if not isinstance(self.num_samples, _int_classes) or isinstance(self.num_samples, bool) or \
                self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))
        '''

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    def __iter__(self):
        label_weights, _ = make_weights_for_balanced_classes(self.img_info['has_label'])
        label_weights = torch.FloatTensor(label_weights)

        img_ids = torch.multinomial(label_weights, len(label_weights), self.replacement).tolist()
        sampled_img = self.img_info.iloc[img_ids].copy()

        if self.val:
            sampled_img = self.img_info.copy()
        # [0, 1, 2, ...., 8250]

        counter = 0
        ret_lesions = []
        for img_id in sampled_img['id']:
            lesions = self.df[self.df['_id'] == img_id].copy()

            # [5, 152, 3998, 4648]

            les_weights, _ = make_weights_for_balanced_classes(lesions['label'])
            les_weights = torch.FloatTensor(les_weights)

            les_ids = torch.multinomial(les_weights, 1, self.replacement).tolist()

            # [5, 5, 5, 5, 1, 2, 3, 4]

            sampled_lesions = lesions.iloc[les_ids].copy()

            if self.val:
                sampled_lesions = lesions

            # [3998, 3998, 3998, 3998, 3998, 125, 32, 6, 1, 8]

            return_lesions = sampled_lesions.index
            ret_lesions += return_lesions.tolist()
            counter += 1

            # Cuantos pacientes incluyo en el batch
            if counter >= 8:
                yield ret_lesions
                counter = 0
                ret_lesions = []

    def __len__(self):
        return len(self.df)

