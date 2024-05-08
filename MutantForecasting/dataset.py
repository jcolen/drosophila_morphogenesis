'''
Dataset objects for pytorch training on Dynamic Atlas
'''

import os
import numpy as np
import pandas as pd
import torch

from scipy.ndimage import gaussian_filter, gaussian_filter1d
from tqdm import tqdm

atlas_dir = '/project/vitelli/jonathan/REDO_fruitfly/src/Public'

class BaseTransformer():
    '''
    Base transformer class
    All transformations apply to the 'value' entry in the sample dict
    or the the sample itself if it's not a dict
    '''
    def transform(self, X):
        raise NotImplementedError

    def __call__(self, sample):
        if isinstance(sample, dict):
            sample['value'] = self.transform(sample['value'])
        else:
            sample = self.transform(sample)

        return sample

class Reshape2DField(BaseTransformer):
    '''
    Reshape field into [C, H, W]
    '''
    def transform(self, X):
        return X.reshape([-1, *X.shape[-2:]])

class LeftRightSymmetrize(BaseTransformer):
    '''
    Left right symmetrize by flipping the DV axis
    '''
    def transform(self, X):
        X_flip = X[:, ::-1, :].copy()
        if X.shape[0] == 2:
            X_flip[0] *= -1
        elif X.shape[0] == 4:
            X_flip[1:3] *= -1

        return 0.5 * (X + X_flip)

class ToTensor(BaseTransformer):
    '''
    Convert field to torch Tensor
    '''
    def transform(self, X):
        return torch.tensor(X, dtype=torch.float32)

class AtlasDataset(torch.utils.data.Dataset):
    '''
    AtlasDataset is a wrapper around the dynamic Atlas
    '''
    def __init__(self,
                 genotype,
                 label,
                 filename,
                 drop_time=False,
                 tmin=-15, tmax=30,
                 transform=ToTensor()):
        '''
        Genotype, label are identical arguments to da.DynamicAtlas
        filename is the information to pull from the atlas folder 
            such as velocity2D, tensor2D, etc.
        drop_time means to ignore timing information and just use
            the movie index to set the timing. IT IS STRONGLY 
            RECOMMENDED that a morphodynamic_offsets.csv file
            exists if you choose this option
        '''

        self.path = os.path.join(atlas_dir, genotype, label)
        self.filename = filename

        if os.path.exists(os.path.join(atlas_dir, genotype, label, 'dynamic_index.csv')):
            self.df = pd.read_csv(os.path.join(atlas_dir, genotype, label, 'dynamic_index.csv'))
        elif os.path.exists(os.path.join(atlas_dir, genotype, label, 'static_index.csv')):
            self.df = pd.read_csv(os.path.join(atlas_dir, genotype, label, 'static_index.csv'))

        if drop_time:
            self.df.time = self.df.eIdx
        else:
            self.df = self.df.dropna(axis=0)

        if os.path.exists(os.path.join(atlas_dir, genotype, label, 'morphodynamic_offsets.csv')):
            morpho = pd.read_csv(os.path.join(atlas_dir, genotype, label, 
                                              'morphodynamic_offsets.csv'), 
                                 index_col='embryoID')
            for eId in self.df.embryoID.unique():
                self.df.loc[self.df.embryoID == eId, 'time'] -= morpho.loc[eId, 'offset']

        if tmin is not None:
            self.df = self.df[self.df.time >= tmin].reset_index(drop=True)
        if tmax is not None:
            self.df = self.df[self.df.time <= tmax].reset_index(drop=True)

        self.values = {}

        folders = self.df.folder.unique()
        for folder in tqdm(folders):
            embryo_ID = int(os.path.basename(folder))
            self.values[embryo_ID] = np.load(
                os.path.join(folder, filename+'.npy'),
                mmap_mode='r')

        self.genotype = genotype
        self.label = label + '_' + filename
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        embryoID = self.df.embryoID[idx]
        index = self.df.eIdx[idx]
        time = self.df.time[idx]

        sample = {
            'embryoID': embryoID,
            'time': time,
            'genotype': self.genotype,
            'value': self.values[embryoID][index],
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class JointDataset(torch.utils.data.Dataset):
    '''
    Think of this as an extension of ConcatDataset but 
    we're careful about which axis we concatenate along
    '''
    def __init__(self,
                 datasets,
                 live_key='vel',
                 ensemble=0):
        self.live_key = live_key
        self.ensemble = ensemble
        self.values = {}
        self.transforms = []

        #Compute alignment between datasets
        df = pd.DataFrame()
        for i in range(len(datasets)):
            key, dataset = datasets[i]
            dfi = dataset.df
            #dfi['time'] = np.round(dfi['time']).astype(int) #Round time to int for merge
            dfi['key'] = key
            dfi['dataset_idx'] = int(i)
            df = df.append(dfi, ignore_index=True)

            # Merge dataset into values dict
            for eId in dataset.values:
                if not eId in self.values:
                    self.values[eId] = {}
                self.values[eId][key] = dataset.values[eId]

            #Add transform function from dataset
            self.transforms.append(dataset.transform)

        #Build an index for unique embryoID/timestamp combinations
        mask = (df.key == self.live_key)
        df.loc[mask, 'merged_index'] = df[mask].groupby(['embryoID', 'time']).ngroup()
        df.loc[~mask, 'merged_index'] = -1
        df.merged_index = df.merged_index.astype(int)
        self.df = df
        self.keys = self.df.key.unique()


    def __len__(self):
        return self.df.merged_index.max()+1

    def __getitem__(self, idx):
        row = self.df[self.df.merged_index == idx] #Only returns embryos with flow

        eId = row.embryoID.values[0]
        time = row.time.values[0]
        rows = self.df[(self.df.embryoID == eId) & (self.df.time == time)]

        sample = {
            'embryoID': eId,
            'time': time
        }

        for i,row in rows.iterrows():
            sample[row.key] = self.values[row.embryoID][row.key][row.eIdx]

            if self.transforms[row.dataset_idx] is not None:
                sample[row.key] = self.transforms[row.dataset_idx](sample[row.key])

        if self.ensemble > 0:
            for key in self.keys:
                if not key in sample:
                    sample[key] = self.ensemble_key(key, sample['time'])

        return sample