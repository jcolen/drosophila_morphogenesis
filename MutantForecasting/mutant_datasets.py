'''
Hardcoded datasets for mutant analyses
Note that some mutant datasets have imaging quirks which require reversal along e.g. the AP axis
Other mutant datasets don't have ventral furrows or GBE, which complicates timing
Rather than make something that generally works, I'm going to hardcode in each dataset for simplicity

Also, things can simplify now that we know we only need myosin, which is measured for every mutant line
'''

import os
import numpy as np
import pandas as pd
import torch

from scipy.ndimage import gaussian_filter, gaussian_filter1d
from tqdm import tqdm

atlas_dir = '/project/vitelli/jonathan/REDO_fruitfly/src/Public'

class BaseTransformer():
    def transform(self, X):
        raise NotImplementedError
    
    def __call__(self, sample):
        sample['sqh'] = self.transform(sample['sqh'])
        sample['vel'] = self.transform(sample['vel'])
        return sample

class ToTensor(BaseTransformer):
    '''
    Convert field to torch Tensor
    '''
    def transform(self, X):
        return torch.tensor(X, dtype=torch.float32)
    
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

class BaseDataset(torch.utils.data.Dataset):
    def get_trajectory(self, embryoID):
        eIdx = self.df.eIdx[self.df.embryoID == embryoID]
        return self.sqh[embryoID][eIdx], self.vel[embryoID][eIdx], self.df.loc[self.df.embryoID == embryoID, 'time'].values

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
            'sqh': self.sqh[embryoID][index],
            'vel': self.vel[embryoID][index],
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class WTDataset(BaseDataset):
    def __init__(self,
                 genotype='Halo_Hetero_Twist[ey53]_Hetero',
                 label='Sqh-GFP',
                 tmin=-15, tmax=30,
                 transform=ToTensor()):
        super().__init__()
        self.genotype = genotype
        self.label = label
        self.tmin = tmin
        self.tmax = tmax
        self.transform = transform
        self.path = os.path.join(atlas_dir, genotype, label)

        # Load dataframe of embryo information
        self.df = pd.read_csv(f'{self.path}/dynamic_index.csv')
        self.df.time = self.df.eIdx
        
        morpho = pd.read_csv(f'{self.path}/morphodynamic_offsets.csv', index_col='embryoID')
        for eId in self.df.embryoID.unique():
            self.df.loc[self.df.embryoID == eId, 'time'] -= morpho.loc[eId, 'offset']

        self.df = self.df[(self.df.time >= tmin) & (self.df.time <= tmax)].reset_index(drop=True)
        
        # Load data
        self.sqh = {}
        self.vel = {}
        for folder in tqdm(self.df.folder.unique()):
            eId = int(os.path.basename(folder))
            self.sqh[eId] = np.load(f'{folder}/tensor2D.npy', mmap_mode='r')
            self.vel[eId] = np.load(f'{folder}/velocity2D.npy', mmap_mode='r')

class TwistDataset(BaseDataset):
    def __init__(self,
                 genotype='Halo_twist[ey53]',
                 label='Sqh-GFP',
                 tmin=-15, tmax=30,
                 transform=ToTensor()):
        super().__init__()
        self.genotype = genotype
        self.label = label
        self.tmin = tmin
        self.tmax = tmax
        self.transform = transform
        self.path = os.path.join(atlas_dir, genotype, label)

        # Load dataframe of embryo information
        self.df = pd.read_csv(f'{self.path}/dynamic_index.csv')
        self.df.time = self.df.eIdx
        
        morpho = pd.read_csv(f'{self.path}/morphodynamic_offsets.csv', index_col='embryoID')
        for eId in self.df.embryoID.unique():
            self.df.loc[self.df.embryoID == eId, 'time'] -= morpho.loc[eId, 'offset']
        
        self.df = self.df[(self.df.time >= tmin) & (self.df.time <= tmax)].reset_index(drop=True)
        
        # Load data
        self.sqh = {}
        self.vel = {}
        for folder in tqdm(self.df.folder.unique()):
            eId = int(os.path.basename(folder))
            if eId in [202007301145, 202007171100, 202007171400]:
                self.sqh[eId] = np.load(f'{folder}/tensor2D.npy')[..., ::-1]
                self.sqh[eId][:, 1, 0] *= -1
                self.sqh[eId][:, 0, 1] *= -1
                self.vel[eId] = np.load(f'{folder}/velocity2D.npy')[..., ::-1]
                self.vel[eId][:, 1] *= -1
            else:
                self.sqh[eId] = np.load(f'{folder}/tensor2D.npy', mmap_mode='r')
                self.vel[eId] = np.load(f'{folder}/velocity2D.npy', mmap_mode='r')

class TollDataset(BaseDataset):
    def __init__(self,
                 genotype='toll[RM9]',
                 label='Sqh-GFP',
                 tmin=0, tmax=45,
                 transform=ToTensor()):
        super().__init__()
        self.genotype = genotype
        self.label = label
        self.tmin = tmin
        self.tmax = tmax
        self.transform = transform
        self.path = os.path.join(atlas_dir, genotype, label)

        # Load dataframe of embryo information
        self.df = pd.read_csv(f'{self.path}/dynamic_index.csv')
        self.df.time = self.df.eIdx

        self.df = self.df[(self.df.time >= tmin) & (self.df.time <= tmax)].reset_index(drop=True)
        
        # Load data
        self.sqh = {}
        self.vel = {}
        for folder in tqdm(self.df.folder.unique()):
            eId = int(os.path.basename(folder))
            self.sqh[eId] = np.load(f'{folder}/tensor2D.npy', mmap_mode='r')
            self.vel[eId] = np.load(f'{folder}/velocity2D.npy', mmap_mode='r')
            
class SpaetzleDataset(BaseDataset):
    def __init__(self,
                 genotype='spaetzle[A]',
                 label='Sqh-GFP',
                 tmin=0, tmax=45,
                 transform=ToTensor()):
        super().__init__()
        self.genotype = genotype
        self.label = label
        self.tmin = tmin
        self.tmax = tmax
        self.transform = transform
        self.path = os.path.join(atlas_dir, genotype, label)

        # Load dataframe of embryo information
        self.df = pd.read_csv(f'{self.path}/dynamic_index.csv')
        self.df.time = self.df.eIdx

        self.df = self.df[(self.df.time >= tmin) & (self.df.time <= tmax)].reset_index(drop=True)
        
        # Load data
        self.sqh = {}
        self.vel = {}
        for folder in tqdm(self.df.folder.unique()):
            eId = int(os.path.basename(folder))
            self.sqh[eId] = np.load(f'{folder}/tensor2D.npy', mmap_mode='r')
            self.vel[eId] = np.load(f'{folder}/velocity2D.npy', mmap_mode='r')