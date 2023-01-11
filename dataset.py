import os
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

class Reshape2DField(object):
	def __call__(self, sample):
		sample['value'] = sample['value'].reshape([-1, *sample['value'].shape[-2:]])
		return sample

class ToTensor(object):
	def __call__(self, sample):
		sample['value'] = torch.tensor(sample['value'].copy(), dtype=torch.float32)
		return sample

atlas_dir = '/project/vitelli/jonathan/REDO_fruitfly/MLData'

class AtlasDataset(torch.utils.data.Dataset):
	def __init__(self,
				 genotype,
				 label,
				 filename,
				 drop_no_time=True,
				 transform=ToTensor()):
		
		self.path = os.path.join(atlas_dir, genotype, label)
		self.filename = filename
		self.df = pd.read_csv(os.path.join(atlas_dir, genotype, label, 'dynamic_index.csv'))
		if drop_no_time:
			self.df = self.df.dropna(axis=0)
		else:
			self.df.loc[np.isnan(self.df.time), 'time'] = self.df.loc[np.isnan(self.df.time), 'eIdx']

		self.values = {}

		folders = self.df.folder.unique()
		for folder in tqdm(folders):
			embryo_ID = int(os.path.basename(folder))
			self.values[embryo_ID] = np.load(os.path.join(folder, filename+'.npy'), mmap_mode='r')

		self.genotype = genotype
		self.label = label + '_' + filename[:-4]
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

class AlignedDataset(torch.utils.data.Dataset):
	def __init__(self,
				 datasets,
				 key_names,
				 align_on=['embryoID', 'time']):
		super(AlignedDataset, self).__init__()

		self.datasets = datasets
		self.key_names = key_names
		self.df = None

		#Build time-alignment between the datasets
		#Find common embryos between datasets and align times
		for i in range(len(self.datasets)):
			dfi = self.datasets[i].df.reset_index()
			dfi = dfi.drop(['folder', 'tiff', 'eIdx'], axis=1)
			dfi = dfi.rename({'index': 'index_'+key_names[i]}, axis=1)

			if self.df is None:
				self.df = dfi
			else:
				self.df = pd.merge(
					self.df, 
					dfi,
					how='inner',
					on=align_on)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		sample = {}
		for key, dataset in zip(self.key_names, self.datasets):
			si = dataset[idx]
			si[key] = si.pop('value')
			sample = {**sample, **si}
		return sample
