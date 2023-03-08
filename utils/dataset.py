import os
import numpy as np
import pandas as pd
import torch

from scipy.ndimage import gaussian_filter
import gc
from tqdm import tqdm

class Reshape2DField(object):
	def __call__(self, sample):
		if isinstance(sample, dict):
			sample['value'] = sample['value'].reshape([-1, *sample['value'].shape[-2:]])
			return sample
		else:
			return sample.reshape([-1, *sample.shape[-2:]])

class Smooth2D(object):
	def __init__(self, sigma=3):
		self.sigma = sigma

	def __call__(self, sample):
		'''
		sample['value'] is a [C, Y, X] field
		'''
		if isinstance(sample, dict):
			sample['value'] = np.stack([gaussian_filter(sample['value'][c], sigma=self.sigma) \
				for c in range(sample['value'].shape[0])], axis=0)
			return sample
		else:
			return np.stack([gaussian_filter(sample[c], sigma=self.sigma) \
				for c in range(sample.shape[0])], axis=0)

class ToTensor(object):
	def __call__(self, sample):
		if isinstance(sample, dict):
			sample['value'] = torch.tensor(sample['value'], dtype=torch.float32)
			return sample
		else:
			return torch.tensor(sample, dtype=torch.float32)

atlas_dir = '/project/vitelli/jonathan/REDO_fruitfly/src/Public'

class AtlasDataset(torch.utils.data.Dataset):
	'''
	AtlasDataset is a wrapper around the dynamic Atlas
	'''
	def __init__(self,
				 genotype,
				 label,
				 filename,
				 drop_time=False,
				 tmin=None, tmax=None,
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

			if drop_time:
				self.df.time = self.df.eIdx
			else:
				self.df = self.df.dropna(axis=0)

			if os.path.exists(os.path.join(atlas_dir, genotype, label, 'morphodynamic_offsets.csv')):
				morpho = pd.read_csv(os.path.join(atlas_dir, genotype, label, 'morphodynamic_offsets.csv'), index_col='embryoID')
				for eId in self.df.embryoID.unique():
					self.df.loc[self.df.embryoID == eId, 'time'] -= morpho.loc[eId, 'offset']
		else:
			#Use the ensemble timeline for static images
			self.df = pd.DataFrame()
			folder = os.path.join(atlas_dir, genotype, label, 'ensemble')
			self.df['time'] = np.load(os.path.join(folder, 't.npy'))
			self.df['embryoID'] = -1
			self.df['folder'] = folder
			self.df['eIdx'] = self.df.index

		if tmin is not None:
			self.df = self.df[self.df.time >= tmin].reset_index(drop=True)
		if tmax is not None:
			self.df = self.df[self.df.time <= tmax].reset_index(drop=True)

		self.values = {}

		folders = self.df.folder.unique()
		for folder in tqdm(folders):
			embryo_ID = os.path.basename(folder)
			try:
				embryo_ID = int(embryo_ID)
			except: #Label ensembles as -1
				embryo_ID = -1
			self.values[embryo_ID] = np.load(os.path.join(folder, filename+'.npy'), mmap_mode='r')

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
				 live_key='v',
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

	def ensemble_key(self, key, time):
		df = self.df[self.df.key == key]
		nearest = df[(df.time - time).abs() < 1]
		#If there are not enough within 1 minute, select the nearest self.ensemble
		if len(nearest) < self.ensemble:
			nearest = df.iloc[(df.time - time).abs().argsort()[:self.ensemble]]
		else: #Otherwise, select self.ensemble random rows
			nearest = nearest.sample(self.ensemble)

		frame = []
		for i, row in nearest.iterrows():
			data = self.values[row.embryoID][key][row.eIdx]
			if self.transforms[row.dataset_idx] is not None:
				data = self.transforms[row.dataset_idx](data)
			frame.append(data)

		if torch.is_tensor(frame[0]):
			frame = torch.mean(torch.stack(frame), dim=0)
		elif isinstance(frame[0], np.ndarray):
			frame = np.mean(np.stack(frame), axis=0)

		return frame
		
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

class TrajectoryDataset(JointDataset):
	'''
	Return a time sequence of multiple fields
	'''
	def __init__(self, *args, seq_sigma=3, **kwargs):
		super(TrajectoryDataset, self).__init__(*args, **kwargs)
		self.seq_sigma = seq_sigma

		df = pd.DataFrame()
		for eId in self.df.embryoID.unique():
			sub = self.df[self.df.embryoID == eId].copy()
			sub['max_len'] = np.max(sub.eIdx) - sub.eIdx
			df = df.append(sub, ignore_index=True)

		#Re-compute merged index
		mask = (df.max_len > 1) & (df.key == self.live_key)
		df.loc[mask, 'sequence_index'] = df[mask].groupby(['embryoID', 'time']).ngroup()
		df.loc[~mask, 'sequence_index'] = -1
		df.sequence_index = df.sequence_index.astype(int)
		self.df = df
	
	def __len__(self):
		return self.df.sequence_index.max()+1
	
	def __getitem__(self, idx):
		row = self.df[self.df.sequence_index == idx].iloc[0]
		seq_len = 2 + np.rint(np.abs(np.random.normal(scale=self.seq_sigma)))
		seq_len = min(int(seq_len), row.max_len)
		seq_len = min(seq_len, 5*self.seq_sigma)

		eId = row.embryoID
		times = np.arange(row.time, row.time+seq_len).astype(int)

		merged_idxs = np.arange(row.merged_index,
								row.merged_index+seq_len).astype(int)

		sample = {}

		for merged_index in merged_idxs:
			try:
				si = super(TrajectoryDataset, self).__getitem__(merged_index)
			except:
				print('Error fetching %d - merged indices: ' % idx, merged_index)
				print(row)
				print(seq_len)
				print(merged_idxs)
				print(eId, times)
				return
			for key in si:
				if not key in sample:
					sample[key] = []
				sample[key].append(si[key])

		for key in sample:
			if torch.is_tensor(sample[key][0]):
				sample[key] = torch.stack(sample[key], dim=0)
			elif isinstance(sample[key][0], np.ndarray):
				sample[key] = np.stack(sample[key])
			else:
				try:
					sample[key] = torch.tensor(sample[key])
				except:
					pass
				
		return sample

	def collate_fn(self, batch0):
		batch = {}
		for key in batch0[0]:
			batch[key] = [b[key] for b in batch0]
			batch['lengths'] = [len(bk) for bk in batch[key]]
			batch[key] = torch.nn.utils.rnn.pad_sequence(batch[key], batch_first=True, padding_value=0)

		return batch
