'''
Dataset objects for pytorch training on Dynamic Atlas
'''

import os
import numpy as np
import pandas as pd
import torch

from scipy.ndimage import gaussian_filter, gaussian_filter1d
from tqdm import tqdm

class BaseTransformer():
	'''
	Base transformer class
	All transformations apply to the 'value' entry in the sample dict
	or the the sample itself if it's not a dict
	'''
	def transform(self, X):
		raise NotImpelentedError

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

class Smooth2D(BaseTransformer):
	'''
	Smooth field with a gaussian filter over space
	Stack everything along channel axis
	'''
	def __init__(self, sigma=3):
		self.sigma = sigma

	def transform(self, X):
		if len(X.shape) == 2:
			X = X[None]
		X = gaussian_filter(X, sigma=(0, self.sigma, self.sigma))
		#X = np.stack([gaussian_filter(X[c], sigma=self.sigma) for c in range(X.shape[0])])
		return X

class DorsalVentralSmooth(BaseTransformer):
	'''
	Smooth along the DV direction with periodic BCs
	'''
	def __init__(self, sigma=1):
		self.sigma = sigma

	def transform(self, X):
		X = gaussian_filter1d(X, sigma=self.sigma, mode='wrap', axis=-2)
		return X

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

class AnteriorPosteriorMask(BaseTransformer):
	'''
	Zero out field at the anterior and posterior poles
	'''
	def __init__(self, anterior=10, posterior=10):
		self.anterior = anterior
		self.posterior = posterior
	
	def transform(self, X):
		X[..., :self.anterior] = 0.
		if self.posterior > 0:
			X[..., -self.posterior:] = 0.
		return X

class ToTensor(BaseTransformer):
	'''
	Convert field to torch Tensor
	'''
	def transform(self, X):
		return torch.tensor(X, dtype=torch.float32)

def load_label_dataframe(path, drop_time=False, ensemble=False, tmin=None, tmax=None):
	'''
	Load dataframes and apply morphodynamic offsets
	'''
	try:
		df = pd.read_csv(os.path.join(path, 'dynamic_index.csv'))
	except FileNotFoundError:
		print('Dynamic index not found, loading static index')
		df = pd.read_csv(os.path.join(path, 'static_index.csv'))

	df = df.drop('tiff', axis=1)

	if drop_time:
		df.time = df.eIdx
	else:
		df = df.dropna(axis=0) #Drop unlabeled timepoints


	morpho = os.path.join(path, 'morphodynamic_offsets.csv')
	try:
		morpho = pd.read_csv(morpho, index_col='embryoID')
		for eId in df.embryoID.unique():
			df.loc[df.embryoID == eId, 'time'] -= morpho.loc[eId, 'offset']
	except FileNotFoundError:
		print('No morphodynamic offsets found')

	
	if ensemble and os.path.exists(os.path.join(path, 'ensemble')):
		t = np.load(os.path.join(path, 'ensemble', 't.npy'))
		df = df.append(pd.DataFrame({
			'folder': os.path.join(path, 'ensemble'),
			'embryoID': 'ensemble',
			'time': t,
			'eIdx': np.arange(len(t), dtype=int)
			}), ignore_index=True)

	if tmin is not None:
		df = df[df.time >= tmin].reset_index(drop=True)
	if tmax is not None:
		df = df[df.time <= tmax].reset_index(drop=True)

	return df

class AtlasDataset(torch.utils.data.Dataset):
	'''
	AtlasDataset is a wrapper around the dynamic Atlas
	'''
	def __init__(self,
				 genotype,
				 label,
				 filename,
				 atlas_dir='/Users/jcolen/Documents/drosophila_morphogenesis/flydrive/',
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
			df = pd.concat([df, dfi], ignore_index=True)
			#df = df.append(dfi, ignore_index=True)

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
	def __init__(self, *args, seq_sigma=3, transform=None,  **kwargs):
		super().__init__(*args, **kwargs)
		self.seq_sigma = seq_sigma
		self.transform = transform

		df = pd.DataFrame()
		for eId in self.df.embryoID.unique():
			sub = self.df[self.df.embryoID == eId].copy()
			sub['max_len'] = np.max(sub.eIdx) - sub.eIdx
			df = pd.concat([df, sub], ignore_index=True)
			#df = df.append(sub, ignore_index=True)

		#Re-compute merged index
		mask = (df.max_len > 1) & (df.key == self.live_key)
		df.loc[mask, 'sequence_index'] = df[mask].groupby(['embryoID', 'time']).ngroup()
		df.loc[~mask, 'sequence_index'] = -1

		df['train_mask'] = 'all'
		df['val_mask'] = 'all'
		df.sequence_index = df.sequence_index.astype(int)
		self.df = df

	def get_mask(self, key, size):
		mask = torch.zeros(size, dtype=bool)
		if key == 'all':
			mask[...] = 1
		elif key == 'left':
			mask[:size[0]//2] = 1
		elif key == 'right':
			mask[size[0]//2:] = 1
		elif key == 'anterior':
			mask[:, :size[1]//2] = 1
		elif key == 'posterior':
			mask[:, size[1]//2:] = 1

		return mask


	def __len__(self):
		return self.df.sequence_index.max()+1
	
	def __getitem__(self, idx):
		row = self.df[self.df.sequence_index == idx].iloc[0]

		#Randomly sample a sequence length
		seq_len = 2 + np.rint(np.abs(np.random.normal(scale=self.seq_sigma)))

		#Sequence length is at most 5 sigma and also at most the max_len for the row
		seq_len = min(int(seq_len), row.max_len)
		seq_len = min(seq_len, 5*self.seq_sigma)

		#These are the indices that we want the superclass to fetch
		merged_idxs = np.arange(row.merged_index,
								row.merged_index+seq_len).astype(int)

		sample = {}

		for merged_index in merged_idxs:
			try:
				si = super().__getitem__(merged_index)
			except Exception as e:
				print(f'Error fetching {idx} - merged index: {merged_index}')
				print(f'Error message: {e}')
				print(f'Attempting to fetch from embryo {row.embryoID}')
				print(f'Trying to get a sequence of length {seq_len}')
				print('Merged indices: ', merged_idxs)
				print('Times: ', np.arange(row.time, row.time+seq_len))
				print(row)
				return None

			for key in si:
				if not key in sample:
					sample[key] = []
				sample[key].append(si[key])

		for key in sample:
			if torch.is_tensor(sample[key][0]):
				sample[key] = torch.stack(sample[key])
			elif isinstance(sample[key][0], np.ndarray):
				sample[key] = np.stack(sample[key])
			else:
				try:
					sample[key] = torch.tensor(sample[key])
				except Exception:
					pass

		sample['train_mask'] = self.get_mask(row.train_mask, sample[self.live_key].shape[-2:])
		sample['val_mask'] = self.get_mask(row.val_mask, sample[self.live_key].shape[-2:])
				
		if self.transform is not None:
			sample = self.transform(sample)

		return sample

	def collate_fn(self, batch0):
		batch = {}
		for key in batch0[0]:
			batch[key] = [b[key] for b in batch0]
			if 'mask' in key:
				batch[key] = torch.stack(batch[key], dim=0) #B, H, W
			else:
				batch['lengths'] = [len(bk) for bk in batch[key]]
				batch[key] = torch.nn.utils.rnn.pad_sequence(batch[key], batch_first=True, padding_value=0)

		return batch
