import pandas as pd
import numpy as np
import h5py
import os
import sys
from tqdm import tqdm

from torchvision.transforms import Compose
from morphogenesis.dataset import *
from morphogenesis.decomposition.decomposition_utils import get_decomposition_model

from morphogenesis.library.derivative_library_utils import *

def build_ensemble_derivative_library(
		directory,
		filename,
		write_library,
		**afl_kwargs):
	'''
	Build a library from ensemble-averaged information
	'''
	with h5py.File(os.path.join(directory, filename), 'a') as h5f:
		#if 'ensemble' in h5f:
		#	del h5f['ensemble']
		group = h5f.require_group('ensemble')
		if not 'time' in group:
			group.create_dataset('time', data=np.load(os.path.join(directory, 'ensemble', 't.npy')))
		afl_kwargs['project'] = False
		write_library(directory, 'ensemble', group, **afl_kwargs)

def build_dynamic_derivative_library(
		directory,
		filename,
		write_library,
		drop_times=False,
		**afl_kwargs):
	'''
	Build a library from dynamic information (live-imaged embryos)
	'''
	index = pd.read_csv(os.path.join(directory, 'dynamic_index.csv'))
	if drop_times:
		index.time = index.eIdx
		print('Dropping times for eIdx')
	if os.path.exists(os.path.join(directory, 'morphodynamic_offsets.csv')):
		print('Adding morphodynamic offsets')
		morpho = pd.read_csv(os.path.join(directory, 'morphodynamic_offsets.csv'), index_col='embryoID')
		for eId in index.embryoID.unique():
				index.loc[index.embryoID == eId, 'time'] -= morpho.loc[eId, 'offset']
	
	with h5py.File(os.path.join(directory, filename), 'a') as h5f:
		for embryoID in tqdm(index.embryoID.unique()):
			#Write embryo info to h5f
			group = h5f.require_group(str(embryoID))
			if not 'time' in group:
				group.create_dataset('time', data=index[index.embryoID==embryoID].time.values)
			write_library(directory, embryoID, group, **afl_kwargs)
	build_ensemble_derivative_library(directory, filename, write_library, **afl_kwargs)

def build_derivative_library(dataset, library_function, key, 
							 keep_frac=0.95,
							 ensemble=True):
	path = os.path.join(dataset.path, 'derivative_library')
	if not os.path.exists(path):
		os.mkdir(path)

	model = get_decomposition_model(dataset)
	
	for embryoID in tqdm(dataset.df.embryoID.unique()):
		with h5py.File(f'{path}/{embryoID}.h5', 'a') as h5f:
			time = dataset.df[dataset.df.embryoID == embryoID].time.values
			eIdx = dataset.df[dataset.df.embryoID == embryoID].eIdx.values
			dv = np.load(f'{dataset.path}/{embryoID}/DV_coordinates.npy', mmap_mode='r')
			ap = np.load(f'{dataset.path}/{embryoID}/AP_coordinates.npy', mmap_mode='r')
			data = dataset.values[embryoID][eIdx]

			library_function(h5f, data, (time, dv, ap), key,
							 project=model, keep_frac=keep_frac)
	
	if ensemble:
		with h5py.File(f'{path}/ensemble.h5', 'a') as h5f:
			# Write ensemble library to dataset
			time = np.load(f'{dataset.path}/ensemble/t.npy')
			dv = np.load(f'{dataset.path}/ensemble/DV_coordinates.npy', mmap_mode='r')
			ap = np.load(f'{dataset.path}/ensemble/AP_coordinates.npy', mmap_mode='r')
			data = np.load(f'{dataset.path}/ensemble/{dataset.filename}.npy', mmap_mode='r')

			library_function(h5f, data, (time, dv, ap), key)


from morphogenesis.dataset import *

if __name__=='__main__':
	transform = Reshape2DField()

	print('eCadherin library')
	cad_dataset = AtlasDataset('WT', 'ECad-GFP', 'raw2D', transform=Compose([transform, Smooth2D(sigma=7)]), tmin=-15, tmax=45)
	cad_vel_dataset = AtlasDataset('WT', 'ECad-GFP', 'velocity2D', transform=transform, tmin=-15, tmax=45)

	build_derivative_library(cad_dataset, scalar_library, 'c')
	build_derivative_library(cad_vel_dataset, vector_library, 'v')

	print('Myosin library')
	sqh_dataset = AtlasDataset('Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP', 'tensor2D', transform=transform, drop_time=True, tmin=-15, tmax=45)
	sqh_vel_dataset = AtlasDataset('Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP', 'velocity2D', transform=transform, drop_time=True, tmin=-15, tmax=45)

	build_derivative_library(sqh_dataset, tensor_library, 'm_ij')
	build_derivative_library(sqh_vel_dataset, vector_library, 'v')

	print('Actin library')
	act_dataset = AtlasDataset('WT', 'Moesin-GFP', 'raw2D', transform=Compose([transform, LeftRightSymmetrize(), Smooth2D(sigma=7)]), tmin=-15, tmax=45)
	act_vel_dataset = AtlasDataset('WT', 'Moesin-GFP', 'velocity2D', transform=transform, tmin=-15, tmax=45)

	build_derivative_library(act_dataset, scalar_library, 'c')
	build_derivative_library(act_vel_dataset, vector_library, 'v')