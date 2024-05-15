import numpy as np
import pandas as pd
import h5py
import os
import glob
import warnings
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import sys
basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))

from library_generation.mesh_scalar_library import build_scalar_library
from library_generation.mesh_vector_library import build_vector_library
from library_generation.mesh_tensor_library import build_tensor_library
from utils.geometry.fenics_utils import FenicsGradient

mesh_grad = FenicsGradient().fit(None)

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
			group = h5f.require_group(str(embryoID))
			if not 'time' in group:
				group.create_dataset('time', data=index[index.embryoID==embryoID].time.values)
			write_library(directory, group, mesh_grad, **afl_kwargs)
		
		group = h5f.require_group('ensemble')
		if not 'time' in group:
			group.create_dataset('time', data=np.load(os.path.join(directory, 'ensemble', 't.npy')))
		write_library(directory, group, mesh_grad, **afl_kwargs)

datadir = '/project/vitelli/jonathan/REDO_fruitfly/src/Public'

sets = [
	('c', ['WT', 'ECad-GFP'], 'raw'),
	('v', ['WT', 'ECad-GFP'], 'velocity'),
	('v', ['Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP'], 'velocity'),
	('m_ij', ['Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP'], 'tensor'),
]

for key, path, base in sets:
	if base == 'tensor':
		libfunc = build_tensor_library
	elif base == 'velocity': 
		libfunc = build_vector_library
	else:
		libfunc = build_scalar_library

	build_dynamic_derivative_library(
		os.path.join(datadir, *path),
		'mesh_derivative_library.h5',
		libfunc,
		drop_times='Sqh-GFP' in path,
		key=key,
		base=base)

