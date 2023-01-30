import pandas as pd
import numpy as np
import h5py
import os
import sys
basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))
from tqdm import tqdm

from tensor_library import scalar2tensor_library, veltensor2tensor_library
from scalar_library import scalar2scalar_library, veltensor2scalar_library
from symmetric_tensor_library import veltensor2symtensor_library, scalar2symtensor_library

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
		**afl_kwargs):
	'''
	Build a library from dynamic information (live-imaged embryos)
	'''
	index = pd.read_csv(os.path.join(directory, 'dynamic_index.csv'))
	with h5py.File(os.path.join(directory, filename), 'a') as h5f:
		for embryoID in tqdm(index.embryoID.unique()):
			print(embryoID)
			#Write embryo info to h5f
			group = h5f.require_group(str(embryoID))
			if not 'time' in group:
				group.create_dataset('time', data=index[index.embryoID==embryoID].time.values)
			write_library(directory, embryoID, group, **afl_kwargs)
	build_ensemble_derivative_library(directory, filename, write_library, **afl_kwargs)


if __name__=='__main__':
	datadir = '/project/vitelli/jonathan/REDO_fruitfly/src/data'

	dirs = {
		'c': ['WT', 'ECad-GFP'],
		'm': ['WT', 'sqh-mCherry'],
	}

	libfuncs = [
		#veltensor2tensor_library,
		#veltensor2scalar_library,
		veltensor2symtensor_library,
	]
	for key in dirs:
		for libfunc in libfuncs:
			build_dynamic_derivative_library(
				os.path.join(datadir, *dirs[key]),
				'library_PCA.h5',
				libfunc,
				key=key)
	dirs = {
		'Rnt': ['WT', 'Runt'],
		'Eve': ['WT', 'Even_Skipped'],
		'Ftz': ['WT', 'Fushi_Tarazu'],
		'Hry': ['WT', 'Hairy'],
		'Slp': ['WT', 'Sloppy_Paired'],
		'Prd': ['WT', 'Paired'],
		'Trt': ['WT', 'Tartan'],
	}
	libfuncs = [
		#scalar2scalar_library,
		scalar2tensor_library,
		scalar2symtensor_library,
	]
	for key in dirs:
		for libfunc in libfuncs:
			build_ensemble_derivative_library(
				os.path.join(datadir, *dirs[key]),
				'library_PCA.h5',
				libfunc,
				key=key)
