import numpy as np
import h5py
import os

import sys

basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))
import warnings
from dataset import *
from utils.pca_translation_utils import *

import pandas as pd
import pickle as pk
from tqdm import tqdm

from sklearn.decomposition import PCA
import pysindy as ps
from scipy.io import loadmat


def unpca_embryo_data(folder, embryoID, base, threshold=0.95):
	'''
	Get the PCA data for a given embryoID
	base - velocity, tensor, etc. tells us what PCA to look for
	return the inverse_transformed PCA data keeping only terms up to a given explained variance
	'''
	#Check if PCA exists for this data
	if not os.path.exists(os.path.join(folder, '%s_PCA.pkl' % base)):
		print('Building PCA for this dataset')
		label = os.path.basename(folder)
		genotype = os.path.basename(folder[:folder.index(label)-1])
		dataset = AtlasDataset(genotype, label, '%s2D' % base, transform=Reshape2DField())
		model, df = get_pca_results(dataset)
	else:
		print('Found PCA for this dataset!')
		model = pk.load(open(os.path.join(folder, '%s_PCA.pkl' % base), 'rb'))
		df = pd.read_csv(os.path.join(folder, '%s_PCA.csv' % base))
	df = df[df.embryoID == embryoID]
	keep = np.cumsum(model.explained_variance_ratio_) <= threshold
	params = df.filter(like='param', axis=1).values[:, keep]

	return unpca(params, model, keep)

def get_derivative_tensors(x, order=2):
	'''
	Return derivative tensors up to specified order
	Return shape is [..., Y, X, I, J, K, ...]
		Assumes the last two axes of x are spatial (Y, X)
		Places the derivative index axes at the end
	'''
	geometry = loadmat('/project/vitelli/jonathan/REDO_fruitfly/flydrive.synology.me/minimalData/vitelli_sharing/pixel_coordinates.mat')
	XX, YY = geometry['XX'][0, :], geometry['YY'][:, 0]

	diffY = ps.SmoothedFiniteDifference(d=1, axis=-2)
	diffX = ps.SmoothedFiniteDifference(d=1, axis=-1)
	diffY.smoother_kws['axis'] = -2
	diffX.smoother_kws['axis'] = -1

	xi = x.copy()

	d = []
	for i in range(1, order+1):
		diffY.smoother_kws['axis'] = -(2+i)
		diffX.smoother_kws['axis'] = -(1+i)
		xi = np.stack([
			diffY._differentiate(xi, YY),
			diffX._differentiate(xi, XX),
		], axis=-1)
		d.append(xi.copy())

	return d

def validate_key_and_derivatives(x, group, key, order=2):
	'''
	Ensure that key and its derivatives up to specified order are present in the dataset
	Compute the derivatives if they are not
	'''
	if not key in group:
		group.create_dataset(key, data=x)
	
	flag = False
	dx = []
	for i in range(order):
		name = 'D%d %s' % (i+1, key)
		if name in group:
			dx.append(group[name])
		else:
			flag = True
	
	if flag:
		print('Computing derivatives!')
		dx = get_derivative_tensors(x, order=order)
		for i in range(order):
			name = 'D%d %s' % (i+1, key)
			if name in group:
				del group[name]
			group.create_dataset(name, data=dx[i])
			
	return dx

def write_library_to_dataset(lib, glib, attrs=None):
	'''
	Write library elements to the dataset if they don't already exist
	'''
	for k in lib:
		if not k in glib:
			glib.create_dataset(k, data=lib[k])
		if attrs is not None:
			glib[k].attrs.update(attrs[k])
	

'''
Tensor libraries for anisotropy fields
'''

def s2t_terms(x, group, key='Rnt'):
	'''
	Compute terms mapping scalars to symmetric tensors
	'''
	d1_x, d2_x = validate_key_and_derivatives(x, group, key, order=2)
	lib = {}
	attrs = {}
	
	lib['grad(grad(%s))' % key] = np.einsum('tyxij->tijyx', d2_x)
	attrs['grad(grad(%s))' % key] = {key: 1, 'space': 2}
	lib['%s grad(grad(%s))' % (key, key)] = np.einsum('tyx,tyxij->tijyx', x, d2_x)
	attrs['%s grad(grad(%s))' % (key, key)] = {key: 2, 'space': 2}
	lib['grad(%s)grad(%s)' % (key, key)] = np.einsum('tyxi,tyxj->tijyx', d1_x, d1_x)
	attrs['grad(%s)grad(%s)' % (key, key)] = {key: 2, 'space': 2}

	write_library_to_dataset(lib, group.require_group('tensor_library'), attrs)


def t2t_terms(x, group, key='m'):
	'''
	Compute terms mapping tensors to symmetric tensors
	'''
	d1_x, d2_x = validate_key_and_derivatives(x, group, key, order=2)
	
	lib = {}
	attrs = {}

	lib[key] = x
	attrs[key] = {key: 1, 'space': 0}

	lib['%s Tr(%s)' % (key, key)] = np.einsum('tkkyx,tijyx->tijyx', x, x)
	attrs['%s Tr(%s)' % (key, key)] = {key: 2, 'space': 0}
	
	lib['grad^2 %s' % key] = np.einsum('tijyxkk->tijyx', d2_x) # 2. d_{kk} x_{ij}
	attrs['grad^2 %s' % key] = {key: 1, 'space': 2}
	lib['grad(div(%s))' % key] = np.einsum('tikyxkj->tijyx', d2_x) # 3. d_{jk} x_{ik}
	attrs['grad(div(%s))' % key] = {key: 1, 'space': 2}
	lib['grad(grad(Tr(%s))' % key] = np.einsum('tkkyxij->tijyx', d2_x) # 4. d_{ij} x_{kk}
	attrs['grad(grad(Tr(%s))' % key] = {key: 1, 'space': 2}
	lib['Tr(%s) grad^2 %s' % (key, key)] = np.einsum('tijyxkk,tllyx->tijyx', d2_x, x) # 5. x_{ll} d_{kk} x_{ij}
	attrs['Tr(%s) grad^2 %s' % (key, key)] = {key: 2, 'space': 2}
	lib['Tr(%s) grad(div(%s))' % (key, key)] = np.einsum('tikyxkj,tllyx->tijyx', d2_x, x) # 6. x_{ll} d_{jk} x_{ik}
	attrs['Tr(%s) grad(div(%s))' % (key, key)] = {key: 2, 'space': 2}
	lib['Tr(%s) grad(grad(Tr(%s)))' % (key, key)] = np.einsum('tijyxkk,tllyx->tijyx', d2_x, x) # 7. x_{ll} d_{ij} x_{kk}
	attrs['Tr(%s) grad(grad(Tr(%s)))' % (key, key)] = {key: 2, 'space': 2}
	
	write_library_to_dataset(lib, group.require_group('tensor_library'), attrs)

def v2t_terms(u, group):
	'''
	Compute terms mapping vectors to symmetric tensors
	'''
	d1_u = validate_key_and_derivatives(u, group, 'v', 1)[0]

	lib = {}
	attrs = {}
	Eij = 0.5 * (np.einsum('tiyxj->tijyx', d1_u) + np.einsum('tjyxi->tijyx', d1_u))
	Oij = 0.5 * (np.einsum('tiyxj->tijyx', d1_u) - np.einsum('tjyxi->tijyx', d1_u))
	
	lib['vv'] = np.einsum('tiyx,tjyx->tijyx', u, u)
	attrs['vv'] = {'v': 2, 'space': 0}
	lib['E'] = Eij
	attrs['E'] = {'v': 1, 'space': 1}
	lib['O'] = Oij
	attrs['O'] = {'v': 1, 'space': 1}
	
	write_library_to_dataset(lib, group.require_group('tensor_library'), attrs)

def vt2t_terms(group, key='m'):
	'''
	Compute terms mapping vectors and tensors to symmetric tensors
	'''
	glib = group.require_group('tensor_library')
	x = group[key]
	u = group['v']
	
	lib = {}
	attrs = {}

	lib['vv Tr(%s)' % key] = np.einsum('tkkyx,tijyx->tijyx', x, glib['vv'])
	attrs['vv Tr(%s)' % key] = {key: 1, 'v': 2, 'space': 0}
	lib['v^2 %s' % key] = np.einsum('tkkyx,tijyx->tijyx', glib['vv'], x)
	attrs['v^2 %s' % key] = {key: 1, 'v': 2, 'space': 0}
	
	vvx = np.einsum('tikyx,tkjyx->tijyx', glib['vv'], x)
	xvv = np.einsum('tikyx,tkjyx->tijyx', x, glib['vv'])
	lib['(vv%s + %svv)' % (key, key)] = vvx + xvv
	attrs['(vv%s + %svv)' % (key, key)] = {key: 1, 'v': 2, 'space': 0}
	
	lib['Tr(%s) E' % key] = np.einsum('tkkyx,tijyx->tijyx', x, glib['E'])
	attrs['Tr(%s) E' % key] = {key: 1, 'v': 1, 'space': 1}
	
	xE = np.einsum('tikyx,tkjyx->tijyx', x, glib['E'])
	Ex = np.einsum('tikyx,tkjyx->tijyx', glib['E'], x)
	lib['(%sE + E%s)' % (key, key)] = xE + Ex
	attrs['(%sE + E%s)' % (key, key)] = {key: 1, 'v': 1, 'space': 1}
	
	xO = np.einsum('tikyx,tkjyx->tijyx', x, glib['O'])
	Ox = np.einsum('tikyx,tkjyx->tijyx', glib['O'], x)
	lib['(%sO - O%s)' % (key, key)] = xO - Ox
	attrs['(%sO - O%s)' % (key, key)] = {key: 1, 'v': 1, 'space': 1}
	
	
	lib['%s div v' % key] = np.einsum('tijyx,tkyxk->tijyx', x, group['D1 v'])
	attrs['%s div v' % key] = {key: 1, 'v': 1, 'space': 1}

	lib['v dot grad %s' % key] = np.einsum('tkyx,tijyxk->tijyx', u, group['D1 %s' % key])
	attrs['v dot grad %s' % key] = {key: 1, 'v': 1, 'space': 1}
	
	lib['%s (%s dot grad v)' % (key, key)] = np.einsum('tijyx,tklyx,tkyxl->tijyx', x, x, group['D1 v'])
	attrs['%s (%s dot grad v)' % (key, key)] = {key: 2, 'v': 1, 'space': 1}
	
	write_library_to_dataset(lib, group.require_group('tensor_library'), attrs)
	
def veltensor2tensor_library(folder, embryoID, group, key='m',
							 pca=True, t_threshold=0.95, v_threshold=0.9):
	'''
	Build a mixed library of velocities and tensors
	'''
	if pca:
		try:
			T = unpca_embryo_data(folder, embryoID, 'tensor', t_threshold)
			V = unpca_embryo_data(folder, embryoID, 'velocity', v_threshold)
		except Exception as e:
			print(e)
			return
	else:
		T = np.load(os.path.join(folder, embryoID, 'tensor2D.npy'), mmap_mode='r')
		V = np.load(os.path.join(folder, embryoID, 'velocity2D.npy'), mmap_mode='r')

	T = T.reshape([T.shape[0], 2, 2, *T.shape[-2:]])

	t2t_terms(T, group, key=key)
	v2t_terms(V, group)
	vt2t_terms(group, key=key)

def scalar2tensor_library(folder, embryoID, group, key='Rnt', 
						  pca=True, s_threshold=0.95):
	'''
	Build a library of just scalar information
	'''
	if pca:
		S = unpca_embryo_data(folder, embryoID, 'raw', s_threshold)
	else:
		S = np.load(os.path.join(folder, embryoID, 'raw2D.npy'), mmap_mode='r')
	
	s2t_terms(S, group, key=key)

'''
Scalar libraries for intensity fields
'''

def s2s_terms(x, group, key='Rnt'):
	d1_x, d2_x = validate_key_and_derivatives(x, group, key, order=2)
	lib = {}
	attrs = {}

	feat = key
	lib[feat] = x
	attrs[feat] = {key: 1, 'space': 0}

	feat = '%s^2' % key
	lib[feat] = x**2
	attrs[feat] = {key: 2, 'space': 0}
	
	feat = 'grad(%s)^2' % key
	lib[feat] = np.einsum('tyxi,tyxi->tyx', d1_x, d1_x)
	attrs[feat] = {key: 2, 'space': 2}

	feat = 'grad^2 %s' % key
	lib[feat] = np.einsum('tyxii->tyx', d2_x)
	attrs[feat] = {key: 1, 'space': 2}
	
	feat = '%s grad^2 %s' % (key, key)
	lib[feat] = np.einsum('tyx,tyxii->tyx', x, d2_x)
	attrs[feat] = {key: 2, 'space': 2}

	write_library_to_dataset(lib, group.require_group('scalar_library'), attrs)

def t2s_terms(x, group, key='m'):
	d1_x, d2_x = validate_key_and_derivatives(x, group, key, order=2)
	lib = {}
	attrs = {}
	
	feat = 'Tr(%s)' % key
	lib[feat] = np.einsum('tiiyx->tyx', x)
	attrs[feat] = {key: 1, 'space': 0}
	
	feat = 'Tr(%s)^2' % key
	lib[feat] = np.einsum('tiiyx->tyx', x)**2
	attrs[feat] = {key: 2, 'space': 0}
	
	feat = 'grad(grad(%s))' % key
	lib[feat] = np.einsum('tijyxij->tyx', d2_x)
	attrs[feat] = {key: 1, 'space': 2}
	
	feat = 'grad^2 Tr(%s)' % key
	lib[feat] = np.einsum('tiiyxjj->tyx', d2_x)
	attrs[feat] = {key: 1, 'space': 2}
	
	feat = 'div(%s) div(%s)' % (key, key)
	lib[feat] = np.einsum('tijyxj,tikyxk->tyx', d1_x, d1_x)
	attrs[feat] = {key: 2, 'space': 2}
	
	feat = 'Tr(%s) grad^2 Tr(%s)' % (key, key)
	lib[feat] = np.einsum('tiiyx,tjjyxkk->tyx', x, d2_x)
	attrs[feat] = {key: 2, 'space': 2}
	
	feat = 'grad(Tr(%s))^2' % key
	lib[feat] = np.einsum('tiiyxj,tkkyxj->tyx', d1_x, d1_x)
	attrs[feat] = {key: 2, 'space': 2}
	
	write_library_to_dataset(lib, group.require_group('scalar_library'), attrs)

def v2s_terms(x, group, key='v'):
	'''
	Compute terms mapping vectors to symmetric tensors
	'''
	d1_x = validate_key_and_derivatives(x, group, key, 1)[0]

	lib = {}
	attrs = {}
	
	feat = '%s^2' % key
	lib[feat] = np.einsum('tiyx,tiyx->tyx', x, x)
	attrs[feat] = {key: 2, 'space': 0}

	feat = 'div %s' % key
	lib[feat] = np.einsum('tiyxi->tyx', d1_x)
	attrs[feat] = {key: 1, 'space': 1}

	feat = '(div %s)^2' % key
	lib[feat] = np.einsum('tiyxi,tjyxj->tyx', d1_x, d1_x)
	attrs[feat] = {key: 2, 'space': 2}

	feat = '(grad %s)^2' % key
	lib[feat] = np.einsum('tiyxj,tiyxj->tyx', d1_x, d1_x)
	attrs[feat] = {key: 2, 'space': 2}

	write_library_to_dataset(lib, group.require_group('scalar_library'), attrs)

def veltensor2scalar_library(folder, embryoID, group, key='m',
							 pca=True, t_threshold=0.95, v_threshold=0.9):
	'''
	Build a mixed library of velocities and tensors
	'''
	if pca:
		try:
			T = unpca_embryo_data(folder, embryoID, 'tensor', t_threshold)
			V = unpca_embryo_data(folder, embryoID, 'velocity', v_threshold)
		except Exception as e:
			print(e)
			return
	else:
		T = np.load(os.path.join(folder, embryoID, 'tensor2D.npy'), mmap_mode='r')
		V = np.load(os.path.join(folder, embryoID, 'velocity2D.npy'), mmap_mode='r')

	T = T.reshape([T.shape[0], 2, 2, *T.shape[-2:]])

	t2s_terms(T, group, key=key)
	v2s_terms(V, group)

def scalar2scalar_library(folder, embryoID, group, key='Rnt', 
						  pca=True, s_threshold=0.95):
	'''
	Build a library of just scalar information
	'''
	if pca:
		S = unpca_embryo_data(folder, embryoID, 'raw', s_threshold)
	else:
		S = np.load(os.path.join(folder, embryoID, 'raw2D.npy'), mmap_mode='r')
	
	s2s_terms(S, group, key=key)

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
		afl_kwargs['pca'] = False
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
	datadir = '/project/vitelli/jonathan/REDO_fruitfly/MLData'
	'''
	dirs = {
		'c': ['WT', 'ECad-GFP'],
		'm': ['WT', 'sqh-mCherry'],
	}
	for key in dirs:
		build_dynamic_derivative_library(
			os.path.join(datadir, *dirs[key]),
			'library_PCA.h5',
			veltensor2tensor_library,
			key=key)
		build_dynamic_derivative_library(
			os.path.join(datadir, *dirs[key]),
			'library_PCA.h5',
			veltensor2scalar_library,
			key=key)
	'''
	dirs = {
		#'Rnt': ['WT', 'Runt'],
		#'Eve': ['WT', 'Even_Skipped'],
		#'Ftz': ['WT', 'Fushi_Tarazu'],
		#'Hry': ['WT', 'Hairy'],
		#'Slp': ['WT', 'Sloppy_Paired'],
		#'Prd': ['WT', 'Paired'],
		'Trt': ['WT', 'Tartan'],
	}
	for key in dirs:
		build_ensemble_derivative_library(
			os.path.join(datadir, *dirs[key]),
			'library_PCA.h5',
			scalar2scalar_library,
			key=key)
		build_ensemble_derivative_library(
			os.path.join(datadir, *dirs[key]),
			'library_PCA.h5',
			scalar2tensor_library,
			key=key)
