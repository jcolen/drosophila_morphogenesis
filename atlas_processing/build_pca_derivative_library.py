import numpy as np
import h5py
import os

import sys

basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))
from dataset import *

import pandas as pd
import pickle as pk
from tqdm import tqdm

from sklearn.decomposition import PCA
import pysindy as ps
from scipy.io import loadmat


def residual(u, v):
	umag = np.linalg.norm(u, axis=-3)												  
	vmag = np.linalg.norm(v, axis=-3)												  

	uavg = np.sqrt((umag**2).mean(axis=(-2, -1), keepdims=True))					
	vavg = np.sqrt((vmag**2).mean(axis=(-2, -1), keepdims=True))					

	res = uavg**2 * vmag**2 + vavg**2 * umag**2 - 2 * uavg * vavg * np.einsum('...ijk,...ijk->...jk', u, v)
	res /= 2 * vavg**2 * uavg**2														
	return res 


'''
Learn a PCA Model on an AtlasDataset object
'''
def build_PCA_model(dataset, n_components=16):
	df = dataset.df.drop(['folder', 'tiff'], axis=1).reset_index()
	test_size = len(df) * 2 // 5
	test_idx = np.random.choice(df.index, test_size, replace=False)

	df['set'] = 'train'
	df.loc[test_idx, 'set'] = 'test'
	df['t'] = df['time']
	df['time'] = df['time'].astype(int)

	y0 = []
	for e in df.embryoID.unique():
		e_data = dataset.values[e]
		t = e_data.shape[0]
		h, w = e_data.shape[-2:]
		y0.append(e_data.reshape([t, -1, h, w]))
	y0 = np.concatenate(y0, axis=0)

	model = PCA(n_components=n_components, whiten=True)
	
	train = y0[df[df.set == 'train'].index]
	model.fit(train.reshape([train.shape[0], -1]))

	params = model.transform(y0.reshape([y0.shape[0], -1]))
	y = model.inverse_transform(params).reshape(y0.shape)

	df['res'] = residual(y, y0).mean(axis=(-1, -2))
	df['mag'] = np.linalg.norm(y, axis=1).mean(axis=(-1, -2))
	df = pd.concat([df, pd.DataFrame(params).add_prefix('param')], axis=1)

	return model, df

'''
Undo PCA using a subset of learned components
'''
def unpca(d, model, keep):
	if d.shape[1] != model.n_components_:
		di = np.zeros([d.shape[0], keep.shape[0]])
		di[:, keep] = d
		d = di
	d = model.inverse_transform(d)
	d = d.reshape([d.shape[0], -1, 236, 200])
	return d

def unpca_embryo_data(folder, base, embryoID, threshold=0.95):
	#Check if PCA exists for this data
	if not os.path.exists(os.path.join(folder, '%s_PCA.pkl' % base)):
		label = os.path.basename(folder)
		genotype = os.path.basename(folder[:folder.index(label)-1])
		dataset = AtlasDataset(genotype, label, '%s2D' % base, transform=Reshape2DField())
		model, df = build_PCA_model(dataset)
		
		pk.dump(model, open(os.path.join(folder, '%s_PCA.pkl' % base), 'wb'))
		df.to_csv(os.path.join(folder, '%s_PCA.csv' % base))
	else:
		model = pk.load(open(os.path.join(folder, '%s_PCA.pkl' % base), 'rb'))
		df = pd.read_csv(os.path.join(folder, '%s_PCA.csv' % base))
	df = df[df.embryoID == embryoID]
	keep = np.cumsum(model.explained_variance_ratio_) <= threshold
	params = df.filter(like='param', axis=1).values[:, keep]

	return unpca(params, model, keep)


'''
Step 4. Compute derivatives and library terms
Terms up to order (N, D) in nonlinearity and derivatives
m (2, 1)
v (1, 1)
c (2, 2)
'''
def pure_tensor_terms(x, d1_x, d2_x=None, key='m'):
	lib = {}
	lib[key] = x
	lib['%s Tr(%s)' % (key, key)] = np.einsum('tkkyx,tijyx->tijyx', x, x)
	
	if d2_x is not None:
		lib['grad^2 %s' % key] = np.einsum('tijyxkk->tijyx', d2_x) # 2. d_{kk} x_{ij}
		lib['grad(div(%s))' % key] = np.einsum('tikyxkj->tijyx', d2_x) # 3. d_{jk} x_{ik}
		lib['grad(grad(Tr(%s))' % key] = np.einsum('tkkyxij->tijyx', d2_x) # 4. d_{ij} x_{kk}
		lib['Tr(%s) grad^2 %s' % (key, key)] = np.einsum('tijyxkk,tllyx->tijyx', d2_x, x) # 5. x_{ll} d_{kk} x_{ij}
		lib['Tr(%s) grad(div(%s))' % (key, key)] = np.einsum('tikyxkj,tllyx->tijyx', d2_x, x) # 6. x_{ll} d_{jk} x_{ik}
		lib['Tr(%s) grad(grad(Tr(%s)))' % (key, key)] = np.einsum('tijyxkk,tllyx->tijyx', d2_x, x) # 7. x_{ll} d_{ij} x_{kk}
		pass
		
	return lib

def pure_vector_terms(u, d1_u):
	lib = {}
	Eij = 0.5 * (np.einsum('tiyxj->tijyx', d1_u) + np.einsum('tjyxi->tijyx', d1_u))
	Oij = 0.5 * (np.einsum('tiyxj->tijyx', d1_u) - np.einsum('tjyxi->tijyx', d1_u))
	
	lib['vv'] = np.einsum('tiyx,tjyx->tijyx', u, u)
	lib['E'] = Eij
	lib['O'] = Oij

	return lib

def vector_tensor_terms(x, u, d1_x, d1_u, past_lib, key='m'):
	lib = {}
	lib['vv Tr(%s)' % key] = np.einsum('tkkyx,tijyx->tijyx', x, past_lib['vv'])
	lib['v^2 %s' % key] = np.einsum('tkkyx,tijyx->tijyx', past_lib['vv'], x)
	
	vvx = np.einsum('tikyx,tkjyx->tijyx', past_lib['vv'], x)
	xvv = np.einsum('tikyx,tkjyx->tijyx', x, past_lib['vv'])
	lib['(vv%s + %svv)' % (key, key)] = vvx + xvv
	
	lib['Tr(%s) E' % key] = np.einsum('tkkyx,tijyx->tijyx', x, past_lib['E'])
	
	xE = np.einsum('tikyx,tkjyx->tijyx', x, past_lib['E'])
	Ex = np.einsum('tikyx,tkjyx->tijyx', past_lib['E'], x)
	lib['(%sE + E%s)' % (key, key)] = xE + Ex
	
	xO = np.einsum('tikyx,tkjyx->tijyx', x, past_lib['O'])
	Ox = np.einsum('tikyx,tkjyx->tijyx', past_lib['O'], x)
	lib['(%sO - O%s)' % (key, key)] = xO - Ox
	
	
	lib['%s div v' % key] = np.einsum('tijyx,tkyxk->tijyx', x, d1_u)
	lib['v dot grad %s' % key] = np.einsum('tkyx,tijyxk->tijyx', u, d1_x)
	lib['%s (%s dot grad v)' % (key, key)] = np.einsum('tijyx,tklyx,tkyxl->tijyx', x, x, d1_u)
	
	return lib

def write_tensor_library(folder, embryoID, group, t_threshold=0.95, v_threshold=0.9, key='m'):
	T = unpca_embryo_data(folder, 'tensor', embryoID, t_threshold)
	T = T.reshape([T.shape[0], 2, 2, *T.shape[-2:]])
	V = unpca_embryo_data(folder, 'velocity', embryoID, v_threshold)

	geometry = loadmat('/project/vitelli/jonathan/REDO_fruitfly/flydrive.synology.me/minimalData/vitelli_sharing/pixel_coordinates.mat')
	XX, YY = geometry['XX'][0, :], geometry['YY'][:, 0]

	diffY = ps.SmoothedFiniteDifference(d=1, axis=-2)
	diffX = ps.SmoothedFiniteDifference(d=1, axis=-1)

	d1_T = np.stack([
		diffY._differentiate(T, YY),
		diffX._differentiate(T, XX),
	], axis=-1)

	d2_T = np.stack([
		diffY._differentiate(d1_T, YY),
		diffX._differentiate(d1_T, XX),
	], axis=-1)

	diffY = ps.SmoothedFiniteDifference(d=1, axis=-2)
	diffX = ps.SmoothedFiniteDifference(d=1, axis=-1)

	d1_V = np.stack([
		diffY._differentiate(V, YY),
		diffX._differentiate(V, XX),
	], axis=-1)

	lib = {
		**pure_tensor_terms(T, d1_T, d2_T, key=key),
		**pure_vector_terms(V, d1_V)
	}
	lib = {**lib, **vector_tensor_terms(T, V, d1_T, d1_V, lib, key=key)}

	glib = group.require_group('tensor_library')
	for k in lib:
		if not k in glib:
			glib.create_dataset(k, data=lib[k])

	if not key in group: 
		group.create_dataset(key, data=T)
		group.create_dataset('D1 %s' % key, data=d1_T)
		group.create_dataset('D2 %s' % key, data=d2_T)
	if not 'v' in group: 
		group.create_dataset('v', data=V)
		group.create_dataset('D1 v', data=d1_V)

def build_PCA_library(directory,
					 filename,
					 write_library,
					 fast_dev_run=False,
					 **afl_kwargs):
	index = pd.read_csv(os.path.join(directory, 'dynamic_index.csv'))
	with h5py.File(os.path.join(directory, filename), 'a') as h5f:
		for embryoID in tqdm(index.embryoID.unique()):
			#Write embryo info to h5f
			group = h5f.require_group(str(embryoID))
			if not 'time' in group:
				group.create_dataset('time', data=index[index.embryoID==embryoID].time.values)
			write_library(directory, embryoID, group)
			
			if fast_dev_run:
				return

if __name__=='__main__':
	datadir = '/project/vitelli/jonathan/REDO_fruitfly/MLData'
	transform = Reshape2DField()
	dirs = {
		'c': ['WT', 'ECad-GFP'],
		'm': ['WT', 'sqh-mCherry'],
	}
	for key in dirs:
		build_PCA_library(os.path.join(datadir, *dirs[key]),
						  'library_PCA.h5',
						  write_tensor_library,
						  fast_dev_run=False,
						  tensor_label=key)
