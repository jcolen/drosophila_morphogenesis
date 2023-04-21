from time import sleep

import numpy as np
import pysindy as ps
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from .fly_sindy import FlySINDy
from ..geometry.geometry_utils import embryo_mesh

overleaf_feature_names = [
	'v dot grad m_ij',
	'[O, m_ij]', 
	'm_ij',
	'c m_ij',
	#'m_ij Tr(m_ij)',
	#'c m_ij Tr(m_ij)',
	'm_ij Tr(m_ij)^2',
	'c m_ij Tr(m_ij)^2',
	'm_ij Tr(E_full)',
	'c m_ij Tr(E_full)',
	'Static_DV Tr(m_ij)',
	'c Static_DV Tr(m_ij)',
]

def collect_decomposed_data(h5f, key, tmin, tmax, feature_names=None):
	'''
	Collect the PCA data from a given h5f library and return X, X_dot, and the feature names
	'''
	if not key in h5f['X_cpt']:
		return None, None, None

	if feature_names is None:
		feature_names = list(h5f['X_cpt'][key].keys())
		#feature_names = [fn for fn in feature_names if not 'E_active' in fn]
		#feature_names = [fn for fn in feature_names if not 'm_ij m_ij' in fn]
		feature_names = [fn for fn in feature_names if not 'Dorsal_Source' in fn]

	data = h5f['X_cpt'][key]
	#Pass 1 - get the proper time range
	for feature in feature_names:
		tmin = max(tmin, np.min(data[feature].attrs['t']))
		tmax = min(tmax, np.max(data[feature].attrs['t']))

	X = []
	#Pass 2 - collect points within that time range
	for feature in feature_names:
		t = data[feature].attrs['t']
		X.append(data[feature][np.logical_and(t >= tmin, t <= tmax), ...])
	X = np.stack(X, axis=-1)
	
	t = h5f['X_dot_cpt'][key].attrs['t']
	X_dot = h5f['X_dot_cpt'][key][np.logical_and(t >= tmin, t <= tmax), ...]

	return X, X_dot, feature_names

def collect_raw_data(h5f, key, tmin, tmax, feature_names=None, keep_frac=0.2):
	'''
	Collect the raw data from a given h5f library and return X, X_dot, and the feature names
	'''
	if not key in h5f['X_cpt']:
		return None, None, None

	if feature_names is None:
		feature_names = list(h5f['X_cpt'][key].keys())
		feature_names = [fn for fn in feature_names if not 'E_active' in fn]
		feature_names = [fn for fn in feature_names if not 'E_passive' in fn]
		feature_names = [fn for fn in feature_names if not 'm_ij m_ij' in fn]

	data = h5f['X_cpt'][key]
	#Pass 1 - get the proper time range
	for feature in feature_names:
		tmin = max(tmin, np.min(data[feature].attrs['t']))
		tmax = min(tmax, np.max(data[feature].attrs['t']))

	X = []
	data = h5f['X_raw']
	#Pass 2 - collect points within that time range
	for feature in feature_names:
		t = data[feature].attrs['t']
		X.append(data[feature][np.logical_and(t >= tmin, t <= tmax), ...])
	X = np.stack(X, axis=-1)
	
	t = h5f['X_dot'][key].attrs['t']
	X_dot = h5f['X_dot'][key][np.logical_and(t >= tmin, t <= tmax), ...]
	
	if keep_frac < 1:
		space_points = X.shape[-3] * X.shape[-2]
		keep_points = int(keep_frac * space_points)
		mask = np.zeros(space_points, dtype=bool)
		mask[np.random.choice(range(space_points), keep_points, replace=False)] = True
		mask = mask.reshape([X.shape[-3], X.shape[-2]])
		X = X[..., mask, :]
		X_dot = X_dot[..., mask, :]

	X = X.reshape([X.shape[0], -1, X.shape[-1]])
	X_dot = X_dot.reshape([X.shape[0], -1, 1])

	return X, X_dot, feature_names

def collect_mesh_data(h5f, key, tmin, tmax, feature_names=None):
	'''
	Collect the raw data from a given h5f library and return X, X_dot, and the feature names
	'''
	data = h5f['X_raw']
	if feature_names is None:
		feature_names = []
		for fn in data.keys():
			if data[fn].shape != data[key].shape:
				continue
			#if 'E_full' in fn: #Use active/passive decomposition
			#	continue
			if 'E_active' in fn or 'E_passive' in fn: 
				continue
			#if key in data[fn].attrs and data[fn].attrs[key] > 2:
			#	continue
			feature_names.append(fn)
		ordered = []
		for fn in feature_names:
			if fn in ordered or fn[0] == 'c':
				continue
			ordered.append(fn)
			if f'c {fn}' in feature_names:
				ordered.append(f'c {fn}')
			
		feature_names = ordered

	key1, key2 = f'X_raw', f'X_dot/{key}'
	#key1, key2 = f'X_tan/{key}', f'X_dot_tan/{key}'
	data = h5f[key1]
	#Pass 1 - get the proper time range
	for feature in feature_names:
		tmin = max(tmin, np.min(data[feature].attrs['t']))
		tmax = min(tmax, np.max(data[feature].attrs['t']))

	X = []
	#Pass 2 - collect points within that time range
	for feature in feature_names:
		t = data[feature].attrs['t']
		X.append(data[feature][np.logical_and(t >= tmin, t <= tmax), ...])
	X = np.stack(X, axis=-1)

	t = h5f[key2].attrs['t']
	X_dot = h5f[key2][np.logical_and(t >= tmin, t <= tmax), ...]

	#Keep only the mesh coordinates away from the poles
	'''
	z = embryo_mesh.coordinates()[:, 2] * 0.2619 #In microns
	zmax = 0.9 * np.ptp(z) / 2
	mask = np.abs(z) <= zmax

	X = X[..., mask, :]
	X_dot = X_dot[..., mask, :]
	'''	
	X = X.reshape([X.shape[0], -1, X.shape[-1]])
	X_dot = X_dot.reshape([X.shape[0], -1, 1])
	return X, X_dot, feature_names

def collect_data(h5f, key, tmin, tmax, 
				 collect_function=collect_mesh_data, 
				 feature_names=None):
	X, X_dot = [], []

	with tqdm(total=len(h5f.keys())) as pbar:
		pbar.set_description('Collecting data')
		for eId in list(h5f.keys()):
			pbar.set_postfix(embryo=eId)
			pbar.update()
			if eId == 'ensemble':
				continue
			x, x_dot, fn = collect_function(h5f[eId], key, tmin, tmax, feature_names=feature_names)
			X.append(x)
			X_dot.append(x_dot)
			feature_names = fn
	
	X = np.concatenate(X, axis=0)
	X_dot = np.concatenate(X_dot, axis=0)

	return X, X_dot, feature_names

def fit_sindy_model(h5f, key, tmin, tmax,
					optimizer=ps.STLSQ(),
					component_weight=None,
					n_models=5,
					n_candidates_to_drop=0,
					subset_fraction=0.2,
					feature_names=None,
					collect_function=collect_mesh_data):
	'''
	Fit a SINDy model on data filled by a given key in an h5py file
	Fit range goes from tmin to tmax, and is applied on a set keep of PCA components
	'''
	#Collect data
	X, X_dot, feature_names = collect_data(h5f, key, tmin, tmax, 
										   collect_function, feature_names)
	
	#Train model
	sindy = FlySINDy(
		optimizer=optimizer,
		feature_names=feature_names,
		n_models=n_models,
		n_candidates_to_drop=n_candidates_to_drop,
		subset_fraction=subset_fraction,
		material_derivative=True,
	)
	sindy.fit(x=X, x_dot=X_dot, component_weight=component_weight)
	
	sindy.print(lhs=[f'D_t {key}'])
	return sindy
