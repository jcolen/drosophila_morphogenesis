from time import sleep

import numpy as np
import pysindy as ps
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from .fly_sindy import FlySINDy

def overleaf_feature_names(key):
	if key == 'c':
		feature_names = [
			'Dorsal_Source', 
			'c', 
			'c Tr(E)',
			'v dot grad c', 
		]
	elif key == 'm_ij':
		feature_names = [
			'v dot grad m_ij',
			'[O, m_ij]', 
			'm_ij',
			'm_ij Tr(E_passive)',
			'Static_DV Tr(m_ij)',
			'm_ij Tr(m_ij)',
			'Dorsal_Source m_ij',
			'Dorsal_Source m_ij Tr(E_passive)',
			'Dorsal_Source Static_DV',
			'Dorsal_Source m_ij Tr(m_ij)',
		]
	return feature_names

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

def fit_sindy_model(h5f, key, tmin, tmax,
					component_weight=None,
					threshold=1e-1, 
					alpha=1e-1, 
					n_models=5,
					n_candidates_to_drop=5,
					subset_fraction=0.2,
					overleaf_only=False,
					collect_function=collect_decomposed_data):
	'''
	Fit a SINDy model on data filled by a given key in an h5py file
	Fit range goes from tmin to tmax, and is applied on a set keep of PCA components
	'''
	with tqdm(total=len(h5f.keys())+6) as pbar:
		pbar.set_description('Processing arguments')
		if overleaf_only:
			pbar.set_postfix(status='Using only overleaf-allowed terms')
			feature_names = overleaf_feature_names(key)
		else:
			pbar.set_postfix(status='Using all terms')
			feature_names = None

		if collect_function == collect_raw_data:
			pbar.set_postfix(status='Setting component weight to None')
			component_weight = None
		sleep(0.2)

		#Collect data
		pbar.update()
		pbar.set_description('Collecting data')
		X, X_dot = [], []
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

		#Shift material derivative terms to LHS
		pbar.update()
		pbar.set_description('Material Derivative to LHS')
		for feat in [f'v dot grad {key}', f'[O, {key}]']:
			pbar.set_postfix(feature=feat)
			if feat in feature_names:
				loc = feature_names.index(feat)
				X_dot += X[..., loc:loc+1]
				X = np.delete(X, loc, axis=-1)
				feature_names.remove(feat)
			to_remove = []
			for i in range(len(feature_names)):
				if feat in feature_names[i]:
					X = np.delete(X, i, axis=-1)
					to_remove.append(i)
			feature_names = [feature_names[i] for i in range(len(feature_names)) if not i in to_remove]
			sleep(0.2)

		#Train test split
		pbar.update()
		pbar.set_description('Applying train/test split')
		if len(X) > 1:
			X, test, X_dot, test_dot = train_test_split(X, X_dot, test_size=0.2)
			pbar.set_postfix(train_size=X.shape[0], test_size=test.shape[0])
			sleep(0.2)

		#Build optimizer (with ensembling)
		pbar.update()
		pbar.set_description('Building optimizer')
		optimizer = ps.STLSQ(threshold=threshold, alpha=alpha, normalize_columns=True)
		if n_models > 1:
			pbar.set_description('Building ensemble optimizer')
			if n_candidates_to_drop == 0:
				n_candidates_to_drop = None
			if subset_fraction is None:
				subset_fraction = 1

			n_subset = int(np.round(X.shape[0] * subset_fraction))
			if component_weight is None:
				n_subset *= int(np.round(X.shape[1] * subset_fraction))

			n_candidates_to_drop = None if n_candidates_to_drop == 0 else n_candidates_to_drop
			
			pbar.set_postfix(n_models=n_models, n_subset=n_subset, n_drop=n_candidates_to_drop)

			optimizer = ps.EnsembleOptimizer(
				opt=optimizer,
				bagging=True,
				library_ensemble=n_candidates_to_drop is not None,
				n_models=n_models,
				n_subset=n_subset,
				n_candidates_to_drop=n_candidates_to_drop)
		sleep(0.2)

		#Train model
		pbar.update()
		pbar.set_description('Training model')
		pbar.set_postfix(threshold=threshold, alpha=alpha)

		sindy = FlySINDy(
			optimizer=optimizer,
			feature_names=feature_names,
		)
		sindy.fit(x=X, x_dot=X_dot, component_weight=component_weight)
		
		#Get test error for each element in ensemble
		pbar.update()
		pbar.set_description('Getting test errors')
		if n_models > 1:
			X, X_dot = test, test_dot
			coef_ = sindy.optimizer.coef_
			coef_list = sindy.optimizer.coef_list
			mses = []
			for i in range(len(coef_list)):
				sindy.optimizer.coef_ = coef_list[i]
				pred = sindy.model.predict(X).reshape(X_dot.shape)
				mse = np.mean((pred - X_dot)**2, axis=(0, -1)) #MSE of each component
				mses.append(mse)

			sindy.coef_list_ = sindy.optimizer.coef_list
			sindy.mses_list_ = mses

			sindy.optimizer.coef_ = coef_

		sindy.print(lhs=[f'D_t {key}'])
		return sindy
