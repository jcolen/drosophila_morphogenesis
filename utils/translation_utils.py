import sys
sys.path.insert(0, '/project/vitelli/jonathan/REDO_fruitfly/src')

from utils.dataset import *
from utils.plot_utils import *
from utils.decomposition_utils import *

import pickle as pk
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from scipy.interpolate import interp1d
from tqdm import tqdm
from time import sleep
from math import ceil
import h5py
import pysindy as ps
from sindy.fly_sindy import FlySINDy

def fill_group_info(data, embryoID, libraries):
	'''
	Fill an embryo group with links, fields, and determine the feature names
	'''
	group = data.require_group(embryoID)
	links = group.require_group('links')
	fields = group.require_group('fields')

	feature_names = set()
	for i, info in enumerate(libraries):
		key, path, library, decomposition = info

		#Is the embryo in this library? Or should we just link the ensemble
		with h5py.File(os.path.join(path, 'derivative_library.h5'), 'r') as f:
			flag = group.name in f.keys()
		
		if flag:
			if key in links: 
				del links[key]
			if key in fields:
				del fields[key]
			links[key] = h5py.ExternalLink(os.path.join(path, 'derivative_library.h5'), group.name)
			fields[key] = h5py.SoftLink(os.path.join(links.name, key, key))
			fields[key].attrs['t'] = np.round(links[key]['time'][()])
			print('Added %s to %s library' % (group.name, key))
		elif key not in links:
			links[key] = h5py.ExternalLink(os.path.join(path, 'derivative_library.h5'), '/ensemble')
			fields[key] = h5py.SoftLink(os.path.join(links.name, key, key))
			fields[key].attrs['t'] = np.round(links[key]['time'][()])
			print('Added %s to %s library' % ('ensemble', key))
	
		sub_features = [ sf for sf in links[key][library] if key in links[key][library][sf].attrs]
		feature_names = feature_names.union(sub_features)
	feature_names = list(feature_names)
	
	return group, feature_names

def collect_library(data,
					libraries,
					feature_names, 
					control_names=None,
					group='tensor_library',
					extra_functions=None,
					control_offset=0):
	'''
	Populate library with links to relevant datasets
	Also compute any mixed or coupled features as necessary
	'''
	links = data['links']
	raw = data.require_group('X_raw')
	for feature in feature_names:
		if not feature in raw:
			for key, _, group, _ in libraries:
				if feature in links[key][group]:
					path = os.path.join(links.name, key, group, feature)
					raw[feature] = h5py.SoftLink(path)
					raw[feature].attrs['t'] = np.round(links[key]['time'][()])
					break

	if extra_functions:
		for extra_function in extra_functions:
			extra_function(data)

def decompose_library(data, 
					  libraries,
					  window_length=5, 
					  t_min=5, 
					  t_max=50,
					  re_model=False):
	'''
	Create PCA libraries from the accumulated data
	Each PCA library should be of shape [T, P, L]
		T = time
		P = num PCA components
		L = library size
	
	Each library FOLDER should have an attribute ['t'] which
		indicates the timestamps for each element of that folder
	
	X_pca - the PCA libraries for fields live-imaged with each embryo (dynamic fields)
	X_dot - the dynamics of live-imaged fields
	X_dot_pca - the PCA libraries of live-imaged dynamics
	
	If the PCA for a given field has already been computed, we don't re-compute it to save time
	'''
	X_cpt = data.require_group('X_cpt')
	X_dot = data.require_group('X_dot')
	X_dot_cpt = data.require_group('X_dot_cpt')
	
	features = list(data['X_raw'].keys())
						 
	for key, path, library, decomposition in libraries:
		if decomposition is None or data['fields'][key].shape[0] != data['fields/v'].shape[0]:
			print('Not computing dynamics for %s' % key)
			continue

		print('Decomposing library for %s' % key)
		model = get_decomposition_model(data, key, decomposition)

		cpt_key = X_cpt.require_group(key)

		for i, feature in enumerate(features):
			if re_model and feature in cpt_key:
				del cpt_key[feature]
			if feature not in cpt_key:
				raw = data['X_raw'][feature]
				if model.can_transform(raw):
					cpt_key[feature] = model.transform(raw, remove_mean=True)
					cpt_key[feature].attrs['t'] = raw.attrs['t']
		
		if key not in X_dot or \
				(key in X_dot and X_dot[key].attrs['window_length'] != window_length):
			print('Computing time derivative for %s' % key)
			smoother_kws = {'window_length': window_length}
			
			diffT = ps.SmoothedFiniteDifference(d=1, axis=0, smoother_kws=smoother_kws)
			x = data['fields'][key]
			dot = diffT._differentiate(x, x.attrs['t'])[..., None]
			
			ds = X_dot.require_dataset(key, shape=dot.shape, dtype=dot.dtype)
			ds[:] = dot
			ds.attrs.update(smoother_kws)
			ds.attrs['t'] = x.attrs['t']

		dot_cpt = model.transform(X_dot[key], remove_mean=True)[..., None]
		ds = X_dot_cpt.require_dataset(key, shape=dot_cpt.shape, dtype=dot_cpt.dtype)
		ds[:] = dot_cpt
		ds.attrs.update(X_dot[key].attrs)

'''
ORGANIZE AND PERFORM FITTING
'''

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
					collect_function=collect_decomposed_data,
					**kwargs):
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
		for feat in ['v dot grad %s' % key, '[O, %s]' % key]:
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
		#optimizer = ps.SSR(alpha=alpha, criteria='model_residual', normalize_columns=True, max_iter=100)
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

		sindy.print(lhs=['D_t %s ' % key])
		return sindy

'''
POSTPROCESS AND EVALUATE
'''

def evolve_rk4_grid(x0, x_dot, tmin=0, tmax=10, step_size=0.2):
	'''
	RK4 evolution of a spatial field given its derivatives
	x0 - initial condition
	xdot - sequence of x-derivatives
	model - pca model describing the subspace (omitting noise) we forecast in
	keep - the set of pca components to keep
	t - timepoints corresponding to xdot
	tmin - start of prediction sequence
	tmax - end of prediction sequence
	step_size - dynamics step size

	Returns x, tt - the predictions and timepoints of those predictions
	'''
	tt = np.arange(tmin, tmax, step_size)
	x = np.zeros([len(tt), *x0.shape])
	x[0] = x0
			
	for ii in range(len(tt)-1):
		k1 = x_dot(tt[ii])
		k2 = x_dot(tt[ii] + 0.5 * step_size)
		k3 = x_dot(tt[ii] + 0.5 * step_size)
		k4 = x_dot(tt[ii] + step_size)
				
		x[ii+1] = x[ii] + (k1 + 2 * k2 + 2 * k3 + k4) * step_size / 6

	return x, tt

def sindy_predict(data, key, sindy, tmin=None, tmax=None):
	'''
	Forecast an embryo using a SINDy model
	Rather than forecasting the PCA components, 
		we integrate the evolution of the fields directly
	'''
	x_true = data['fields'][key]
	x_int = interp1d(x_true.attrs['t'], x_true, axis=0)

	#Pass 1 - get the proper time range
	for feature in sindy.feature_names:
		t = data['X_raw'][feature].attrs['t']
		tmin = max(tmin, np.min(t))
		tmax = min(tmax, np.max(t))
	time = x_true.attrs['t']
	time = time[np.logical_and(time >= tmin, time <= tmax)]

	
	#Pass 2 - add in the terms
	x_dot_pred = np.zeros([tmax-tmin+1, *x_true.shape[1:]])
	coefs = sindy.coefficients()[0]
	for i, feature in enumerate(sindy.feature_names):
		x = data['X_raw'][feature]
		t_mask = np.logical_and(x.attrs['t'] >= tmin, x.attrs['t'] <= tmax)
		x_dot_pred += coefs[i] * x[t_mask, ...]
	
	if sindy.material_derivative_:
		for feature in ['v dot grad %s' % key, '[O, %s]' % key]:
			if feature in data['X_raw']:
				x = data['X_raw'][feature]
				t_mask = np.logical_and(x.attrs['t'] >= tmin, x.attrs['t'] <= tmax)
				x_dot_pred -= x[t_mask, ...]
	
	ic = x_int(tmin)
	x_dot_int = interp1d(time, x_dot_pred, axis=0)
	x_pred, times = evolve_rk4_grid(ic, x_dot_int, tmin, tmax, step_size=0.2)

	return x_pred, x_int, times

def sindy_predictions_plot(x_pred, x_int, times, plot_fn=plot_tensor2D):
	'''
	Plot a summary of the predictions of a sindy model
	'''
	step = 5 #minutes
	ncols = int(np.ceil(np.ptp(times) / step))
	dt = times[1] - times[0]
	step = int(step // dt)

	x_true = x_int(times).reshape(x_pred.shape)
	x_norm = np.linalg.norm(x_true, axis=1)
	vmin = x_norm.min()
	vmax = x_norm.max()

	res = mean_norm_residual(x_pred, x_true)

	fig, ax = plt.subplots(3, ncols, figsize=(1*ncols, 3))

	for i in range(ncols):
		ii = min(i*step, len(x_pred)-1)
		plot_fn(ax[0, i], x_pred[ii], vmin=vmin, vmax=vmax)
		plot_fn(ax[1, i], x_true[ii], vmin=vmin, vmax=vmax)
		color_2D(ax[2, i], res[ii], vmin=0, vmax=1, cmap='jet')

		ax[0, i].set_title('t=%d' % times[ii])

	ax[0, 0].set_ylabel('SINDy')
	ax[1, 0].set_ylabel('Experiment')
	ax[2, 0].set_ylabel('Error')

	plt.tight_layout()
	
def decomposed_predictions_plot(x_pred, x_int, times, x_model, keep):
	'''
	Plot a summary of the PCA component evolution for a given SINDy model
	'''
	cpt_pred = x_model.transform(x_pred.reshape([x_pred.shape[0], -1]))
	cpt_true = x_model.transform(x_int(times).reshape([cpt_pred.shape[0], -1]))

	cpt_pred = cpt_pred[:, keep]
	cpt_true = cpt_true[:, keep]
	
	#Print R2 score
	print('PCA Component R2=%g\tMSE=%g' % (r2_score(cpt_true, cpt_pred), mean_squared_error(cpt_true, cpt_pred)))

	ncols = min(cpt_pred.shape[1], 4)
	nrows = ceil(cpt_pred.shape[1] / ncols)
	fig, ax = plt.subplots(nrows, ncols, 
						   figsize=(ncols, nrows),
						   sharey=False, sharex=True, dpi=200)
	ax = ax.flatten()

	for i in range(cpt_pred.shape[1]):
		ax[i].plot(times, cpt_pred[:, i], color='red')
		ax[i].plot(times, cpt_true[:, i], color='black')
		ax[i].set_ylabel('Component %d' % i, fontsize=6)
		ax[i].text(0.02, 0.98, 'R2=%.3g' % r2_score(cpt_true[:, i], cpt_pred[:, i]),
				   fontsize=5, color='blue',
				   transform=ax[i].transAxes, va='top', ha='left')
		ax[i].tick_params(which='both', labelsize=4)
		ax[i].set_yticks([])
	plt.tight_layout()
