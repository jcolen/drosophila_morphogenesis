import sys
sys.path.insert(0, '/project/vitelli/jonathan/REDO_fruitfly/src')

from utils.dataset import *
from utils.plot_utils import *
from utils.decomposition_utils import *

import pickle as pk
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from tqdm import tqdm
from math import ceil
import h5py
import pysindy as ps
from sindy.fly_sindy import FlySINDy

def fill_group_info(data, embryoID, libraries, prgs):
	'''
	Fill an embryo group with links, fields, and determine the feature names
	'''
	group = data.require_group(embryoID)
	links = group.require_group('links')
	fields = group.require_group('fields')

	control_names = set()
	for key in prgs:
		if not key in links:
			links[key] = h5py.ExternalLink(os.path.join(prgs[key], 'derivative_library.h5'), '/ensemble')
			fields[key] = h5py.SoftLink(os.path.join(links.name, key, key))
		control_names = control_names.union(links[key][library].keys())
	control_names = list(control_names)

	feature_names = set()
	for i, info in enumerate(libraries):
		key, path, library, decomposition = info
		if i == 0:
			if 't' in group:
				del group['t']
			group['t'] = h5py.SoftLink(os.path.join(links.name, key, 'time'))
		if not key in links:
			links[key] = h5py.ExternalLink(os.path.join(path, 'derivative_library.h5'), group.name)
			fields[key] = h5py.SoftLink(os.path.join(links.name, key, key))
		
		sub_features = [ sf for sf in links[key][library] if key in links[key][library][sf].attrs]
		feature_names = feature_names.union(sub_features)
		#feature_names = feature_names.union(links[key][library].keys())
	feature_names = list(feature_names)
	
	return group, feature_names, control_names

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
					break

	if control_names is not None:
		control = data.require_group('U_raw')
		control.attrs['offset'] = control_offset
		for feature in control_names:
			if feature in control:
				continue
				#del control[feature]
			for key in links:
				if feature in links[key][group]:
					path = os.path.join(links.name, key, group, feature)
					control[feature] = h5py.SoftLink(path)
					control[feature].attrs['t'] = links[key]['time'][()]
					break

	if extra_functions:
		for extra_function in extra_functions:
			extra_function(data)

	'''
	Allow establishing units for each field
	Do this by writing the statistics of each field as attributes
	'''
	fields = data.require_group('fields')
	for field in fields:
		f = fields[field]
		f.attrs['std'] = np.std(f)
		f.attrs['range'] = np.ptp(f)
		f.attrs['mean'] = np.mean(f)
		f.attrs['min'] = np.min(f)
		f.attrs['max'] = np.max(f)

def get_control_timeline(data, libraries):
	'''
	Determine the timeline of control parameters U
	'''
	links = data.require_group('links')
	t_min, t_max = -1e5, 1e5
	for key in links:
		if key in libraries: #Skip dynamic fields here
			continue
		time = links[key]['time'] + data['U_raw'].attrs['offset']
		t_min = max(np.min(time), t_min)
		t_max = min(np.max(time), t_max)
	print('Control time range: [%d, %d]' % (t_min, t_max))
	return np.arange(t_min, t_max+1)

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
	U_pca - the PCA libraries for static-imaged ensemble-calculated fields (control fields)
	X_dot - the dynamics of live-imaged fields
	X_dot_pca - the PCA libraries of live-imaged dynamics
	
	If the PCA for a given field has already been computed, we don't re-compute it to save time
	'''
	X_cpt = data.require_group('X_cpt')
	X_dot = data.require_group('X_dot')
	X_dot_cpt = data.require_group('X_dot_cpt')

	X_cpt.attrs['t'] = data['t']
	X_dot.attrs['t'] = data['t']
	X_dot_cpt.attrs['t'] = data['t']
	
	if 'U_raw' in data:
		U_cpt = data.require_group('U_cpt')
		U_cpt.attrs['t'] = get_control_timeline(data, libraries)

		t_min, t_max = np.min(U_cpt.attrs['t']), np.max(U_cpt.attrs['t'])
		controls = list(data['U_raw'].keys())
	else:
		controls = []

	features = list(data['X_raw'].keys())
						 
	for key, path, library, decomposition in libraries:
		if decomposition is None:
			continue
		model = get_decomposition_model(data, key, decomposition)
		
		def overwrite_dataset(group, key, data, features):
			if key in group:
				del group[key]
			group.create_dataset(key, data=data)
			group[key].attrs['feature_names'] = features

		X = np.empty([X_cpt.attrs['t'].shape[0], model.n_components, len(features)])
		not_included = []
		for i, feature in enumerate(features):
			if not re_model and \
					key in X_cpt and \
					feature in X_cpt[key].attrs['feature_names']:
				loc = np.argwhere(X_cpt[key].attrs['feature_names'] == feature)[0, 0]
				X[..., i] = X_cpt[key][..., loc][()]
			else:
				if model.can_transform(data['X_raw'][feature]):
					X[..., i] = model.transform(data['X_raw'][feature], remove_mean=True)
				else:
					not_included.append(i)
		X = np.delete(X, not_included, axis=-1)
		overwrite_dataset(X_cpt, key, X, [f for i, f in enumerate(features) if not i in not_included])

		if 'U_raw' in data:
			U = np.empty([t_max-t_min+1, model.n_components, len(controls)])
			for i, feature in enumerate(controls):
				if not re_model and \
						key in U_cpt and \
						feature in U_cpt[key].attrs['feature_names']:
					loc = np.argwhere(U_cpt[key].attrs['feature_names'] == feature)[0, 0]
					U[..., i] =U_cpt[key][..., loc][()]
				else:
					ui = data['U_raw'][feature]
					t = ui.attrs['t'] + data['U_raw'].attrs['offset']
					ui = ui[np.logical_and(t >= t_min, t <= t_max), ...]
					U[..., i] = model.transform(data['U_raw'][feature], remove_mean=True)
			overwrite_dataset(U_cpt, key, U, controls)
		
		compute = False
		if key not in X_dot:
			compute = True
		elif X_dot[key].attrs['window_length'] != window_length:
			print('Window length has changed!')
			compute = True
						
		if compute:
			print('Computing time derivatives')
			smoother_kws = {'window_length': window_length}
			
			diffT = ps.SmoothedFiniteDifference(d=1, axis=0, smoother_kws=smoother_kws)
			dot = diffT._differentiate(data['fields'][key], data['t'][()])[..., None]
			
			ds = X_dot.require_dataset(key, shape=dot.shape, dtype=dot.dtype)
			ds[:] = dot
			ds.attrs.update(smoother_kws)

		dot_cpt = model.transform(X_dot[key], remove_mean=True)[..., None]
		ds = X_dot_cpt.require_dataset(key, shape=dot_cpt.shape, dtype=dot_cpt.dtype)
		ds[:] = dot_cpt
		ds.attrs.update(X_dot[key].attrs)

def collect_overleaf_data(h5f, key, tmin, tmax):
	'''
	Collect only those terms which correspond to the proposed Overleaf equations
	'''
	if key == 'c':
		feature_names = [
			'Dorsal_Source', 
			'c', 
			'c Tr(E)',
			#'c Tr(m_ij)',
			#'c^2', 
			#'c^2 Tr(m_ij)',
			'v dot grad c', 
		]
	elif key == 'm_ij':
		feature_names = [
			'v dot grad m_ij',
			'[O, m_ij]', 
			'm_ij',
			'{m_ij, E_passive}',
			'c {m_ij, E_passive}',
			#'Static_DV',
			'c Static_DV',
			'Static_DV Tr(m_ij)',
			'c m_ij',
			'c m_ij Tr(m_ij)',
		]
	else:
		raise RuntimeError('Overleaf model not proposed for %s' % key)

	return collect_decomposed_data(h5f, key, tmin, tmax, feature_names)

def collect_decomposed_data(h5f, key, tmin, tmax, feature_names=None):
	'''
	Collect the PCA data from a given h5f library and return X, X_dot, and the feature names
	'''
	if feature_names is None:
		feature_names = list(h5f['X_cpt'][key].attrs['feature_names'])
	
	t_X = h5f['t'][()].astype(int)
	t_mask = np.logical_and(t_X >= tmin, t_X <= tmax)

	data = h5f['X_cpt'][key]

	base_names = list(data.attrs['feature_names'])
	X = []
	for feature in feature_names:
		loc = base_names.index(feature)
		X.append(data[..., loc])

	X = np.stack(X, axis=-1)[t_mask, ...]
	X_dot = h5f['X_dot_cpt'][key][t_mask, ...]

	return X, X_dot, feature_names

def repeat_components(x, N):
	'''
	Repeat PCA components during fitting procedure to over-weight them
	'''
	N[N == 0] = 1 #Avoid eliminating components
	X = []
	for i in range(x.shape[1]):
		X.append(np.repeat(x[:, i:i+1], N[i], axis=1))
	return np.concatenate(X, axis=1)

def fit_sindy_model(h5f, key, tmin, tmax,
					component_weight=None,
					threshold=1e-1, 
					alpha=1e-1, 
					n_models=5,
					n_candidates_to_drop=5,
					n_subset=None,
					material_derivative=False, 
					component_repeat=None,
					overleaf_only=False,
					**kwargs):
	'''
	Fit a SINDy model on data filled by a given key in an h5py file
	Fit range goes from tmin to tmax, and is applied on a set keep of PCA components
	'''
	if overleaf_only:
		print('Using only overleaf-allowed terms')
		collect_function = collect_overleaf_data
	else:
		collect_function = collect_decomposed_data

	#Collect data
	print('Collecting data')
	X, X_dot = [], []
	for eId in list(h5f.keys()):
		x, x_dot, feature_names = collect_function(h5f[eId], key, tmin, tmax)
		X.append(x)
		X_dot.append(x_dot)
	X = np.concatenate(X, axis=0)
	X_dot = np.concatenate(X_dot, axis=0)

	#Shift material derivative terms to LHS
	if material_derivative:
		print('Adding Material Derivative terms to LHS')
		for feat in ['v dot grad %s' % key, '[O, %s]' % key]:
			if feat in feature_names:
				loc = feature_names.index(feat)
				X_dot += X[..., loc:loc+1]
				X = np.delete(X, loc, axis=-1)
				feature_names.remove(feat)
	
	#HARDCODED overweighting of component 0
	if component_repeat is not None:
		print('Repeating components by this factor %s' % str(component_repeat))
		assert component_repeat.shape[0] == X.shape[1]
		X = repeat_components(X, component_repeat)
		X_dot = repeat_components(X_dot, component_repeat)
		component_weight = repeat_components(component_weight[None], component_repeat)[0]

	if len(X) > 1:
		print('Applying train/test split')
		X, _, X_dot, _ = train_test_split(X, X_dot, test_size=0.33)

	#Build optimizer (with ensembling)
	optimizer = ps.STLSQ(threshold=threshold, alpha=alpha, normalize_columns=True)
	if n_models > 1:
		bagging = True
		if n_candidates_to_drop == 0:
			n_candidates_to_drop = None
		if n_subset is None:
			n_subset = X.shape[0]

		n_candidates_to_drop = None if n_candidates_to_drop == 0 else n_candidates_to_drop
		optimizer = ps.EnsembleOptimizer(
			opt=optimizer,
			bagging=True,
			library_ensemble=n_candidates_to_drop is not None,
			n_models=n_models,
			n_subset=n_subset,
			n_candidates_to_drop=n_candidates_to_drop)
	sindy = FlySINDy(
		optimizer=optimizer,
		feature_names=feature_names,
	)
	
	#Fit and print
	sindy.fit(x=X, x_dot=X_dot, component_weight=component_weight)
	sindy.material_derivative_ = material_derivative
	sindy.print(lhs=['D_t %s ' % key])
	return sindy

	
def evolve_rk4_grid(x0, x_dot, model, keep, tmin=0, tmax=10, step_size=0.2):
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
				
		xtt = x[ii] + (k1 + 2 * k2 + 2 * k3 + k4) * step_size / 6
		
		#Project onto PCA components
		xtt = xtt.reshape([1, -1])
		xtt = model.inverse_transform(model.transform(xtt), keep)
		x[ii+1] = xtt.reshape(x[ii+1].shape)

	return x, tt

def sindy_predict(data, key, sindy, model, keep, tmin=None, tmax=None):
	'''
	Forecast an embryo using a SINDy model
	Rather than forecasting the PCA components, 
		we integrate the evolution of the fields directly
	'''
	#Collect library from h5py dataset
	t_X = data['t'][()].astype(int)
	if 'U_cpt' in data:	
		t_U = data['U_cpt'].attrs['t']
		if tmin is None:
			tmin = max(np.min(t_X), np.min(t_U))
		if tmax is None:
			tmax = min(np.max(t_X), np.max(t_U))
	else:
		if tmin is None: tmin = np.min(t_X)
		if tmax is None: tmax = np.max(t_X)
	
	time = t_X[np.logical_and(t_X >= tmin, t_X <= tmax)]

	x_true = data['fields'][key]
	x_int = interp1d(t_X, x_true, axis=0)
	ic = x_int(tmin)
	
	x_dot_pred = np.zeros([tmax-tmin+1, *x_true.shape[1:]])
	coefs = sindy.coefficients()[0]
	for i, feature in enumerate(sindy.feature_names):
		if feature in data['X_raw']:
			t_mask = np.logical_and(t_X >= tmin, t_X <= tmax)
			x_dot_pred += coefs[i] * data['X_raw'][feature][t_mask, ...]
		else:
			t_mask = np.logical_and(t_U >= tmin, t_U <= tmax)
			x_dot_pred += coefs[i] * data['U_raw'][feature][t_mask, ...]
	
	if sindy.material_derivative_:
		adv = 'v dot grad %s' % key
		if adv in data['X_raw']:
			t_mask = np.logical_and(t_X >= tmin, t_X <= tmax)
			x_dot_pred -= data['X_raw'][adv][t_mask, ...]

		cor = '[O, %s]' % key
		if cor in data['X_raw']:
			t_mask = np.logical_and(t_X >= tmin, t_X <= tmax)
			x_dot_pred -= data['X_raw'][cor][t_mask, ...]
	
	x_dot_int = interp1d(time, x_dot_pred, axis=0)
	x_pred, times = evolve_rk4_grid(ic, x_dot_int, model, keep,
									tmin=tmin, tmax=tmax, step_size=0.2)

	return x_pred, x_int, times

def sindy_predictions_plot(x_pred, x_int, x_model, times, keep, data, plot_fn=plot_tensor2D):
	'''
	Plot a summary of the predictions of a sindy model
	'''
	ncols = 4
	step = x_pred.shape[0] // ncols

	x_true = x_int(times)
	x_true = x_model.inverse_transform(x_model.transform(x_true)[:, keep], keep)

	mask = x_model['masker'].mask_
	ys, xs = np.where(mask != 0)
	crop_mask = np.s_[..., min(ys):max(ys)+1, min(xs):max(xs)+1]
	mask = mask[crop_mask]

	x_pred = x_pred[crop_mask]
	x_true = x_true[crop_mask]

	x_true = x_true.reshape([x_true.shape[0], -1, *x_true.shape[-2:]])
	x_norm = np.linalg.norm(x_true, axis=1)[:, mask]
	vmin = x_norm.min()
	vmax = x_norm.max()

	res = mean_norm_residual(x_pred, x_true)
	error = res.mean(axis=(-2, -1))
	error = res[..., mask].mean(axis=-1)

	x_true[..., ~mask] = np.nan
	x_pred[..., ~mask] = np.nan
	res[..., ~mask] = np.nan

	v2 = np.linalg.norm(data['fields']['v'], axis=1).mean(axis=(1, 2))

	fig, ax = plt.subplots(1, 1, figsize=(2, 2))
	ax.plot(times, error)
	ax.set(ylim=[0, 1], 
		   ylabel='Error Rate',
		   xlabel='Time')
	ax2 = ax.twinx()
	ax2.plot(data['t'], v2, color='red')
	ax2.set_yticks([])
	ax2.set_ylabel('$v^2$', color='red')
	ax.set_xlim([times.min(), times.max()])

	axis = ax

	fig, ax = plt.subplots(3, ncols, figsize=(1*ncols, 3))

	offset = min(5, x_pred.shape[0] - ncols*step) 
	for i in range(ncols):
		ii = i*step + offset
		plot_fn(ax[0, i], x_pred[ii], vmin=vmin, vmax=vmax)
		plot_fn(ax[1, i], x_true[ii], vmin=vmin, vmax=vmax)
		color_2D(ax[2, i], res[ii], vmin=0, vmax=1, cmap='jet')

		axis.axvline(times[ii], zorder=-1, color='black', linestyle='--')

	for a in ax.flatten():
		a.set_aspect('auto')

	ax[0, 0].set_ylabel('SINDy')
	ax[1, 0].set_ylabel('Experiment')
	ax[2, 0].set_ylabel('Error')

	plt.tight_layout()
	
def decomposed_predictions_plot(x_pred, x_int, x_model, times, keep):
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
