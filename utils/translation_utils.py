import sys
sys.path.insert(0, '/project/vitelli/jonathan/REDO_fruitfly/src')

from utils.dataset import *
from utils.plot_utils import *
from utils.decomposition_utils import *

import pickle as pk
from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import interp1d
from tqdm import tqdm
from math import ceil
import h5py
import pysindy as ps

def fill_group_info(data, embryoID, libraries, prgs, library):
	'''
	Fill an embryo group with links, fields, and determine the feature names
	'''
	group = data.require_group(embryoID)
	links = group.require_group('links')
	fields = group.require_group('fields')

	control_names = set()
	for key in prgs:
		if not key in links:
			links[key] = h5py.ExternalLink(os.path.join(prgs[key], 'library_PCA.h5'), '/ensemble')
			fields[key] = h5py.SoftLink(os.path.join(links.name, key, key))
		control_names = control_names.union(links[key][library].keys())
	control_names = list(control_names)

	feature_names = set()
	for i, key in enumerate(libraries.keys()):
		if i == 0:
			if 't' in group:
				del group['t'], fields['v']
			group['t'] = h5py.SoftLink(os.path.join(links.name, key, 'time'))
			fields['v'] = h5py.SoftLink(os.path.join(links.name, key, 'v'))
		if not key in links:
			links[key] = h5py.ExternalLink(os.path.join(libraries[key], 'library_PCA.h5'), group.name)
			fields[key] = h5py.SoftLink(os.path.join(links.name, key, library, key))
		feature_names = feature_names.union(links[key][library].keys())
	feature_names = list(feature_names)
	
	return group, feature_names, control_names

def collect_library(data, priority, feature_names, control_names, 
					group='tensor_library',
					mixed_features=None,
					mixed_features_attrs=None,
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
			for key in priority:
				if feature in links[key][group]:
					path = os.path.join(links.name, key, group, feature)
					raw[feature] = h5py.SoftLink(path)
					break
	
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
	
	if mixed_features:
		for feature in mixed_features:
			feature_names.append(feature)
			if feature not in raw:
				raw[feature] = mixed_features[feature](raw)
			if mixed_features_attrs is not None:
				raw[feature].attrs.update(mixed_features_attrs[feature])

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

def decompose_library(data, libraries, library, window_length=5, re_model=False):
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
	U_cpt = data.require_group('U_cpt')

	X_cpt.attrs['t'] = data['t']
	X_dot.attrs['t'] = data['t']
	X_dot_cpt.attrs['t'] = data['t']
	U_cpt.attrs['t'] = get_control_timeline(data, libraries)

	t_min, t_max = np.min(U_cpt.attrs['t']), np.max(U_cpt.attrs['t'])

	features = list(data['X_raw'].keys())
	controls = list(data['U_raw'].keys())
						 
	for key in libraries:
		model = get_decomposition_model(data, key, library)

		X = np.empty([X_cpt.attrs['t'].shape[0], model.n_components, len(features)])
		U = np.empty([t_max-t_min+1, model.n_components, len(controls)])

		for i, feature in enumerate(features):
			if not re_model and \
					key in X_cpt and \
					feature in X_cpt[key].attrs['feature_names']:
				loc = np.argwhere(X_cpt[key].attrs['feature_names'] == feature)[0, 0]
				X[..., i] = X_cpt[key][..., loc][()]
			else:
				X[..., i] = decompose_trajectory(data['X_raw'][feature][()], model)

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
				U[..., i] = decompose_trajectory(ui[()], model)

		def overwrite_dataset(group, key, data, features):
			if key in group:
				del group[key]
			group.create_dataset(key, data=data)
			group[key].attrs['feature_names'] = features
		
		overwrite_dataset(X_cpt, key, X, features)
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
			
		dot_cpt = decompose_trajectory(X_dot[key][()], model)[..., None]
		ds = X_dot_cpt.require_dataset(key, shape=dot_cpt.shape, dtype=dot_cpt.dtype)
		ds[:] = dot_cpt
		ds.attrs.update(X_dot[key].attrs)

def collect_decomposed_data(h5f, key, tmin, tmax, keep, scale_units=False, material_derivative=True):
	'''
	Collect the PCA data froma	given h5f library and return X, U, X_dot, and the relevant variable names

	New: Include the option to rescale components by their units
		Each field has a characteristic unit and we include the option to unscale everything

		the returned scale_array is the unit factor we need to multiply by to get to units of key / time
		these are the units of the feature
		Each feature should be multiplied by these units, and its coefficient should be divided by them
	'''
	t_X = h5f['t'][()].astype(int)
	t_U = h5f['U_cpt'].attrs['t']
	
	feature_names = list(h5f['X_cpt'][key].attrs['feature_names'])
	control_names = list(h5f['U_cpt'][key].attrs['feature_names'])

	u_mask = np.logical_and(t_X >= np.min(t_U), t_X <= np.max(t_U))
	x_mask = np.logical_and(t_X >= tmin, t_X <= tmax)
	t_mask = np.logical_and(x_mask, u_mask)
	times = t_X[t_mask]

	X = h5f['X_cpt'][key][t_mask, ...][..., keep, :][()]
	X_dot = h5f['X_dot_cpt'][key][t_mask, ...][..., keep, :][()]

	if material_derivative:
		adv = 'v dot grad %s' % key
		if adv in feature_names:
			loc = feature_names.index(adv)
			X_dot += X[..., loc:loc+1]
			X = np.delete(X, loc, axis=-1)
			feature_names.remove(adv)

		cor = '[O, %s]' % key
		if cor in feature_names:
			loc = feature_names.index(cor)
			X_dot -= X[..., loc:loc+1]
			X = np.delete(X, loc, axis=-1)
			feature_names.remove(cor)

	#Eliminate other advection terms because they involve gradients of proteins
	adv = [feature for feature in feature_names if 'v dot grad' in feature]
	for a in adv:
		loc = feature_names.index(a)
		X = np.delete(X, loc, axis=-1)
		feature_names.remove(a)

	x_mask = np.logical_and(t_U >= np.min(t_X), t_U <= np.max(t_X))
	u_mask = np.logical_and(t_U >= tmin, t_U <= tmax)
	t_mask = np.logical_and(x_mask, u_mask)
	U = h5f['U_cpt'][key][t_mask, ...][..., keep, :][()]

	scale_array = np.ones(X.shape[-1] + U.shape[-1])
	if scale_units:
		fields = h5f['fields']
		base = fields[key].attrs['std']
		def find_scales(group, features):
			scales = np.ones(len(features)) #* base #Units of key
			for i, feature in enumerate(features):
				attrs = group[feature].attrs
				for unit in attrs:
					if unit in fields and not unit == 'v':
						scales[i] /= (fields[unit].attrs['std'] / base) ** attrs[unit]
					#if unit in fields:
					#	scales[i] /= fields[unit].attrs['std'] * attrs[unit]
					#elif unit == 'space': #Spatial derivative means multiply by a spatial unit
					#	scales[i] *= fields['v'].attrs['std'] ** attrs[unit]
			return scales

		scale_array[:X.shape[-1]] = find_scales(h5f['X_raw'], feature_names)
		scale_array[X.shape[-1]:] = find_scales(h5f['U_raw'], control_names)
		
	return X, U, X_dot, times, feature_names, control_names, scale_array

def fit_sindy_model(h5f, key, tmin, tmax, keep, threshold=1e-1, alpha=1e-1, scale_units=False, material_derivative=False):
	'''
	Fit a SINDy model on data filled by a given key in an h5py file
	Fit range goes from tmin to tmax, and is applied on a set keep of PCA components

	New: Include the option to rescale components by their units
		Each field has a characteristic unit and we include the option to unscale everything
		In general, the goal should be to use v as a reference
	'''
	X, U, X_dot, scales = [], [], [], []
	eIds = list(h5f.keys())
	for eId in eIds:
		x, u, x_dot, times, feature_names, control_names, scale = collect_decomposed_data(
			h5f[eId], 
			key, 
			tmin,
			tmax,
			keep,
			scale_units,
			material_derivative)
		X.append(x * scale[:len(feature_names)])
		U.append(u * scale[len(feature_names):])
		X_dot.append(x_dot)
		scales.append(scale)

	scales = np.mean(scales, axis=0)
	sindy = ps.SINDy(
		feature_library=ps.IdentityLibrary(),
		optimizer=ps.STLSQ(threshold=threshold, alpha=alpha),
		feature_names=feature_names+control_names,
	)
	sindy.fit(x=X, x_dot=X_dot, u=U, multiple_trajectories=True)
	sindy.model.steps[-1][1].optimizer.coef_[:] *= scales

	if material_derivative:
		sindy.print(lhs=['D_t %s ' % key])
		sindy.material_derivative_ = True
	else:
		sindy.print(lhs=['d_t %s' % key])
		sindy.material_derivative_ = False
	return sindy
	
def evolve_rk4_grid(x0, xdot, model, keep, t, tmin=0, tmax=10, step_size=0.2):
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
			
	interp = interp1d(t, xdot, axis=0)
	#for ii in tqdm(range(len(tt)-1)):
	for ii in range(len(tt)-1):
		k1 = interp(tt[ii])
		k2 = interp(tt[ii] + 0.5 * step_size)
		k3 = interp(tt[ii] + 0.5 * step_size)
		k4 = interp(tt[ii] + step_size)
				
		xtt = x[ii] + (k1 + 2 * k2 + 2 * k3 + k4) * step_size / 6
		
		#Project onto PCA components
		xtt = xtt.reshape([1, -1])
		xtt = model.inverse_transform(model.transform(xtt)[:, keep], keep)
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
	t_U = data['U_cpt'].attrs['t']
	
	if tmin is None:
		tmin = max(np.min(t_X), np.min(t_U))
	if tmax is None:
		tmax = min(np.max(t_X), np.max(t_U))
	time = t_X[np.logical_and(t_X >= tmin, t_X <= tmax)]

	x_true = data['fields'][key]
	if model.crop > 0:
		crop = model.crop
		x_true = x_true[..., crop:-crop, crop:-crop]
	x_int = interp1d(t_X, x_true, axis=0)
	ic = x_int(tmin)
	
	x_dot_pred = np.zeros([tmax-tmin+1, *x_true.shape[1:]])
	coefs = sindy.coefficients()[0]
	for i, feature in enumerate(sindy.feature_names):
		if feature in data['X_raw']:
			t_mask = np.logical_and(t_X >= tmin, t_X <= tmax)
			x_dot_pred += coefs[i] * data['X_raw'][feature][t_mask, ...][..., crop:-crop, crop:-crop]
		else:
			t_mask = np.logical_and(t_U >= tmin, t_U <= tmax)
			x_dot_pred += coefs[i] * data['U_raw'][feature][t_mask, ...][..., crop:-crop, crop:-crop]
	
	if sindy.material_derivative_:
		adv = 'v dot grad %s' % key
		if adv in data['X_raw']:
			#print('Subtracting advection')
			t_mask = np.logical_and(t_X >= tmin, t_X <= tmax)
			x_dot_pred -= data['X_raw'][adv][t_mask, ...][..., crop:-crop, crop:-crop]

		cor = '[O, %s]' % key
		if cor in data['X_raw']:
			#print('Adding co-rotation')
			t_mask = np.logical_and(t_X >= tmin, t_X <= tmax)
			x_dot_pred += data['X_raw'][cor][t_mask, ...][..., crop:-crop, crop:-crop]
	
	x_pred, times = evolve_rk4_grid(ic, x_dot_pred, model, keep,
									t=time, tmin=tmin, tmax=tmax, step_size=0.2)

	return x_pred, x_int, times

def sindy_predictions_plot(x_pred, x_int, x_model, times, keep, data, plot_fn=plot_tensor2D):
	'''
	Plot a summary of the predictions of a sindy model
	'''
	ncols = 4
	step = x_pred.shape[0] // ncols

	fig = plt.figure(dpi=150, figsize=(1*ncols, 5))
	gs = fig.add_gridspec(4, ncols)

	ax = fig.add_subplot(gs[:2, :])

	xi = x_int(times)
	xi = x_model.inverse_transform(x_model.transform(xi)[:, keep], keep)

	error = residual(x_pred, xi).mean(axis=(-2, -1))

	v2 = np.linalg.norm(data['fields']['v'], axis=1).mean(axis=(1, 2))

	ax.plot(times, error)
	ax.set(ylim=[0, 1], 
		   ylabel='Error Rate',
		   xlabel='Time')
	ax2 = ax.twinx()
	ax2.plot(data['t'], v2, color='red')
	ax2.set_yticks([])
	ax2.set_ylabel('$v^2$', color='red')
	ax.set_xlim([times.min(), times.max()])

	offset = min(5, x_pred.shape[0] - ncols*step) 
	for i in range(ncols):
		ii = i*step + offset
		plot_fn(fig.add_subplot(gs[2, i]), x_pred[ii])
		if i == 0:
			plt.gca().set_ylabel('SINDy')

		plot_fn(fig.add_subplot(gs[3, i]), xi[ii])
		if i == 0:
			plt.gca().set_ylabel('Truth')
		ax.axvline(times[ii], zorder=-1, color='black', linestyle='--')

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
						   sharey=True, sharex=True, dpi=200)
	ax = ax.flatten()

	for i in range(cpt_pred.shape[1]):
		ax[i].plot(times, cpt_pred[:, i], color='red')
		ax[i].plot(times, cpt_true[:, i], color='black')
		ax[i].text(0.98, 0.02, 'Component %d' % i, 
				   fontsize=6, color='blue',
				   transform=ax[i].transAxes, va='bottom', ha='right')
		ax[i].text(0.98, 0.98, 'R2=%g' % r2_score(cpt_true[:, i], cpt_pred[:, i]),
				   fontsize=6, color='blue',
				   transform=ax[i].transAxes, va='top', ha='right')
		ax[i].tick_params(which='both', labelsize=6)
	plt.tight_layout()
