import sys
sys.path.insert(0, '/project/vitelli/jonathan/REDO_fruitfly/src')

from dataset import *
from utils.plot_utils import *

import seaborn as sns
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import interp1d
from tqdm import tqdm
from math import ceil
import h5py
import pysindy as ps

def residual(u, v):
	'''
	Residual metric from Streichan eLife paper
	'''
	umag = np.linalg.norm(u, axis=-3)												  
	vmag = np.linalg.norm(v, axis=-3)												  

	uavg = np.sqrt((umag**2).mean(axis=(-2, -1), keepdims=True))					
	vavg = np.sqrt((vmag**2).mean(axis=(-2, -1), keepdims=True))					

	res = uavg**2 * vmag**2 + vavg**2 * umag**2 - 2 * uavg * vavg * np.einsum('...ijk,...ijk->...jk', u, v)
	res /= 2 * vavg**2 * uavg**2														
	return res 

def unpca(d, model, keep):
	'''
	Undo PCA using a subset of learned components
	'''
	if d.shape[1] != model.n_components_:
		di = np.zeros([d.shape[0], keep.shape[0]])
		di[:, keep] = d
		d = di
	d = model.inverse_transform(d)
	d = d.reshape([d.shape[0], -1, 236, 200])
	return d

def build_PCA_model(dataset, n_components=16):
	'''
	Learn a PCA Model on an AtlasDataset object
	'''
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

def get_pca_results(dataset, 
					n_components=16):
	'''
	Get PCA results on a dataset, loading from file if it already exists
	Create new PCA model and save it if none is found on that folder
	'''
	base = dataset.filename[:-2]
	if os.path.exists(os.path.join(dataset.path, '%s_PCA.pkl' % base)):
		model = pk.load(open(os.path.join(dataset.path, '%s_PCA.pkl' % base), 'rb'))
		if model.n_components == n_components:
			print('Found PCA for this dataset!')
			df = pd.read_csv(os.path.join(dataset.path, '%s_PCA.csv' % base))
		else:
			print('Overwriting PCA for this dataset')
			model, df = build_PCA_model(dataset, n_components=n_components)
			pk.dump(model, open(os.path.join(dataset.path, '%s_PCA.pkl' % base), 'wb'))
			df.to_csv(os.path.join(dataset.path, '%s_PCA.csv' % base))
			
	else:
		print('Building PCA for this dataset')
		model, df = build_PCA_model(dataset, n_components=n_components)
		pk.dump(model, open(os.path.join(dataset.path, '%s_PCA.pkl' % base), 'wb'))
		df.to_csv(os.path.join(dataset.path, '%s_PCA.csv' % base))
		
	return model, df

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['lines.markersize'] = 2

def pca_summary_plot(model, df, dataset, show_tt, show_magnitude=True, cutoff=0.9, plot_fn=plot_tensor2D):
	'''
	Summarize PCA results with an accuracy plot, explained variance plot, and example prediction
	'''
	fig = plt.figure(figsize=(10, 2), dpi=200)
	gs = fig.add_gridspec(1, 5)
	
	ax = fig.add_subplot(gs[0, 0])
	sns.lineplot(data=df, x='time', y='res', color='blue', ax=ax)
	ax.set_ylabel('Error Rate', color='blue')
	ax.set(ylim=[0, 1])
	
	if show_magnitude:
		ax2 = ax.twinx()
		sns.lineplot(data=df, x='time', y='mag', ax=ax2, color='red')
		ax2.set_ylabel('Magnitude', color='red')
		ax2.set_yticks([])
		   
	ax = fig.add_subplot(gs[0, 1:3])
	ax.plot(np.arange(model.n_components_), np.cumsum(model.explained_variance_ratio_))
	ax.bar(np.arange(model.n_components_), model.explained_variance_ratio_)
	ax.set(xlabel='Component', ylabel='Explained Variance')
	ax.axhline(cutoff, color='black', linestyle='--')
   
	offset = 0.03
	bb1 = ax.get_position()
	bb1.x0 = bb1.x0 + offset
	ax.set_position(bb1)
	
	idx = df[df.set == 'test'].iloc[show_tt]['index']
	y0 = dataset[idx]['value']
	y = model.inverse_transform(model.transform(y0.reshape([1, -1]))).reshape(y0.shape)    
	plot_fn(fig.add_subplot(gs[:, -2]), y0)
	plt.title('Experiment')
	plot_fn(fig.add_subplot(gs[:, -1]), y)
	plt.title('ML')

def pca_point_plot(model, df, cutoff=0.9):
	'''
	Generate a seaborn-style pointplot of the parameter values up to a given explained variance
	Points are colored by their inclusion in the train or test set
	'''
	explained = np.cumsum(model.explained_variance_ratio_)
	keep = np.argwhere(explained <= cutoff).flatten()

	dfi = pd.wide_to_long(df.reset_index(), stubnames=['param'], i='index', j='param_num').reset_index(level=1)
	dfi = dfi[dfi.param_num.isin(keep)]

	nrows, ncols = len(keep), len(keep)
	ss = 1.5
	fig, ax = plt.subplots(nrows, ncols,
						   sharey='row',
						   figsize=(ss*ncols, ss*nrows), dpi=150)

	for ii in range(nrows):
		for jj in range(ncols):
			if ii == jj:
				sns.lineplot(
					data=dfi[dfi.param_num==ii],
					x='time',
					y='param',
					hue='embryoID',
					palette='viridis',
					ax=ax[ii,jj],
					legend=False)
				ax[ii,jj].set(xlabel='Time', ylabel='', yticklabels=[])
			else:
				data = df[df.set == 'train']
				ax[ii, jj].scatter(data['param%d' % ii], data['param%d' % jj], c=data['time'], cmap='Blues')
				data = df[df.set == 'test']
				ax[ii, jj].scatter(data['param%d' % ii], data['param%d' % jj], c=data['time'], cmap='Reds')

				if ii == 0:
					ax[ii,jj].xaxis.tick_top()
					ax[ii, jj].set_xlabel('Param %d' % jj)
					ax[ii,jj].xaxis.set_label_position('top')
				elif ii == nrows-1 and jj == 0:
					ax[ii, jj].set_xlabel('Param %d' % jj)

				ax[ii,jj].set(xticklabels=[], yticklabels=[])

			if jj == 0:
				ax[ii,jj].set_ylabel('Param %d' % ii)

	plt.tight_layout()

def pca_sweep_plot(model, df, cutoff=0.9, plot_fn=plot_tensor2D):
	'''
	Plot a series of independent sweeps over each PCA parameter
	'''
	explained = np.cumsum(model.explained_variance_ratio_)
	keep = np.argwhere(explained <= cutoff).flatten()
	
	cols = ['param%d' % k for k in keep]
	values = df[cols].values

	fig = plt.figure(figsize=(6, len(keep)), dpi=150)
	gs = fig.add_gridspec(len(keep), 5)

	for i in range(len(keep)):
		points = np.zeros([gs.ncols-1, model.n_components])
		points[:, :values.shape[1]] = np.mean(values, axis=0, keepdims=True)
		r0, r1 = np.percentile(values[:, i], [10, 90])
		points[:, i] = np.linspace(0, 1, points.shape[0]) * (r1-r0) + r0

		z = model.inverse_transform(points).reshape([points.shape[0], -1, 236, 200])

		colors = plt.cm.Purples(np.linspace(0, 1, points.shape[0]+1))[1:]

		vmax = np.max(np.linalg.norm(z, axis=1))

		for j in range(len(colors)):
			a = fig.add_subplot(gs[i, j])
			plot_fn(a, z[j], vmax=vmax)
			for spine in a.spines.values():
				spine.set_edgecolor(colors[j])
				spine.set_linewidth(2)

	plt.tight_layout()

def pca_trajectory(x, x_model, whiten=True):
	'''
	Perform PCA on a trajectory of shape [T, *C, Y, X]
	whiten performs mean-subtraction on the resulting trajectory
	This is relevant for computing dynamics, as all we really care about 
		is the induced change in the PCA latent space
	'''
	xi= x.reshape([x.shape[0], -1]) # [T, C*Y*X]
	xi = x_model.transform(xi) # [T, P]
	if whiten: 
		xi -= np.mean(xi, axis=0, keepdims=True)
	return xi

def collect_library(data, priority, feature_names, control_names, 
					group='tensor_library',
					mixed_features=None,
					mixed_features_attrs=None,
					control_offset=0):
	'''
	Populate library with links to relevant datasets
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
	if mixed_features:
		for feature in mixed_features:
			feature_names.append(feature)
			if feature not in raw:
				raw[feature] = mixed_features[feature](raw)
			if mixed_features_attrs is not None:
				raw[feature].attrs.update(mixed_features_attrs[feature])
	
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

def pca_library(data, pcas, window_length=5, re_pca=False):
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
	links = data.require_group('links')
	X_pca = data.require_group('X_pca')
	X_pca.attrs['t'] = data['t']
	X_dot = data.require_group('X_dot')
	X_dot.attrs['t'] = data['t']
	X_dot_pca = data.require_group('X_dot_pca')
	X_dot_pca.attrs['t'] = data['t']
	
	U_pca = data.require_group('U_pca')
	t_min, t_max = -1e5, 1e5
	for key in links:
		if key in pcas:  #Skip dynamic fields when assessing control field timeline
			continue
		time = links[key]['time'] + data['U_raw'].attrs['offset']
		t_min = max(np.min(time), t_min)
		t_max = min(np.max(time), t_max)
	print('Control time range: [%d, %d]' % (t_min, t_max))
	if 't' in U_pca.attrs:
		u_mask = np.logical_and(U_pca.attrs['t'] >= t_min, U_pca.attrs['t'] <= t_max)
	U_pca.attrs['t'] = np.arange(t_min, t_max+1)
	
	feature_names = list(data['X_raw'].keys())
	control_names = list(data['U_raw'].keys())
						 
	for key in pcas:
		X = np.empty([X_pca.attrs['t'].shape[0], pcas[key].n_components, len(feature_names)])
		
		for i, feature in enumerate(feature_names):
			if not re_pca and key in X_pca and feature in X_pca[key].attrs['feature_names']:
				X[..., i] = X_pca[key][..., np.argwhere(X_pca[key].attrs['feature_names'] == feature)[0, 0]][()]
			else:
				X[..., i] = pca_trajectory(data['X_raw'][feature][()], pcas[key])
		
		if key in X_pca:
			del X_pca[key]
			
		dataset = X_pca.create_dataset(key, data=X)
		dataset.attrs['feature_names'] = feature_names
		
				

		U = np.empty([t_max-t_min+1, pcas[key].n_components, len(control_names)])
		
		for i, feature in enumerate(control_names):
			if not re_pca and key in U_pca and feature in U_pca[key].attrs['feature_names']:
				ui = U_pca[key][..., np.argwhere(U_pca[key].attrs['feature_names'] == feature)[0, 0]][()]
				ui = ui[u_mask, ...]
				U[..., i] = ui															  
			else:
				ui = data['U_raw'][feature]
				t = ui.attrs['t'] + data['U_raw'].attrs['offset']
				ui = ui[np.logical_and(t >= t_min, t <= t_max), ...]
				U[..., i] = pca_trajectory(ui[()], pcas[key])
		
		if key in U_pca:
			del U_pca[key]
		dataset = U_pca.create_dataset(key, data=U)
		dataset.attrs['feature_names'] = control_names
		
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
			
			dot_pca = pca_trajectory(dot, pcas[key])[..., None]
			ds = X_dot_pca.require_dataset(key, shape=dot_pca.shape, dtype=dot_pca.dtype)
			ds[:] = dot_pca
			ds.attrs.update(smoother_kws)

def collect_pca_data(h5f, key, tmin, tmax, keep):
	'''
	Collect the PCA data froma	given h5f library and return X, U, X_dot, and the relevant variable names
	'''
	t_X = h5f['t'][()].astype(int)
	t_U = h5f['U_pca'].attrs['t']
	
	feature_names = list(h5f['X_pca'][key].attrs['feature_names'])
	control_names = list(h5f['U_pca'][key].attrs['feature_names'])

	u_mask = np.logical_and(t_X >= np.min(t_U), t_X <= np.max(t_U))
	x_mask = np.logical_and(t_X >= tmin, t_X <= tmax)
	t_mask = np.logical_and(x_mask, u_mask)
	times = t_X[t_mask]

	X = h5f['X_pca'][key][t_mask, ...][..., keep, :][()]
	X_dot = h5f['X_dot_pca'][key][t_mask, ...][..., keep, :][()]

	x_mask = np.logical_and(t_U >= np.min(t_X), t_U <= np.max(t_X))
	u_mask = np.logical_and(t_U >= tmin, t_U <= tmax)
	t_mask = np.logical_and(x_mask, u_mask)
	U = h5f['U_pca'][key][t_mask, ...][..., keep, :][()]
		
	return X, U, X_dot, times, feature_names, control_names

def fit_sindy_model(h5f, key, tmin, tmax, keep, threshold=1e-1, alpha=1e-1):
	'''
	Fit a SINDy model on data filled by a given key in an h5py file
	Fit range goes from tmin to tmax, and is applied on a set keep of PCA components
	'''
	X, U, X_dot = [], [], []
	eIds = list(h5f.keys())
	for eId in eIds:
		x, u, x_dot, times, feature_names, control_names = collect_pca_data(
			h5f[eId], 
			key, 
			tmin,
			tmax,
			keep)
		X.append(x)
		U.append(u)
		X_dot.append(x_dot)
		#print(eId, X[-1].shape, U[-1].shape, X_dot[-1].shape)
			
	sindy = ps.SINDy(
		feature_library=ps.IdentityLibrary(),
		optimizer=ps.STLSQ(threshold=threshold, alpha=alpha),
		feature_names=feature_names+control_names,
	)
	sindy.fit(x=X, x_dot=X_dot, u=U, multiple_trajectories=True)
	sindy.print(lhs=['(%s)\'' % key])
	return sindy
	
def evolve_rk4_grid(x0, xdot, pca, keep, t, tmin=0, tmax=10, step_size=0.2):
	'''
	RK4 evolution of a spatial field given its derivatives
	x0 - initial condition
	xdot - sequence of x-derivatives
	pca - pca model describing the subspace (omitting noise) we forecast in
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
	for ii in tqdm(range(len(tt)-1)):
		k1 = interp(tt[ii])
		k2 = interp(tt[ii] + 0.5 * step_size)
		k3 = interp(tt[ii] + 0.5 * step_size)
		k4 = interp(tt[ii] + step_size)
				
		xtt = x[ii] + (k1 + 2 * k2 + 2 * k3 + k4) * step_size / 6
		
		#Project onto PCA components
		xtt = xtt.reshape([1, -1])
		xtt = unpca(pca.transform(xtt)[:, keep], pca, keep).reshape(x[ii].shape)
		x[ii+1] = xtt

		
	return x, tt

def sindy_predict(data, key, sindy, pca, keep, tmin=None, tmax=None):
	'''
	Forecast an embryo using a SINDy model
	Rather than forecasting the PCA components, 
		we integrate the evolution of the fields directly
	'''
	#Collect library from h5py dataset
	t_X = data['t'][()].astype(int)
	t_U = data['U_pca'].attrs['t']
	
	if tmin is None:
		tmin = max(np.min(t_X), np.min(t_U))
	if tmax is None:
		tmax = min(np.max(t_X), np.max(t_U))
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
	
	
	x_pred, times = evolve_rk4_grid(ic, x_dot_pred, pca, keep,
									t=time, tmin=tmin, tmax=tmax, step_size=0.2)

	return x_pred, x_int, times

def sindy_predictions_plot(x_pred, x_int, x_model, times, keep, data):
	'''
	Plot a summary of the predictions of a sindy model
	'''
	ncols = 4
	step = x_pred.shape[0] // ncols

	fig = plt.figure(dpi=150, figsize=(1*ncols, 5))
	gs = fig.add_gridspec(4, ncols)

	ax = fig.add_subplot(gs[:2, :])

	xi = x_int(times)
	xi = x_model.transform(xi.reshape([times.shape[0], -1]))[:, keep]
	xi = unpca(xi, x_model, keep)

	error = r2_score(
		x_pred.reshape([x_pred.shape[0], -1]).T,
		xi.reshape([x_pred.shape[0], -1]).T,
		multioutput='raw_values'
	)

	v2 = np.linalg.norm(data['fields']['v'], axis=1).mean(axis=(1, 2))

	ax.plot(times, error)
	ax.set(ylim=[0, 1],
		   ylabel='R2 Score',
		   xlabel='Time')
	ax2 = ax.twinx()
	ax2.plot(data['t'], v2, color='red')
	ax2.set_yticks([])
	ax2.set_ylabel('$v^2$', color='red')
	ax.set_xlim([times.min(), times.max()])


	offset = 5
	for i in range(ncols):
		ii = i*step + offset
		plot_tensor2D(fig.add_subplot(gs[2, i]), x_pred[ii])
		if i == 0:
			plt.gca().set_ylabel('SINDy')

		plot_tensor2D(fig.add_subplot(gs[3, i]), xi[ii])
		if i == 0:
			plt.gca().set_ylabel('Truth')
		ax.axvline(times[ii], zorder=-1, color='black', linestyle='--')

	plt.tight_layout()
	
def pca_predictions_plot(x_pred, x_int, x_model, times, keep):
	'''
	Plot a summary of the PCA component evolution for a given SINDy model
	'''
	pca_pred = x_model.transform(x_pred.reshape([x_pred.shape[0], -1]))
	pca_true = x_model.transform(x_int(times).reshape([pca_pred.shape[0], -1]))

	pca_pred = pca_pred[:, keep]
	pca_true = pca_true[:, keep]
	
	#Print R2 score
	print('PCA Component R2=%g\tMSE=%g' % (r2_score(pca_true, pca_pred), mean_squared_error(pca_true, pca_pred)))

	ncols = min(pca_pred.shape[1], 4)
	nrows = ceil(pca_pred.shape[1] / ncols)
	fig, ax = plt.subplots(nrows, ncols, 
						   figsize=(ncols, nrows),
						   sharey=True, sharex=True, dpi=200)
	ax = ax.flatten()

	for i in range(pca_pred.shape[1]):
		ax[i].plot(times, pca_pred[:, i], color='red')
		ax[i].plot(times, pca_true[:, i], color='black')
		ax[i].text(0.98, 0.02, 'Component %d' % i, 
				   fontsize=6, color='blue',
				   transform=ax[i].transAxes, va='bottom', ha='right')
		ax[i].text(0.98, 0.98, 'R2=%g' % r2_score(pca_true[:, i], pca_pred[:, i]),
				   fontsize=6, color='blue',
				   transform=ax[i].transAxes, va='top', ha='right')
	plt.tight_layout()
