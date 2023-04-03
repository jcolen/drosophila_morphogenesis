import sys
sys.path.insert(0, '/project/vitelli/jonathan/REDO_fruitfly/src')
from utils.dataset import *
from utils.plot_utils import *

import numpy as np
import pandas as pd
import seaborn as sns
import pickle as pk

from math import ceil
import h5py

from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class StandardShaper(BaseEstimator, TransformerMixin):
	def fit(self, X, y0=None):
		'''
		Figure out how to reshape X to shape n_samples, _, Y, X
		'''
		self.data_shape_ = X.shape[-2:]
		self.n_samples_  = X.shape[0]

		X = X.reshape([self.n_samples_, -1, *self.data_shape_])
		self.n_vec_components_ = X.shape[1]

		return self

	@property
	def n_features_in_(self):
		return self.n_vec_components_ * np.prod(self.data_shape_)

	def transform(self, X):
		n_samples = X.shape[0]
		return X.reshape([n_samples, self.n_vec_components_, *self.data_shape_])
	
	def inverse_transform(self, X):
		return X
			

class LeftRightSymmetrize(BaseEstimator, TransformerMixin):
	def fit(self, X, y0=None):
		'''
		Assume X has shape [n_samples, _, Y, X]
		'''
		self.n_vec_components_ = X.shape[1]
		return self

	def transform(self, X):
		X_flip = X[..., ::-1, :].copy() #Flip DV axis

		#Invert DV component of vector field
		if self.n_vec_components_ == 2:
			X_flip[..., 0, :, :] *= -1

		#Invert off-diagonal components of tensor field
		elif self.n_vec_components_ == 4:
			X_flip[:, 1:3] *= -1
		
		return 0.5 * (X +  X_flip)

	def inverse_transform(self, X):
		return X

class Masker(BaseEstimator, TransformerMixin):
	def __init__(self, crop=0, mask=None):
		self.crop = crop
		self.mask = mask
		super(Masker, self).__init__()
	
	def fit(self, X, y0=None):
		self.mask_ = np.ones(X.shape[-2:], dtype=np.bool)

		if self.mask is not None:
			self.mask_ = np.logical_and(self.mask_, self.mask)

		if self.crop > 0:
			mask = np.zeros_like(self.mask_)
			mask[self.crop:-self.crop, self.crop:-self.crop] = True
			self.mask_ = np.logical_and(self.mask_, mask)

		self.n_features_out_ = np.count_nonzero(self.mask_)
		return self
		
	def transform(self, X):
		Xt = X.reshape([X.shape[0], -1, *self.mask_.shape])
		Xt = Xt[..., self.mask_]
		return Xt.reshape([Xt.shape[0], -1])

	def inverse_transform(self, X):
		'''
		Re-apply spatial structure and fill in masked regions with zeros
		'''
		X = X.reshape([X.shape[0], -1, self.n_features_out_])
		Xt = np.zeros([*X.shape[:2], *self.mask_.shape])
		Xt[..., self.mask_] = X

		return Xt

class SVDPipeline(Pipeline):
	def __init__(self, n_components=16, whiten=True, crop=0, mask=None, lrsym=True):
		self.n_components = n_components
		self.whiten = whiten
		self.crop = crop
		self.mask = mask
		self.lrsym = lrsym
		steps = [
			('shaper', StandardShaper()),
			('masker', Masker(crop=crop, mask=mask)),
			('scaler', StandardScaler(with_std=False)),
			('svd', TruncatedSVD(n_components=n_components)),
		]

		if lrsym:
			steps.insert(1, ('leftright', LeftRightSymmetrize()))
		
		super(SVDPipeline, self).__init__(steps)
	
	def __getitem__(self, key):
		return self.named_steps[key]

	def can_transform(self, X):
		return self.n_features_in_ == np.prod(X.shape[1:])
	
	def fit(self, X, y0):
		'''
		X is the data used for fitting the Truncated SVD
		y0 is the data used for fitting the Standard Scaler
		'''
		for i in range(len(self.steps)):
			if self.steps[i][0] == 'scaler':
				self.steps[i][1].fit(y0)
			else:
				self.steps[i][1].fit(X, y0)
			
			X = self.steps[i][1].transform(X)
			y0 = self.steps[i][1].transform(y0)
			
		return self

	def transform(self, X, remove_mean=False):
		if isinstance(X, h5py.Dataset):
			X = X[()]

		if remove_mean:
			mean = self.inverse_transform(np.zeros([1, self['svd'].n_components]))
			mean = mean.reshape([-1, *X.shape[1:]])
			X += mean[0]

		Xt = super(SVDPipeline, self).transform(X)
		if self.whiten:
			return Xt / np.sqrt(self['svd'].explained_variance_)
		return Xt

	def inverse_transform(self, Xt, keep=None):
		if self.whiten:
			factor = np.sqrt(self['svd'].explained_variance_)
		else:
			factor = np.ones_like(self['svd'].explained_variance_)

		if keep is not None:
			X = np.zeros([Xt.shape[0], self['svd'].n_components])
			if Xt.shape[-1] == keep.shape[0]:
				X[:, keep] = Xt[:, keep]
			else:
				X[:, keep] = Xt
			Xt = X

		return super(SVDPipeline, self).inverse_transform(Xt * factor)

	def score(self, X, metric=residual, multioutput='raw_values'):
		y = self.inverse_transform(self.transform(X))

		#Only score in the masked regions
		X = X[..., self['masker'].mask].squeeze()
		y = y[..., self['masker'].mask].squeeze()

		if metric == residual:
			score = residual(X, y)
			if multioutput == 'uniform_average': 
				return np.mean(score)
			else:
				return score
		else:
			score = metric(X, y, multioutput=multioutput)
			return score
		
from torchvision.transforms import Compose
def build_decomposition_model(dataset, model_type=SVDPipeline, tmin=-15, tmax=45, **model_kwargs):
	'''
	Learn a decomposition model on an AtlasDataset object
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
		e_idx = df[df.embryoID == e].eIdx.values
		e_data = e_data[e_idx]
		t = e_data.shape[0]
		h, w = e_data.shape[-2:]
		y0.append(e_data.reshape([t, -1, h, w]))
	y0 = np.concatenate(y0, axis=0)

	if isinstance(dataset.transform, Compose) and isinstance (dataset.transform.transforms[1], Smooth2D):
		sigma = dataset.transform.transforms[1].sigma
		y0 = np.stack([
			np.stack([
				gaussian_filter(y0[t, c], sigma=sigma) \
				for c in range(y0.shape[1])]) \
			for t in range(y0.shape[0])])

	model = model_type(whiten=True, **model_kwargs)

	train_mask = (df.set == 'train') & (df.time >= tmin) & (df.time <= tmax)
	train_mask = (df.time >= tmin) & (df.time <= tmax)
	train = y0[df[train_mask].index]
	
	if dataset.filename == 'velocity2D':
		scaler_train = np.zeros([1, *train.shape[-3:]])
	else:
		scaler_train = y0[df[(train_mask) & (df.time < 0)].index]
	
	model.fit(train, scaler_train)

	params = model.transform(y0)
	y = model.inverse_transform(params)
	df['res'] = model.score(y0, metric=residual).mean(axis=(-2, -1))
	df['mag'] = np.linalg.norm(y, axis=1).mean(axis=(-1, -2))
	df = pd.concat([df, pd.DataFrame(params).add_prefix('param')], axis=1)

	return model, df

def get_decomposition_results(dataset, 
							  overwrite=False,
							  model_type=SVDPipeline,
							  model_name=None,
							  **model_kwargs):
	'''
	Get results on a dataset, loading from file if it already exists
	Create new model and save it if none is found on that folder
	'''
	base = dataset.filename[:-2]
	if model_name is None:
		model_name = model_type.__name__
	if not os.path.exists(os.path.join(dataset.path, 'decomposition_models')):
		os.mkdir(os.path.join(dataset.path, 'decomposition_models'))
	path = os.path.join(dataset.path, 'decomposition_models', '%s_%s' % (base, model_name))
	if not overwrite and os.path.exists(path+'.pkl'):
		print('Found SVDPipeline for this dataset!')
		model = pk.load(open(path+'.pkl', 'rb'))
		df = pd.read_csv(path+'.csv')
	else:
		print('Building new SVDPipeline for this dataset')
		model, df = build_decomposition_model(dataset, model_type, **model_kwargs)
		pk.dump(model, open(path+'.pkl', 'wb'))
		df.to_csv(path+'.csv')
		print(path)
		
	return model, df

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['lines.markersize'] = 2

def decomposition_point_plot(model, df, cutoff=0.9):
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

def decomposition_sweep_plot(model, df, cutoff=0.9, plot_fn=plot_tensor2D):
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

		z = model.inverse_transform(points)

		colors = plt.cm.Purples(np.linspace(0, 1, points.shape[0]+1))[1:]

		for j in range(len(colors)):
			a = fig.add_subplot(gs[i, j])
			plot_fn(a, z[j])
			for spine in a.spines.values():
				spine.set_edgecolor(colors[j])
				spine.set_linewidth(2)

	plt.tight_layout()

def decompose_trajectory(x, x_model, zero_mean=True):
	'''
	Perform PCA on a trajectory of shape [T, *C, Y, X]
	whiten performs mean-subtraction on the resulting trajectory
	This is relevant for computing dynamics, as all we really care about 
		is the induced change in the PCA latent space
	'''
	xi = x_model.transform(x) # [T, P]
	if zero_mean: 
		xi -= np.mean(xi, axis=0, keepdims=True)
	return xi

def get_decomposition_model(data, key, model_name):
	'''
	Collect a decomposition model given an h5py dataset
	'''
	path = os.path.dirname(data['links'].get(key, getlink=True).filename)
	model_name = os.path.join(path, 'decomposition_models', model_name)
	return pk.load(open(model_name, 'rb'))

