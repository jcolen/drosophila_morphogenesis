import sys
sys.path.insert(0, '/project/vitelli/jonathan/REDO_fruitfly/src')
from utils.dataset import *
from utils.plot_utils import *

import numpy as np
import pandas as pd
import seaborn as sns
import pickle as pk

from math import ceil

from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class SVDPipeline(Pipeline):
	def __init__(self, n_components=16, whiten=True, crop=0):
		self.n_components = n_components
		self.whiten = whiten
		self.crop = crop
		steps = [
			('scaler', StandardScaler(with_std=False)),
			('fitter', TruncatedSVD(n_components=n_components)),
		]
		super(SVDPipeline, self).__init__(steps)
		
	
	def fit(self, X, y0):
		'''
		X is the data used for fitting the Truncated SVD
		y0 is the data used for fitting the Standard Scaler
		'''
		if self.crop > 0:
			Xt = X[..., self.crop:-self.crop, self.crop:-self.crop]
			y0t = y0[..., self.crop:-self.crop, self.crop:-self.crop]
		else:
			Xt = X
			y0t = y0

		self.data_shape_ = Xt.shape[1:]
		
		if not y0t.shape[1:] == self.data_shape_:
			raise ValueError('X and y0 should have the same shape after n_samples')
		
		Xt = Xt.reshape([Xt.shape[0], -1])
		y0t = y0t.reshape([y0t.shape[0], -1])
		
		self.steps[0][1].fit(y0t)
		
		Xt = self.steps[0][1].transform(Xt)
		self.steps[1][1].fit(Xt)
		
		# Standard Scaler attributes
		self.scale_ = self.steps[0][1].scale_
		self.mean_ = self.steps[0][1].mean_
		self.var_ = self.steps[0][1].var_
		
		# Truncated SVD attributes
		self.components_ = self.steps[1][1].components_
		self.n_components_ = self.n_components
		self.explained_variance_ = self.steps[1][1].explained_variance_
		self.explained_variance_ratio_ = self.steps[1][1].explained_variance_ratio_
		self.singular_values_ = self.steps[1][1].singular_values_
		self.n_samples_ = X.shape[0]

		return self

	def transform(self, X):
		#Check that we can reshape to data_shape without changing number of samples
		if np.prod(X.shape[1:]) != np.prod(self.data_shape_):
			uncropped_shape = [*self.data_shape_[:-2], self.data_shape_[-2]+2*self.crop, self.data_shape_[-1]+2*self.crop]
			if np.prod(X.shape[1:]) != np.prod(uncropped_shape):
				raise ValueError('X should be resizable to [1, %s] or croppable with size %d' % (str(self.data_shape_), self.crop))
			Xt = X.reshape([X.shape[0], *uncropped_shape])
			if self.crop > 0:
				Xt = Xt[..., self.crop:-self.crop, self.crop:-self.crop]
		else:
			Xt = X
		
		Xt = Xt.reshape([Xt.shape[0], -1])
		Xt = super(SVDPipeline, self).transform(Xt)
		
		if self.whiten:
			Xt /= np.sqrt(self.steps[1][1].explained_variance_)
		
		return Xt

	def inverse_transform(self, Xt, keep=None):
		if keep is not None:
			X = np.zeros([Xt.shape[0], self.n_components_])
			X[:, keep] = Xt
			Xt = X
		
		if self.whiten:
			X = Xt * np.sqrt(self.steps[1][1].explained_variance_)
			X = super(SVDPipeline, self).inverse_transform(X)
		else:
			X = super(SVDPipeline, self).inverse_transform(Xt)
		X = X.reshape([X.shape[0], *self.data_shape_])
		return X

	def score(self, X, metric=residual, multioutput='raw_values'):
		y = self.inverse_transform(self.transform(X))
		if self.crop > 0:
			X = X[..., self.crop:-self.crop, self.crop:-self.crop]

		if metric == residual:
			score = residual(X, y)
			if multioutput == 'uniform_average': 
				return np.mean(score)
			else:
				return score
		else:
			score = metric(X, y, multioutput=multioutput)
			return score

		
def build_decomposition_model(dataset, model_type=SVDPipeline, tmin=0, tmax=60, **model_kwargs):
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
		t = e_data.shape[0]
		h, w = e_data.shape[-2:]
		y0.append(e_data.reshape([t, -1, h, w]))
	y0 = np.concatenate(y0, axis=0)

	model = model_type(whiten=True, **model_kwargs)

	train_mask = (df.set == 'train') & (df.time >= tmin) & (df.time <= tmax)
	train = y0[df[train_mask].index]
	
	if dataset.filename == 'velocity2D':
		scaler_train = np.zeros([1, *train.shape[-3:]])
	else:
		scaler_train = y0[df[(train_mask) & (np.abs(df.time - 5) < 10)].index]
	
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
							  **model_kwargs):
	'''
	Get results on a dataset, loading from file if it already exists
	Create new model and save it if none is found on that folder
	'''
	base = dataset.filename[:-2]
	model_name = model_type.__name__
	path = os.path.join(dataset.path, '%s_%s' % (base, model_name))
	if not overwrite and os.path.exists(path+'.pkl'):
		print('Found SVDPipeline for this dataset!')
		model = pk.load(open(path+'.pkl', 'rb'))
		df = pd.read_csv(path+'.csv')
	else:
		print('Building new SVDPipeline for this dataset')
		model, df = build_decomposition_model(dataset, model_type, **model_kwargs)
		pk.dump(model, open(path+'.pkl', 'wb'))
		df.to_csv(path+'.csv')
		
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

def get_decomposition_model(data, key, library, model_type=SVDPipeline):
	'''
	Collect a decomposition model given an h5py dataset
	'''
	lib_key = library[:library.index('library')]
	path = os.path.dirname(data['links'].get(key, getlink=True).filename)
	if lib_key == 'symmetric_':
		lib_key = 'tensor_'
	model_name = os.path.join(path, lib_key+'%s.pkl' % model_type.__name__)
	return pk.load(open(model_name, 'rb'))

