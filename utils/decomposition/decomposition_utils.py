'''
Utils for building and fetching SVD Decomposition models
'''
import os
import pickle as pk
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter
from torchvision.transforms import Compose

from ..dataset import Smooth2D
from ..plot_utils import residual
from .decomposition_model import SVDPipeline

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
		y0.append(e_data.reshape([e_data.shape[0], -1, *e_data.shape[-2:]]))
	y0 = np.concatenate(y0, axis=0)

	if isinstance(dataset.transform, Compose) and \
	   isinstance (dataset.transform.transforms[1], Smooth2D):
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
	df['res'] = model.score(y0, metric=residual).mean(axis=(-2, -1))
	df['mag'] = np.linalg.norm(model.inverse_transform(params), axis=1).mean(axis=(-1, -2))
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
	path = os.path.join(
		dataset.path,
		'decomposition_models',
		f'{base}_{model_name}')
	if not overwrite and os.path.exists(path+'.pkl'):
		print('Found SVDPipeline for this dataset!')
		with open(path+'.pkl', 'rb') as f:
			model = pk.load(f)
		df = pd.read_csv(path+'.csv')
	else:
		print('Building new SVDPipeline for this dataset')
		model, df = build_decomposition_model(dataset, model_type, **model_kwargs)
		with open(path+'.pkl', 'wb') as f:
			pk.dump(model, f)
		df.to_csv(path+'.csv')
		print(path)

	return model, df

def get_decomposition_model(data, key, model_name):
	'''
	Collect a decomposition model given an h5py dataset
	'''
	path = os.path.dirname(data['links'].get(key, getlink=True).filename)
	model_name = os.path.join(path, 'decomposition_models', model_name)
	return pk.load(open(model_name, 'rb'))
