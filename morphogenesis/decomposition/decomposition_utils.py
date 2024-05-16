'''
Utils for building and fetching SVD Decomposition models
'''
import os
import pickle as pk
import numpy as np
import pandas as pd

from .decomposition_model import SVDPipeline

def get_decomposition_model(dataset, model_type=SVDPipeline, model_name=None, **model_kwargs):
	'''
	Get a decomposition model from a dataset
	'''
	base = dataset.filename[:-2]
	if model_name is None:
		model_name = model_type.__name__
	path = os.path.join(dataset.path, 'decomposition_models', f'{base}_{model_name}')
	return pk.load(open(path+'.pkl', 'rb'))

def get_decomposition_results(dataset, model_type=SVDPipeline, model_name=None, **model_kwargs):
	base = dataset.filename[:-2]
	if model_name is None:
		model_name = model_type.__name__
	path = os.path.join(dataset.path, 'decomposition_models', f'{base}_{model_name}')
	return pd.read_csv(path+'.csv')