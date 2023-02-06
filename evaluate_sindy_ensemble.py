import numpy as np
import pandas as pd
import h5py
import sys
import os
import glob
import warnings
import pysindy as ps
import matplotlib.pyplot as plt

from tqdm import tqdm

basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))

from utils.translation_utils import *
from utils.decomposition_utils import *

warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score
from utils.translation_utils import evolve_rk4_grid

lib_key, lib_path = 'm', 'data/WT/sqh-mCherry'
lib_key, lib_path = 'c', 'data/WT/ECad-GFP'
tmin, tmax = 15, 45 #30 minute prediction window

model = pk.load(open(os.path.join(lib_path, 'tensor_SVDPipeline.pkl'), 'rb'))
keep=np.cumsum(model.explained_variance_ratio_) <= 0.9

results_df = pd.read_csv('SINDy_ensemble_%s.csv' % lib_key)
coefs = results_df.filter(like='coef_')
coefs = coefs.rename(columns=lambda x: x[5:])
feature_names = coefs.columns

results_df['mse'] = 0.
results_df['residual'] = 0.
results_df['r2_score'] = 0.

with h5py.File('data/symmetric_dynamics_fitting.h5', 'r') as h5f:
	data = h5f['ensemble']
	
	#Collect library from h5py dataset
	t_X = data['t'][()].astype(int)
	t_mask = np.logical_and(t_X >= tmin, t_X <= tmax)
	time = t_X[t_mask]

	x_true = data['fields'][lib_key]
	if model.crop > 0:
		crop = model.crop
		x_true = x_true[..., crop:-crop, crop:-crop]
	x_int = interp1d(t_X, x_true, axis=0)
	ic = x_int(tmin)
	
	material_derivative = np.zeros([tmax-tmin+1, *x_true.shape[1:]])
	adv = 'v dot grad %s' % lib_key
	cor = '[O, %s]' % lib_key
	material_derivative -= data['X_raw'][adv][t_mask, ...][..., crop:-crop, crop:-crop]
	material_derivative += data['X_raw'][cor][t_mask, ...][..., crop:-crop, crop:-crop]
	
	for row in tqdm(range(len(results_df))):
		x_dot_pred = np.zeros([tmax-tmin+1, *x_true.shape[1:]])
		for feature in feature_names:
			t_mask = np.logical_and(t_X >= tmin, t_X <= tmax)
			x_dot_pred += coefs.loc[row, feature] * data['X_raw'][feature][t_mask, ...][..., crop:-crop, crop:-crop]

		if results_df.loc[row, 'material_derivative']:
			x_dot_pred += material_derivative

		x_pred, times = evolve_rk4_grid(ic, x_dot_pred, model, keep,
										t=time, tmin=tmin, tmax=tmax, step_size=0.2)
		x_pred = x_pred.reshape([x_pred.shape[0], -1, *x_pred.shape[-2:]])
		x_0 = x_int(times)
		x_0 = model.inverse_transform(model.transform(x_0)[:, keep], keep)

		results_df.loc[row, 'mse'] = np.linalg.norm(x_0 - x_pred, axis=(1, 2)).mean()
		results_df.loc[row, 'residual'] = residual(x_0, x_pred).mean()
		results_df.loc[row, 'r2_score'] = r2_score(
			x_0.reshape([x_0.shape[0], -1]),
			x_pred.reshape([x_pred.shape[0], -1])
		)
		
		results_df.to_csv('SINDy_ensemble_%s.csv' % lib_key)
