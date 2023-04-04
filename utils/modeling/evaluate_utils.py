from math import ceil

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import interp1d

from ..plot_utils import mean_norm_residual, color_2D, plot_tensor2D

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
		for feature in [f'v dot grad {key}', f'[O, {key}]']: 
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

		ax[0, i].set_title(f't={times[ii]:.0f}')

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
	r2 = r2_score(cpt_true, cpt_pred)
	mse = mean_squared_error(cpt_true, cpt_pred)
	print(f'PCA Component R2={r2:g}\tMSE={mse:g}')

	ncols = min(cpt_pred.shape[1], 4)
	nrows = ceil(cpt_pred.shape[1] / ncols)
	fig, ax = plt.subplots(nrows, ncols, 
						   figsize=(ncols, nrows),
						   sharey=False, sharex=True, dpi=200)
	ax = ax.flatten()

	for i in range(cpt_pred.shape[1]):
		ax[i].plot(times, cpt_pred[:, i], color='red')
		ax[i].plot(times, cpt_true[:, i], color='black')
		ax[i].set_ylabel(f'Component {i}', fontsize=6)
		r2 = r2_score(cpt_true[:, i], cpt_pred[:, i])
		ax[i].text(0.02, 0.98, f'R2={r2:.3g}',
				   fontsize=5, color='blue',
				   transform=ax[i].transAxes, va='top', ha='left')
		ax[i].tick_params(which='both', labelsize=4)
		ax[i].set_yticks([])
	plt.tight_layout()
