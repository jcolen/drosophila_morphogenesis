'''
Plotting utilities for drosophila embryo data
'''

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 8

#Units of microns!
ap_min = 3.230
ap_max = 455.701
dv_min = 3.227
dv_max = 536.891
dv_midpoint = dv_min + 0.5 * (dv_max - dv_min)

def plot_scalar(ax, f, vmax_std=None, cmap='viridis', **im_kwargs):
	'''
	Colormap on 2D projection of embryo surface
	'''
	im_kwargs['extent'] = [ap_min, ap_max, dv_min, dv_max]
	im_kwargs['cmap'] = cmap
	if vmax_std:
		im_kwargs['vmax'] = vmax_std * np.std(f)
	if len(f.shape) > 2:
		ax.imshow(np.linalg.norm(f, axis=0), **im_kwargs)
	else:
		ax.imshow(f, **im_kwargs)
	ax.set(xticks=[], yticks=[])
	ax.set(xlim=[ap_min, ap_max], ylim=[dv_min, dv_midpoint])
	ax.set_aspect('equal')

def plot_vector(ax, f, skip=10, color='black', width=0.005, **kwargs):
	'''
	plot a vector field on the 2D embryo projection
	'''
	X = np.linspace(ap_min, ap_max, f.shape[-1])[::skip]
	Y = np.linspace(dv_min, dv_max, f.shape[-2])[::skip]
	Y, X = np.meshgrid(Y, X, indexing='ij')
	X = X.flatten()
	Y = Y.flatten()
	f0 = f[:, ::skip, ::skip].reshape([2, -1])
	ax.quiver(X, Y, f0[1], f0[0], pivot='middle', color=color, width=width, **kwargs)
	ax.set(xticks=[], yticks=[])
	ax.set(xlim=[ap_min, ap_max], ylim=[dv_min, dv_midpoint])
	ax.set_aspect('equal')

def plot_tensor(ax, f0, skip=20, both=False, 
				  linecolor='white', linewidth=0.007, 
				  cmap='inferno', **im_kwargs):
	'''
	Plot the dominant eigenvectors and intensity of a tensor field on 2D projected embryo
	'''
	f = f0.copy()
	f = f.reshape([4, *f.shape[-2:]])
	plot_scalar(ax, np.linalg.norm(f, axis=0), cmap=cmap, **im_kwargs)

	X = np.linspace(ap_min, ap_max, f.shape[-1])[::skip]
	Y = np.linspace(dv_min, dv_max, f.shape[-2])[::skip]
	Y, X = np.meshgrid(Y, X, indexing='ij')
	X = X.flatten()
	Y = Y.flatten()

	f = f.reshape([2, 2, *f.shape[-2:]])
	f = np.nan_to_num(f)
	#Ensure we're using deviatoric part
	trf = np.einsum('kkyx->yx', f)
	if np.mean(trf) != 0:
		f -= 0.5 * np.eye(2)[..., None, None] * trf

	f = f.transpose(2, 3, 0, 1)[::skip, ::skip]
	f = f.reshape([-1, *f.shape[-2:]])
	el, ev = np.linalg.eig(f)
	ev *= el[:, None, :]

	qwargs = dict(pivot='middle', color=linecolor, width=linewidth,
				  headwidth=0, headlength=0, headaxislength=0)

	mask = np.all(el != 0, axis=1)
	if np.sum(mask) == 0:
		return
	el = el[mask]
	ev = ev[mask]
	X = X[mask]
	Y = Y[mask]

	if both:
		for i in range(ev.shape[-1]):
			ax.quiver(X, Y, ev[:, 1, i], ev[:, 0, i], **qwargs)
	else:
		order = np.argmax(el, axis=-1)
		ev = np.array([ev[i, :, ei] for i, ei in enumerate(order)])
		ax.quiver(X, Y, ev[:, 1], ev[:, 0], **qwargs)

	ax.set(xticks=[], yticks=[])
	ax.set(xlim=[ap_min, ap_max], ylim=[dv_min, dv_midpoint])
	ax.set_aspect('equal')