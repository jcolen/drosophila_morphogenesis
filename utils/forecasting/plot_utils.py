import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d
from matplotlib.colors import Normalize

from ..plot_utils import plot_tensor2D, color_2D
from ..plot_utils import ap_min, ap_max, dv_min, dv_max, dv_midpoint
from ..plot_utils import residual, mean_norm_residual

plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 250
plt.rcParams['figure.frameon'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['legend.framealpha'] = 0.
plt.rcParams['legend.handlelength'] = 1.
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['axes.linewidth'] = 1.
plt.rcParams['image.origin'] = 'lower'

def get_cut(z, N0=80, N1=120):
	'''
	Return the average DV cut between N0 and N1 along the AP axi
	'''
	znorm = np.linalg.norm(z.reshape([-1, *z.shape[-2:]]), axis=0)
	base = np.ones_like(znorm)
	base[np.isnan(znorm)] = 0
	znorm = np.nan_to_num(znorm)
	cut = np.sum(znorm[:, N0:N1], axis=1) / np.sum(base[:, N0:N1], axis=1)
	return cut
	
def sqh_vel_plot(m, v, t, dt=5, skip=20):
	'''
	Evolution plot for myosin and flow
	'''
	dynamic_mask = np.load('Public/Masks/Dynamic_PMG_CF_mask.npy', mmap_mode='r')
	dynamic_time = np.load('Public/Masks/Dynamic_PMG_CF_time.npy', mmap_mode='r')
	msk = interp1d(dynamic_time, dynamic_mask, axis=0, fill_value='extrapolate')

	N = int(np.ceil(len(t) / dt))

	fig, ax = plt.subplots(1, N+1, figsize=(N+1, 1), dpi=300)

	alpha = np.zeros(m.shape[-2:])
	midpoint = alpha.shape[0] // 2
	top = np.s_[..., midpoint:, :]
	bot = np.s_[..., :midpoint, :]
	dv = np.linspace(dv_min, dv_midpoint, midpoint)
	
	slc = np.s_[..., ::skip, ::skip]
	X = np.linspace(ap_min, ap_max, v.shape[-1])
	Y = np.linspace(dv_min, dv_max, v.shape[-2])
	Y, X = np.meshgrid(Y, X, indexing='ij')

	alpha[top] = 1
	X = X[bot][slc]
	Y = Y[bot][slc]
	vwargs = dict(pivot='middle', width=0.005, color='black')

	mnorm = np.linalg.norm(m, axis=(1, 2))[..., 20:-20]
	mmin = np.min(mnorm)
	mmax = np.max(mnorm)
	mwargs = dict(vmin=mmin, vmax=mmax, alpha=alpha)

	cmap = plt.get_cmap('Reds')
		
	axes = []
	for i in range(N):
		ii = dt*i
		ax[i].set_title(f't = {t[ii]:.0f}')

		plot_tensor2D(ax[i], m[ii], both=t[ii]<0, **mwargs)
		ax[i].quiver(X, Y, v[ii, 1][bot][slc], v[ii, 0][bot][slc], **vwargs)

		ax[i].set(xlim=[ap_min, ap_max], ylim=[dv_min, dv_max])

		cut = get_cut(m[ii])
		color = cmap((t[ii]+20)/(t.max()+20))
		ax[-1].plot(dv, cut[:midpoint], color=color)

		cut = cut[midpoint:]
		ax_Y = ax[i].inset_axes([-0.25, 0, 0.2, 1], sharey=ax[i])
		ax_Y.plot(cut, dv+dv_midpoint, color=color, lw=0.5)
		ax_Y.fill_betweenx(dv+dv_midpoint, 0, cut, color=color, alpha=0.7)
		axes.append(ax_Y)

	ax[-1].set_xlim([dv_min+20, dv_midpoint])
	ax[-1].set(xticks=[], yticks=[], xlabel='DV')
	ax[-1].autoscale(axis='y')
	for axis in axes:
		axis.set_xlim(ax[-1].get_ylim())
		axis.invert_xaxis()
		axis.axis('off')
	
	fig.subplots_adjust(right=0.98)
	norm = Normalize(vmin=-20, vmax=np.max(t))
	plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
				 ax=ax[-1], label='Time (min)', ticks=[])

def comparison_plot(t, *fields,
					dt=5,
					mask=np.s_[..., 20:-20, 0:-25],
					al=0.7, labelpad=16):
	'''
	Plot a comparison plot of multiple fields
	NOTE: assumes the last field is the velocity field and won't plot DV cuts
	Instead, it'll do a residual plot in the last row last column
	'''
	N = int(np.ceil(len(t) / dt))
	n_rows = len(fields)
	fig, ax = plt.subplots(n_rows, N+1, 
						   figsize=(N+1, n_rows*1.2),
						   constrained_layout=True)

	vfc = 40
	skip = 20
	ylim = [dv_min, dv_max]
	xlim = [ap_min, ap_max]

	colors = ['Reds', 'Blues', 'Greys']

	for i, (field, z, z0) in enumerate(fields):
		znorm = z0.reshape([z0.shape[0], -1, *z0.shape[-2:]])
		n_channels = znorm.shape[1]
		znorm = np.linalg.norm(znorm[mask], axis=1)[..., :-20] #ignore posterior pole
		kwargs = dict(vmin=np.min(znorm), vmax=np.max(znorm))
		
		alpha = np.zeros(z.shape[-2:])
		alpha0 = np.zeros_like(alpha)
		midpoint = alpha.shape[0] // 2
		alpha[midpoint:] = 1
		alpha0[:midpoint] = al

		dv = np.linspace(dv_min, dv_midpoint, midpoint)
		
		slc = np.s_[..., ::skip, ::skip]
		X = np.linspace(ap_min, ap_max, z.shape[-1])
		Y = np.linspace(dv_min, dv_max, z.shape[-2])
		Y, X = np.meshgrid(Y, X, indexing='ij')
	
		top = np.s_[..., midpoint:, :]
		bot = np.s_[..., :midpoint, :]
		vwargs = dict(pivot='middle', width=0.005)
		
		axes = []
		cmap = plt.get_cmap(colors[i])
		for j in range(N):
			jj = dt * j
			if i == 0:
				ax[i, j].set_title(f't = {t[jj]:.0f}')
			
			color = cmap((t[jj] + 20) / (t.max() + 20))	

			if n_channels == 1:
				color_2D(ax[i, j], z[jj], alpha=alpha, **kwargs)
				color_2D(ax[i, j], z0[jj], alpha=alpha0, **kwargs)
			elif n_channels == 2:
				ax[i, j].quiver(X[bot][slc], Y[bot][slc],
								z0[jj, 1][bot][slc], z0[jj, 0][bot][slc],
								color='grey', **vwargs)
				ax[i, j].quiver(X[top][slc], Y[top][slc],
								z[jj, 1][top][slc], z[jj, 0][top][slc],
								color='black', **vwargs)
				ax[i, j].set(xticks=[], yticks=[])
			elif n_channels == 4:
				kwargs['both'] = t[jj] < 0
				plot_tensor2D(ax[i, j], z[jj], alpha=alpha, **kwargs)
				plot_tensor2D(ax[i, j], z0[jj], alpha=alpha0, **kwargs)

			ax[i, j].set(xlim=xlim, ylim=ylim)
			
			if i == n_rows - 1:
				continue


			cut = get_cut(z[jj])
			cut0 = get_cut(z0[jj])
			ax[i, -1].plot(dv, cut[:midpoint], color=color)
			ax[i, -1].plot(dv, cut0[:midpoint], color=color, linestyle='--')
			
			cut = cut[midpoint:]
			cut0 = cut0[:midpoint]
			ax_Y = ax[i, j].inset_axes([-0.25, 0, 0.2, 1], sharey=ax[i, j])
			ax_Y.plot(cut, dv+dv_midpoint, color=color, lw=0.5)
			ax_Y.fill_betweenx(dv+dv_midpoint, 0, cut, color=color, alpha=0.7)
			ax_Y.plot(cut0, dv, color=color, lw=0.5, alpha=al)
			ax_Y.fill_betweenx(dv, 0, cut0, color=color, alpha=0.7*al)
			axes.append(ax_Y)

		if n_channels == 4:
			res = mean_norm_residual(z0[mask], z[mask]).mean(axis=(1, 2))
			ax[-1, -1].plot(t, res, color=color, label=field)
			labelcolor=color
		else:
			pass
			res = residual(z0[mask], z[mask]).mean(axis=(1, 2))

		if n_channels == 2:
			znorm = znorm.mean(axis=(1, 2))
			ax2 = ax[-1, -1].twinx()
			ax2.plot(t, znorm, color='black')
			ax2.set_yticklabels([])
			ax2.set_ylabel('Mean flow\n($\\mu$m / min)', labelpad=0)


		ax[i, 0].set_ylabel(field, labelpad=labelpad)

		if i == n_rows - 1:
			continue
		ax[i, -1].set_xlim([dv_min+vfc, dv_midpoint])
		ax[i, -1].set(xticks=[], yticks=[], xlabel='DV')
		ax[i, -1].autoscale(axis='y')
		for axis in axes:
			axis.set(xlim=ax[i, -1].get_ylim())
			axis.invert_xaxis()
			axis.axis('off')

	ax[-1, -1].set_ylabel('Error Rate', labelpad=0, color=labelcolor)
	ax[-1, -1].set(ylim=[-0.05, 1.05])
	
	norm = Normalize(vmin=-20, vmax=np.max(t))
	fig.subplots_adjust(right=0.97, wspace=0.35, hspace=0.3)
	for i in range(n_rows-1):
		pos = ax[i, 0].get_position()
		cax = fig.add_axes([0.98, pos.y0, 0.01, pos.y1-pos.y0])
		plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(colors[i])),
					 cax=cax, label='Time (min)', ticks=[])
