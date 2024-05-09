import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d
from matplotlib.colors import Normalize

import sys
import os
basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'release'))

from utils.plot_utils import plot_tensor2D, color_2D
from utils.plot_utils import ap_min, ap_max, dv_min, dv_max, dv_midpoint
from utils.plot_utils import residual, mean_norm_residual

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

def sqh_vel_plot(t,
                 m, m0,
                 v, v0,
                 dt=10,
                 mask=np.s_[..., 20:-20, 0:-25],
                 mmin=None, mmax=None,
                 al=0.7, labelpad=16):

    vfc = 40
    skip = 20
    slc = np.s_[..., ::skip, ::skip]
    X = np.linspace(ap_min, ap_max, 200)
    Y = np.linspace(dv_min, dv_max, 236)
    Y, X = np.meshgrid(Y, X, indexing='ij')

    top = np.s_[..., 118:, :]
    bot = np.s_[..., :118, :]
    vwargs = dict(pivot='middle', width=0.005, scale=5e1, color='black')
    
    dv = np.linspace(dv_min, dv_max, 236)

    colors = ['Reds', 'Greys']
    
    alpha = np.zeros([236, 200])
    alpha[:118, :] = 1
    
    if mmin is None and mmax is None:
        mnorm = m0.reshape([-1, 4, 236, 200])
        mnorm = np.linalg.norm(mnorm, axis=1)[..., :-20] #Ignore posterior pole
        mmin = np.min(mnorm)
        mmax = np.max(mnorm)
    
    N = int(np.ceil(len(t) / dt))
    fig, ax = plt.subplots(2, N+1, 
                           figsize=(N+1, 2*1.2),
                           constrained_layout=False)
    
    cmap = plt.cm.Reds
    
    ax[0,0].set_ylabel('Myosin')
    ax[1,0].set_ylabel('Velocity')
    
    for j in range(N):
        jj = dt * j
        ax[0, j].set_title(f't = {t[jj]:.0f}')
        
        color = cmap((t[jj] + 20) / (t.max() + 20))
        
        plot_tensor2D(ax[0,j], m0[jj] * alpha, alpha=alpha, vmin=mmin, vmax=mmax, both=t[jj]<0)
        plot_tensor2D(ax[0,j], m[jj] * (1-alpha), alpha=(1-alpha), vmin=mmin, vmax=mmax, both=t[jj]<0)
        
        ax[0,j].set(xlim=[ap_min, ap_max], ylim=[dv_min, dv_max])
        
        ax[1,j].quiver(X[bot][slc], Y[bot][slc], 
                       v0[jj,1][bot][slc], v0[jj,0][bot][slc], **vwargs)
        ax[1,j].quiver(X[top][slc], Y[top][slc], 
                       v[jj,1][top][slc], v[jj,0][top][slc], **vwargs)
        ax[1,j].set(xlim=[ap_min, ap_max], ylim=[dv_min, dv_max], aspect=ax[0,j].get_aspect(), xticks=[], yticks=[])


        cut = get_cut(m[jj])
        cut0 = get_cut(m0[jj])
        ax[0, -1].plot(dv, cut, color=color)
        ax[0, -1].plot(dv, cut0, color=color, linestyle='--')
        

    vnorm = np.linalg.norm(v, axis=1).mean(axis=(1, 2))
    v0norm = np.linalg.norm(v0, axis=1).mean(axis=(1,2))
    
    ax[1,-1].plot(t, v0norm, color='black', linestyle='--')
    ax[1,-1].plot(t, vnorm,  color='black', linestyle='-')
    
    ax[0,-1].set(xlim=[dv_min+vfc, dv_max-vfc])
    ax[0,-1].set(xticks=[dv_min+vfc, dv_midpoint, dv_max-vfc], xticklabels=['D', 'V', 'D'])
    ax[0,-1].set(ylim=[0, 0.25], yticks=[0, 0.1, 0.2])
    ax[0,-1].set_ylabel('Intensity')
    
    ax[1, -1].set_xlabel('Time (min)')
    ax[1, -1].set(xlim=[-10, 20], ylim=[0, 4], xticks=[-10, 0, 10, 20], yticks=[0, 2, 4])
    ax[1, -1].set_ylabel('Mean flow')

    norm = Normalize(vmin=-20, vmax=np.max(t))
    fig.subplots_adjust(right=0.97)
    pos = ax[0,0].get_position()
    cax = fig.add_axes([0.98, pos.y0, 0.01, pos.y1-pos.y0])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Reds'),
                 cax=cax, label='Time (min)', ticks=[])
    plt.tight_layout()

    fig1, ax = plt.subplots(1, 1, figsize=(1, 1))
    colors = ['firebrick', 'black']
    ax.plot(t, residual(m0, m).mean(axis=(1,2)), color='firebrick', label='Myosin')
    ax.plot(t, residual(v0, v).mean(axis=(1,2)), color='black', label='Velocity')
    ax.axhline(0.25, zorder=0, color='grey', linestyle='--')
    ax.text(1.01, 0.25, '25%', transform=ax.transAxes,
            color='grey', va='center', ha='left')
    ax.set_ylabel('Error Rate')
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Time')
    ax.set_xticks([-10, 0, 10, 20])
    ax.legend(fontsize=6)

    return fig, fig1


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

    colors = ['Reds', 'Greys']

    res = []

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
                color_2D(ax[i, j], z[jj, ::-1], alpha=alpha, **kwargs)
                color_2D(ax[i, j], z0[jj], alpha=alpha0, **kwargs)
            elif n_channels == 2:
                #vwargs['scale'] = 5e1 if t[jj] > 5 else 2e1
                vwargs['scale'] = 5e1
                ax[i, j].quiver(X[bot][slc], Y[bot][slc],
                                z0[jj, 1][bot][slc], z0[jj, 0][bot][slc],
                                color='grey', **vwargs)
                ax[i, j].quiver(X[top][slc], Y[top][slc],
                                z[jj, 1, ::-1][top][slc], -z[jj, 0, ::-1][top][slc],
                                color='black', **vwargs)
                ax[i, j].set(xticks=[], yticks=[])
            elif n_channels == 4:
                kwargs['both'] = t[jj] < 0
                zj = z[jj, :, :, ::-1].copy()
                zj[0, 1] *= -1
                zj[1, 0] *= -1
                plot_tensor2D(ax[i, j], zj * alpha, alpha=alpha, **kwargs)
                plot_tensor2D(ax[i, j], z0[jj] * np.sign(alpha0), alpha=alpha0, **kwargs)

            ax[i, j].set(xlim=xlim, ylim=ylim)

            if i == n_rows - 1:
                continue


            cut = get_cut(z[jj, ..., ::-1, :])
            cut0 = get_cut(z0[jj])
            ax[i, -1].plot(dv, cut[::-1][:midpoint], color=color)
            ax[i, -1].plot(dv, cut0[:midpoint], color=color, linestyle='--')

            cut = cut[midpoint:]
            cut0 = cut0[:midpoint]
            ax_Y = ax[i, j].inset_axes([-0.25, 0, 0.2, 1], sharey=ax[i, j])
            ax_Y.plot(cut, dv+dv_midpoint, color=color, lw=0.5)
            ax_Y.fill_betweenx(dv+dv_midpoint, 0, cut, color=color, alpha=0.7)
            ax_Y.plot(cut0, dv, color=color, lw=0.5, alpha=al)
            ax_Y.fill_betweenx(dv, 0, cut0, color=color, alpha=0.7*al)
            axes.append(ax_Y)

        ax[i, 0].set_ylabel(field, labelpad=labelpad)
        #res.append(residual(z0[mask], z[mask]).mean(axis=(1, 2)))
        if i > 0:
            res.append(residual(z0[mask], z[mask]).mean(axis=(1, 2)))
        else:
            res.append(mean_norm_residual(z0[mask], z[mask]).mean(axis=(1, 2)))

        if n_channels == 2:
            znorm = znorm.mean(axis=(1, 2))
            z0norm = np.linalg.norm(
                z0.reshape([z0.shape[0], -1, *z0.shape[-2:]])[..., :-20], axis=1)
            znorm = np.linalg.norm(
                z.reshape([z.shape[0], -1, *z.shape[-2:]])[..., :-20], axis=1)
            ax[i, -1].plot(t, z0norm.mean(axis=(-1, -2)), color='black', linestyle='--')
            ax2 = ax[i, -1]
            ax2.plot(t, znorm.mean(axis=(-1, -2)), color='black')
            ax2.set_yticks([])
            ax2.set_yticklabels([])
            ax[i, -1].set_xlabel('Time (min)')
            ax[i, -1].set_xticks([-10, 0, 10, 20])
            ax[i, -1].set_ylabel('Mean flow\n($\\mu$m / min)', labelpad=0)
        else:
            ax[i, -1].set_xlim([dv_min+vfc, dv_midpoint])
            ax[i, -1].set(xticks=[], yticks=[], xlabel='DV')
            ax[i, -1].autoscale(axis='y')
            for axis in axes:
                axis.set(xlim=ax[i, -1].get_ylim())
                axis.invert_xaxis()
                axis.axis('off')

    norm = Normalize(vmin=-20, vmax=np.max(t))
    fig.subplots_adjust(right=0.97, wspace=0.35, hspace=0.3)
    for i in range(n_rows-1):
        pos = ax[i, 0].get_position()
        cax = fig.add_axes([0.98, pos.y0, 0.01, pos.y1-pos.y0])
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(colors[i])),
                     cax=cax, label='Time (min)', ticks=[])

    fig1, ax = plt.subplots(1, 1, figsize=(1, 1))
    colors = ['firebrick', 'black']
    for i in range(2):
        ax.plot(t, res[i], color=colors[i], label=fields[i][0])
    ax.axhline(0.25, zorder=0, color='grey', linestyle='--')
    ax.text(1.01, 0.25, '25%', transform=ax.transAxes,
            color='grey', va='center', ha='left')
    ax.set_ylabel('Error Rate')
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Time')
    ax.set_xticks([-10, 0, 10, 20])
    ax.legend(fontsize=6)

    return fig, fig1
