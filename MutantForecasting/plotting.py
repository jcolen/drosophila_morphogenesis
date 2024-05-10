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
        
        print(mmin, mmax)
    
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
    
    ax[0,-1].set(xlim=[dv_min+vfc, dv_midpoint])
    ax[0,-1].set(xticks=[dv_min+vfc, dv_midpoint], xticklabels=['D', 'V'])
    #ax[0,-1].set(xticks=[dv_min+vfc, dv_midpoint, dv_max-vfc], xticklabels=['D', 'V', 'D'])
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

def residual_plot(t, m, m0, v, v0):
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

from scipy.interpolate import RectBivariateSpline as interp2d
from math import floor, ceil
def get_velocity(v, tt, X0, Y0, x, y):
    idxA, idxB = floor(tt), ceil(tt)
    frac = 1 - (tt - idxA)
    vA, vB = v[idxA], v[idxB]
        
    vx = interp2d(Y0, X0, frac * vA[1] + (1 - frac) * vB[1])(y, x, grid=False)
    vy = interp2d(Y0, X0, frac * vA[0] + (1 - frac) * vB[0])(y, x, grid=False)
    return vx, vy

def rk4DynamicVelocity2D(pts, X0v, Y0v, v, hh):
    tt = np.arange(0, v.shape[0]-1, hh)
    xyPathsMat = np.zeros([pts.shape[0], len(tt), 2])

    x = pts[:, 0]
    y = pts[:, 1]
    xyPathsMat[:, 0, 0] = x
    xyPathsMat[:, 0, 1] = y
    
    Xmin, Xmax = X0v[0], X0v[-1]
    Ymin, Ymax = Y0v[0], Y0v[-1]
    
    for ii in range(len(tt) - 1):
        k1x, k1y = get_velocity(v, tt[ii], X0v, Y0v, x, y)
        k2x, k2y = get_velocity(v, tt[ii] + 0.5 * hh, X0v, Y0v, x + 0.5 * hh * k1x, y + 0.5 * hh * k1y)
        k3x, k3y = get_velocity(v, tt[ii] + 0.5 * hh, X0v, Y0v, x + 0.5 * hh * k2x, y + 0.5 * hh * k2y)
        k4x, k4y = get_velocity(v, tt[ii] + hh, X0v, Y0v, x + hh * k3x, y + hh * k3y)
        
        #Main equation
        x = x + (k1x + 2 * k2x + 2 * k3x + k4x) * hh / 6.
        y = y + (k1y + 2 * k2y + 2 * k3y + k4y) * hh / 6.
        
        x[x > Xmax] = Xmax
        x[x < Xmin] = Xmin
        y[y > Ymax] = Ymax
        y[y < Ymin] = Ymin
        
        xyPathsMat[:, ii, 0] = x
        xyPathsMat[:, ii, 1] = y
            
    return xyPathsMat

def trajectory_plot(v0, v1, nDV=8, nAP=8, hh=0.2):
    '''
    Plot trajectories of cells in test flow field
    '''
    DV = np.linspace(dv_min + 25, dv_max - 25, nDV)
    AP = np.linspace(ap_min + 25, ap_max - 25, nAP)
    pts = np.stack(np.meshgrid(AP, DV, indexing='xy'), axis=-1).reshape([-1, 2])

    DV = np.linspace(dv_min, dv_max, 236)
    AP = np.linspace(ap_min, ap_max, 200)
    
    y0Paths = rk4DynamicVelocity2D(pts, AP, DV, v0, hh)
    y1Paths = rk4DynamicVelocity2D(pts, AP, DV, v1, hh)
    
    fig, ax = plt.subplots(1, 1, dpi=200)
    nPts = y0Paths.shape[0]
    nTps = y0Paths.shape[1]
    for tpId in range(nTps - 1):
        ax.scatter(y0Paths[:, tpId, 0], y0Paths[:, tpId, 1], s=3,
                   c=np.ones(nPts) * tpId,
                   cmap='Blues', vmin=-10, vmax=nTps*1.2)
        ax.scatter(y1Paths[:, tpId, 0], y1Paths[:, tpId, 1], s=3,
                   c=np.ones(nPts) * tpId,
                   cmap='Reds', vmin=-10, vmax=nTps*1.2)
    
    ax.set(xlim=[ap_min, ap_max], xticks=[ap_min, ap_max], xticklabels=['A', 'P'])
    ax.set(ylim=[dv_min, dv_max], yticks=[dv_min, dv_midpoint, dv_max], yticklabels=['D', 'V', 'D'])
    ax.set_aspect('equal')
    
    ax.text(0.2, 1.02, 'Experiment', color='Blue', va='bottom', ha='center', transform=ax.transAxes)
    ax.text(0.8, 1.02, 'SINDy', color='Red', va='bottom', ha='center', transform=ax.transAxes)
    