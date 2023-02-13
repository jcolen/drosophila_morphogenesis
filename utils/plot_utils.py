import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import get_cmap
import matplotlib
import numpy as np
from .geometry_utils import mesh

plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.cmap'] = 'inferno'

def residual(u0, v0):
	'''
	Residual metric from Streichan eLife paper
	'''
	u = u0.reshape([u0.shape[0], -1, *u0.shape[-2:]])
	v = v0.reshape([v0.shape[0], -1, *v0.shape[-2:]])

	umag = np.linalg.norm(u, axis=-3)												  
	vmag = np.linalg.norm(v, axis=-3)												  

	uavg = np.sqrt((umag**2).mean(axis=(-2, -1), keepdims=True))					
	vavg = np.sqrt((vmag**2).mean(axis=(-2, -1), keepdims=True))					

	res = uavg**2 * vmag**2 + vavg**2 * umag**2 - 2 * uavg * vavg * np.einsum('...ijk,...ijk->...jk', u, v)
	res /= 2 * vavg**2 * uavg**2														
	return res 


'''
2D plotting
'''

def color_2D(ax, f, vmax_std=None, **im_kwargs):
	if vmax_std:
		im_kwargs['vmax'] = vmax_std * np.std(f)
	if len(f.shape) > 2:
		ax.imshow(np.linalg.norm(f, axis=0), **im_kwargs)
	else:
		ax.imshow(f, **im_kwargs)
	ax.set(xticks=[], yticks=[])

def plot_tensor2D(ax, f0, skip=20, both=False, linecolor='white', linewidth=0.007, **im_kwargs):
	f = f0.copy()
	f = f.reshape([4, *f.shape[-2:]])
	color_2D(ax, np.linalg.norm(f, axis=0), **im_kwargs)

	Y, X = np.mgrid[0:f.shape[-2]:skip, 0:f.shape[-1]:skip]
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

	if both:
		for i in range(ev.shape[-1]):
			ax.quiver(X, Y, ev[:, 1, i], ev[:, 0, i], **qwargs)
	else:
		order = np.argmax(el, axis=-1)
		ev = np.array([ev[i, :, ei] for i, ei in enumerate(order)])
		ax.quiver(X, Y, ev[:, 1], ev[:, 0], **qwargs)

def plot_vector2D(ax, f, skip=10, color='black', width=0.005, **kwargs):
	Y, X = np.mgrid[0:f.shape[-2]:skip, 0:f.shape[-1]:skip]
	X = X.flatten()
	Y = Y.flatten()
	f0 = f[:, ::skip, ::skip].reshape([2, -1])
	ax.quiver(X, Y, f0[1], f0[0], pivot='middle', color=color, width=width, **kwargs)
	ax.set(xticks=[], yticks=[])

'''
3D plotting
'''
def format_ax(ax, mesh=mesh, a0=-90, a1=-90, title=''):
	verts = mesh.coordinates()
	ax.view_init(a0, a1)
	ax.set_box_aspect((
		np.ptp(verts[:, 2]),
		np.ptp(verts[:, 0]),
		np.ptp(verts[:, 1])))  # aspect ratio is 1:1:1 in data space#set_axes_equal(ax)
	axmin = np.min(verts, axis=0)
	axmax = np.max(verts, axis=0)
	ax.set(
		xlim=[axmin[2], axmax[2]],
		ylim=[axmin[0], axmax[0]],
		zlim=[axmin[1], axmax[1]],
	)
	ax.dist = 10.
	ax.patch.set_alpha(0.)
	ax.axis('off')
	ax.text2D(0.5, 0.7, title,
		  transform=ax.transAxes,
		  va='bottom', ha='center')


def color_mesh(ax, f, norm=None, cmap='magma', linewidth=0., title='', offset=np.zeros(3)):
	verts = mesh.coordinates().copy()
	verts += offset
	faces = mesh.cells()
	p3dc = ax.plot_trisurf(
		verts[:, 2], verts[:, 0], verts[:, 1],
		edgecolor='lightgray', linewidth=linewidth, triangles=faces)
	f_faces = np.zeros(faces.shape[0])
	for fid, face in enumerate(faces):
		f_faces[fid] = np.mean(f[face])

	if norm is None:
		norm = Normalize()
	colors = get_cmap(cmap)(norm(f_faces))
	p3dc.set_fc(colors)
	format_ax(ax, title=title)

def plot_vector3D(ax, f, title='', normalize=True):
	verts = mesh.coordinates()
	color_mesh(ax, np.linalg.norm(f, axis=0), title=title)
	skip = 3

	mask = verts[:, 1] < 0
	fv = f[:, mask][:, ::skip]
	rv = verts[mask][::skip, :].T

	qwargs = dict(pivot='middle', color='white', length=1e2, normalize=normalize, linewidth=.5)
	ax.quiver(rv[2], rv[0], rv[1],
			  fv[2], fv[0], fv[1],
			  **qwargs)

def plot_tensor3D(ax, f0, skip=3, title='', norm=None, vmax=None):
	verts = mesh.coordinates()
	f = f0.copy().reshape([3, 3, -1]).transpose(2, 0, 1)
	fnorm = np.linalg.norm(f, axis=(1, 2))
	fnorm = np.einsum('vii->v', f)
	if norm is None:
		norm = Normalize(vmin=0, vmax=vmax*np.std(fnorm)) if vmax else None
	color_mesh(ax, fnorm, norm=norm, title=title)
	el, ev = np.linalg.eig(f)

	order = np.argmax(el, axis=-1) #Find position of largest eigenvector
	ev = np.array([ev[i, :, ei] for i, ei in enumerate(order)])
	qwargs = dict(pivot='middle', color='white', length=100, normalize=True, arrow_length_ratio=0.5, linewidth=0.5)
	mask = verts[:, 1] < 0
	ev = ev[mask][::skip, :].T
	rm = verts[mask][::skip, :].T
	ax.quiver(rm[2], rm[0], rm[1],
			  ev[2], ev[0], ev[1],
			  **qwargs)

'''
Tangent space plotting
'''

from .geometry_utils import pull_vector_from_tangent_space
from .geometry_utils import e1, e2, pull_tensor_from_tangent_space

def plot_tangent_space_vector(ax, f0, **kwargs):
	plot_vector3D(ax, pull_vector_from_tangent_space(f0), **kwargs)


def plot_tangent_space_tensor(ax, f0, e1=e1, e2=e2, offset=np.zeros(3), skip=3, norm=None, vmax=None, title=''):
	verts = mesh.coordinates().copy()
	verts += offset
	f = f0.copy().reshape([2, 2, -1]).transpose(2, 0, 1)
	fnorm = np.linalg.norm(f, axis=(1, 2))
	if norm is None:
		norm = Normalize(vmin=0, vmax=vmax*np.std(fnorm)) if vmax else None
	color_mesh(ax, fnorm, norm=norm, title=title, offset=offset)
	el, ev = np.linalg.eig(f)
	
	order = np.argmax(el, axis=-1) #Find position of largest eigenvector
	ev = np.array([ev[i, :, ei] for i, ei in enumerate(order)])
	qwargs = dict(pivot='middle', color='white', length=100, normalize=True, arrow_length_ratio=0.5, linewidth=0.25)
	mask = verts[:, 1] - offset[1] < 0
	rm = verts[mask][::skip, :].T
	ev = ev[:, 0] * e1.T + ev[:, 1] * e2.T
	ev = ev[:, mask][:, ::skip]
	ax.quiver(rm[2], rm[0], rm[1], 
			  ev[2], ev[0], ev[1],
			  **qwargs)
