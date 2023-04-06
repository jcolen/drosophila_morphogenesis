import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

from .geometry_utils import embryo_mesh

'''
3D plotting
'''
def format_ax(ax, a0=-90, a1=-90, title=''):
	verts = embryo_mesh.coordinates()
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


def color_3D(ax, f, 
			   cmap='viridis', 
			   linewidth=0., 
			   title='', 
			   vmin=None, vmax=None):
	verts = embryo_mesh.coordinates()
	faces = embryo_mesh.cells()
	p3dc = ax.plot_trisurf(
		verts[:, 2], verts[:, 0], verts[:, 1],
		edgecolor='lightgray', linewidth=linewidth, triangles=faces)
	f_faces = np.zeros(faces.shape[0])
	for fid, face in enumerate(faces):
		f_faces[fid] = np.mean(f[face])


	if vmin is None:
		vmin = np.min(f)
		vmax = np.max(f)
	norm = Normalize(vmin=vmin, vmax=vmax)

	colors = get_cmap(cmap)(norm(f_faces))
	p3dc.set_fc(colors)
	format_ax(ax, title=title)

def plot_vector3D(ax, f, title='', cmap='Greys', normalize=True):
	verts = embryo_mesh.coordinates()
	color_3D(ax, np.zeros(verts.shape[0]), title=title, cmap=cmap)
	skip = 3

	mask = verts[:, 1] < 0
	fv = f[:, mask][:, ::skip]
	rv = verts[mask][::skip, :].T

	qwargs = dict(pivot='middle', color='black', length=1e2, normalize=normalize, linewidth=.5)
	ax.quiver(rv[2], rv[0], rv[1],
			  fv[2], fv[0], fv[1],
			  **qwargs)

def plot_tensor3D(ax, f0, skip=6, title='', cmap='inferno', vmin=None, vmax=None):
	f = f0.copy().reshape([3, 3, -1]).transpose(2, 0, 1)
	fnorm = np.linalg.norm(f, axis=(1, 2))
	color_3D(ax, fnorm, title=title, cmap=cmap, vmin=vmin, vmax=vmax)

	el, ev = np.linalg.eig(f)
	order = np.argmax(el, axis=-1) #Find position of largest eigenvector
	ev = np.array([ev[i, :, ei] for i, ei in enumerate(order)])
	qwargs = dict(pivot='middle', color='white', length=100, normalize=True, arrow_length_ratio=0.5, linewidth=0.5)
	
	verts = embryo_mesh.coordinates()
	mask = verts[:, 1] < 0
	ev = ev[mask][::skip, :].T
	rm = verts[mask][::skip, :].T
	ax.quiver(rm[2], rm[0], rm[1],
			  ev[2], ev[0], ev[1],
			  **qwargs)
