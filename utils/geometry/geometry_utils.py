import os
import numpy as np

from scipy.io import loadmat
from scipy import sparse
from scipy.interpolate import RectBivariateSpline, griddata
from sklearn.neighbors import KDTree

from fenics import Mesh

basedir = '/project/vitelli/jonathan/REDO_fruitfly/src'
geo_dir = os.path.join(
	'/project/vitelli/jonathan/REDO_fruitfly/',
	'flydrive.synology.me/Public/dynamic_atlas/embryo_geometry')

embryo_mesh = Mesh(os.path.join(geo_dir, 'embryo_coarse_noll.xml'))
tangent_space_data = loadmat(os.path.join(geo_dir, 'embryo_3D_geometry.mat'))

z_emb = tangent_space_data['z'].squeeze()
phi_emb = tangent_space_data['ph'].squeeze()

e1 = tangent_space_data['e2'] #We use y-first convention
e2 = tangent_space_data['e1'] #We use y-first convention
num_vertices = e1.shape[0]
E11, E12, E21, E22 = [], [], [], []
E1, E2 = [], []
for i in range(3):
	E1.append(sparse.spdiags(e1[:, i], 0, num_vertices, num_vertices))
	E2.append(sparse.spdiags(e2[:, i], 0, num_vertices, num_vertices))
	for j in range(3):
		E11.append(sparse.spdiags(e1[:, i]*e1[:, j], 0, num_vertices, num_vertices))
		E12.append(sparse.spdiags(e1[:, i]*e2[:, j], 0, num_vertices, num_vertices))
		E21.append(sparse.spdiags(e2[:, i]*e1[:, j], 0, num_vertices, num_vertices))
		E22.append(sparse.spdiags(e2[:, i]*e2[:, j], 0, num_vertices, num_vertices))

N_vector = sparse.vstack([sparse.hstack(E1), sparse.hstack(E2)])
N_tensor = sparse.vstack([sparse.hstack(E11), sparse.hstack(E12), 
						  sparse.hstack(E21), sparse.hstack(E22)])

def left_right_symmetrize(pc, r_verts=embryo_mesh.coordinates()):
	#Build coordinate points for flipped mesh
	l_faces = r_verts.copy()
	l_faces[..., 1] *= -1
	
	tree = KDTree(l_faces)
	dist, ind = tree.query(r_verts, k=1)
	ind = ind[:, 0]
	
	reflected = pc[:, ind]
	
	if reflected.shape[0] == 3:
		reflected[1] *= -1
	elif reflected.shape[0] == 9:
		reflected[1] *= -1
		reflected[3] *= -1
		reflected[5] *= -1
		reflected[7] *= -1
	
	return 0.5 * (pc + reflected)

def interpolate_grid_to_mesh_vertices(f0):
	header_shape = f0.shape[:-2]
	f1 = f0.reshape([-1, *f0.shape[-2:]])

	#Interpolate to vertex points
	Z_AP = np.linspace(z_emb.min(), z_emb.max(), f0.shape[-1])
	Phi_DV = np.linspace(phi_emb.min(), phi_emb.max(), f0.shape[-2]) #This goes the opposite direction of Y

	f = []
	for i in range(f1.shape[0]):
		fi = RectBivariateSpline(Phi_DV, Z_AP, f1[i])(phi_emb, z_emb, grid=False)
		f.append(fi)
	f = np.stack(f).reshape([*header_shape, -1])

	#Account for switch of DV direction
	if len(f.shape) == 2: #It's a vector
		f[0] *= -1 #Since we inverted phi

	elif len(f.shape) == 3: #It's a tensor
		f[1, 0] *= -1
		f[0, 1] *= -1

	return f

def pull_vector_from_tangent_space(M):
	Mi = M.flatten()
	return np.einsum('Vi,V->iV', e1, Mi[:num_vertices]) + \
		   np.einsum('Vi,V->iV', e2, Mi[num_vertices:])

def pull_tensor_from_tangent_space(M):
	Mi = M.flatten()
	return np.einsum('Vi,V,Vj->ijV', e1, Mi[:num_vertices], e1) + \
		   np.einsum('Vi,V,Vj->ijV', e1, Mi[num_vertices:2*num_vertices], e2) + \
		   np.einsum('Vi,V,Vj->ijV', e2, Mi[2*num_vertices:3*num_vertices], e1) + \
		   np.einsum('Vi,V,Vj->ijV', e2, Mi[3*num_vertices:], e2)

def pull_from_tangent_space(f0):
	if f0.shape[-1] != num_vertices:
		f = interpolate_grid_to_mesh_vertices(f0)
	else:
		f = f0

	#Now convert using embryo surface basis vectors
	if len(f.shape) == 2: #Transforms like a vector
		return pull_vector_from_tangent_space(f.flatten())
	elif len(f.shape) == 3: #Transforms like a tensor
		return pull_tensor_from_tangent_space(f.flatten())
	else:
		return f

def interpolate_mesh_vertices_to_grid(f0, nDV=236, nAP=200):
	f1 = f0.reshape([-1, num_vertices])
	
	#Interpolate to vertex points
	Z_AP = np.linspace(z_emb.min(), z_emb.max(), nAP)
	Phi_DV = np.linspace(phi_emb.min(), phi_emb.max(), nDV) #This goes the opposite direction of Y

	f = []
	for i in range(f1.shape[0]):
		fi = griddata(
			(phi_emb, z_emb), f1[i], 
			(Phi_DV[:, None], Z_AP[None, :])
		)
		fi_nearest = griddata(
			(phi_emb, z_emb), f1[i], 
			(Phi_DV[:, None], Z_AP[None, :]),
			method='nearest'
		)
		fi[np.isnan(fi)] = fi_nearest[np.isnan(fi)]
			
		f.append(fi)
	f = np.stack(f)
	
	if f.shape[0] == 2: #It's a vector
		f[0] *= -1
	elif f.shape[0] == 4: #It's a tensor
		f[1:3] *= -1
	
	return f

def push_to_tangent_space(f0, grid=True):
	if np.prod(f0.shape) // num_vertices == 3: #Vector
		f = N_vector.dot(f0.flatten()[:, None])
	elif np.prod(f0.shape) // num_vertices == 9: #Tensor
		f = N_tensor.dot(f0.flatten()[:, None])
	else:
		f = f0

	if grid:
		return interpolate_mesh_vertices_to_grid(f)
	
	return f
