import os
import numpy as np
import torch
from torch.nn import Parameter, ParameterList, ParameterDict, Module

from string import ascii_lowercase

from scipy.io import loadmat
from scipy import sparse
from scipy.interpolate import RectBivariateSpline, griddata

from sklearn.base import BaseEstimator, TransformerMixin

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

class MeshInterpolator(BaseEstimator, TransformerMixin):
	'''
	Interpolates grid points to mesh vertices
	Inverse transform interpolates mesh vertices to grid points

	This transformer only operates in cpu mode
	'''
	def __init__(self, 
				 mesh_name='embryo_coarse_noll',
				 cutoff=0.05):
		self.mesh_name = mesh_name
		self.cutoff = cutoff

	def fit(self, X, y0=None):
		mesh = Mesh(os.path.join(geo_dir, f'{self.mesh_name}.xml'))
		tangent_space_data = loadmat(os.path.join(geo_dir, f'{self.mesh_name}.mat'))

		self.z_emb = tangent_space_data['z'].squeeze()
		self.phi_emb = tangent_space_data['ph'].squeeze()
		self.n_vertices_ = mesh.coordinates().shape[0]	

		return self

	def transform(self, X):
		header_shape = X.shape[:-2]
		x0 = X.reshape([-1, *X.shape[-2:]])

		if torch.is_tensor(X):
			mode = 'torch'
			x0 = x0.numpy()
		else: 
			mode = 'numpy'

		#Interpolate to vertex points
		Z_AP = np.linspace(self.z_emb.min(), 
						   self.z_emb.max(), 
						   X.shape[-1])
		Phi_DV = np.linspace(self.phi_emb.min(), 
							 self.phi_emb.max(), 
							 X.shape[-2]) #This goes the opposite direction of Y

		mask = np.logical_and(Z_AP >= self.cutoff, Z_AP <= 1 - self.cutoff)

		x1 = []
		for i in range(x0.shape[0]):
			xi = RectBivariateSpline(Phi_DV, Z_AP[mask], x0[i][..., mask])(self.phi_emb, self.z_emb, grid=False)
			x1.append(xi)
		x1 = np.stack(x1)

		#Account for switch of DV direction
		if x1.shape[0] == 2: #It's a vector
			x1[0] *= -1 #Since we inverted phi

		elif x1.shape[0] == 4: #It's a tensor
			x1[1:3] *= -1
			x1 = x1.reshape([2, 2, -1])
		else:
			x1 = x1.squeeze()
		
		if mode == 'torch':
			x1 = torch.from_numpy(x1)

		return x1

	def inverse_transform(self, X, nDV=236, nAP=200):
		x0 = X.reshape([-1, self.n_vertices_])
		
		if torch.is_tensor(X):
			mode = 'torch'
			x0 = x0.numpy()
		else: 
			mode = 'numpy'

		#Interpolate to vertex points
		Z_AP = np.linspace(self.z_emb.min(), 
						   self.z_emb.max(), 
						   nAP)
		Phi_DV = np.linspace(self.phi_emb.min(), 
							 self.phi_emb.max(), 
							 nDV) #This goes the opposite direction of Y
		
		mask = np.logical_and(self.z_emb >= self.cutoff, self.z_emb <= 1 - self.cutoff)

		x1 = []
		for i in range(x0.shape[0]):
			xi = griddata(
				(self.phi_emb[mask], self.z_emb[mask]), x0[i][mask], 
				(Phi_DV[:, None], Z_AP[None, :])
			)
			xi_nearest = griddata(
				(self.phi_emb[mask], self.z_emb[mask]), x0[i][mask], 
				(Phi_DV[:, None], Z_AP[None, :]),
				method='nearest'
			)
			xi[np.isnan(xi)] = xi_nearest[np.isnan(xi)]
				
			x1.append(xi)
		x1 = np.stack(x1)
		
		if x1.shape[0] == 2: #It's a vector
			x1[0] *= -1
		elif x1.shape[0] == 4: #It's a tensor
			x1[1:3] *= -1
			x1 = x1.reshape([2, 2, nDV, nAP])
		
		if mode == 'torch':
			x1 = torch.from_numpy(x1)
		
		return x1

class TangentSpaceTransformer(BaseEstimator, TransformerMixin, Module):
	'''
	Pushes quantities from the tangent space to 3D space
	Inverse transform pulls quantities back to tangent space

	Assumes all quantities are defined at the mesh vertices

	For memory reasons, this transformer only operates in cpu mode
	'''
	def __init__(self, 
				 mesh_name='embryo_coarse_noll'):
		Module.__init__(self)
		self.mesh_name = mesh_name

	def build_N_matrix(self, order=1):
		if order == 0:
			N = sparse.identity(self.n_vertices_)
		else:
			E = [ [] for i in range(2**order) ]
			for ij in range(3**order):
				IJ = np.array(list(np.base_repr(ij, 3).zfill(order))).astype(int)

				for ab in range(2**order):
					AB = np.array(list(np.binary_repr(ab, width=order))).astype(int)
					es = 1
					for a, i in zip(AB, IJ):
						es *= self.e[a][:, i]
					E[ab].append(sparse.spdiags(es, 0, self.n_vertices_, self.n_vertices_))

			N = sparse.vstack([sparse.hstack(Ei) for Ei in E])

		if self.mode == 'torch':
			N = torch.from_numpy(N.todense()).to_sparse()
		
		self.N[order] = N
	
	def fit(self, X, y0=None):
		mesh = Mesh(os.path.join(geo_dir, f'{self.mesh_name}.xml'))
		tangent_space_data = loadmat(os.path.join(geo_dir, f'{self.mesh_name}.mat'))

		self.n_vertices_ = mesh.coordinates().shape[0]
		self.e = [tangent_space_data['e2'], tangent_space_data['e1']] #Use y-first convention
		self.N = {}

		if torch.is_tensor(X):
			self.mode = 'torch'
			self.transpose = torch.t
		else:
			self.mode = 'numpy'
			self.transpose = np.transpose
		
		#Build transformation jacobians
		for order in range(3):
			self.build_N_matrix(order)

		return self

	def transform(self, X):
		'''
		Transforms from tangent space to 3D
		'''
		n_components = np.prod(X.shape) // self.n_vertices_
		order = int(np.log2(n_components))
		if not order in self.N:
			self.build_N_matrix(order)

		x = self.transpose(self.N[order]) @ X.flatten()[:, None]
		x = x.reshape([3,] * order + [self.n_vertices_,])

		return x


	def inverse_transform(self, X):
		'''
		Transforms from 3D to tangent space

		Note that the transformation operations do not commute
		That is, inverse_transform(transform(X)) = X
		However, transform(inverse_transform(X)) =/= X

		Be aware of this when using repeated mesh interpolations
		'''
		n_components = np.prod(X.shape) // self.n_vertices_
		order = int(np.log(n_components) / np.log(3))
		if not order in self.N:
			self.build_N_matrix(order)
		x = self.N[order] @ X.flatten()[:, None]
		x = x.reshape([2,] * order + [self.n_vertices_,])

		return x
	
	def forward(self, X):
		return self.transform(X)
