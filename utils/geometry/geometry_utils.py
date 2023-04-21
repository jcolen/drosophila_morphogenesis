import os
import numpy as np
import torch
from torch.nn import Parameter

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
	'''
	def __init__(self, 
				 mesh=embryo_mesh,
				 z_emb=z_emb,
				 phi_emb=phi_emb):
		self.mesh = mesh
		self.z_emb = z_emb
		self.phi_emb = phi_emb

	def fit(self, X, y0=None):
		if torch.is_tensor(X):
			self.mode = 'torch'
		else:
			self.mode = 'numpy'

		self.n_vertices_ = self.mesh.coordinates().shape[0]	
		return self

	def transform(self, X):
		header_shape = X.shape[:-2]
		x0 = X.reshape([-1, *X.shape[-2:]])

		#Interpolate to vertex points
		if self.mode == 'torch':
			x0 = x0.cpu().numpy()

		Z_AP = np.linspace(self.z_emb.min(), 
						   self.z_emb.max(), 
						   X.shape[-1])
		Phi_DV = np.linspace(self.phi_emb.min(), 
							 self.phi_emb.max(), 
							 X.shape[-2]) #This goes the opposite direction of Y

		x1 = []
		for i in range(x0.shape[0]):
			xi = RectBivariateSpline(Phi_DV, Z_AP, x0[i])(phi_emb, z_emb, grid=False)
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

		if self.mode == 'torch':
			x1 = torch.from_numpy(x1).to(X.device)

		return x1

	def inverse_transform(self, X, nDV=236, nAP=200):
		x0 = X.reshape([-1, self.n_vertices_])

		if self.mode == 'torch':
			x0 = x0.cpu().numpy()
		
		#Interpolate to vertex points
		Z_AP = np.linspace(self.z_emb.min(), 
						   self.z_emb.max(), 
						   nAP)
		Phi_DV = np.linspace(self.phi_emb.min(), 
							 self.phi_emb.max(), 
							 nDV) #This goes the opposite direction of Y

		x1 = []
		for i in range(x0.shape[0]):
			xi = griddata(
				(phi_emb, z_emb), x0[i], 
				(Phi_DV[:, None], Z_AP[None, :])
			)
			xi_nearest = griddata(
				(phi_emb, z_emb), x0[i], 
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

		if self.mode == 'torch':
			x1 = torch.from_numpy(x1).to(X.device)
		
		return x1

class TangentSpaceTransformer(BaseEstimator, TransformerMixin, torch.nn.Module):
	'''
	Pulls quantities from the tangent space to 3D space
	Inverse transform pushes quantities back to tangent space

	Assumes all quantities are defined at the mesh vertices
	'''
	def __init__(self, 
				 mesh=embryo_mesh,
				 e1=e1,
				 e2=e2,
				 N_vector=N_vector,
				 N_tensor=N_tensor):
		torch.nn.Module.__init__(self)
		self.mesh = mesh
		self.e1 = e1
		self.e2 = e2
		self.N_vector = N_vector
		self.N_tensor = N_tensor

	def fit(self, X, y0=None):
		self.n_vertices_ = self.mesh.coordinates().shape[0]
		if torch.is_tensor(X):
			self.mode = 'torch'
			self.einsum_ = torch.einsum
			convert = lambda x: Parameter(torch.from_numpy(x.todense()).to_sparse(), requires_grad=False)

			self.e1 = Parameter(torch.from_numpy(self.e1), requires_grad=False)
			self.e2 = Parameter(torch.from_numpy(self.e2), requires_grad=False)
			self.N_vector = convert(N_vector)
			self.N_tensor = convert(N_tensor)
			
		else:
			self.mode = 'numpy'
			self.einsum_ = np.einsum

		return self

	def transform(self, X):
		n_components = np.prod(X.shape) // self.n_vertices_
		x0 = X.flatten()

		if n_components == 2: #Transforms like a vector
			x1	= self.einsum_('Vi,V->iV', self.e1, x0[:self.n_vertices_])
			x1 += self.einsum_('Vi,V->iV', self.e2, x0[self.n_vertices_:])
		elif n_components == 4: #Transforms like a tensor
			x1	= self.einsum_('Vi,V,Vj->ijV', self.e1, x0[:self.n_vertices_], self.e1)
			x1 += self.einsum_('Vi,V,Vj->ijV', self.e1, x0[self.n_vertices_:2*self.n_vertices_], self.e2)
			x1 += self.einsum_('Vi,V,Vj->ijV', self.e2, x0[2*self.n_vertices_:3*self.n_vertices_], self.e1)
			x1 += self.einsum_('Vi,V,Vj->ijV', self.e2, x0[3*self.n_vertices_:], self.e2)

		else: #Scalar
			x1 = X

		return x1


	def inverse_transform(self, X):
		n_components = np.prod(X.shape) // self.n_vertices_
		x0 = X.flatten()[:, None]

		if n_components == 3: #Vector
			x = self.N_vector @ x0
			x = x.reshape([2, self.n_vertices_])
		elif n_components == 9: #Tensor
			x = self.N_tensor @ x0
			x = x.reshape([2, 2, self.n_vertices_])
		else:
			x = X

		return x
	
	def forward(self, X):
		return self.transform(X)
