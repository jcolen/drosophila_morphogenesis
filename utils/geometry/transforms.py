import numpy as np
import torch

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KDTree
from .geometry_utils import embryo_mesh


class InputProcessor(BaseEstimator, TransformerMixin):
	'''
	Proceses inputs and shapes them correctly
	'''
	def __init__(self, mesh=embryo_mesh):
		self.mesh = mesh

	def fit(self, X, y=None):
		x = X.reshape([-1, self.mesh.coordinates().shape[0]])
		self.n_components_ = x.shape[0]
		self.data_shape_ = x.shape[1:]

		if torch.is_tensor(X):
			self.mode_ = 'torch'
		else:
			self.mode_ = 'numpy'

		return self
	
	def transform(self, X):
		X = X.reshape([-1, self.n_components_, *self.data_shape_])
		m = X[:, :4].reshape([-1, 2, 2, *self.data_shape_]).squeeze()
		s = X[:, 4:].squeeze()
		return m, s

	def inverse_transform(self, m, s):
		if self.mode_ == 'torch':
			X = torch.cat([
				m.reshape([-1, 4, *self.data_shape_]),
				s.reshape([-1, 1, *self.data_shape_])
			], dim=1)
		elif self.mode_ == 'numpy':
			X = np.concatenate([
				m.reshape([-1, 4, *self.data_shape_]),
				s.reshape([-1, 1, *self.data_shape_])
			], axis=1)
			X = X.flatten()
		return X

class LeftRightSymmetrize(BaseEstimator, TransformerMixin):
	'''
	Symmetrize the embryo left/right
	'''
	def __init__(self, mesh=embryo_mesh):
		self.mesh = mesh
	
	def fit(self, X, y=None):
		l_verts = self.mesh.coordinates()
		r_verts = l_verts.copy()
		r_verts[..., 1] *= -1 #Flip the y-coordinate in 3D

		tree = KDTree(l_verts)
		_, ind = tree.query(r_verts, k=1) #Find nearest neighbor in flipped system
		self.ind_ = ind.squeeze()

		return self

	def transform(self, X):
		reflected = X[..., self.ind_]
		if len(reflected.shape) == 2: #Vector
			reflected[1] *= -1 #Flip y coordinate
		elif len(reflected.shape) == 3: #Tensor
			reflected[0, 1] *= -1 #Flip xy
			reflected[1, 0] *= -1 #Flip yx
			reflected[1, 2] *= -1 #Flip yz
			reflected[2, 1] *= -1 #Flip zy

		return 0.5 * (X + reflected)

class ActiveStrainDecomposition(BaseEstimator, TransformerMixin):
	'''
	Determine active (myosin-induced) strain rate as decsribed in the manuscript
	'''
	def fit(self, X, y0=None):
		if torch.is_tensor(X):
			self.mode_ = 'torch'
		else:
			self.mode_ = 'numpy'

		return self
	
	def transform(self, E, m):
		if self.mode_ == 'numpy':
			deviatoric = m - 0.5 * np.einsum('kk...,ij->ij...', m, np.eye(3))
			dev_mag = np.linalg.norm(deviatoric, axis=(0, 1), keepdims=True)

			m_0 = np.linalg.norm(m, axis=(0, 1), keepdims=True)
			m_0 = m_0.mean(axis=(-1), keepdims=True)

			devE = np.einsum('kl...,kl...->...', deviatoric, E)[None, None]
			E_active = E - np.sign(devE) * devE * deviatoric / dev_mag**2
			E_active = 0.5 * E_active * dev_mag / m_0

		elif self.mode_ == 'torch':
			deviatoric = m - 0.5 * torch.einsum('kk...,ij->ij...', m, torch.eye(3, device=m.device))
			dev_mag = torch.linalg.norm(deviatoric, axis=(0, 1), keepdims=True)

			m_0 = torch.linalg.norm(m, dim=(0, 1), keepdims=True)
			m_0 = m_0.mean(dim=(-1), keepdims=True)

			devE = torch.einsum('kl...,kl...->...', deviatoric, E)[None, None]
			E_active = E - torch.sign(devE) * devE * deviatoric / dev_mag**2
			E_active = 0.5 * E_active * dev_mag / m_0 

		return E_active
