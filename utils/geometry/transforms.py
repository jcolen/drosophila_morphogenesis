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
		_, ind = tree.querty(r_verts, k=1) #Find nearest neighbor in flipped system
		self.ind_ = ind.squeeze()

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
