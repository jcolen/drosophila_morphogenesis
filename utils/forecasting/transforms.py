import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.ndimage import correlate1d
from scipy.io import loadmat
from scipy.interpolate import RectBivariateSpline

from ..plot_utils import dv_min, dv_max, ap_min, ap_max

def gaussian_kernel1d(sigma, order, radius):
	"""
	Computes a 1-D Gaussian convolution kernel.
	Copied from the scipy v1.10 source code on github, since the function is 
		private in their implementation
	Here I need to copy it because I need to pass the kernel to torch
	"""
	if order < 0:
		raise ValueError('order must be non-negative')
	exponent_range = np.arange(order + 1)
	sigma2 = sigma * sigma
	x = np.arange(-radius, radius+1)
	phi_x = np.exp(-0.5 / sigma2 * x ** 2)
	phi_x = phi_x / phi_x.sum()

	if order == 0:
		return phi_x
	else:
		# f(x) = q(x) * phi(x) = q(x) * exp(p(x))
		# f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
		# p'(x) = -1 / sigma ** 2
		# Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
		# coefficients of q(x)
		q = np.zeros(order + 1)
		q[0] = 1
		D = np.diag(exponent_range[1:], 1)	# D @ q(x) = q'(x)
		P = np.diag(np.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
		Q_deriv = D + P
		for _ in range(order):
			q = Q_deriv.dot(q)
		q = (x[:, None] ** exponent_range).dot(q)
		return q * phi_x

torch_pad_keys = [
	'constant',
	'reflect',
	'replicate',
	'circular'
]

numpy_pad_keys = [
	'constant',
	'reflect',
	'edge',
	'wrap'
]

class EmbryoPadder(BaseEstimator, TransformerMixin):
	'''
	Pads a field defined over an embryo 
	Defaults to periodic padding in the DV direction 
	Defaults to edge padding in the AP direction
	'''
	def __init__(self, 
				 dv_mode='circular', 
				 dv_pad=4,
				 ap_mode='edge',
				 ap_pad=4):
		
		assert dv_mode in torch_pad_keys or dv_mode in numpy_pad_keys
		assert ap_mode in torch_pad_keys or ap_mode in numpy_pad_keys

		self.dv_mode = dv_mode
		self.dv_pad = dv_pad

		self.ap_mode = ap_mode
		self.ap_pad = ap_pad
	
	def fit(self, X, y0=None):
		if torch.is_tensor(X):
			self.mode_ = 'torch'
			if self.dv_mode in numpy_pad_keys:
				self.dv_mode_ = torch_pad_keys[numpy_pad_keys.index(self.dv_mode)]
			else:
				self.dv_mode_ = self.dv_mode
			if self.ap_mode in numpy_pad_keys:
				self.ap_mode_ = torch_pad_keys[numpy_pad_keys.index(self.ap_mode)]
			else:
				self.ap_mode_ = self.ap_mode
		
		else:
			self.mode_ = 'numpy'
			if self.dv_mode in torch_pad_keys:
				self.dv_mode_ = numpy_pad_keys[torch_pad_keys.index(self.dv_mode)]
			else:
				self.dv_mode_ = self.dv_mode
			if self.ap_mode in torch_pad_keys:
				self.ap_mode_ = numpy_pad_keys[torch_pad_keys.index(self.ap_mode)]
			else:
				self.ap_mode_ = self.ap_mode

		return self

	def pad_AP(self, X):
		if self.mode_ == 'numpy':
			return np.pad(X, ((0, 0), (0, 0), (self.ap_pad, self.ap_pad)), mode=self.ap_mode_)

		elif self.mode_ == 'torch':
			return F.pad(X, (self.ap_pad, self.ap_pad), mode=self.ap_mode_)
		
		return None
	
	def pad_DV(self, X):
		if self.mode_ == 'numpy':
			return np.pad(X, ((0, 0), (self.dv_pad, self.dv_pad), (0, 0)), mode=self.dv_mode_)

		elif self.mode_ == 'torch':
			X = X.permute(0, 2, 1)
			X = F.pad(X, (self.dv_pad, self.dv_pad), mode=self.dv_mode_)
			X = X.permute(0, 2, 1)
			return X

		return None

	def transform(self, X):
		c = X.shape[:-2]
		h, w = X.shape[-2:]
		x = X.reshape([-1, h, w])

		x = self.pad_AP(x)
		x = self.pad_DV(x)
		return x.reshape([*c, *x.shape[-2:]])
	
	def forward(self, X):
		return self.transform(X)

	def crop_AP(self, X):
		return X[..., :, self.ap_pad:-self.ap_pad]
	
	def crop_DV(self, X):
		return X[..., self.dv_pad:-self.dv_pad, :]

	def inverse_transform(self, X):
		c = X.shape[:-2]
		h0, w0 = X.shape[-2:]
		x = X.reshape([-1, h0, w0])

		x = self.crop_AP(x)
		x = self.crop_DV(x)
		return x.reshape([*c, *x.shape[-2:]])

class EmbryoGradient(BaseEstimator, TransformerMixin, torch.nn.Module):
	'''
	Compute gradient on the embryo surface
	'''
	def __init__(self, 
				 sigma=3,
				 dv_mode='periodic',
				 ap_mode='edge'):
		torch.nn.Module.__init__(self)
		self.sigma = sigma
		self.dv_mode = dv_mode
		self.ap_mode = ap_mode

	def fit(self, X, y0=None):
		nDV = X.shape[-2]
		nAP = X.shape[-1]

		self.dDV_ = (dv_max - dv_min) / nDV
		self.dAP_ = (ap_max - ap_min) / nAP

		if isinstance(self.sigma, tuple):
			sigma = self.sigma
		else:
			sigma = (self.sigma, self.sigma)

		#Define kernels
		self.dv_kernel_ = gaussian_kernel1d(
			sigma=sigma[0],
			order=1,
			radius=int(4*sigma[0]+0.5),
		)[::-1]

		self.ap_kernel_ = gaussian_kernel1d(
			sigma=sigma[1],
			order=1,
			radius=int(4*sigma[1]+0.5),
		)[::-1]

		#Instantiate the padding object
		self.padder_ = EmbryoPadder(
			dv_mode=self.dv_mode,
			dv_pad=self.dv_kernel_.shape[0] // 2,
			ap_mode=self.ap_mode,
			ap_pad=self.ap_kernel_.shape[0] // 2,
		)
		self.padder_.fit(X)

		#Define operation mode
		if torch.is_tensor(X):
			self.mode_ = 'torch'
			self.dv_kernel_ = torch.nn.Parameter(
				torch.from_numpy(self.dv_kernel_.copy()[None, None]),
				requires_grad=False
			)
			self.ap_kernel_ = torch.nn.Parameter(
				torch.from_numpy(self.ap_kernel_.copy()[None, None]),
				requires_grad=False
			)
		else:
			self.mode_ = 'numpy'

		return self

	def grad_DV(self, X):
		if self.mode_ == 'numpy':
			grad = correlate1d(X, self.dv_kernel_, axis=-2, mode='nearest')
			grad = self.padder_.inverse_transform(grad) #Remove padding

		elif self.mode_ == 'torch':
			c, h, w = X.shape
			grad = X.permute(0, 2, 1).reshape([-1, 1, h])
			grad = F.conv1d(grad, self.dv_kernel_)
			grad = grad.reshape([c, w, -1]).permute(0, 2, 1)
			grad = self.padder_.crop_AP(grad) #Remove AP padding

		return grad / self.dDV_

	def grad_AP(self, X):
		if self.mode_ == 'numpy':
			grad = correlate1d(X, self.ap_kernel_, axis=-1, mode='nearest')
			grad = self.padder_.inverse_transform(grad) #Remove padding
		elif self.mode_ == 'torch':
			c, h, w = X.shape
			grad = X.reshape([-1, 1, w])
			grad = F.conv1d(grad, self.ap_kernel_)
			grad = grad.reshape([c, h, -1])
			grad = self.padder_.crop_DV(grad) #Remove DV padding

		return grad / self.dAP_
	
	def transform(self, X):
		c = X.shape[:-2]
		h, w = X.shape[-2:]
		x = X.reshape([-1, h, w])

		#Pad the embryo
		x = self.padder_.transform(x)
		
		dY = self.grad_DV(x)
		dX = self.grad_AP(x)

		if self.mode_ == 'numpy':
			grad = np.stack([dY, dX], axis=-1)
		elif self.mode_ == 'torch':
			grad = torch.stack([dY, dX], dim=-1)

		grad = grad.reshape([*c, h, w, 2])
		return grad

	def forward(self, X):
		return self.transform(X)

	def __call__(self, X):
		return self.transform(X)

class CovariantEmbryoGradient(EmbryoGradient):
	'''
	Covariant corrections using the embryo surface metric and kartoffel symbols
	'''
	def __init__(self,
				 sigma=3,
				 dv_mode='periodic',
				 ap_mode='edge'):
		super().__init__(sigma=sigma, dv_mode=dv_mode, ap_mode=ap_mode)

		#Load geometric information
		geo_dir = os.path.join(
			'/project/vitelli/jonathan',
			'REDO_fruitfly/flydrive.synology.me/Public',
			'dynamic_atlas/embryo_geometry'
		)
		self.Gijk = np.load(os.path.join(geo_dir, 'christoffel_symbols.npy'), mmap_mode='r')
		geometry = loadmat(os.path.join(geo_dir, 'embryo_rectPIVscale_fundamentalForms.mat'),
						   simplify_cells=True)
		X = geometry['X0'] * 0.2619 / 0.4
		Y = geometry['Y0'] * 0.2619 / 0.4

		self.ap_min, self.ap_max = X.min(), X.max()
		self.dv_min, self.dv_max = Y.min(), Y.max()

	def fit(self, X, y0=None):
		super().fit(X)

		#Interpolate metric onto gridpoints
		dv_g = np.linspace(self.dv_min, self.dv_max, self.Gijk.shape[0])
		ap_g = np.linspace(self.ap_min, self.ap_max, self.Gijk.shape[1])

		dv = np.linspace(dv_min, dv_max, X.shape[-2])
		ap = np.linspace(ap_min, ap_max, X.shape[-1])

		Gijk = self.Gijk.reshape([*self.Gijk.shape[:2], -1])
		interpolated = []
		for i in range(Gijk.shape[-1]):
			interpolated.append(RectBivariateSpline(dv_g, ap_g, Gijk[..., i])(dv, ap))

		interpolated = np.stack(interpolated, axis=-1)
		interpolated = interpolated.reshape([*interpolated.shape[:2], 2, 2, 2])

		#Define operation mode
		if torch.is_tensor(X):
			self.mode_ = 'torch'
			self.Gijk_ = torch.nn.Parameter(
				torch.from_numpy(interpolated), requires_grad=False)
			self.einsum_ = torch.einsum
		else:
			self.mode_ = 'numpy'
			self.Gijk_ = interpolated
			self.einsum_ = np.einsum

		return self

	def transform(self, X):
		grad = super().transform(X)
		
		if len(X.shape) == 3: #Vector
			grad += self.einsum_('yxijk,kyx->iyxj', self.Gijk_, X)
		elif len(X.shape) == 4: #Tensor
			grad += self.einsum_('yxikl,ljyx->ijyxk', self.Gijk_, X) + \
					self.einsum_('yxjkl,ilyx->ijyxk', self.Gijk_, X)

		return grad

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
			deviatoric = m - 0.5 * np.einsum('kkyx,ij->ijyx', m, np.eye(2))
			dev_mag = np.linalg.norm(deviatoric, axis=(0, 1), keepdims=True)

			m_0 = np.linalg.norm(m, axis=(0, 1), keepdims=True)
			m_0 = m_0.mean(axis=(2, 3), keepdims=True)

			devE = np.einsum('klyx,klyx->yx', deviatoric, E)[None, None]
			E_active = E - np.sign(devE) * devE * deviatoric / dev_mag**2
			E_active = 0.5 * E_active * dev_mag / m_0

		elif self.mode_ == 'torch':
			deviatoric = m - 0.5 * torch.einsum('kkyx,ij->ijyx', m, torch.eye(2, device=m.device))
			dev_mag = torch.linalg.norm(deviatoric, axis=(0, 1), keepdims=True)

			m_0 = torch.linalg.norm(m, dim=(0, 1), keepdims=True)
			m_0 = m_0.mean(dim=(2, 3), keepdims=True)

			devE = torch.einsum('klyx,klyx->yx', deviatoric, E)[None, None]
			E_active = E - torch.sign(devE) * devE * deviatoric / dev_mag**2
			E_active = 0.5 * E_active * dev_mag / m_0 

		return E_active
