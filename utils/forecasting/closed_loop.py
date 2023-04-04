import torch
from torch import nn

import numpy as np

from scipy.integrate import solve_ivp
from sklearn.base import BaseEstimator
from torchdiffeq import odeint

from ..decomposition.decomposition_model import StandardShaper
from .transforms import CovariantEmbryoGradient, ActiveStrainDecomposition

class ClosedFlyLoop(BaseEstimator, nn.Module):
	'''
	A closed fly loop can use scipy solve_ivp or torchdiffeq to integrate
		a machine-learned equation of motion from initial conditions using the fly

	I've done my best to abstract this as much as possible for simpler extension

	TODO: accept a RHS method after we decide what variables it accepts
	TODO: account for posterior pole distortion, either by swap padding or imposing a maximum ydot
	'''
	def __init__(self,
				 v_model=None,
				 v_thresh=0.,
				 sigma=3):
		nn.Module.__init__(self)
		self.v_model = v_model
		self.sigma = sigma
		self.v_thresh = v_thresh
		
		self.shaper = StandardShaper()
		self.active = ActiveStrainDecomposition()
		self.gradient = CovariantEmbryoGradient(sigma=sigma,
												dv_mode='circular', 
												ap_mode='replicate')
		
	def fit(self, X, y0=None):
		self.gamma_dv_ = np.array([
				[1., 0.], 
				[0., 0.]
		])[..., None, None]
		
		self.shaper.fit(X)
		self.gradient.fit(X)
		self.active.fit(X)
		
		if torch.is_tensor(X):
			self.mode_ = 'torch'
			self.einsum_ = torch.einsum
			self.gamma_dv_ = torch.nn.Parameter(torch.from_numpy(self.gamma_dv_), requires_grad=False)
		else:
			self.mode_ = 'numpy'
			self.einsum_ = np.einsum
		
		return self
	
	def get_velocity(self, t, y):
		v = self.v_model(t, y)
		
		#Left right symmetrize
		if self.mode_ == 'torch':
			v_lr = torch.flip(v, (-2,))
		elif self.mode_ == 'numpy':
			v_lr = np.flip(v, (-2,)).copy()
		v_lr[..., 0, :, :] *= -1
		v = 0.5 * (v + v_lr)
		
		#Threshold
		if self.mode_ == 'torch':
			vnorm = torch.linalg.norm(v, dim=-3, keepdims=True)
		elif self.mode_ == 'numpy':
			vnorm = np.linalg.norm(v, axis=-3, keepdims=True)
		v *= vnorm >= self.v_thresh
				
		return v
		
	def forward(self, t, y):
		#Get myosin and source
		y = self.shaper.transform(y[None]).squeeze()
		m = y[:4].reshape([2, 2, *y.shape[-2:]])
		s = y[4:].squeeze()
		
		#Compute flow from myosin/cadherin
		v = self.get_velocity(t, y[:4]).squeeze()

		#Gradients
		d1_m = self.gradient(m)
		d1_s = self.gradient(s)
		d1_v = self.gradient(v)
		
		#Source is passively advected
		sdot = -1.000 * self.einsum_('iyx,yxi->yx', v, d1_s)

		#Flow derivative tensors
		O = -0.5 * (self.einsum_('iyxj->ijyx', d1_v) - \
					self.einsum_('jyxi->ijyx', d1_v))
		E = 0.5 * (self.einsum_('iyxj->ijyx', d1_v) + \
				   self.einsum_('jyxi->ijyx', d1_v))

		#Active/Passive strain decomposition
		E_active = self.active.transform(E, m)
		E_passive = E - E_active

		trm = self.einsum_('kkyx->yx', m)[None, None]
		s = s[None, None]
		
		#Myosin dynamics
		mdot =	-self.einsum_('kyx,ijyxk->ijyx', v, d1_m)
		mdot -=  self.einsum_('ikyx,kjyx->ijyx', O, m)
		mdot -= -self.einsum_('ikyx,kjyx->ijyx', m, O)	
		
		mdot += -(0.085 - 0.077 * s) * m
		mdot +=  (0.717 - 0.528 * s) * trm * m
		mdot +=  (0.051 - 0.037 * s) * trm * self.gamma_dv_
		mdot +=  (1.465 - 0.216 * s) * m * self.einsum_('kkyx->yx', E)
		mdot += -(1.814 - 1.003 * s) * m * self.einsum_('kkyx->yx', E_passive) 
		
		if self.mode_ == 'torch':
			ydot = torch.cat([
				mdot.reshape([4, *mdot.shape[-2:]]),
				sdot.reshape([1, *sdot.shape[-2:]]),
			])
		else:
			ydot = np.concatenate([
				mdot.reshape([4, *mdot.shape[-2:]]),
				sdot.reshape([1, *sdot.shape[-2:]]),
			]).flatten()
		
		return ydot
	
	def integrate(self, y0, t):
		if self.mode_ == 'torch':
			with torch.no_grad():
				y = odeint(self, y0, t, method='rk4')
				m = y[:, :4].reshape([-1, 2, 2, *y.shape[-2:]]).cpu().numpy()
				s = y[:, 4:].cpu().numpy()
				v = self.get_velocity(t, y[:, :4]).cpu().numpy()

		elif self.mode_ == 'numpy':
			out = solve_ivp(self.forward, [t[0], t[-1]], y0.flatten(), method='RK45', t_eval=t)
			y = self.shaper.transform(out['y'].T)
			m = y[:, :4].reshape([-1, 2, 2, *y.shape[-2:]])
			s = y[:, 4:]
			v = self.get_velocity(t, y[:, :4])
		
		return m, s, v
