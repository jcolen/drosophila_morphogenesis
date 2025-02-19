import torch
from torch import nn

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, NotFittedError

from scipy.integrate import solve_ivp
from torchdiffeq import odeint

from .transforms import InputProcessor, EmbryoSmoother
from .transforms import EmbryoGradient, CovariantEmbryoGradient

class ClosedFlyLoop(BaseEstimator, nn.Module):
	'''
	A closed fly loop can use scipy solve_ivp or torchdiffeq to integrate
		a machine-learned equation of motion from initial conditions using the fly

	I've done my best to abstract this as much as possible for simpler extension
	'''
	def __init__(self,
				 v_model=None,
				 rhs=None,
				 sigma=3,
				 sym_vel=True,
				 dv_mode='circular',
				 ap_mode='replicate',
				 geo_dir='/Users/jcolen/Documents/drosophila_morphogenesis/flydrive/embryo_geometry'):
		nn.Module.__init__(self)
		
		self.v_model = v_model
		self.sigma = sigma
		self.ap_mode = ap_mode
		self.dv_mode = dv_mode
		self.geo_dir = geo_dir
		self.sym_vel = sym_vel
		if rhs is None:
			print('Using default RHS')
			self.rhs = lambda *args: ClosedFlyLoop.rhs_WT(self, *args)
		else:
			print('Using custom RHS')
			self.rhs = lambda *args: rhs(self, *args)
		
	def fit(self, X, y0=None):
		self.gamma_dv_ = np.array([
				[1., 0.], 
				[0., 0.]
		])[..., None, None] #Constant over space
		
		self.inputs = InputProcessor().fit(X)
		if self.geo_dir is None:
			print('Using standard gradient')
			self.gradient = EmbryoGradient(
				sigma=self.sigma,
				dv_mode=self.dv_mode,
				ap_mode=self.ap_mode,
			).fit(X)
		else:
			print('Using covariant gradient')
			self.gradient = CovariantEmbryoGradient(
				sigma=self.sigma,
				dv_mode=self.dv_mode,
				ap_mode=self.ap_mode,
				geo_dir=self.geo_dir,
			).fit(X)

		self.smoother = EmbryoSmoother(
			sigma=self.sigma,
			dv_mode=self.dv_mode,
			ap_mode=self.ap_mode,
		).fit(X)
		
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
		
		if self.sym_vel:
			#Left right symmetrize the flow
			if self.mode_ == 'torch':
				v_lr = torch.flip(v, (-2,))
			elif self.mode_ == 'numpy':
				v_lr = np.flip(v, (-2,)).copy()
			v_lr[..., 0, :, :] *= -1
			v = 0.5 * (v + v_lr)
				
		return v
	
	def rhs_WT(self, m, s, v, E):
		'''
		Compute the right hand side of the myosin dynamics
		'''
		trm = self.einsum_('kkyx->yx', m)
		trE = self.einsum_('kkyx->yx', E)

		rhs  = -(0.066 - 0.061 * s) * m #Detachment
		rhs +=	(0.489 + 0.318 * s) * m * trE #Strain recruitment
		rhs +=	(0.564 - 0.393 * s) * trm * m #Tension recruitment
		rhs +=	(0.047 - 0.037 * s) * trm * self.gamma_dv_ #Hoop stress recruitment

		return rhs
		
	def forward(self, t, y):
		#Get myosin and source
		m, s = self.inputs.transform(y)
		
		#Compute flow from myosin/cadherin
		v = self.get_velocity(t, y).squeeze()

		#Gradients
		d1_m = self.gradient(m)
		d1_s = self.gradient(s)
		d1_v = self.gradient(v)
		
		#Source dynamics - passive advection
		sdot = -self.einsum_('iyx,yxi->yx', v, d1_s)

		#Flow derivative tensors
		O = -0.5 * (self.einsum_('iyxj->ijyx', d1_v) - \
					self.einsum_('iyxj->jiyx', d1_v))
		E =  0.5 * (self.einsum_('iyxj->ijyx', d1_v) + \
					self.einsum_('iyxj->jiyx', d1_v))

		#Myosin dynamics - comoving derivative
		lhs  =	self.einsum_('kyx,ijyxk->ijyx', v, d1_m)
		lhs +=	self.einsum_('ikyx,kjyx->ijyx', O, m)
		lhs -=	self.einsum_('ikyx,kjyx->ijyx', m, O)

		#Myosin dynamics - right hand side
		rhs  =	self.rhs(m, s, v, E)

		mdot = -lhs + rhs

		mdot = self.postprocess(mdot)
		sdot = self.postprocess(sdot)
		
		ydot = self.inputs.inverse_transform(mdot, sdot)
		return ydot

	def postprocess(self, f, ap_cut=15):
		'''
		Because of edge distortions and discrete effects, 
		we zero out the poles (15/200 = 7.5%) and smooth
		with a gaussian filter
		'''
		#fill edges 
		f[..., :ap_cut] = 0
		f[..., -ap_cut:] = 0
		
		#smooth 
		f = self.smoother.transform(f)

		return f

	
	def integrate(self, y0, t):
		print('Initializing')
		try: 
			check_is_fitted(self)
		except NotFittedError:
			self.fit(y0)

		if self.mode_ == 'torch':
			print('Using torchdiffeq solver')
			with torch.no_grad():
				self.to(y0.device)
				if not torch.is_tensor(t):
					t = torch.FloatTensor(t).to(y0.device)
				y = odeint(self, y0, t, method='rk4')
				v = self.get_velocity(t, y).cpu().numpy()
				y = y.cpu().numpy()

		elif self.mode_ == 'numpy':
			print('Using scipy solve_ivp')
			y = solve_ivp(self.forward, [t[0], t[-1]], y0.flatten(), method='RK45', t_eval=t)
			y = y['y'].T.reshape([-1, self.inputs.n_components_, *self.inputs.data_shape_])
			v = self.get_velocity(t, y)
		
		m, s = self.inputs.transform(y)
		
		return m, s, v
