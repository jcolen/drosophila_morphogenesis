import torch
from torch import nn
import numpy as np

import sys
sys.path.insert(0, '/project/vitelli/jonathan/REDO_fruitfly/src/')

from utils.forecasting.closed_loop import ClosedFlyLoop
from run_MyosinDynamicsPINN import *
from myosin_dynamics_pinn import *

class ClosedPINNLoop(ClosedFlyLoop):
	'''
	Implementation of a closed loop integrator which uses a PINN
	for the velocity field
	'''
	def __init__(self,
				 v_model=None,
				 load_path='',
				 sigma=3,
				 dv_mode='circular',
				 ap_mode='replicate'):
		super().__init__(v_model, sigma, dv_mode, ap_mode)

		data = load_data()
		info = torch.load(load_path, map_location='cpu')
		v_model = v_model(**data)
		v_model.load_state_dict(info['state_dict'], strict=False)

		self.v_model = v_model
	
	def fit(self, X, y0=None):
		super().fit(X, y0)

		assert self.mode_ == 'torch'

		#Set up spatial evaluation points for the model
		y = np.linspace(self.v_model.lb[1], self.v_model.ub[1], X.shape[-2])
		x = np.linspace(self.v_model.lb[2], self.v_model.ub[2], X.shape[-1])

		Y, X = np.meshgrid(y, x, indexing='ij')
		self.Y = to_param(Y, requires_grad=False)
		self.X = to_param(X, requires_grad=False)

		return self

	def get_velocity(self, t, y):
		t = t.reshape([-1]).float()
		tt = torch.broadcast_to(t[:, None, None], (t.shape[0], *self.Y.shape))
		yy = torch.broadcast_to(self.Y[None], (t.shape[0], *self.Y.shape))
		xx = torch.broadcast_to(self.X[None], (t.shape[0], *self.Y.shape))
		vel = self.v_model(tt.flatten()[:, None],
						   yy.flatten()[:, None],
						   xx.flatten()[:, None])[1].double()

		vel = vel.reshape([-1, *y.shape[-2:], 2]).permute(0, 3, 1, 2)

		return vel
	
	def rhs(self, m, s, v, E):
		'''
		Compute the right hand side of the myosin dynamics
		'''
		trm = self.einsum_('kkyx->yx', m)
		trE = self.einsum_('kkyx->yx', E)

		rhs  = -0.065 * (1 - 0.8 * s.mean()) * m #Detachment
		rhs += 0.5 * (1 + 1.1 * s) * m * trE #Strain recruitment
		rhs += 0.5 * (1 - 0.7 * s.mean()) * trm * m #Tension recruitment
		rhs += 0.5 * (1 - 0.7 * s.mean()) * trm * self.gamma_dv_ / 10 #Hoop stress recruitment

		return rhs
	
	def rhs_PINN(self, m, s, v, E):
		'''
		Compute the right hand side of the myosin dynamics
		'''
		trm = self.einsum_('kkyx->yx', m)
		trE = self.einsum_('kkyx->yx', E)
		k = self.v_model.get_myosin_coefficients()
		
		rhs  = k[0] * (1 - k[1] * s) * m #Detachment
		rhs += k[2] * (1 - k[3] * s) * m * trE #Strain recruitment
		rhs += k[4] * (1 - k[5] * s) * trm * m #Tension recruitment
		rhs += k[6] * (1 - k[7] * s) * trm * self.gamma_dv_ #Hoop stress recruitment

		return rhs
	
	def forward_PINN(self, t, y):
		#Get myosin and source
		m, s = self.inputs.transform(y)
		
		#Compute flow from myosin/cadherin
		v = self.get_velocity(t, y).squeeze()
		
		#Source dynamics - passive advection
		d1_s = self.gradient(s)
		sdot = -self.einsum_('iyx,yxi->yx', v, d1_s)
		
		t = t.reshape([-1]).float()
		tt = torch.broadcast_to(t[:, None, None], (t.shape[0], *self.Y.shape))
		yy = torch.broadcast_to(self.Y[None], (t.shape[0], *self.Y.shape))
		xx = torch.broadcast_to(self.X[None], (t.shape[0], *self.Y.shape))

		with torch.enable_grad():
			tt = tt.flatten()[:, None]
			tt.requires_grad = True
			sqh = self.v_model(tt,
							   yy.flatten()[:, None],
							   xx.flatten()[:, None])[0]
			
			mdot = self.v_model.gradient(sqh, tt)
			mdot = mdot.reshape(m.shape).double()

		mdot = self.postprocess(mdot)
		sdot = self.postprocess(sdot)
		
		ydot = self.inputs.inverse_transform(mdot, sdot)
		return ydot
