import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from myosin_dynamics_pinn import *

class IncompressibleStokesPINN(MyosinDynamicsPINN):
	'''
	Require flow to come from the Incompressible Stokes equation
	This means we drop the MSE loss on the flow and add another PHYS loss
	'''
	def init_model(self, n_hidden_layers=7, hidden_width=64, n_outs=8):
		super().init_model(n_hidden_layers, hidden_width, n_outs=7)

		self.vel_coefs = to_param(torch.zeros(1), requires_grad=True)
		self.model.register_parameter('vel_coefs', self.vel_coefs)
	
	def get_velocity_coefficients(self):
		return self.vel_coefs.exp()

	def forward(self, t, y, x):
		'''
		This predicts a 8-component vector:
			Myosin, Cadherin, Flow, Pressure
		'''
		X = torch.cat([t, y, x], dim=-1)
		H = 2. * (X - self.lb) / (self.ub - self.lb) - 1.0
		svp = self.model(H)
		sqh = svp[:, 0:4].reshape([svp.shape[0], 2, 2])
		vel = svp[:, 4:6]
		pre = svp[:, 6:7].squeeze()
		return sqh, vel, pre
	
	def mse_loss(self, N=5000):
		idx = np.random.choice(self.t_train.shape[0], N, replace=False)
		t, y, x = self.t_train[idx], self.y_train[idx], self.x_train[idx]
		sqh, vel, pre = self(t, y, x)
					
		#MSE loss without any constraint on velocity
		sqh_loss = (sqh - self.sqh_train[idx]).pow(2).sum() / N

		mse = sqh_loss / self.sqh_scale

		return mse
	
	def phys_loss(self, N=5000):
		idx = np.random.choice(self.t_train.shape[0], N, replace=False)
		t, y, x = self.t_train[idx], self.y_train[idx], self.x_train[idx]
		sqh, vel, pre = self(t, y, x)
		cad = self.cad_train[idx][:, None, None]

		#Myosin dynamics
		dt_sqh = self.gradient(sqh, t)
		grad_sqh = self.gradient(sqh, y, x)
		grad_v = self.gradient(vel, y, x)
		
		O = 0.5 * (torch.einsum('bji->bij', grad_v) - torch.einsum('bij->bij', grad_v))
		E = 0.5 * (torch.einsum('bij->bij', grad_v) + torch.einsum('bji->bij', grad_v))
		mTrE = torch.einsum('bij,bkk->bij', sqh, E)
		Trm  = torch.einsum('bkk->b', sqh)[:, None, None]

		
		lhs =  dt_sqh + torch.einsum('bk,bijk->bij', vel, grad_sqh) #advection
		lhs += torch.einsum('bik,bkj->bij', O, sqh) #co-rotation
		lhs -= torch.einsum('bik,bkj->bij', sqh, O) #co-rotation

		k = self.get_myosin_coefficients()
		rhs  = k[0] * (1 - k[1] * cad) * sqh
		rhs += k[2] * (1 - k[3] * cad) * mTrE
		rhs += k[4] * (1 - k[5] * cad) * Trm * sqh
		rhs += k[6] * (1 - k[7] * cad) * Trm * self.gamma_dv

		sqh_dyn = lhs - rhs

		#Incompressible stokes equation
		lapl_v = self.gradient(grad_v[..., 0], y) + self.gradient(grad_v[..., 1], x)
		grad_p = self.gradient(pre, y, x)
		div_m  = torch.einsum('bijj->bi', grad_sqh)

		alpha = self.get_velocity_coefficients()[0]
		stokes = lapl_v - grad_p + alpha * div_m

		phys = sqh_dyn.pow(2).sum() / self.sqh_scale + \
			   stokes.pow(2).sum() / self.vel_scale
		return phys / N
	
	def print(self):
		super().print()
		k = self.get_velocity_coefficients()[0].item()
		outstr = f'\tgrad^2 v - grad(p) = -{k:.3g} div(m)'
		print(outstr, flush=True)

class CompressibleStokesPINN(IncompressibleStokesPINN):
	'''
	CompressibleStokes equation with a time-varying B(t) as in eLife
	B(t) needs to be a neural network
	'''
	def forward(self, t, y, x):
		'''
		This predicts a 8-component vector:
			Myosin, Cadherin, Flow, Pressure
		'''
		X = torch.cat([t, y, x], dim=-1)
		H = 2. * (X - self.lb) / (self.ub - self.lb) - 1.0
		svB = self.model(H)
		sqh = svB[:, 0:4].reshape([svB.shape[0], 2, 2])
		vel = svB[:, 4:6]
		B = svB[:, 6:7].exp() #B is positive (pressure can be anything)
		return sqh, vel, B
	
	def phys_loss(self, N=5000):
		idx = np.random.choice(self.t_train.shape[0], N, replace=False)
		t, y, x = self.t_train[idx], self.y_train[idx], self.x_train[idx]
		sqh, vel, B = self(t, y, x)
		cad = self.cad_train[idx][:, None, None]

		#Myosin dynamics
		dt_sqh = self.gradient(sqh, t)
		grad_sqh = self.gradient(sqh, y, x)
		grad_v = self.gradient(vel, y, x)
		
		O = 0.5 * (torch.einsum('bji->bij', grad_v) - torch.einsum('bij->bij', grad_v))
		E = 0.5 * (torch.einsum('bij->bij', grad_v) + torch.einsum('bji->bij', grad_v))
		mTrE = torch.einsum('bij,bkk->bij', sqh, E)
		Trm  = torch.einsum('bkk->b', sqh)[:, None, None]

		
		lhs =  dt_sqh + torch.einsum('bk,bijk->bij', vel, grad_sqh) #advection
		lhs += torch.einsum('bik,bkj->bij', O, sqh) #co-rotation
		lhs -= torch.einsum('bik,bkj->bij', sqh, O) #co-rotation

		k = self.get_myosin_coefficients()
		rhs  = k[0] * (1 - k[1] * cad) * sqh
		rhs += k[2] * (1 - k[3] * cad) * mTrE
		rhs += k[4] * (1 - k[5] * cad) * Trm * sqh
		rhs += k[6] * (1 - k[7] * cad) * Trm * self.gamma_dv

		sqh_dyn = lhs - rhs

		#Incompressible stokes equation
		lapl_v = self.gradient(grad_v[..., 0], y) + self.gradient(grad_v[..., 1], x)
		gdiv_v = self.gradient(torch.einsum('bii->b', grad_v), y, x)
		div_m  = torch.einsum('bijj->bi', grad_sqh)

		alpha = self.get_velocity_coefficients()[0]
		stokes = lapl_v + B * gdiv_v + alpha * div_m

		#B must be independent of x, y
		grad_B = self.gradient(B, y, x)

		phys = sqh_dyn.pow(2).sum() / self.sqh_scale + \
			   stokes.pow(2).sum() / self.vel_scale + \
			   grad_B.pow(2).sum()
		return phys / N
	
	def print(self):
		super().print()
		k = self.get_velocity_coefficients()[0].item()
		outstr = f'\tgrad^2 v + B(t) grad(div(v)) = -{k:.3g} div(m)'
		print(outstr, flush=True)

