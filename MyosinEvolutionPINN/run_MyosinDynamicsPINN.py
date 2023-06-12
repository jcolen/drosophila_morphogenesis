import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from scipy.ndimage import gaussian_filter

def load_data():
	loaddir = '../Public/WT/ECad-GFP/ensemble/'
	cad = np.load(os.path.join(loaddir, 'raw2D.npy'), mmap_mode='r')

	loaddir = '../Public/Halo_Hetero_Twist[ey53]_Hetero/Sqh-GFP/ensemble/'
	sqh = np.load(os.path.join(loaddir, 'tensor2D.npy'), mmap_mode='r')
	vel = np.load(os.path.join(loaddir, 'velocity2D.npy'), mmap_mode='r')
	t = np.load(os.path.join(loaddir, 't.npy'), mmap_mode='r')
	y = np.load(os.path.join(loaddir, 'DV_coordinates.npy'), mmap_mode='r')
	x = np.load(os.path.join(loaddir, 'AP_coordinates.npy'), mmap_mode='r')
	
	t_min = -10
	t_max = 30
	mask = np.logical_and(t >= t_min, t <= t_max)
	t = t[mask]

	sqh = sqh[mask].transpose(0, 3, 4, 1, 2)
	cad = gaussian_filter(cad[mask], sigma=(0, 8, 8))
	cad = 0.5 * (cad + cad[:, ::-1])
	vel = vel[mask].transpose(0, 2, 3, 1)

	print('Data shapes')
	print(f'Sqh: {sqh.shape}')
	print(f'Cad: {cad.shape}')
	print(f'Vel: {vel.shape}')

	nAP = x.shape[1]
	nDV = y.shape[0]
	nTP = t.shape[0]

	XX = np.broadcast_to(x[None], (nTP, nDV, nAP))
	YY = np.broadcast_to(y[None], (nTP, nDV, nAP))
	TT = np.broadcast_to(t[:, None, None], (nTP, nDV, nAP))

	print('Coordinates shapes')
	print(f'XX: {XX.shape} Range [{XX.min():g}, {XX.max():g}] um')
	print(f'YY: {YY.shape} Range [{YY.min():g}, {YY.max():g}] um')
	print(f'TT: {TT.shape} Range [{TT.min():g}, {TT.max():g}] um')

	lower_bound = np.array([TT.min(), YY.min(), XX.min()])
	upper_bound = np.array([TT.max(), YY.max(), XX.max()])

	t = TT.flatten()[:, None]
	y = YY.flatten()[:, None]
	x = XX.flatten()[:, None]

	sqh_train = sqh.reshape([-1, *sqh.shape[3:]])
	cad_train = cad.reshape([-1, 1])
	vel_train = vel.reshape([-1, *vel.shape[3:]])

	print(t.shape, y.shape, x.shape)
	print(sqh_train.shape, cad_train.shape, vel_train.shape)

	N_train = 100000
	idx = np.random.choice(nAP*nDV*nTP, N_train, replace=False)

	x_train = x[idx, :]
	y_train = y[idx, :]
	t_train = t[idx, :]

	sqh_train = sqh_train[idx, :]
	cad_train = cad_train[idx, :]
	vel_train = vel_train[idx, :]

	print('Training data')
	print(t_train.shape, y_train.shape, x_train.shape)
	print(sqh_train.shape, cad_train.shape, vel_train.shape)

	return {
		'sqh_train': sqh_train,
		'cad_train': cad_train,
		'vel_train': vel_train,
		'x_train': x_train,
		'y_train': y_train,
		't_train': t_train,
		'lower_bound': lower_bound,
		'upper_bound': upper_bound
	}

def to_param(x, **kwargs):
	if isinstance(x, np.ndarray):
		return nn.Parameter(torch.from_numpy(x).float(), **kwargs)
	else:
		return nn.Parameter(x, **kwargs)

class Sin(nn.Module):
	def forward(self, x):
		return torch.sin(x)
	
class MyosinDynamicsPINN(nn.Module):
	'''
	For some reason, we can't even memorize cadherin????
	So we're taking it as given rather than as an output
	'''
	def __init__(self,
				 t_train, y_train, x_train,
				 sqh_train, cad_train, vel_train,
				 lower_bound, upper_bound, 
				 n_hidden_layers=7, hidden_width=128,
				 beta=1e0, lr=1e-3):
		super().__init__()
		
		self.init_model(n_hidden_layers, hidden_width)
		self.init_data(t_train, y_train, x_train, 
					   sqh_train, cad_train, vel_train,
					   lower_bound, upper_bound)	
		
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
		
		self.beta = beta

	def init_model(self, n_hidden_layers=7, hidden_width=64, n_outs=6):
		self.n_hidden_layers = n_hidden_layers
		self.hidden_width = hidden_width
		act = Sin
		layers = [3,] + ([hidden_width,] * n_hidden_layers) + [n_outs,]
		lst = []
		for i in range(len(layers)-2):
			lst.append(nn.Linear(layers[i], layers[i+1]))
			lst.append(act())
		lst.append(nn.Linear(layers[-2], layers[-1]))
		self.model = nn.Sequential(*lst)

		self.apply(self.init_weights) 

		self.sqh_coefs = to_param(torch.zeros(8), requires_grad=True)
		self.model.register_parameter('sqh_coefs', self.sqh_coefs)
		
	def init_weights(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight)
			m.bias.data.fill_(0.0)
	
	def init_data(self, 
				  t_train, y_train, x_train, 
				  sqh_train, cad_train, vel_train,
				  lower_bound, upper_bound):
		self.t_train = to_param(t_train, requires_grad=True)
		self.y_train = to_param(y_train, requires_grad=True)  
		self.x_train = to_param(x_train, requires_grad=True)
		
		self.sqh_train = to_param(sqh_train, requires_grad=False)
		self.cad_train = to_param(cad_train, requires_grad=False)
		self.vel_train = to_param(vel_train, requires_grad=False)
			
		self.lb = to_param(lower_bound, requires_grad=False)
		self.ub = to_param(upper_bound, requires_grad=False)

		#Constant DV-aligned source term
		self.gamma_dv = to_param(
			np.array([[1, 0],
					  [0, 0]])[None], 
			requires_grad=False)

		self.sqh_scale = np.mean(np.linalg.norm(sqh_train, axis=(1, 2)))
		self.vel_scale = np.mean(np.linalg.norm(vel_train, axis=1))
		self.cad_scale = np.mean(cad_train)
	
	def forward(self, t, y, x):
		'''
		The basic version predicts a 7-component vector:
			Myosin, Cadherin, Flow
		'''
		X = torch.cat([t, y, x], dim=-1)
		H = 2. * (X - self.lb) / (self.ub - self.lb) - 1.0
		sv = self.model(H)
		sqh = sv[:, 0:4].reshape([sv.shape[0], 2, 2])
		vel = sv[:, 4:6]
		return sqh, vel
	
	def gradient(self, x, *y):
		'''
		Take the gradient of a field x with respect to variables y
		Populates a tensor of size [*x.shape, len(y)] such that
		gradient[..., i] is the gradient of x with respect to y[i]

		x is assumed to have shape [N, C] or is reshaped to have that
		'''
		x0 = x.view([x.shape[0], -1])
		dxdy = []
		for yi in y:
			dxdyi = torch.stack([
				autograd.grad(x0[..., i], y, grad_outputs=torch.ones_like(x0[..., i]),
							  retain_graph=True, create_graph=True)[0] 
				for i in range(x0.shape[-1])], dim=-1)
			dxdyi = dxdyi.reshape(x.shape)
			dxdy.append(dxdyi)
		return torch.stack(dxdy, dim=-1).squeeze()
	
	def mse_loss(self, N=5000):
		idx = np.random.choice(self.t_train.shape[0], N, replace=False)
		t, y, x = self.t_train[idx], self.y_train[idx], self.x_train[idx]
		sqh, vel = self(t, y, x)
					
		#MSE loss
		sqh_loss = (sqh - self.sqh_train[idx]).pow(2).sum() / N
		vel_loss = (vel - self.vel_train[idx]).pow(2).sum() / N

		#print('MSE loss (sqh, cad, vel)', sqh_loss.item(), vel_loss.item())

		mse = sqh_loss / self.sqh_scale + \
			  vel_loss / self.vel_scale

		return mse

	def get_myosin_coefficients(self):
		'''
		This is abstracted slightly to allow us to fix the signs of the coefficients later
		'''
		return self.sqh_coefs
	
	def phys_loss(self, N=5000):
		idx = np.random.choice(self.t_train.shape[0], N, replace=False)
		t, y, x = self.t_train[idx], self.y_train[idx], self.x_train[idx]
		sqh, vel = self(t, y, x)
		cad = self.cad_train[idx][:, None, None]

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

		phys = sqh_dyn.pow(2).sum() / self.sqh_scale / N
		return phys
	
	def print(self):
		k = self.get_myosin_coefficients().detach().cpu().numpy()
		outstr  = f'\tD_t m = '
		outstr += f'{k[0]:.3g} (1 - {k[1]:.3g} c) m + '
		outstr += f'{k[2]:.3g} (1 - {k[3]:.3g} c) m Tr(E) + '
		outstr += f'{k[4]:.3g} (1 - {k[5]:.3g} c) m Tr(m) + '
		outstr += f'{k[6]:.3g} (1 - {k[7]:.3g} c) Gamma^DV Tr(m)'
		print(outstr, flush=True)
	
	def train(self, num_iter, save_every=1000):
		save_name = f'{self.__class__.__name__}_beta={self.beta:.0e}.ckpt'

		for step in range(num_iter):
			self.optimizer.zero_grad()

			mse = self.mse_loss()
			phys = self.phys_loss()
			loss = mse + self.beta * phys
			loss.backward()

			self.optimizer.step()

			if step % save_every == 0:
				print(f'Iteration {step:d}\tLoss: {loss.item():e}, MSE: {mse.item():e}, Phys: {phys.item():e}')
				self.print()
				torch.save({
					'state_dict': self.state_dict(), 
					'mse_loss': mse,
					'phys_loss': phys,
				}, save_name)



		self.print()
		torch.save({
			'state_dict': self.state_dict(), 
			'mse_loss': mse,
			'phys_loss': phys,
		}, save_name)

class PositiveCoefficientsPINN(MyosinDynamicsPINN):
	'''
	Require only positive cadherin coefficients, so effect of cadherin must be against myosin
	'''
	def get_myosin_coefficients(self):
		'''
		Coefficients can only be positive
		'''
		coefs = torch.ones_like(self.sqh_coefs) * self.sqh_coefs
		coefs[1] = coefs[1].exp()
		coefs[3] = coefs[3].exp()
		coefs[5] = coefs[5].exp()
		coefs[7] = coefs[7].exp()
		return coefs

class DorsalSourcePINN(MyosinDynamicsPINN):
	'''
	Two changes 
		1. Require a DV-graded source that is not necessarily cadherin (no cadherin MSE loss)
		2. Source must be counter-running myosin (coefficients are positive)
	We must require that this source is advected for some minimal constraint
	So that we don't let it just dominate everything, we'll also require it live in the 
		range of cadherin intensity
	To prevent the coefficients from overtaking everything, we'll restrict them to live in the
		range from 0 to 1
	'''
	def init_model(self, n_hidden_layers=7, hidden_width=64, n_outs=8):
		super().init_model(n_hidden_layers, hidden_width, n_outs=7)

	def init_data(self, 
				  t_train, y_train, x_train, 
				  sqh_train, cad_train, vel_train,
				  lower_bound, upper_bound):
		super().init_data(t_train, y_train, x_train,
						  sqh_train, cad_train, vel_train,
						  lower_bound, upper_bound)
		self.cad_min = np.min(cad_train)
		self.cad_max = np.max(cad_train)
		print('Allowed cadherin range: ', self.cad_min, self.cad_max)

	def forward(self, t, y, x):
		'''
		The basic version predicts a 7-component vector:
			Myosin, Cadherin, Flow
		'''
		X = torch.cat([t, y, x], dim=-1)
		H = 2. * (X - self.lb) / (self.ub - self.lb) - 1.0
		svc = self.model(H)
		sqh = svc[:, 0:4].reshape([svc.shape[0], 2, 2])
		vel = svc[:, 4:6]
		cad = svc[:, 6:7].squeeze()
		cad = self.cad_min + (1 + cad.tanh()) * (self.cad_max - self.cad_min) / 2.
		return sqh, vel, cad

	def mse_loss(self, N=5000):
		idx = np.random.choice(self.t_train.shape[0], N, replace=False)
		t, y, x = self.t_train[idx], self.y_train[idx], self.x_train[idx]
		sqh, vel, cad = self(t, y, x)
					
		#MSE loss without cadherin loss
		sqh_loss = (sqh - self.sqh_train[idx]).pow(2).sum() / N
		vel_loss = (vel - self.vel_train[idx]).pow(2).sum() / N

		mse = sqh_loss / self.sqh_scale + \
			  vel_loss / self.vel_scale

		return mse
	
	def get_myosin_coefficients(self):
		'''
		Coefficients can only be positive and in the range [0, 1]
		'''
		coefs = torch.ones_like(self.sqh_coefs) * self.sqh_coefs
		coefs[1] = (1 + coefs[1].tanh()) / 2.
		coefs[3] = (1 + coefs[3].tanh()) / 2.
		coefs[5] = (1 + coefs[5].tanh()) / 2.
		coefs[7] = (1 + coefs[7].tanh()) / 2.
		return coefs
	
	def phys_loss(self, N=5000):
		#Evaluate at random bulk points since we're not beholden to cadherin
		t = torch.rand(N, 
					   dtype=self.t_train.dtype, 
					   device=self.t_train.device, 
					   requires_grad=True)[:, None]
		t = self.lb[0] + (self.ub[0] - self.lb[0]) * t
		y = self.lb[1] + (self.ub[1] - self.lb[1]) * torch.rand_like(t, requires_grad=True)
		x = self.lb[2] + (self.ub[2] - self.lb[2]) * torch.rand_like(t, requires_grad=True)
		sqh, vel, cad = self(t, y, x)

		#Source is conserved
		dt_cad = self.gradient(cad, t)
		div_cv = self.gradient(cad * vel[:, 0], y) + self.gradient(cad * vel[:, 1], x)
		cad_dyn = dt_cad + div_cv

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
		cadh = cad[:, None, None]
		rhs  = k[0] * (1 - k[1] * cadh) * sqh
		rhs += k[2] * (1 - k[3] * cadh) * mTrE
		rhs += k[4] * (1 - k[5] * cadh) * Trm * sqh
		rhs += k[6] * (1 - k[7] * cadh) * Trm * self.gamma_dv

		sqh_dyn = lhs - rhs

		phys = sqh_dyn.pow(2).sum() / self.sqh_scale + \
			   cad_dyn.pow(2).sum() / self.cad_scale
		return phys / N

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


from argparse import ArgumentParser
if __name__=='__main__':
	parser = ArgumentParser()
	parser.add_argument('--beta', type=float, default=1)
	args = vars(parser.parse_args())
	
	print('Loading data', flush=True)
	data = load_data()
	print('\nStarting to train', flush=True)	

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = MyosinDynamicsPINN(**data, **args)
	model = PositiveCoefficientsPINN(**data, **args)
	model = DorsalSourcePINN(**data, **args)
	model = IncompressibleStokesPINN(**data, **args)
	model = CompressibleStokesPINN(**data, **args)
	model.to(device)
	model.train(250000)
