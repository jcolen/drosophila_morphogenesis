import numpy as np
import os
import sys
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))

from utils.translation_utils import *
from utils.decomposition_utils import *
from utils.plot_utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

def to_param(x, **kwargs):
	if isinstance(x, np.ndarray):
		return nn.Parameter(torch.from_numpy(x).float(), **kwargs)
	else:
		return nn.Parameter(x, **kwargs)

class Sin(nn.Module):
	def forward(self, x):
		return torch.sin(x)
	
class ClosedLoopPINN(nn.Module):
	'''
	Closed Loop PINN tries to learn the entire dynamics of the system
	Accepts [t, y, x] coordinate and predicts
	myosin tensor [2, 2] = 4 components [KNOWN]
	cadherin scalar = 1 component		[KNOWN]
	velocity field = 2 components		[KNOWN]
	dorsal source = 1 component			[UNKNOWN]
	pressure field = 1 component		[UNKNOWN]
	
	Subject to the closed loop conditions
	This includes an undetermined advected dorsal source and pressure field
	This also requires simultaneous learning of several parameters
	'''
	def __init__(self,
				 t_train, y_train, x_train,
				 sqh_train, cad_train, vel_train,
				 lower_bound, 
				 upper_bound,
				 hidden_width=100,
				 n_hidden_layers=7,
				 beta_0=1,
				 dorsal_weight=1e2,
				 save_every=1000):
		super(ClosedLoopPINN, self).__init__()
		self.save_every = save_every
		self.beta = beta_0
		self.dorsal_weight = dorsal_weight
		self.n_hidden_layers = n_hidden_layers
		self.hidden_width = hidden_width
		act = Sin
		layers = [3,] + ([hidden_width,] * n_hidden_layers) + [9,]
		lst = []
		for i in range(len(layers)-2):
			lst.append(nn.Linear(layers[i], layers[i+1]))
			lst.append(act())
		lst.append(nn.Linear(layers[-2], layers[-1]))
		self.model = nn.Sequential(*lst)
				
		self.cad_coefs = to_param(torch.zeros(3), requires_grad=True)
		self.sqh_coefs = to_param(torch.zeros(5), requires_grad=True)
		self.vel_coefs = to_param(torch.zeros(1), requires_grad=True) 
		self.model.register_parameter('cad_coefs', self.cad_coefs)
		self.model.register_parameter('sqh_coefs', self.sqh_coefs)
		self.model.register_parameter('vel_coefs', self.vel_coefs)
		
		self.gamma_dv = to_param(np.array([[1,0],[0, 0]])[None], requires_grad=False)
		
		self.t_train = to_param(t_train, requires_grad=True)
		self.y_train = to_param(y_train, requires_grad=True)  
		self.x_train = to_param(x_train, requires_grad=True)
		
		self.sqh_train = to_param(sqh_train, requires_grad=False)
		self.cad_train = to_param(cad_train, requires_grad=False)  
		self.vel_train = to_param(vel_train, requires_grad=False)
			
		self.lb = to_param(lower_bound, requires_grad=False)
		self.ub = to_param(upper_bound, requires_grad=False)
		
		self.optimizer_Adam = torch.optim.Adam(self.model.parameters(), lr=1e-3)
		self.optimizer_LBFGS = torch.optim.LBFGS(
			self.model.parameters(), 
			lr=1e0, 
			max_iter=50000, 
			max_eval=50000, 
			history_size=50,
			tolerance_grad=1e-5, 
			tolerance_change=1.0 * np.finfo(float).eps,
			line_search_fn="strong_wolfe"		# can be "strong_wolfe"
		)
		self.optimizer = self.optimizer_LBFGS
		self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_LBFGS, gamma=0.95)
		
		self.iter = 0
		self.apply(self.init_weights) 
		
	def init_weights(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight)
			m.bias.data.fill_(0.0)
	
	def forward(self, t, y, x):
		X = torch.cat([t, y, x], dim=-1)
		H = 2. * (X - self.lb) / (self.ub - self.lb) - 1.0
		scvdp = self.model(H)
		sqh = scvdp[:, 0:4].reshape([scvdp.shape[0], 2, 2])
		cad = scvdp[:, 4:5]
		vel = scvdp[:, 5:7]
		dor = 0.5*(torch.tanh(scvdp[:, 7:8].squeeze()) + 1) #range [0, 1]
		pre = scvdp[:, 8:9].squeeze()
		return sqh, cad, vel, dor, pre 
	
	def gradient(self, x, *y):
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
	
	def training_step(self, step_size=5000):  
		idx = np.random.choice(self.t_train.shape[0], step_size, replace=False)
		t, y, x = self.t_train[idx], self.y_train[idx], self.x_train[idx]
		sqh, cad, vel, dor, pre = self(t, y, x)
		
		dt_sqh = self.gradient(sqh, t)
		grad_sqh = self.gradient(sqh, y, x)
		
		dt_cad = self.gradient(cad, t)
		grad_cad = self.gradient(cad, y, x)
		
		dt_dor = self.gradient(dor, t)
		grad_dor = self.gradient(dor, y, x)
		
		grad_p = self.gradient(pre, y, x)
		grad_v = self.gradient(vel, y, x)
		lapl_v = self.gradient(grad_v[..., 0], y) + \
				 self.gradient(grad_v[..., 1], x)

		stokes_loss = self.vel_coefs[0].exp() * lapl_v - grad_p + torch.einsum('bijj->bi', grad_sqh)
		dor_dyn = dt_dor + torch.einsum('bj,bj->b', vel, grad_dor) #advection
	
		#Cadherin dynamics includes a dorsal source term which we will round to 0 or 1
		#Otherwise, pretty much all of the physics gets shoved into this field
		#So we won't let it store too much information
		cad_dyn =  dt_cad + torch.einsum('bj,bj->b', vel, grad_cad) #advection
		cad_dyn -= -torch.exp(self.cad_coefs[0]) * cad.squeeze() #detachment
		cad_dyn -= -torch.exp(self.cad_coefs[1]) * cad.squeeze() * torch.einsum('bkk->b', grad_v) #dilution
		#cad_dyn -= +torch.exp(self.cad_coefs[2]) * dor #dorsal source
		cad_dyn -= dor #Dorsal source (no coefficient now)
		 
		'''
		Active/Passive strain decomposition
		'''			  
		O = -0.5 * (torch.einsum('bij->bij', grad_v) - torch.einsum('bji->bij', grad_v))
		E = 0.5 *  (torch.einsum('bij->bij', grad_v) + torch.einsum('bji->bij', grad_v))
		
		deviatoric = sqh - 0.5 * torch.einsum('bkk,ij->bij', sqh, torch.eye(2, device=sqh.device))
		sqh_0 = torch.linalg.norm(sqh, dim=(1, 2)).mean()
		dev_mag = torch.linalg.norm(deviatoric, dim=(1, 2), keepdims=True)
		devE = torch.einsum('bkl,bkl->b', deviatoric, E)[:, None, None]
			
		E_active = E - torch.sign(devE) * devE * deviatoric / dev_mag**2
		E_active = 0.5 * E_active * dev_mag / sqh_0 
		E_passive = E - E_active		
		
		mE = torch.einsum('bik,bkj->bij', sqh, E_passive) + \
			 torch.einsum('bik,bkj->bij', E_passive, sqh)
		
		trm = torch.einsum('bkk->b', sqh)[:, None, None]
		
		sqh_dyn =  dt_sqh + torch.einsum('bk,bijk->bij', vel, grad_sqh) #advection
		sqh_dyn += torch.einsum('bik,bkj->bij', O, sqh) #co-rotation
		sqh_dyn -= torch.einsum('bik,bkj->bij', sqh, O) #co-rotation
		
		sqh_dyn -= -torch.exp(self.sqh_coefs[0]) * sqh #detachment
		sqh_dyn -= +torch.exp(self.sqh_coefs[1]) * mE #passive strain coupling
		sqh_dyn -= -torch.exp(self.sqh_coefs[2]) * cad[:, :, None] * mE #cadherin passive strain coupling
		sqh_dyn -= +torch.exp(self.sqh_coefs[3]) * trm * self.gamma_dv #DV source
		sqh_dyn -= +torch.exp(self.sqh_coefs[4]) * trm * sqh
					 
		sqh_loss = (sqh - self.sqh_train[idx]).pow(2).sum()
		cad_loss = (cad - self.cad_train[idx]).pow(2).sum()
		vel_loss = (vel - self.vel_train[idx]).pow(2).sum()

		cad_scale = self.cad_train[idx].pow(2).sum().item() / step_size
		sqh_scale = self.sqh_train[idx].pow(2).sum().item() / step_size
		vel_scale = self.vel_train[idx].pow(2).sum().item() / step_size
		
		#Scale MSE losses by magnitudes
		mse = sqh_loss / sqh_scale + \
			  cad_loss / cad_scale + \
			  vel_loss / vel_scale
		phys = stokes_loss.pow(2).sum() / vel_scale + \
			   dor_dyn.pow(2).sum() * self.dorsal_weight + \
			   cad_dyn.pow(2).sum() / cad_scale + \
			   sqh_dyn.pow(2).sum() / sqh_scale
		   
		return mse, phys
	
	def print(self, loss=None, mse=None, phys=None):
		outstr = 'Iteration %d\t' % self.iter
		if loss is not None:
			outstr += 'Loss: %e, MSE: %e, Phys: %e\n' % \
				(
					loss.item() if loss else 0., 
					mse.item() if loss else 0., 
					phys.item() if loss else 0.,
				)
		outstr += '\t%.3f grad^2 v - grad p = -div(m)\n' % self.vel_coefs[0].exp().item()
		outstr += '\tD_t c = -%.3g c + %.3g c div(v) + Gamma^{D}\n' % (
			self.cad_coefs[0].exp().item(),
			self.cad_coefs[1].exp().item(),
		)
		outstr += '\tD_t m = -%.3g m + (%.3g - %.3g c){m, E_p} + Tr(m) (%.3g Gamma^{DV} + %.3g m)' % (
			self.sqh_coefs[0].exp().item(),
			self.sqh_coefs[1].exp().item(),
			self.sqh_coefs[2].exp().item(),
			self.sqh_coefs[3].exp().item(),
			self.sqh_coefs[4].exp().item(),
		)
		print(outstr, flush=True)
	
	def loss_func(self):
		self.optimizer.zero_grad()
		mse, phys = self.training_step()
		loss = mse + self.beta * phys
		loss.backward()
		
		if self.iter % self.save_every == 0:
			self.print(loss, mse, phys)
			torch.save(self.state_dict(), self.save_name)

		self.iter += 1

		return loss
	
	def train(self, lbfgsIter, adamIter, inc_beta=int(1e5)):
		self.optimizer = self.optimizer_LBFGS
		self.save_name = 'pinn_closed_loop_beta=%.0e.ckpt' % self.beta
		for epoch in range(lbfgsIter):
			self.optimizer.step(self.loss_func)
			self.scheduler.step()
			
		self.print()
		torch.save(self.state_dict(), self.save_name)
		print('Finished L-BFGS optimization', flush=True)

		self.optimizer = self.optimizer_Adam
		for epoch in range(adamIter):
			loss = self.loss_func()
			self.optimizer.step()

			if epoch > 0 and epoch % inc_beta == 0:
				self.beta *= 10
				self.save_name = 'pinn_closed_loop_beta=%.0e.ckpt' % self.beta
				print('Incrementing beta to %e' % self.beta)
		
		self.print()
		torch.save(self.state_dict(), self.save_name)


if __name__=='__main__':
	plt.rcParams['font.size'] = 5
	atlas_dir = '/project/vitelli/jonathan/REDO_fruitfly/src/Public'

	loaddir = os.path.join(atlas_dir, 'WT/ECad-GFP/ensemble/')
	cad = np.load(os.path.join(loaddir, 'cyt2D.npy'), mmap_mode='r')
	vel = np.load(os.path.join(loaddir, 'velocity2D.npy'), mmap_mode='r')

	loaddir = os.path.join(atlas_dir, 'Halo_Hetero_Twist[ey53]_Hetero/Sqh-GFP/ensemble/')
	sqh = np.load(os.path.join(loaddir, 'tensor2D.npy'), mmap_mode='r')
	vel = np.load(os.path.join(loaddir, 'velocity2D.npy'), mmap_mode='r')
	t = np.load(os.path.join(loaddir, 't.npy'), mmap_mode='r')
	y = np.load(os.path.join(loaddir, 'DV_coordinates.npy'), mmap_mode='r')
	x = np.load(os.path.join(loaddir, 'AP_coordinates.npy'), mmap_mode='r')


	sqh = sqh * 3e1
	cad = cad * 1.5e0

	print('Data shapes')

	sqh = sqh.transpose(0, 3, 4, 1, 2)
	cad = cad.transpose(0, 1, 2)

	from scipy.ndimage import gaussian_filter
	from atlas_processing.anisotropy_detection import cell_size
	cad = np.stack([
		gaussian_filter(cad[i], sigma=cell_size) for i in range(cad.shape[0])])

	vel = vel.transpose(0, 2, 3, 1)

	print('Sqh: ', sqh.shape)
	print('Cad: ', cad.shape)
	print('Vel: ', vel.shape)

	nAP = x.shape[1]
	nDV = y.shape[0]
	nTP = t.shape[0]

	XX = np.broadcast_to(x[None], (nTP, nDV, nAP))
	YY = np.broadcast_to(y[None], (nTP, nDV, nAP))
	TT = np.broadcast_to(t[:, None, None], (nTP, nDV, nAP))

	print('Coordinates shapes')
	print('XX: ', XX.shape, 'Range [%g, %g] um' % (XX.min(), XX.max()))
	print('YY: ', YY.shape, 'Range [%g, %g] um' % (YY.min(), YY.max()))
	print('TT: ', TT.shape, 'Range [%g, %g] min' % (TT.min(), TT.max()))

	lower_bound = np.array([TT.min(), YY.min(), XX.min()])
	upper_bound = np.array([TT.max(), YY.max(), XX.max()])

	t = TT.flatten()[:, None]
	y = YY.flatten()[:, None]
	x = XX.flatten()[:, None]

	sqh_train = sqh.reshape([-1, *sqh.shape[3:]])
	cad_train = cad.reshape([-1, 1])
	vel_train = vel.reshape([-1, *vel.shape[3:]])

	print('Flattened shapes')
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
	print('Training shapes')
	print(t_train.shape, y_train.shape, x_train.shape)
	print(sqh_train.shape, cad_train.shape, vel_train.shape)
	print('\nStarting to train', flush=True)
	

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = ClosedLoopPINN(
		t_train, y_train, x_train,
		sqh_train, cad_train, vel_train,
		lower_bound, upper_bound, beta_0=1e1)
	model.to(device)
	model.train(0, int(1.2e6), inc_beta=int(3e5))