import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from scipy.ndimage import gaussian_filter

def to_param(x, **kwargs):
	if isinstance(x, np.ndarray):
		return nn.Parameter(torch.from_numpy(x).float(), **kwargs)
	else:
		return nn.Parameter(x, **kwargs)

class Sin(nn.Module):
	def forward(self, x):
		return torch.sin(x)
	
class MyoVelPINN(nn.Module):
	'''
	Closed Loop PINN tries to learn the entire dynamics of the system
	Accepts [t, y, x] coordinate and predicts
	myosin tensor [2, 2] = 4 components [KNOWN]
	velocity field = 2 components		[KNOWN]
	dorsal source = 1 component			[UNKNOWN]
	pressure field = 1 component		[UNKNOWN]
	
	Subject to the closed loop conditions
	This includes an undetermined advected dorsal source and pressure field
	This also requires simultaneous learning of several parameters
	'''
	def __init__(self,
				 t_train, y_train, x_train,
				 sqh_train, vel_train,
				 lb, ub, 
				 hidden_width=100,
				 n_hidden_layers=7,
				 n_outs=7,
				 beta=1,
				 save_every=1000):
		super().__init__()
		
		self.init_model(n_hidden_layers, hidden_width, n_outs=n_outs)
		self.init_data(t_train, y_train, x_train, sqh_train, vel_train, lb, ub)	
		
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
		
		self.iter = 0
		self.beta = beta
		self.save_every = save_every

	def init_model(self, n_hidden_layers=7, hidden_width=64, n_outs=7):
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
				
		self.vel_coefs = to_param(torch.zeros(2), requires_grad=True) 
		self.model.register_parameter('vel_coefs', self.vel_coefs)
		
		self.apply(self.init_weights) 
	
	def init_data(self, t_train, y_train, x_train, sqh_train, vel_train, lb, ub):
		self.t_train = to_param(t_train, requires_grad=True)
		self.y_train = to_param(y_train, requires_grad=True)  
		self.x_train = to_param(x_train, requires_grad=True)
		
		self.sqh_train = to_param(sqh_train, requires_grad=False)
		self.vel_train = to_param(vel_train, requires_grad=False)
			
		self.lb = to_param(lb, requires_grad=False)
		self.ub = to_param(ub, requires_grad=False)
		
	def init_weights(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight)
			m.bias.data.fill_(0.0)
	
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
	
	def train(self, num_iter):
		self.save_name = f'{self.__class__.__name__}_beta={self.beta:.0e}.ckpt'
		for step in range(num_iter):
			loss = self.loss_func()
			self.optimizer.step()

		self.print()
		torch.save(self.state_dict(), self.save_name)
	
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

class CompressibleStokesPINN(MyoVelPINN):
	'''
	Predicts flow subject to the compressible stokes equation
	'''
	def __init__(self, *args, **kwargs):
		super().__init__(*args, n_outs=7, **kwargs)

	def get_boundary_points(self, Nb=20):
		bTDA = torch.rand([Nb, 3], device=self.t_train.device) * (self.ub - self.lb) + self.lb

		boundary_t = bTDA[:, 0].repeat(Nb*4)[:, None]
		boundary_y = torch.zeros([Nb, Nb * 4], device=self.t_train.device)
		boundary_x = torch.zeros([Nb, Nb * 4], device=self.t_train.device)

		boundary_y[:, :Nb] = bTDA[:, 1]
		boundary_x[:, :Nb] = self.lb[2]

		boundary_y[:, Nb:2*Nb] = bTDA[:, 1]
		boundary_x[:, Nb:2*Nb] = self.ub[2]

		boundary_y[:, 2*Nb:3*Nb] = self.lb[1]
		boundary_x[:, 2*Nb:3*Nb] = bTDA[:, 2]

		boundary_y[:, 3*Nb:4*Nb] = self.ub[1]
		boundary_x[:, 3*Nb:4*Nb] = bTDA[:, 2]	

		boundary_x = boundary_x.reshape([-1, 1])
		boundary_y = boundary_y.reshape([-1, 1])

		return boundary_t, boundary_y, boundary_x

	def forward(self, t, y, x):
		X = torch.cat([t, y, x], dim=-1)
		H = 2. * (X - self.lb) / (self.ub - self.lb) - 1.0
		scvdp = self.model(H)
		sqh = scvdp[:, 0:4].reshape([scvdp.shape[0], 2, 2])
		vel = scvdp[:, 4:6]
		pre = scvdp[:, 6:7]
		return sqh, vel, pre
	
	def training_step(self, step_size=5000):  
		idx = np.random.choice(self.t_train.shape[0], step_size, replace=False)
		t, y, x = self.t_train[idx], self.y_train[idx], self.x_train[idx]
		sqh, vel, pre = self(t, y, x)
					
		#MSE loss
		sqh_loss = (sqh - self.sqh_train[idx]).pow(2).sum()
		vel_loss = (vel - self.vel_train[idx]).pow(2).sum()
		mse = sqh_loss + vel_loss
	
		#Stokes loss
		grad_sqh = self.gradient(sqh, y, x)
		grad_v = self.gradient(vel, y, x)
		grad_p = self.gradient(pre, y, x)

		lapl_v = self.gradient(grad_v[..., 0], y) + \
				 self.gradient(grad_v[..., 1], x)
		div_m  = torch.einsum('bijj->bi', grad_sqh)
		
		alpha_over_nu = self.vel_coefs[0].exp()

		stokes_loss = lapl_v - grad_p + alpha_over_nu * div_m
		phys = stokes_loss.pow(2).sum()
		
		#Periodic boundary loss
		Nb = 20
		bt, by, bx = self.get_boundary_points(Nb)
		bm, bv, bp = self(bt, by, bx)
		bvals = torch.cat([
			bm.reshape([Nb, 4*Nb, 4]),
			bv.reshape([Nb, 4*Nb, 2]),
			bp.reshape([Nb, 4*Nb, 1]),
		], dim=-1)
		boundary_loss = (bvals[:, :Nb] - bvals[:, Nb:2*Nb]).pow(2).sum() + \
						(bvals[:, 2*Nb:3*Nb] - bvals[:, 3*Nb:]).pow(2).sum()
		
		   
		return mse+boundary_loss, phys
	
	def print(self, loss=None, mse=None, phys=None):
		outstr = 'Iteration %d\t' % self.iter
		if loss is not None:
			outstr += 'Loss: %e, MSE: %e, Phys: %e\n' % \
				(
					loss.item() if loss else 0., 
					mse.item() if loss else 0., 
					phys.item() if loss else 0.,
				)
		outstr += '\tgrad^2 v - grad p = -%.3f div(m)\n' % (
			self.vel_coefs[0].exp().item(),
		)
		print(outstr, flush=True)


class BulkCompressibleStokesPINN(CompressibleStokesPINN):
	def training_step(self, step_size=5000):  
		idx = np.random.choice(self.t_train.shape[0], step_size, replace=False)
		t, y, x = self.t_train[idx], self.y_train[idx], self.x_train[idx]
		sqh, vel, pre = self(t, y, x)
					
		#MSE loss
		sqh_loss = (sqh - self.sqh_train[idx]).pow(2).sum()
		vel_loss = (vel - self.vel_train[idx]).pow(2).sum()
		mse = sqh_loss + vel_loss
	
		#Stokes loss
		grad_sqh = self.gradient(sqh, y, x)
		grad_v = self.gradient(vel, y, x)
		grad_p = self.gradient(pre, y, x)

		lapl_v = self.gradient(grad_v[..., 0], y) + \
				 self.gradient(grad_v[..., 1], x)
		gdiv_v = self.gradient(torch.einsum('bjj->b', grad_v), y, x)
		div_m  = torch.einsum('bijj->bi', grad_sqh)
		
		alpha = self.vel_coefs[0].exp()
		beta = self.vel_coefs[1].exp()
		

		stokes_loss = lapl_v + beta * gdiv_v - grad_p + alpha * div_m
		phys = stokes_loss.pow(2).sum()

		#Periodic boundary loss
		Nb = 20
		bt, by, bx = self.get_boundary_points(Nb)
		bm, bv, bp = self(bt, by, bx)
		bvals = torch.cat([
			bm.reshape([Nb, 4*Nb, 4]),
			bv.reshape([Nb, 4*Nb, 2]),
			bp.reshape([Nb, 4*Nb, 1]),
		], dim=-1)
		boundary_loss = (bvals[:, :Nb] - bvals[:, Nb:2*Nb]).pow(2).sum() + \
						(bvals[:, 2*Nb:3*Nb] - bvals[:, 3*Nb:]).pow(2).sum()
		
		   
		return mse+boundary_loss, phys
	
	def print(self, loss=None, mse=None, phys=None):
		outstr = 'Iteration %d\t' % self.iter
		if loss is not None:
			outstr += 'Loss: %e, MSE: %e, Phys: %e\n' % \
				(
					loss.item() if loss else 0., 
					mse.item() if loss else 0., 
					phys.item() if loss else 0.,
				)
		outstr += '\tgrad^2 v + %.3f grad(div(v) - grad p = -%.3f div(m)\n' % (
			self.vel_coefs[1].exp().item(),
			self.vel_coefs[0].exp().item(),
		)
		print(outstr, flush=True)

class NoEndpointsCompressibleStokesPINN(CompressibleStokesPINN):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


from argparse import ArgumentParser
if __name__=='__main__':
	parser = ArgumentParser()
	parser.add_argument('--beta', type=float, default=1)
	args = parser.parse_args()

	loaddir = '/project/vitelli/jonathan/REDO_fruitfly/test_data'
	t = np.load(os.path.join(loaddir, 'time.npy'), mmap_mode='r')
	t_min = -10
	t_max = 25
	mask = np.logical_and(t >= t_min, t <= t_max)
	t = t[mask]
	sqh = np.load(os.path.join(loaddir, 'myosin.npy'), mmap_mode='r')[mask]
	sqh = sqh.reshape([sqh.shape[0], 2, 2, *sqh.shape[-2:]])
	vel = np.load(os.path.join(loaddir, 'velocity.npy'), mmap_mode='r')[mask]
	y = np.load(os.path.join(loaddir, 'DV_coordinates.npy'), mmap_mode='r')
	x = np.load(os.path.join(loaddir, 'AP_coordinates.npy'), mmap_mode='r')

	#For NoEndpoints case, crop the AP poles
	sqh = sqh.copy()
	sqh[..., -25:] = 0.
	sqh[..., :25] = 0.

	print('Data shapes')

	sqh = sqh.transpose(0, 3, 4, 1, 2)
	vel = vel.transpose(0, 2, 3, 1)

	print('Sqh: ', sqh.shape)
	print('Vel: ', vel.shape)

	nAP = x.shape[1]
	nDV = y.shape[0]
	nTP = t.shape[0]

	XX = np.broadcast_to(x[None], (nTP, nDV, nAP))
	YY = np.broadcast_to(y[None], (nTP, nDV, nAP))
	TT = np.broadcast_to(t[:, None, None], (nTP, nDV, nAP))

	print('Coordinates shapes')
	print(f'XX: {XX.shape} Range [{XX.min():g}, {XX.max():g}] um')
	print(f'YY: {YY.shape} Range [{YY.min():g}, {YY.max():g}] um')
	print(f'TT: {TT.shape} Range [{TT.min():g}, {TT.max():g}] min')

	lower_bound = np.array([TT.min(), YY.min(), XX.min()])
	upper_bound = np.array([TT.max(), YY.max(), XX.max()])

	t = TT.flatten()[:, None]
	y = YY.flatten()[:, None]
	x = XX.flatten()[:, None]

	sqh_train = sqh.reshape([-1, *sqh.shape[3:]])
	vel_train = vel.reshape([-1, *vel.shape[3:]])

	print('Flattened shapes')
	print(t.shape, y.shape, x.shape)
	print(sqh_train.shape, vel_train.shape)

	N_train = 100000
	idx = np.random.choice(nAP*nDV*nTP, N_train, replace=False)

	x_train = x[idx, :]
	y_train = y[idx, :]
	t_train = t[idx, :]

	sqh_train = sqh_train[idx, :]
	vel_train = vel_train[idx, :]
	print('Training shapes')
	print(t_train.shape, y_train.shape, x_train.shape)
	print(sqh_train.shape, vel_train.shape)
	print('\nStarting to train', flush=True)
	

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = NoEndpointsCompressibleStokesPINN(
		t_train, y_train, x_train,
		sqh_train, vel_train,
		lower_bound, upper_bound, beta=args.beta)
	model.to(device)
	model.train(int(5e5))
