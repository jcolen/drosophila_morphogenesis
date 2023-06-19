import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from myosin_dynamics_pinn import *
from myosin_stokes_pinn import *

from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger('run_MyosinDynamicsPINN')

def load_data(N_train=100000, output=True):
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

	logger.debug('Data shapes')
	logger.debug(f'Sqh: {sqh.shape}')
	logger.debug(f'Cad: {cad.shape}')
	logger.debug(f'Vel: {vel.shape}')

	nAP = x.shape[1]
	nDV = y.shape[0]
	nTP = t.shape[0]

	XX = np.broadcast_to(x[None], (nTP, nDV, nAP))
	YY = np.broadcast_to(y[None], (nTP, nDV, nAP))
	TT = np.broadcast_to(t[:, None, None], (nTP, nDV, nAP))

	logger.debug('Coordinates shapes')
	logger.debug(f'XX: {XX.shape} Range [{XX.min():g}, {XX.max():g}] um')
	logger.debug(f'YY: {YY.shape} Range [{YY.min():g}, {YY.max():g}] um')
	logger.debug(f'TT: {TT.shape} Range [{TT.min():g}, {TT.max():g}] um')

	lower_bound = np.array([TT.min(), YY.min(), XX.min()])
	upper_bound = np.array([TT.max(), YY.max(), XX.max()])

	t = TT.flatten()[:, None]
	y = YY.flatten()[:, None]
	x = XX.flatten()[:, None]

	sqh_train = sqh.reshape([-1, *sqh.shape[3:]])
	cad_train = cad.reshape([-1, 1])
	vel_train = vel.reshape([-1, *vel.shape[3:]])

	logger.debug(t.shape, y.shape, x.shape)
	logger.debug(sqh_train.shape, cad_train.shape, vel_train.shape)

	idx = np.random.choice(nAP*nDV*nTP, N_train, replace=False)

	x_train = x[idx, :]
	y_train = y[idx, :]
	t_train = t[idx, :]

	sqh_train = sqh_train[idx, :]
	cad_train = cad_train[idx, :]
	vel_train = vel_train[idx, :]

	logger.debug('Training data')
	logger.debug(t_train.shape, y_train.shape, x_train.shape)
	logger.debug(sqh_train.shape, cad_train.shape, vel_train.shape)

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
	model = CadherinPlusSourcePINN(**data, **args)
	model.to(device)
	model.train(250000)
