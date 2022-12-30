import numpy as np
import pandas as pd
import h5py
import sys
import os
import glob
import warnings
import gc

warnings.filterwarnings('ignore')

from tqdm import tqdm

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose

basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))

from dataset import *
from convnext_models import *
	
def residual(u, v):
	umag = torch.linalg.norm(u, dim=-3)
	vmag = torch.linalg.norm(v, dim=-3)
	
	uavg = torch.sqrt(umag.pow(2).mean(dim=(-2, -1), keepdims=True))
	vavg = torch.sqrt(vmag.pow(2).mean(dim=(-2, -1), keepdims=True))

	res = uavg**2 * vmag**2 + vavg**2 * umag**2 - 2 * uavg * vavg * torch.einsum('...ijk,...ijk->...jk', u, v)
	res /= 2 * vavg**2 * uavg**2
	return res

def kld_loss(params, mu, logvar):
	kld = mu.pow(2) + logvar.exp() - logvar - 1
	kld = 0.5 * kld.sum(axis=-1).mean()
	return kld

'''
Following arXiv:1804.03599, we run a sweep on the KL divergence with a high beta
This ensures we obtain models at different levels of the KL divergence (rather than relative to the data itself)
There are some proposed orthogonality pressures which should make our latent spaces cleaner?
'''
def run_C_sweep(dataset, 
			    logdir, 
			    model_kwargs, 
			    dl_kwargs,
			    grad_clip=0.5,
			    epochs=1000,
				save_every=50,
			    C_min=10.,
			    C_max=50.):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	val_size = len(dataset) // 5
	train, val = random_split(dataset, [len(dataset)-val_size, val_size])
	val_indices = val.indices
	val_df = dataset.df.iloc[val_indices]
	train_loader = DataLoader(train, **dl_kwargs)
	val_loader = DataLoader(val, **dl_kwargs)

	model = VAE(**model_kwargs)
	model.to(device)
	print(model_kwargs['input'], model_kwargs['output'])
	optimizer = torch.optim.Adam(
		filter(lambda p: p.requires_grad, model.parameters()), 
		lr=model_kwargs['lr'])

	model_logdir = os.path.join(logdir, 
		'_'.join(['DisentangledAdam',model.__class__.__name__,model_kwargs['input'], model_kwargs['output']]))
	if not os.path.exists(model_logdir):
		os.mkdir(model_logdir)

	C = C_min
	C_step = (C_max - C_min) / epochs / len(train_loader)
	
	for epoch in range(epochs):
		for bb, batch in enumerate(train_loader):
			if isinstance(input, list):
				x = torch.cat([batch[i] for i in model_kwargs['input']], axis=-3).to(device)
			else:
				x = batch[model_kwargs['input']].to(device)
			y0 = batch[model_kwargs['output']].to(device)
			optimizer.zero_grad()
			y, pl = model.forward(x)
			res = residual(y, y0).mean()
			mag = (y.pow(2).sum(dim=(1,2,3)) - y0.pow(2).sum(dim=(1,2,3))).abs().mean()
			kld = kld_loss(*pl)
			loss = res + model_kwargs['alpha'] * mag + model_kwargs['beta'] * (kld-C).abs()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			optimizer.step()

			C = C + C_step

		val_loss = 0.
		res_val = 0.
		mag_val = 0.
		kld_val = 0.
		with torch.no_grad():
			for bb, batch in enumerate(val_loader):
				if isinstance(input, list):
					x = torch.cat([batch[i] for i in model_kwargs['input']], axis=-3).to(device)
				else:
					x = batch[model_kwargs['input']].to(device)
				y0 = batch[model_kwargs['output']].to(device)
				
				y, pl = model(x)
				res = residual(y, y0).mean()
				mag = (y.pow(2).sum(dim=(1,2,3)) - y0.pow(2).sum(dim=(1,2,3))).abs().mean()
				kld = kld_loss(*pl)
				loss = res + model_kwargs['alpha'] * mag + model_kwargs['beta'] * (kld-C).abs()
				val_loss += loss.item() / len(val_loader)
				res_val += res.item() / len(val_loader)
				mag_val += mag.item() / len(val_loader)
				kld_val += kld.item() / len(val_loader)
	
		outstr = 'Epoch %d\tVal Loss=%g' % (epoch, val_loss)
		outstr += '\tRes=%g\tKLD=%g\tC=%g' % (res_val, kld_val, C)
		print(outstr)
		if epoch % save_every == 0:
			save_dict = {
				'state_dict': model.state_dict(),
				'hparams': model_kwargs,
				'epoch': epoch,
				'loss': val_loss,
				'val_df': val_df,
			}
			torch.save(
				save_dict, 
				os.path.join(model_logdir, 'C=%.2g.ckpt' % C))

'''
Sweeping on C didn't seem to work, so instead we'll sweep on beta
'''
def run_beta_sweep(dataset, 
			    logdir, 
			    model_kwargs, 
			    dl_kwargs,
			    grad_clip=0.5,
			    epochs=1000,
				save_every=25,
				log_beta_max=0,
				log_beta_min=-5):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	val_size = len(dataset) // 5
	train, val = random_split(dataset, [len(dataset)-val_size, val_size])
	val_indices = val.indices
	val_df = dataset.df.iloc[val_indices]
	train_loader = DataLoader(train, **dl_kwargs)
	val_loader = DataLoader(val, **dl_kwargs)

	model = VAE(**model_kwargs)
	model.to(device)
	print(model_kwargs['input'], model_kwargs['output'])
	optimizer = torch.optim.Adam(
		filter(lambda p: p.requires_grad, model.parameters()), 
		lr=model_kwargs['lr'])

	model_logdir = os.path.join(logdir, 
		'_'.join(['DisentangledAdam',model.__class__.__name__,model_kwargs['input'], model_kwargs['output']]))
	if not os.path.exists(model_logdir):
		os.mkdir(model_logdir)

	log_beta = log_beta_max
	log_beta_step = (log_beta_min-log_beta_max) / epochs / len(train_loader)

	best_res = 10
	
	for epoch in range(epochs):
		for bb, batch in enumerate(train_loader):
			if isinstance(input, list):
				x = torch.cat([batch[i] for i in model_kwargs['input']], axis=-3).to(device)
			else:
				x = batch[model_kwargs['input']].to(device)
			y0 = batch[model_kwargs['output']].to(device)
			optimizer.zero_grad()
			y, pl = model.forward(x)
			res = residual(y, y0).mean()
			mag = (y.pow(2).sum(dim=(1,2,3)) - y0.pow(2).sum(dim=(1,2,3))).abs().mean()
			kld = kld_loss(*pl)
			loss = res + model_kwargs['alpha'] * mag + np.power(10., log_beta) * kld
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			optimizer.step()

			log_beta = log_beta + log_beta_step

		val_loss = 0.
		res_val = 0.
		mag_val = 0.
		kld_val = 0.
		with torch.no_grad():
			for bb, batch in enumerate(val_loader):
				if isinstance(input, list):
					x = torch.cat([batch[i] for i in model_kwargs['input']], axis=-3).to(device)
				else:
					x = batch[model_kwargs['input']].to(device)
				y0 = batch[model_kwargs['output']].to(device)
				
				y, pl = model(x)
				res = residual(y, y0).mean()
				mag = (y.pow(2).sum(dim=(1,2,3)) - y0.pow(2).sum(dim=(1,2,3))).abs().mean()
				kld = kld_loss(*pl)
				loss = res + model_kwargs['alpha'] * mag + np.power(10., log_beta) * kld
				val_loss += loss.item() / len(val_loader)
				res_val += res.item() / len(val_loader)
				mag_val += mag.item() / len(val_loader)
				kld_val += kld.item() / len(val_loader)
	
		outstr = 'Epoch %d\tVal Loss=%g' % (epoch, val_loss)
		outstr += '\tRes=%g\tKLD=%g\tbeta=%g' % (res_val, kld_val, np.power(10, log_beta))
		print(outstr)
		if epoch > 0 and epoch % save_every == 0:
			if res_val < best_res:
				save_dict = {
					'state_dict': model.state_dict(),
					'hparams': model_kwargs,
					'epoch': epoch,
					'loss': val_loss,
					'val_df': val_df,
				}
				torch.save(
					save_dict, 
					os.path.join(model_logdir, 'log10beta=%.2g.ckpt' % log_beta))
				best_res = res_val
	
if __name__ == '__main__':
	transform=Compose([Reshape2DField(), ToTensor()])
	dl_kwargs = dict(
		num_workers=2, 
		batch_size=8, 
		shuffle=True, 
		pin_memory=True
	)

	
	model_kwargs = dict(
		lr=1e-4,
		num_latent=16,
		stage_dims=[[32,32],[64,64],[128,128],[256,256]],
		alpha=0.,
	)

	logdir = '/project/vitelli/jonathan/REDO_fruitfly/tb_logs/'
	
	cad = AtlasDataset('WT', 'ECad-GFP', 'tensor2D', transform=transform)
	sqh = AtlasDataset('WT', 'sqh-mCherry', 'tensor2D', transform=transform)
	vel = AtlasDataset('WT', 'ECad-GFP', 'velocity2D', transform=transform)
	dataset = AlignedDataset([sqh, cad, vel], ['sqh', 'cad', 'vel'])
	
	model_kwargs['in_channels'] = 2
	model_kwargs['out_channels'] = 2
	model_kwargs['input'] = 'vel'
	model_kwargs['output'] = 'vel'
	run_beta_sweep(dataset, logdir, model_kwargs, dl_kwargs, log_beta_max=0., log_beta_min=-5.)	
	
	
	model_kwargs['in_channels'] = 4
	model_kwargs['out_channels'] = 4
	model_kwargs['input'] = 'sqh'
	model_kwargs['output'] = 'sqh'
	run_beta_sweep(dataset, logdir, model_kwargs, dl_kwargs, log_beta_max=-1., log_beta_min=-6.)

	model_kwargs['input'] = 'cad'
	model_kwargs['output'] = 'cad'
	run_beta_sweep(dataset, logdir, model_kwargs, dl_kwargs, log_beta_max=-1., log_beta_min=-6.)

	runt = AtlasDataset('WT', 'Runt', 'raw2D', transform=transform)
	dataset = AlignedDataset([runt], ['runt'])
	model_kwargs['input'] = 'runt'
	model_kwargs['output'] = 'runt'
	model_kwargs['in_channels'] = 1
	model_kwargs['out_channels'] = 1
	run_beta_sweep(dataset, logdir, model_kwargs, dl_kwargs, log_beta_max=-3., log_beta_min=-8.)
	
