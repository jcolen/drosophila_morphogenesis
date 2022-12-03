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
from gradient_descent_the_ultimate_optimizer import gdtuo

basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))

from dataset import *
from convnext_models import *
	
def residual(u, v):
	umag = torch.linalg.norm(u, dim=-3)
	vmag = torch.linalg.norm(v, dim=-3)
	
	uavg = torch.sqrt(umag.pow(2).mean(dim=(-2, -1), keepdims=True))
	vavg = torch.sqrt(vmag.pow(2).mean(dim=(-2, -1), keepdims=True))

	res = uavg**2 * vmag**2 + vavg**2 * umag**2 -			2 * uavg * vavg * torch.einsum('...ijk,...ijk->...jk', u, v)
	res /= 2 * vavg**2 * uavg**2
	return res

def kld_loss(params, logvar):
	kld = params.pow(2) + logvar.exp() - logvar - 1
	kld = kld.sum(axis=-1).mean()
	return 0.5 * kld

def run_beta_sweep(dataset, 
				   betas, 
				   logdir, 
				   model_kwargs, 
				   dl_kwargs,
				   grad_clip=0.5,
				   max_epochs=500, 
				   patient=10):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	val_size = len(dataset) // 5
	train, val = random_split(dataset, [len(dataset)-val_size, val_size])
	val_indices = val.indices
	val_df = dataset.df.iloc[val_indices]
	train_loader = DataLoader(train, **dl_kwargs)
	val_loader = DataLoader(val, **dl_kwargs)

	for beta in betas:
		model = VAE(**model_kwargs)
		model.to(device)
		print(beta, model_kwargs['input'], model_kwargs['output'])
		opt1 = gdtuo.Adam(1e-3, optimizer=gdtuo.SGD(1e-5))
		mw = gdtuo.ModuleWrapper(model, optimizer=opt1)
		mw.initialize()

		model_logdir = os.path.join(logdir, 
			'_'.join(['GDTUO',model.__class__.__name__,model_kwargs['input'], model_kwargs['output']]))
		if not os.path.exists(model_logdir):
			os.mkdir(model_logdir)
		model_kwargs['beta'] = beta	
		min_loss = 1e10
		best_epoch = -1

		for epoch in range(max_epochs):
			for bb, batch in enumerate(train_loader):
				mw.begin()
				if isinstance(input, list):
					x = torch.cat([batch[i] for i in model_kwargs['input']], axis=-3).to(device)
				else:
					x = batch[model_kwargs['input']].to(device)
				y0 = batch[model_kwargs['output']].to(device)

				y, pl = mw.forward(x)
				res = residual(y, y0).mean()
				mag = (y.pow(2).sum(dim=(1,2,3)) - y0.pow(2).sum(dim=(1,2,3))).abs().mean()
				kld = kld_loss(*pl)
				loss = res + model_kwargs['alpha'] * mag + model_kwargs['beta'] * kld
				mw.zero_grad()
				loss.backward(create_graph=True)
				torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
				mw.step()

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
					loss = res + model_kwargs['alpha'] * mag + model_kwargs['beta'] * kld
					val_loss += loss.item() / len(val_loader)
					res_val += res.item() / len(val_loader)
					mag_val += mag.item() / len(val_loader)
					kld_val += kld.item() / len(val_loader)
		
			outstr = 'Epoch %d\tVal Loss=%g\tLR=%g' % (epoch, val_loss, mw.optimizer.parameters['alpha'].item())
			outstr += '\tRes=%g\tMag=%g\tKLD=%g' % (res_val, mag_val, kld_val)
			print(outstr)

			if val_loss < min_loss:
				save_dict = {
					'state_dict': model.state_dict(),
					'hparams': model_kwargs,
					'epoch': epoch,
					'loss': loss,
					'val_df': val_df,
				}
				torch.save(
					save_dict, 
					os.path.join(model_logdir, 'beta=%g.ckpt' % beta))
				min_loss = val_loss
				best_epoch = epoch

			#Early stopping
			if epoch - best_epoch > patient:
				print('Stoppping at epoch %d' % epoch)
				break
		
if __name__ == '__main__':
	transform=Compose([Reshape2DField(), ToTensor()])
	betas = np.power(10., np.arange(-10, -1))
	dl_kwargs = dict(
		num_workers=2, 
		batch_size=8, 
		shuffle=True, 
		pin_memory=True
	)
		
	model_kwargs = dict(
		num_latent=16,
		stage_dims=[[32,32],[64,64],[128,128],[256,256]],
		alpha=1e-8,
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
	run_beta_sweep(dataset, betas, logdir, model_kwargs, dl_kwargs)

	model_kwargs['alpha'] = 0.
	
	model_kwargs['in_channels'] = 4
	model_kwargs['out_channels'] = 4
	model_kwargs['input'] = 'sqh'
	model_kwargs['output'] = 'sqh'
	run_beta_sweep(dataset, betas, logdir, model_kwargs, dl_kwargs)

	model_kwargs['input'] = 'cad'
	model_kwargs['output'] = 'cad'
	run_beta_sweep(dataset, betas, logdir, model_kwargs, dl_kwargs)


	runt = AtlasDataset('WT', 'Runt', 'raw2D', transform=transform)
	dataset = AlignedDataset([runt], ['runt'])
	model_kwargs['input'] = 'runt'
	model_kwargs['output'] = 'runt'
	model_kwargs['in_channels'] = 1
	model_kwargs['out_channels'] = 1
	run_beta_sweep(dataset, betas, logdir, model_kwargs, dl_kwargs)
