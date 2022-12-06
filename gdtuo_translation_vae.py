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

from translation_vae import residual, kld_loss

def run_C_sweep(dataset, 
			   C_values, 
			   logdir, 
			   model_kwargs, 
			   dl_kwargs,
			   grad_clip=0.5,
			   epochs_per_step=5):
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

	opt1 = gdtuo.Adam(1e-4, optimizer=gdtuo.SGD(1e-6))
	mw = gdtuo.ModuleWrapper(model, optimizer=opt1)
	mw.initialize()

	model_logdir = os.path.join(logdir, 
		'_'.join(['GDTUO_DisentangledAdam',model.__class__.__name__,model_kwargs['input'], model_kwargs['output']]))
	if not os.path.exists(model_logdir):
		os.mkdir(model_logdir)
	
	for C in C_values:
		print('Target KLD = %g ' % C)
		min_loss = 1e10

		for epoch in range(epochs_per_step):
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
				loss = res + model_kwargs['alpha'] * mag + model_kwargs['beta'] * (kld-C).abs()
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
					
					y, pl = mw.forward(x)
					res = residual(y, y0).mean()
					mag = (y.pow(2).sum(dim=(1,2,3)) - y0.pow(2).sum(dim=(1,2,3))).abs().mean()
					kld = kld_loss(*pl)
					loss = res + model_kwargs['alpha'] * mag + model_kwargs['beta'] * (kld-C).abs()
					val_loss += loss.item() / len(val_loader)
					res_val += res.item() / len(val_loader)
					mag_val += mag.item() / len(val_loader)
					kld_val += kld.item() / len(val_loader)
		
				outstr = 'Epoch %d\tVal Loss=%g\tLR=%g' % (epoch, val_loss, mw.optimizer.parameters['alpha'].detach().item())
				outstr += '\tRes=%g\tMag=%g\tKLD=%g' % (res_val, mag_val, kld_val)
				print(outstr)

				if val_loss < min_loss:
					save_dict = {
						'state_dict': model.state_dict(),
						'hparams': model_kwargs,
						'epoch': epoch,
						'loss': val_loss,
						'val_df': val_df,
					}
					torch.save(
						save_dict, 
						os.path.join(model_logdir, 'C=%g.ckpt' % C))
					min_loss = val_loss
					best_epoch = epoch
	
def run_beta_sweep(dataset, 
				   betas, 
				   logdir, 
				   model_kwargs, 
				   dl_kwargs,
				   grad_clip=0.5,
				   max_epochs=500,
				   patient=20):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	val_size = len(dataset) // 5
	train, val = random_split(dataset, [len(dataset)-val_size, val_size])
	val_indices = val.indices
	val_df = dataset.df.iloc[val_indices]
	train_loader = DataLoader(train, **dl_kwargs)
	val_loader = DataLoader(val, **dl_kwargs)
		
	model = VAE(**model_kwargs)
	model.to(device)
	opt1 = gdtuo.AdamBaydin(1e-3, optimizer=gdtuo.SGD(1e-5))
	mw = gdtuo.ModuleWrapper(model, optimizer=opt1)
	mw.initialize()

	#Rather than retrain from scratch, do a slow beta sweep

	for beta in betas:
		print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/2**30))
		print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/2**30))
		print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/2**30))

		model_kwargs['beta'] = beta
		print(model_kwargs['beta'], model_kwargs['input'], model_kwargs['output'])


		model_logdir = os.path.join(logdir, 
			'_'.join(['GDTUO',model.__class__.__name__,model_kwargs['input'], model_kwargs['output']]))
		if not os.path.exists(model_logdir):
			os.mkdir(model_logdir)
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
					
					y, pl = mw.forward(x)
					res = residual(y, y0).mean()
					mag = (y.pow(2).sum(dim=(1,2,3)) - y0.pow(2).sum(dim=(1,2,3))).abs().mean()
					kld = kld_loss(*pl)
					loss = res + model_kwargs['alpha'] * mag + model_kwargs['beta'] * kld
					val_loss += loss.detach().item() / len(val_loader)
					res_val += res.detach().item() / len(val_loader)
					mag_val += mag.detach().item() / len(val_loader)
					kld_val += kld.detach().item() / len(val_loader)
		
				outstr = 'Epoch %d\tVal Loss=%g\tLR=%g' % (epoch, val_loss, mw.optimizer.parameters['alpha'].detach().item())
				outstr += '\tRes=%g\tMag=%g\tKLD=%g' % (res_val, mag_val, kld_val)
				print(outstr)

				if val_loss < min_loss:
					save_dict = {
						'state_dict': model.state_dict(),
						'hparams': model_kwargs,
						'epoch': epoch,
						'loss': val_loss,
						'val_df': val_df,
					}
					torch.save(
						save_dict, 
						os.path.join(model_logdir, 'beta=%g.ckpt' % model_kwargs['beta']))
					min_loss = val_loss
					best_epoch = epoch

				#Early stopping
				if epoch - best_epoch > patient:
					print('Stoppping at epoch %d' % epoch)
					break
		
if __name__ == '__main__':
	transform=Compose([Reshape2DField(), ToTensor()])
	dl_kwargs = dict(
		num_workers=2, 
		batch_size=8, 
		shuffle=True, 
		pin_memory=True
	)
		
	model_kwargs = dict(
		beta=1,
		num_latent=32,
		stage_dims=[[32,32],[64,64],[128,128],[256,256]],
		alpha=0.,
	)

	logdir = '/project/vitelli/jonathan/REDO_fruitfly/tb_logs/'
	
	Cs = np.linspace(0., 100., 20)
	
	cad = AtlasDataset('WT', 'ECad-GFP', 'tensor2D', transform=transform)
	sqh = AtlasDataset('WT', 'sqh-mCherry', 'tensor2D', transform=transform)
	vel = AtlasDataset('WT', 'ECad-GFP', 'velocity2D', transform=transform)
	dataset = AlignedDataset([sqh, cad, vel], ['sqh', 'cad', 'vel'])
	
	model_kwargs['in_channels'] = 2
	model_kwargs['out_channels'] = 2
	model_kwargs['input'] = 'vel'
	model_kwargs['output'] = 'vel'
	run_C_sweep(dataset, Cs, logdir, model_kwargs, dl_kwargs)

	model_kwargs['alpha'] = 0.
	
	model_kwargs['in_channels'] = 4
	model_kwargs['out_channels'] = 4
	model_kwargs['input'] = 'sqh'
	model_kwargs['output'] = 'sqh'
	run_C_sweep(dataset, Cs, logdir, model_kwargs, dl_kwargs)

	model_kwargs['input'] = 'cad'
	model_kwargs['output'] = 'cad'
	run_C_sweep(dataset, Cs, logdir, model_kwargs, dl_kwargs)

	runt = AtlasDataset('WT', 'Runt', 'raw2D', transform=transform)
	dataset = AlignedDataset([runt], ['runt'])
	model_kwargs['input'] = 'runt'
	model_kwargs['output'] = 'runt'
	model_kwargs['in_channels'] = 1
	model_kwargs['out_channels'] = 1
	run_C_sweep(dataset, Cs, logdir, model_kwargs, dl_kwargs)
