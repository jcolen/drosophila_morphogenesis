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

import pytorch_lightning as pl
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

	res = uavg**2 * vmag**2 + vavg**2 * umag**2 -			2 * uavg * vavg * torch.einsum('...ijk,...ijk->...jk', u, v)
	res /= 2 * vavg**2 * uavg**2
	return res

def kld_loss(params, logvar):
	kld = params.pow(2) + logvar.exp() - logvar - 1
	kld = kld.sum(axis=-1).mean()
	return 0.5 * kld

class TranslationVAE(pl.LightningModule):
	def __init__(self,
				 input='sqh',
				 output='vel',
				 in_channels=4,
				 out_channels=2,
				 lr=1e-3,
				 beta=1e-2,
				 num_latent=16,
				 input_size=(236,200),
				 stage_dims=[[32,32],[64,64],[128,128]],
				 *args, **kwargs):
		super(TranslationVAE, self).__init__()
		self.model = VAE(
			num_latent=num_latent, 
			in_channels=in_channels,
			out_channels=out_channels,
			input_size=input_size,
			stage_dims=stage_dims,
			**kwargs)
		self.save_hyperparameters(
			'input', 'in_channels', 'output', 'out_channels', 'num_latent',
			'lr', 'beta', 'input_size', 'stage_dims')
	
	def getxy(self, batch):
		if isinstance(input, list):
			x = torch.cat([batch[i] for i in self.hparams['input']], axis=-3)
		else:
			x = batch[self.hparams['input']]
		y0 = batch[self.hparams['output']]

		return x.to(self.device), y0.to(self.device)

	def compute_loss(self, batch):
		x, y0 = self.getxy(batch)
		y, pl = self(x)
		res_loss = residual(y, y0).mean()
		mse_loss = F.mse_loss(y, y0)
		vae_loss = kld_loss(*pl)
		return res_loss, mse_loss, vae_loss
		
	def forward(self, x):
		return self.model(x)

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
		model = TranslationVAE(beta=beta, **model_kwargs)
		model.to(device)
		print(beta, model.hparams['input'], model.hparams['output'])
		optimizer = torch.optim.Adam(
			filter(lambda p: p.requires_grad, model.parameters()), 
			lr=model.hparams['lr'])
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
		model_logdir = os.path.join(logdir, model.__class__.__name__ + '_'+model.hparams['input']+'_'+model.hparams['output'])
		
		min_loss = 1e10
		best_epoch = -1

		for epoch in range(max_epochs):
			for bb, batch in enumerate(train_loader):
				optimizer.zero_grad()
				mse, res, kld = model.compute_loss(batch)
				loss = mse + model.hparams['beta'] * kld
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
				optimizer.step()

			val_loss = 0.
			with torch.no_grad():
				for bb, batch in enumerate(val_loader):
					mse, res, kld = model.compute_loss(batch)
					loss = mse + model.hparams['beta'] * kld
					val_loss += loss.item() / len(val_loader)
			
			print('Epoch %d\tVal Loss=%g' % (epoch, val_loss))
			scheduler.step()

			if val_loss < min_loss:
				save_dict = {
					'state_dict': model.state_dict(),
					'hparams': model.hparams,
					'epoch': epoch,
					'loss': loss,
					'val_df': val_df,
				}
				torch.save(
					save_dict, 
					os.path.join(model_logdir, 'beta=%g.ckpt' % beta))
				min_loss = loss
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
		lr=1e-3, 
		num_latent=16,
		stage_dims=[[32,32],[64,64],[128,128],[256,256]],
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
