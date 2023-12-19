import numpy as np
import pandas as pd
import h5py
import sys
import os
import glob
import warnings
import gc

warnings.filterwarnings('ignore')

from tqdm.auto import tqdm

from scipy.stats import sem
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import Compose
from copy import deepcopy

from ..dataset import *
from .convnext_models import *

import gc
import psutil
from argparse import ArgumentParser

def masked_residual(u, v, mask=None):
	'''
	We assume u is the INPUT and v is the TARGET

	'''
	umag = torch.linalg.norm(u, dim=-3)
	vmag = torch.linalg.norm(v, dim=-3)

	uavg = torch.sqrt(umag.pow(2).mean(dim=(-2, -1), keepdims=True))
	vavg = torch.sqrt(vmag.pow(2).mean(dim=(-2, -1), keepdims=True))

	res = uavg**2 * vmag**2 + vavg**2 * umag**2 - 2 * uavg * vavg * torch.einsum('...ijk,...ijk->...jk', u, v)
	denom = 2 * vavg**2 * uavg**2
	denom[denom == 0] += 1
	res /= denom

	if mask is None:
		mask = torch.ones(res.shape, device=res.device, dtype=bool)

	res = res * mask #Only use the masked spatial region
	res = res.sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)) #This is a sum over space

	return res

def masked_mse(u, v, mask=None):
	mse = (u - v).pow(2).sum(dim=-3)
	
	if mask is None:
		mask = torch.ones(mse.shape, device=mse.device, dtype=bool)
	mse = mse * mask #Only use the masked spatial region
	mse = mse.sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)) #Sum over space

	return mse

def kld_loss(params, mu, logvar):
	kld = mu.pow(2) + logvar.exp() - logvar - 1
	kld = 0.5 * kld.sum(axis=-1).mean()
	return kld

def get_argument_parser():
	parser = ArgumentParser()
	parser.add_argument('--num_latent', type=int, default=64)
	parser.add_argument('--hidden_size', type=int, default=128)
	parser.add_argument('--lstm_layers', type=int, default=2)
	parser.add_argument('--beta', type=float, default=0)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--mode', type=str, choices=['embryo', 'LR', 'AP'], default='embryo')
	return parser

def train_val_split(dataset, mode='embryo', random_state=42):
	live_df = dataset.df[
		(dataset.df.sequence_index >= 0) & \
		(dataset.df.key == dataset.live_key)
	]
	
	embryos = live_df.embryoID.unique()
	embryos = dataset.df[dataset.df.sequence_index >= 0].embryoID.unique()
	assert len(embryos) > 1

	set1, set2 = train_test_split(embryos, test_size=0.5, random_state=random_state)
	
	if mode == 'embryo':
		#Split on embryo
		train, val = set1, set2
		print('Train embryos: ', train)
		print('Val embryos: ', val)
		train_idxs = live_df[live_df.embryoID.isin(train)].sequence_index.values
		val_idxs = live_df[live_df.embryoID.isin(val)].sequence_index.values
		train = Subset(deepcopy(dataset), train_idxs)
		val = Subset(deepcopy(dataset), val_idxs)

		#Mimic an embryo split by taking every other from static imaged embryos
		static_keys = dataset.df[~dataset.df.embryoID.isin(embryos)].key.unique()
		for key in static_keys:
			idxs = dataset.df[dataset.df.key == key].index.values
			trains = idxs[::2]
			vals = idxs[1::2]

			train.dataset.df = train.dataset.df.drop(vals, axis=0)
			val.dataset.df = val.dataset.df.drop(trains, axis=0)

	elif mode == 'LR':
		left, right = set1, set2
		print('Left embryos: ', left)
		print('Right embryos: ', right)
		dataset.df.loc[dataset.df.embryoID.isin(left), 'train_mask'] = 'left'
		dataset.df.loc[dataset.df.embryoID.isin(left), 'val_mask'] = 'right'
		dataset.df.loc[dataset.df.embryoID.isin(right), 'train_mask'] = 'right'
		dataset.df.loc[dataset.df.embryoID.isin(right), 'val_mask'] = 'left'
		
		train = Subset(dataset, np.arange(len(dataset), dtype=int))
		val = Subset(dataset, np.arange(len(dataset), dtype=int))

	elif mode == 'AP':
		ant, post = set1, set2
		print('Anterior embryos: ', ant)
		print('Posterior embryos: ', post)
		dataset.df.loc[dataset.df.embryoID.isin(ant), 'train_mask'] = 'anterior'
		dataset.df.loc[dataset.df.embryoID.isin(ant), 'val_mask'] = 'posterior'
		dataset.df.loc[dataset.df.embryoID.isin(post), 'train_mask'] = 'posterior'
		dataset.df.loc[dataset.df.embryoID.isin(post), 'val_mask'] = 'anterior'
		
		train = Subset(dataset, np.arange(len(dataset), dtype=int))
		val = Subset(dataset, np.arange(len(dataset), dtype=int))
	
	print('Train size: ', len(train))
	print('Val size: ', len(val))
	return train, val

def run_train(dataset,
			  model_kwargs,
		      logdir='/project/vitelli/jonathan/REDO_fruitfly/tb_logs/November2023',
			  grad_clip=0.5,
			  epochs=100):
	
	dl_kwargs = dict(
		num_workers=4, 
		batch_size=8, 
		shuffle=True, 
		pin_memory=True
	)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	split_mode = model_kwargs.get('mode', 'embryo')
	train, val = train_val_split(dataset, mode=split_mode)
	train_loader = DataLoader(train, **dl_kwargs, collate_fn=dataset.collate_fn)
	val_loader = DataLoader(val, **dl_kwargs, collate_fn=dataset.collate_fn)
	
	#Model parameters
	model_kwargs['stage_dims'] = [[32,32],[64,64],[128,128],[256,256]]
	model_kwargs['out_channels'] = 2
	model_kwargs['output'] = 'vel'
	
	#Retrain a model to improve magnitude
	#info = torch.load(os.path.join(logdir, 'March2023', 'MaskedVAE_Evolver_sqh_beta=0_split=embryo'))
	#model_kwargs = info['hparams']
	
	model = MaskedVAE_Evolver(**model_kwargs)
	#model.load_state_dict(info['state_dict'])

	model.to(device)
	print('Training ', model_kwargs['input'], model_kwargs['output'])
	optimizer = torch.optim.AdamW(
		filter(lambda p: p.requires_grad, model.parameters()), 
		lr=model_kwargs['lr'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, min_lr=1e-5)
	
	model_path = os.path.join(logdir, 
		'_'.join([
			model.__class__.__name__,
			','.join(model_kwargs['input']), 'beta=%.2g' % model_kwargs['beta'],
			'split=%s' % split_mode,
		]))

	print(os.path.basename(model_path))

	best_res = 1e5

	if 'epochs' in model_kwargs:
		epochs = model_kwargs['epochs']
	
	for epoch in range(epochs):
		model.train()
		for batch in train_loader:
			x = torch.cat(
				[batch[i][:, 0] for i in model_kwargs['input']],
				axis=-3
			)
			if not torch.is_tensor(batch['lengths']):
				batch['lengths'] = torch.tensor(batch['lengths']).to(device)
			y0 = batch[model_kwargs['output']].to(device)
			mask = batch['train_mask'].to(device)[:, None]

			optimizer.zero_grad()
			y, pl = model.forward(x.to(device), batch['lengths'])
			res = masked_residual(y, y0, mask).mean()
			#res = masked_mse(y, y0, mask).mean()
			kld = kld_loss(*pl)
			loss = res + model_kwargs['beta'] * kld
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			optimizer.step()

		gc.collect()
		torch.cuda.empty_cache()

		val_loss = 0.
		res_val = 0.
		kld_val = 0.
		residuals = []
		model.eval()
		with torch.no_grad():
			for batch in val_loader:
				x = torch.cat(
					[batch[i][:, 0] for i in model_kwargs['input']],
					axis=-3
				)
				if not torch.is_tensor(batch['lengths']):
					batch['lengths'] = torch.tensor(batch['lengths']).to(device)
				y0 = batch[model_kwargs['output']].to(device)
				mask = batch['val_mask'].to(device)[:, None]
				
				y, pl = model.forward(x.to(device), batch['lengths'])
				res = masked_residual(y, y0, mask).mean()
				#res = masked_mse(y, y0, mask).mean()
				kld = kld_loss(*pl)
				loss = res + model_kwargs['beta'] * kld
				val_loss += loss.item() / len(val_loader)
				res_val += res.item() / len(val_loader)
				kld_val += kld.item() / len(val_loader)

				residuals.append(res.item())

				gc.collect()

		scheduler.step(val_loss)

		res_val = np.mean(residuals)
	
		outstr = 'Epoch %d: Val Loss=%.3f' % (epoch, val_loss)
		outstr += '\tRes=%.3f  KLD=%.3f' % (res_val, kld_val)
		outstr += '\tRAM=%.3f GB' % (
			psutil.Process(os.getpid()).memory_info().rss / 1e9)
		outstr += '\tGPU=%.3f GB' % (
			torch.cuda.memory_allocated() / 1e9)
		print(outstr)
		if res_val < best_res:
			save_dict = {
				'state_dict': model.state_dict(),
				'hparams': model_kwargs,
				'epoch': epoch,
				'loss': val_loss,
				'res': res_val,
				'res_sem': sem(residuals),
				'res_std': np.std(residuals),
				'val_df': dataset.df.iloc[val.indices],
			}
			torch.save(save_dict, model_path)
			best_res = res_val
		
			#if input('Press q to quit. Press any key to continue: ') == 'q':
			#	break

		gc.collect()
		torch.cuda.empty_cache()
