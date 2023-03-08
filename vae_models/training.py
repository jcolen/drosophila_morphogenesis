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

from utils.dataset import *
from convnext_models import *

import gc
import psutil

def run_train(dataset,
			  model_kwargs,
			  dl_kwargs,
		      logdir='/project/vitelli/jonathan/REDO_fruitfly/tb_logs/',
			  grad_clip=0.5,
			  epochs=100):

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	val_size = len(dataset) // 5
	train, val = random_split(
		dataset,
		[len(dataset)-val_size, val_size], 
		generator=torch.Generator().manual_seed(42))
	train_loader = DataLoader(train, **dl_kwargs, collate_fn=dataset.collate_fn)
	val_loader = DataLoader(val, **dl_kwargs, collate_fn=dataset.collate_fn)

	model = VAE_Evolver(**model_kwargs)
	model.to(device)
	print('Training ', model_kwargs['input'], model_kwargs['output'])
	optimizer = torch.optim.Adam(
		filter(lambda p: p.requires_grad, model.parameters()), 
		lr=model_kwargs['lr'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
	
	model_path = os.path.join(logdir, 
		'_'.join([model.__class__.__name__,','.join(model_kwargs['input']), 'beta=%.2g.ckpt' % model_kwargs['beta']]))

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
			y0 = batch[model_kwargs['output']].to(device)

			optimizer.zero_grad()
			y, pl = model.forward(x.to(device), batch['lengths'])
			res = residual(y, y0).mean()
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
		model.eval()
		with torch.no_grad():
			for batch in val_loader:
				x = torch.cat(
					[batch[i][:, 0] for i in model_kwargs['input']],
					axis=-3
				)
				y0 = batch[model_kwargs['output']].to(device)
				
				y, pl = model.forward(x.to(device), batch['lengths'])
				res = residual(y, y0).mean()
				kld = kld_loss(*pl)
				loss = res + model_kwargs['beta'] * kld
				val_loss += loss.item() / len(val_loader)
				res_val += res.item() / len(val_loader)
				kld_val += kld.item() / len(val_loader)
				gc.collect()

		scheduler.step(val_loss)
	
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
				'val_df': dataset.df.iloc[val.indices],
			}
			torch.save(save_dict, model_path)
			best_res = res_val
		
		gc.collect()
		torch.cuda.empty_cache()
