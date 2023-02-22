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
from atlas_processing.anisotropy_detection import *

import gc
import psutil
	
def run_train(dataset,
			  model_kwargs,
			  dl_kwargs,
		      logdir='/project/vitelli/jonathan/REDO_fruitfly/tb_logs/',
			  grad_clip=0.5,
			  epochs=250):

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
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

	model_logdir = os.path.join(logdir, 
		'_'.join([model.__class__.__name__,str(model_kwargs['input']), model_kwargs['output']]))
	if not os.path.exists(model_logdir):
		os.mkdir(model_logdir)

	best_res = 1e5
	
	for epoch in range(epochs):
		model.train()
		with tqdm(train_loader, unit='batch') as ttrain:
			for batch in ttrain:
				x = torch.cat(
					[batch[i] for i in model_kwargs['input']],
					axis=-3
				)
				y0 = batch[model_kwargs['output']].to(device)

				optimizer.zero_grad()
				y, pl = model.forward(x.to(device))
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
			with tqdm(val_loader, unit='batch') as tval:
				for batch in tval:
					x = torch.cat(
						[batch[i] for i in model_kwargs['input']],
						axis=-3
					)
					y0 = batch[model_kwargs['output']].to(device)
					y, pl = model(x.to(device))
					
					res = residual(y, y0).mean()
					kld = kld_loss(*pl)
					loss = res + model_kwargs['beta'] * kld
					val_loss += loss.item() / len(val_loader)
					res_val += res.item() / len(val_loader)
					kld_val += kld.item() / len(val_loader)
					
					gc.collect()

			scheduler.step(val_loss)

			outstr = 'Epoch %d\tVal Loss=%.3g' % (epoch, val_loss)
			outstr += '\tRes=%.3g\tKLD=%.3g' % (res_val, kld_val)
			outstr += '\tMem Usage=%.3f GB' % (
				psutil.Process(os.getpid()).memory_info().rss / 1e9)
			print(outstr)
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
					os.path.join(model_logdir, 'beta=%.2g.ckpt' % model_kwargs['beta']))
				best_res = res_val

			gc.collect()
			torch.cuda.empty_cache()

from argparse import ArgumentParser
if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--num_latent', type=int, default=32)
	parser.add_argument('--alpha', type=float, default=0)
	parser.add_argument('--beta', type=float, default=1e-3)
	parser.add_argument('--lr', type=float, default=1e-4)
	model_kwargs = vars(parser.parse_args())

	transform=Compose([Reshape2DField(), ToTensor()])
	dl_kwargs = dict(
		num_workers=2, 
		batch_size=8, 
		shuffle=True, 
		pin_memory=True
	)
	model_kwargs['stage_dims'] = [[32,32],[64,64],[128,128],[256,256]]
	model_kwargs['out_channels'] = 2
	model_kwargs['output'] = 'vel'
	
	cad = AtlasDataset('WT', 'ECad-GFP', 'cyt2D', 
		transform=Compose([Reshape2DField(), Smooth2D(sigma=cell_size), ToTensor()]))
	cad_vel = AtlasDataset('WT', 'ECad-GFP', 'velocity2D', 
		transform=Compose([Reshape2DField(), ToTensor()]))

	sqh = AtlasDataset('Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP', 'tensor2D',
		transform=Compose([Reshape2DField(), ToTensor()]), drop_time=True)
	sqh_vel = AtlasDataset('Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP', 'velocity2D',
		transform=Compose([Reshape2DField(), ToTensor()]), drop_time=True)

	'''
	Myosin and cadherin
	'''
	dataset = JointDataset(
		datasets=[
			('sqh', sqh),
			('vel', sqh_vel),
			('cad', cad),
			('vel', cad_vel),
		],
		ensemble=3,
	)
	model_kwargs['in_channels'] = 5
	model_kwargs['input'] = ['sqh', 'cad']
	run_train(dataset, model_kwargs, dl_kwargs)
