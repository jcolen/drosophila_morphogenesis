import numpy as np
import pandas as pd
import h5py
import sys
import os
import glob
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose
from argparse import ArgumentParser

basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))

from utils.dataset import *
from convnext_models import *
from training import run_train


from argparse import ArgumentParser
if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--num_latent', type=int, default=32)
	parser.add_argument('--hidden_size', type=int, default=64)
	parser.add_argument('--lstm_layers', type=int, default=2)
	parser.add_argument('--beta', type=float, default=0)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--epochs', type=int, default=100)
	model_kwargs = vars(parser.parse_args())

	dl_kwargs = dict(
		num_workers=4, 
		batch_size=8, 
		shuffle=True, 
		pin_memory=True
	)
	model_kwargs['stage_dims'] = [[32,32],[64,64],[128,128],[256,256]]
	model_kwargs['out_channels'] = 2
	model_kwargs['output'] = 'vel'

	transform = Compose([Reshape2DField(), Smooth2D(sigma=3), ToTensor()])

	rnt_vel = AtlasDataset('WT', 'Runt', 'velocity2D', 
		transform=Compose([Reshape2DField(), ToTensor()]))
	
	eve_vel = AtlasDataset('WT', 'Even_Skipped', 'velocity2D', 
		transform=Compose([Reshape2DField(), ToTensor()]), drop_time=True)

	rnt = AtlasDataset('WT', 'Runt', 'raw2D', transform=transform)
	eve = AtlasDataset('WT', 'Even_Skipped', 'raw2D', transform=transform, drop_time=True)
	ftz = AtlasDataset('WT', 'Fushi_Tarazu', 'raw2D', transform=transform)
	slp = AtlasDataset('WT', 'Sloppy_Paired', 'raw2D', transform=transform)
	prd = AtlasDataset('WT', 'Paired', 'raw2D', transform=transform)
	trt = AtlasDataset('WT', 'Tartan', 'raw2D', transform=transform)

	#Runt + Eve
	dataset = TrajectoryDataset(
		datasets=[
			('rnt', rnt),
			('eve', eve),
			('vel', rnt_vel),
			('vel', eve_vel),
		], live_key='vel',
		ensemble=1)
	model_kwargs['in_channels'] = 2
	model_kwargs['input'] = ['rnt', 'eve']
	run_train(dataset, model_kwargs, dl_kwargs)

	#Runt + PRG
	dataset = TrajectoryDataset(
		datasets=[
			('rnt', rnt),
			('vel', rnt_vel),
			('ftz', ftz),
			('slp', slp),
			('prd', prd),
			('trt', trt),
		], 
		live_key='vel', 
		ensemble=1)

	for key in ['ftz', 'slp', 'prd', 'trt']:
		model_kwargs['in_channels'] = 2
		model_kwargs['input'] = ['rnt', key]
		run_train(dataset, model_kwargs, dl_kwargs)
	
	'''
	#Eve + PRG
	dataset = TrajectoryDataset(
		datasets=[
			('eve', eve),
			('vel', eve_vel),
			('ftz', ftz),
			('slp', slp),
			('prd', prd),
			('trt', trt),
		], 
		live_key='vel', 
		ensemble=1)

	for key in ['ftz', 'slp', 'prd', 'trt']:
		model_kwargs['in_channels'] = 2
		model_kwargs['input'] = ['eve', key]
		run_train(dataset, model_kwargs, dl_kwargs)
	'''
