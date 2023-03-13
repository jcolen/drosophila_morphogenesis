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

	'''
	Define datasets
	'''
	cadRaw = AtlasDataset('WT', 'ECad-GFP', 'raw2D',
		transform=Compose([Reshape2DField(), Smooth2D(sigma=8), ToTensor()]))
	#cadCyt = AtlasDataset('WT', 'ECad-GFP', 'cyt2D', 
	#	transform=Compose([Reshape2DField(), Smooth2D(sigma=8), ToTensor()]))
	cad_vel = AtlasDataset('WT', 'ECad-GFP', 'velocity2D', 
		transform=Compose([Reshape2DField(), ToTensor()]))

	sqh = AtlasDataset('Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP', 'tensor2D',
		transform=Compose([Reshape2DField(), ToTensor()]), drop_time=True)
	sqh_vel = AtlasDataset('Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP', 'velocity2D',
		transform=Compose([Reshape2DField(), ToTensor()]), drop_time=True)

	#Myosin
	dataset = TrajectoryDataset(
		datasets=[
			('sqh', sqh),
			('vel', sqh_vel),
		],
		live_key='vel',
	)
	model_kwargs['in_channels'] = 4
	model_kwargs['input'] = ['sqh']
	run_train(dataset, model_kwargs, dl_kwargs)

	#Cadherin
	dataset = TrajectoryDataset(
		datasets=[
			('cadRaw', cadRaw),
			('vel', cad_vel),
		],
		live_key='vel',
	)
	model_kwargs['in_channels'] = 1
	model_kwargs['input'] = ['cadRaw']
	run_train(dataset, model_kwargs, dl_kwargs)

	'''
	dataset = TrajectoryDataset(
		datasets=[
			('cadCyt', cadCyt),
			('vel', cad_vel),
		],
		live_key='vel',
	)
	model_kwargs['in_channels'] = 1
	model_kwargs['input'] = ['cadCyt']
	run_train(dataset, model_kwargs, dl_kwargs)
	'''
