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
import gc

basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'release'))

from utils.dataset import *
from utils.vae.convnext_models import *
from utils.vae.training import *

if __name__ == '__main__':
	parser = get_argument_parser()
	model_kwargs = vars(parser.parse_args())

	#Model parameters
	model_kwargs['stage_dims'] = [[32,32],[64,64],[128,128],[256,256]]
	model_kwargs['out_channels'] = 2
	model_kwargs['output'] = 'vel'
	model_kwargs['num_latent'] = 64

	'''
	Define datasets
	'''
	cad = AtlasDataset('WT', 'ECad-GFP', 'raw2D',
		transform=Compose([Reshape2DField(), Smooth2D(sigma=7), ToTensor()]))
	cad_vel = AtlasDataset('WT', 'ECad-GFP', 'velocity2D', 
		transform=Compose([Reshape2DField(), ToTensor()]))

	sqh = AtlasDataset('Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP', 'tensor2D',
		transform=Compose([Reshape2DField(), ToTensor()]), drop_time=True)
	sqh_vel = AtlasDataset('Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP', 'velocity2D',
		transform=Compose([Reshape2DField(), ToTensor()]), drop_time=True)
	
	rnt = AtlasDataset('WT', 'Runt', 'raw2D', 
		transform=Compose([Reshape2DField(), Smooth2D(sigma=3), ToTensor()]))
	eve = AtlasDataset('WT', 'Even_Skipped', 'raw2D', 
		transform=Compose([Reshape2DField(), Smooth2D(sigma=3), ToTensor()]), drop_time=True)

	'''
	Myosin + Genes
	'''
	dataset = TrajectoryDataset(
		datasets=[
			('sqh', sqh),
			('vel', sqh_vel),
			('rnt', rnt),
			('eve', eve),
		],
		live_key='vel',
		ensemble=2,
	)
	model_kwargs['in_channels'] = 5
	model_kwargs['input'] = ['sqh', 'rnt']
	run_train(dataset, model_kwargs)

	gc.collect()
	
	model_kwargs['input'] = ['sqh', 'eve']
	run_train(dataset, model_kwargs)

	gc.collect()
	
	'''
	Cadherin + Genes
	'''
	dataset = TrajectoryDataset(
		datasets=[
			('cad', cad),
			('vel', cad_vel),
			('rnt', rnt),
			('eve', eve),
		],
		live_key='vel',
		ensemble=2,
	)
	model_kwargs['in_channels'] = 2
	model_kwargs['input'] = ['cad', 'rnt']
	run_train(dataset, model_kwargs)

	gc.collect()

	model_kwargs['input'] = ['cad', 'eve']
	run_train(dataset, model_kwargs)

	gc.collect()
