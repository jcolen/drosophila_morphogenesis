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
from training import *

if __name__ == '__main__':
	parser = get_argument_parser()
	model_kwargs = vars(parser.parse_args())

	#Model parameters
	model_kwargs['stage_dims'] = [[32,32],[64,64],[128,128],[256,256]]
	model_kwargs['out_channels'] = 2
	model_kwargs['output'] = 'vel'

	#Base datasets
	cad = AtlasDataset('WT', 'ECad-GFP', 'raw2D',
		transform=Compose([Reshape2DField(), Smooth2D(sigma=8), ToTensor()]))
	cad_vel = AtlasDataset('WT', 'ECad-GFP', 'velocity2D', 
		transform=Compose([Reshape2DField(), ToTensor()]))

	sqh = AtlasDataset('Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP', 'tensor2D',
		transform=Compose([Reshape2DField(), ToTensor()]), drop_time=True)
	sqh_vel = AtlasDataset('Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP', 'velocity2D',
		transform=Compose([Reshape2DField(), ToTensor()]), drop_time=True)

	'''
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
	run_train(dataset, model_kwargs)
	'''

	#Cadherin
	#dataset = TrajectoryDataset(
	dataset = SequenceDataset(
		datasets=[
			('cad', cad),
			('vel', cad_vel),
		],
		live_key='vel',
	)
	model_kwargs['in_channels'] = 1
	model_kwargs['input'] = ['cad']
	run_train(dataset, model_kwargs)
