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

	rnt = AtlasDataset('WT', 'Runt', 'raw2D', 
		transform=Compose([Reshape2DField(), Smooth2D(sigma=3), ToTensor()]))
	rnt_vel = AtlasDataset('WT', 'Runt', 'velocity2D', 
		transform=Compose([Reshape2DField(), ToTensor()]))
	
	eve = AtlasDataset('WT', 'Even_Skipped-YFP', 'raw2D', 
		transform=Compose([Reshape2DField(), Smooth2D(sigma=3), ToTensor()]), drop_time=True)
	eve_vel = AtlasDataset('WT', 'Even_Skipped-YFP', 'velocity2D', 
		transform=Compose([Reshape2DField(), ToTensor()]), drop_time=True)
	
	#Eve
	dataset = TrajectoryDataset(
		datasets=[
			('eve', eve),
			('vel', eve_vel),
		],
		live_key='vel',
	)
	model_kwargs['in_channels'] = 1
	model_kwargs['input'] = ['eve']
	run_train(dataset, model_kwargs)

	print(torch.cuda.memory_allocated() / 1e9)
	torch.cuda.empty_cache()
	print(torch.cuda.memory_allocated() / 1e9)

	#Runt
	dataset = TrajectoryDataset(
		datasets=[
			('rnt', rnt),
			('vel', rnt_vel),
		],
		live_key='vel',
	)
	model_kwargs['in_channels'] = 1
	model_kwargs['input'] = ['rnt']
	run_train(dataset, model_kwargs)
	
