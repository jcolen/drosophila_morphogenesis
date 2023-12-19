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
sys.path.insert(0, os.path.join(basedir, 'release'))

from utils.dataset import *
from utils.vae.convnext_models import *
from utils.vae.training import *


if __name__ == '__main__':
	parser = get_argument_parser()
	model_kwargs = vars(parser.parse_args())

	model_kwargs['stage_dims'] = [[32,32],[64,64],[128,128],[256,256]]
	model_kwargs['out_channels'] = 2
	model_kwargs['output'] = 'vel'

	transform = Compose([Reshape2DField(), Smooth2D(sigma=3), ToTensor()])

	rnt_vel = AtlasDataset('WT', 'Runt', 'velocity2D', 
		transform=Compose([Reshape2DField(), ToTensor()]))
	eve_vel = AtlasDataset('WT', 'Even_Skipped-YFP', 'velocity2D', 
		transform=Compose([Reshape2DField(), ToTensor()]), drop_time=True)

	rnt = AtlasDataset('WT', 'Runt', 'raw2D', transform=transform)
	eve = AtlasDataset('WT', 'Even_Skipped-YFP', 'raw2D', transform=transform, drop_time=True)
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
			('ftz', ftz),
			('slp', slp),
			('prd', prd),
			('trt', trt),
		], live_key='vel',
		ensemble=2)
	model_kwargs['in_channels'] = 2
	model_kwargs['input'] = ['rnt', 'eve']
	run_train(dataset, model_kwargs)

	for key in ['ftz', 'slp', 'prd', 'trt']:
		model_kwargs['input'] = ['rnt', key]
		run_train(dataset, model_kwargs)
		
		model_kwargs['input'] = ['eve', key]
		run_train(dataset, model_kwargs)

		#model_kwargs['in_channels'] = 3
		#model_kwargs['input'] = ['rnt', 'eve', key]
		#run_train(dataset, model_kwargs)
