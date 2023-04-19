import os
import sys
basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))

from torchvision.transforms import Compose

from utils.dataset import *
from utils.vae.convnext_models import *
from utils.vae.training import *

if __name__ == '__main__':
	parser = get_argument_parser()
	model_kwargs = vars(parser.parse_args())

	transform = Compose([
		Reshape2DField(),
		LeftRightSymmetrize(),
		#AnteriorPosteriorMask(),
		ToTensor()
	])

	#Base datasets
	cad = AtlasDataset('WT', 'ECad-GFP', 'raw2D',
		transform=Compose([Smooth2D(sigma=7), transform]))
	cad_vel = AtlasDataset('WT', 'ECad-GFP', 'velocity2D', 
		transform=transform)

	#Cadherin
	dataset = TrajectoryDataset(
		datasets=[
			('cad', cad),
			('vel', cad_vel),
		],
		live_key='vel',
	)
	model_kwargs['in_channels'] = 1
	model_kwargs['input'] = ['cad']
	run_train(dataset, model_kwargs)
