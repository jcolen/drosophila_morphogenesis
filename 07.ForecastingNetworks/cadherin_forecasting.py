import os
import sys

from torchvision.transforms import Compose

from morphogenesis.dataset import *
from morphogenesis.flow_networks.forecasting_models import *
from training import *

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
