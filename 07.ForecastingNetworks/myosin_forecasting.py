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
	sqh = AtlasDataset('Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP', 'tensor2D',
		transform=transform, drop_time=True)
	sqh_vel = AtlasDataset('Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP', 'velocity2D',
		transform=transform, drop_time=True)

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
