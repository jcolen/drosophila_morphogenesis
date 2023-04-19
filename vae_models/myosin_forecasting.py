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
