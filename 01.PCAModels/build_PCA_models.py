import os
import pickle as pk
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter
from torchvision.transforms import Compose

from morphogenesis.dataset import *
from morphogenesis.decomposition.decomposition_model import SVDPipeline

def build_decomposition_model(dataset, model_type=SVDPipeline, tmin=-15, tmax=45, save=True, **model_kwargs):
	'''
	Learn a decomposition model on an AtlasDataset object
	'''
	df = dataset.df.drop(['folder', 'tiff'], axis=1).reset_index()
	test_size = len(df) * 2 // 5
	test_idx = np.random.choice(df.index, test_size, replace=False)

	df['set'] = 'train'
	df.loc[test_idx, 'set'] = 'test'
	df['t'] = df['time']
	df['time'] = df['time'].astype(int)

	y0 = []
	for e in df.embryoID.unique():
		e_data = dataset.values[e]
		e_idx = df[df.embryoID == e].eIdx.values
		e_data = e_data[e_idx]
		y0.append(e_data.reshape([e_data.shape[0], -1, *e_data.shape[-2:]]))
	y0 = np.concatenate(y0, axis=0)

	if isinstance(dataset.transform, Compose) and \
	   isinstance (dataset.transform.transforms[1], Smooth2D):
		sigma = dataset.transform.transforms[1].sigma
		y0 = np.stack([
			np.stack([
				gaussian_filter(y0[t, c], sigma=sigma) \
				for c in range(y0.shape[1])]) \
			for t in range(y0.shape[0])])

	model = model_type(whiten=True, **model_kwargs)

	train_mask = (df.set == 'train') & (df.time >= tmin) & (df.time <= tmax)
	train_mask = (df.time >= tmin) & (df.time <= tmax)
	train = y0[df[train_mask].index]

	if dataset.filename == 'velocity2D':
		scaler_train = np.zeros([1, *train.shape[-3:]])
	else:
		scaler_train = y0[df[(train_mask) & (df.time < 0)].index]

	model.fit(train, scaler_train)

	params = model.transform(y0)
	df['mag'] = np.linalg.norm(model.inverse_transform(params), axis=1).mean(axis=(-1, -2))
	df = pd.concat([df, pd.DataFrame(params).add_prefix('param')], axis=1)

	if save:
		path = os.path.join(dataset.path, 'decomposition_models')
		if not os.path.exists(path):
			os.mkdir(path)
		path = os.path.join(path, f'{dataset.filename[:-2]}_{model.__class__.__name__}')
		with open(path+'.pkl', 'wb') as f:
			pk.dump(model, f)
		
		df.to_csv(path+'.csv')

	return model, df

if __name__ == '__main__':
    transform = Reshape2DField()

    print('eCadherin dataset')
    cad_dataset = AtlasDataset('WT', 'ECad-GFP', 'raw2D', transform=Compose([transform, Smooth2D(sigma=7)]), tmin=-15, tmax=45)
    cad_vel_dataset = AtlasDataset('WT', 'ECad-GFP', 'velocity2D', transform=transform, tmin=-15, tmax=45)

    build_decomposition_model(cad_dataset, crop=10)
    build_decomposition_model(cad_vel_dataset, crop=10)

    print('Myosin dataset')
    sqh_dataset = AtlasDataset('Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP', 'tensor2D', transform=transform, drop_time=True, tmin=-15, tmax=45)
    sqh_vel_dataset = AtlasDataset('Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP', 'velocity2D', transform=transform, drop_time=True, tmin=-15, tmax=45)

    build_decomposition_model(sqh_dataset, crop=10)
    build_decomposition_model(sqh_vel_dataset, crop=10)

    print('Actin dataset')
    act_dataset = AtlasDataset('WT', 'Moesin-GFP', 'raw2D', transform=Compose([transform, LeftRightSymmetrize(), Smooth2D(sigma=7)]), tmin=-15, tmax=45)
    act_vel_dataset = AtlasDataset('WT', 'Moesin-GFP', 'velocity2D', transform=transform, tmin=-15, tmax=45)

    build_decomposition_model(act_dataset, crop=10)
    build_decomposition_model(act_vel_dataset, crop=10)

    print('Other datasets')
    runt_dataset = AtlasDataset('WT', 'Runt', 'raw2D', transform=Compose([transform, Smooth2D(sigma=3)]), tmin=-15, tmax=45)
    eve_dataset = AtlasDataset('WT', 'Even_Skipped', 'raw2D', transform=Compose([transform, Smooth2D(sigma=3)]), drop_time=True, tmin=-15, tmax=45)

    build_decomposition_model(runt_dataset, crop=10, lrsym=False)
    build_decomposition_model(eve_dataset, crop=10, lrsym=False)