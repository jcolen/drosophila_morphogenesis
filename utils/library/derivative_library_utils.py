import os
import pickle as pk
import numpy as np
import pandas as pd
import pysindy as ps

from ..decomposition.decomposition_model import SVDPipeline

def project_embryo_data(folder, embryoID, base, threshold=0.95, model_type=SVDPipeline):
	'''
	Get the PCA data for a given embryoID
	base - velocity, tensor, etc. tells us what PCA to look for
	return the inverse_transformed PCA data keeping only terms up to a given explained variance
	'''
	#Check if SVDPipeline exists for this data
	path = os.path.join(folder, 'decomposition_models', f'{base}_{model_type.__name__}')
	print(f'Checking for {model_type.__name__} for this dataset')
	
	with open(f'{path}.pkl', 'rb') as f:
		model = pk.load(f)

	df = pd.read_csv(path+'.csv')
	
	df = df[df.embryoID == embryoID]
	keep = np.cumsum(model['svd'].explained_variance_ratio_) <= threshold
	params = df.filter(like='param', axis=1).values[:, keep]

	return model.inverse_transform(params, keep)

def get_derivative_tensors(x, YY, XX, order=2):
	'''
	Return derivative tensors up to specified order
	Return shape is [..., Y, X, I, J, K, ...]
		Assumes the last two axes of x are spatial (Y, X)
		Places the derivative index axes at the end

	We now pass coordinates from the folder which are in units of microns

	Deprecated note:
		Here we use the coordinates from vitelli_sharing/pixel_coordinates.mat
		Thus, we compute spatial derivatives in units of PIV scale pixels NOT microns
		The PIV image size is 0.4 x [original dimensions], so each PIV pixel is 
			equivalent to 2.5 x original pixel size 
		The original units are 1pix=0.2619 um, so 1 PIV pix = 0.65479 um
	'''
	diffY = ps.SmoothedFiniteDifference(d=1, axis=-2)
	diffX = ps.SmoothedFiniteDifference(d=1, axis=-1)
	diffY.smoother_kws['axis'] = -2
	diffX.smoother_kws['axis'] = -1

	xi = x.copy()

	d = []
	for i in range(1, order+1):
		diffY.smoother_kws['axis'] = -(2+i)
		diffX.smoother_kws['axis'] = -(1+i)
		xi = np.stack([
			diffY._differentiate(xi, YY[:, 0]),
			diffX._differentiate(xi, XX[0, :]),
		], axis=-1)
		d.append(xi.copy())

	return d

def validate_key_and_derivatives(x, group, YY, XX, key, order=2):
	'''
	Ensure that key and its derivatives up to specified order are present in the dataset
	Compute the derivatives if they are not
	'''
	if not key in group:
		group.create_dataset(key, data=x)
	
	flag = False
	dx = []
	for i in range(order):
		name = f'D{i+1} {key}' % (i+1, key)
		if name in group:
			dx.append(group[name])
		else:
			flag = True
	
	if flag:
		print('Computing derivatives!')
		dx = get_derivative_tensors(x, YY, XX, order=order)
		for i in range(order):
			name = f'D{i+1} {key}' % (i+1, key)
			if name in group:
				del group[name]
			group.create_dataset(name, data=dx[i])

		if 'AP_coordinates' in group:
			del group['AP_coordinates']
		group.create_dataset('AP_coordinates', data=XX)
		
		if 'DV_coordinates' in group:
			del group['DV_coordinates']
		group.create_dataset('DV_coordinates', data=YY)
			
	return dx

def write_library_to_dataset(lib, glib, attrs=None):
	'''
	Write library elements to the dataset if they don't already exist
	'''
	for k in lib:
		if not k in glib:
			glib.create_dataset(k, data=lib[k])
		elif not np.array_equal(glib[k], lib[k]):
			del glib[k]
			glib.create_dataset(k, data=lib[k])
		if attrs is not None:
			glib[k].attrs.update(attrs[k])
