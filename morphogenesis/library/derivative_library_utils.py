import os
import pickle as pk
import numpy as np
import pandas as pd
import pysindy as ps

from scipy.ndimage import gaussian_filter

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

def get_gradient(data, YY, XX):
	'''
	Compute the gradient of the data
	'''
	y_shape = YY.shape[0]
	x_shape = XX.shape[1]
	y_axis = np.argwhere(np.array(data.shape) == y_shape)[0][0]
	x_axis = np.argwhere(np.array(data.shape) == x_shape)[0][0]

	diffY = ps.SmoothedFiniteDifference(d=1, axis=y_axis)
	diffX = ps.SmoothedFiniteDifference(d=1, axis=x_axis)
	diffY.smoother_kws['axis'] = y_axis
	diffX.smoother_kws['axis'] = x_axis

	return np.stack([
		diffY._differentiate(data, YY[:, 0]),
		diffX._differentiate(data, XX[0, :]),
	], axis=-1)

def validate_key_and_derivatives(h5f, data, key, order=2):
	'''
	Ensure that key and its derivatives up to specified order are present in the dataset
	Compute the derivatives if they are not
	'''
	try:
		ds = h5f.require_dataset(key, shape=data.shape, dtype=data.dtype)
		ds[:] = data
	except:
		del h5f[key]
		h5f.create_dataset(key, data=data)

	xi = data.copy()
	dx = []
	for i in range(order):
		name = f'D{i+1} {key}'
		xi = get_gradient(xi, h5f['DV_coordinates'], h5f['AP_coordinates'])
		dx.append(xi.copy())
		try:
			ds = h5f.require_dataset(name, shape=xi.shape, dtype=xi.dtype)
			ds[:] = xi
		except:
			del h5f[name]
			h5f.create_dataset(name, data=xi)

	return dx

def validate_coordinates(h5f, TT, YY, XX):
	'''
	Ensure that coordinates are present in the dataset
	'''
	try:
		tt = h5f.require_dataset('time', shape=TT.shape, dtype=TT.dtype)
		tt[:] = TT
	except:
		del h5f['time']
		h5f.create_dataset('time', data=TT)
	
	try:
		yy = h5f.require_dataset('DV_coordinates', shape=YY.shape, dtype=YY.dtype)
		yy[:] = YY
	except:
		del h5f['DV_coordinates']
		h5f.create_dataset('DV_coordinates', data=YY)
	
	try:
		xx = h5f.require_dataset('AP_coordinates', shape=XX.shape, dtype=XX.dtype)
		xx[:] = XX
	except:
		del h5f['AP_coordinates']
		h5f.create_dataset('AP_coordinates', data=XX)

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

def scalar_library(h5f, 
				   data, 
				   coords, 
				   key='c', 
				   project=None, 
				   keep_frac=0.95, 
				   sigma=7):
	'''
	h5f - h5py.File object
	data - scalar data of shape [T, 1, Y, X]
	coords - (T, Y, X) tuple of coordinates
	key - key to use for scalar field when saving
	project - decomposition model to project data onto, or None 
	threshold - explained variance threshold for decomposition model
	sigma - smoothing kernel size for data
	'''
	if project is None:
		print(f'No projection function provided, smoothing with cell size {sigma}')
		data = data.reshape([-1, *data.shape[-2:]])
		data = np.stack([gaussian_filter(data[i], sigma=sigma) for i in range(data.shape[0])])
	else:
		keep = np.cumsum(project['svd'].explained_variance_ratio_) <= keep_frac
		data = project.inverse_transform(project.transform(data), keep)[:, 0]
	
	validate_coordinates(h5f, *coords)
	d1_x, d2_x = validate_key_and_derivatives(h5f, data, key, order=2)
	lib = {}
	attrs = {}

	feat = key
	lib[feat] = data
	attrs[feat] = {key: 1, 'space': 0}

	feat = 'grad(%s)^2' % key
	lib[feat] = np.einsum('tyxi,tyxi->tyx', d1_x, d1_x)
	attrs[feat] = {key: 2, 'space': 2}

	feat = 'grad^2 %s' % key
	lib[feat] = np.einsum('tyxii->tyx', d2_x)
	attrs[feat] = {key: 1, 'space': 2}
	
	write_library_to_dataset(lib, h5f.require_group('scalar_library'), attrs)

def vector_library(h5f,
				   data,
				   coords,
				   key='v',
				   project=None,
				   keep_frac=0.95):
	'''
	The velocity derivative terms we care about are VORTICITY and STRAIN RATE
	We will decompose them further and couple them with tensors later in the pipeline

	h5f - h5py.File object
	data - vector data of shape [T, 2, Y, X]
	coords - (T, Y, X) tuple of coordinates
	key - key to use for scalar field when saving
	project - decomposition model to project data onto, or None 
	threshold - explained variance threshold for decomposition model
	'''
	if project is None:
		print(f'No projection function provided, using raw data')
	else:
		keep = np.cumsum(project['svd'].explained_variance_ratio_) <= keep_frac
		data = project.inverse_transform(project.transform(data), keep)
	
	validate_coordinates(h5f, *coords)
	d1_u = validate_key_and_derivatives(h5f, data, key, order=1)[0]

	lib = {}
	attrs = {}
	Eij = 0.5 * (np.einsum('tiyxj->tijyx', d1_u) + np.einsum('tjyxi->tijyx', d1_u))
	Oij = 0.5 * (np.einsum('tiyxj->tijyx', d1_u) - np.einsum('tjyxi->tijyx', d1_u))
	
	lib['O'] = Oij
	attrs['O'] = {'v': 1, 'space': 1}

	lib['E'] = Eij
	attrs['E'] = {'v': 1, 'space': 1}
	
	write_library_to_dataset(lib, h5f.require_group('tensor_library'), attrs)

def tensor_library(h5f,
				   data,
				   coords,
				   key='m',
				   project=None,
				   keep_frac=0.95):
	'''
	Compute terms mapping tensors to symmetric tensors

	h5f - h5py.File object
	data - tensor data of shape [T, 4, Y, X]
	coords - (T, Y, X) tuple of coordinates
	key - key to use for scalar field when saving
	project - decomposition model to project data onto, or None 
	threshold - explained variance threshold for decomposition model
	'''
	if project is None:
		print(f'No projection function provided, using raw data')
	else:
		keep = np.cumsum(project['svd'].explained_variance_ratio_) <= keep_frac
		data = project.inverse_transform(project.transform(data), keep)
	
	data = data.reshape([data.shape[0], 2, 2, *data.shape[-2:]])
	validate_coordinates(h5f, *coords)
	validate_key_and_derivatives(h5f, data, key, order=1)
	
	lib = {}
	attrs = {}

	lib[key] = data
	attrs[key] = {key: 1, 'space': 0}

	lib['%s Tr(%s)' % (key, key)] = np.einsum('tkkyx,tijyx->tijyx', data, data)
	attrs['%s Tr(%s)' % (key, key)] = {key: 2, 'space': 0}

	lib['%s %s' % (key, key)] = np.einsum('tikyx,tkjyx->tijyx', data, data)
	attrs['%s %s' % (key, key)] = {key: 2, 'space': 0}
	
	write_library_to_dataset(lib, h5f.require_group('tensor_library'), attrs)