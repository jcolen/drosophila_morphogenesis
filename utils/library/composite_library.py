import os
from time import sleep

import h5py
import numpy as np
import pysindy as ps
from tqdm.auto import tqdm

from ..decomposition.decomposition_utils import get_decomposition_model
from ..geometry.geometry_utils import TangentSpaceTransformer

def fill_group_info(data, embryoID, libraries, filename='derivative_library.h5'):
	'''
	Fill an embryo group with links, fields, and determine the feature names
	'''
	group = data.require_group(embryoID)
	links = group.require_group('links')
	fields = group.require_group('fields')

	feature_names = set()
	with tqdm(total=len(libraries), leave=False) as pbar:
		pbar.set_description('Collecting links and metadata')
		for info in libraries:
			key, path, library, decomposition = info

			#Is the embryo in this library? Or should we just link the ensemble
			with h5py.File(os.path.join(path, filename), 'r') as f:
				flag = group.name in f.keys()
			
			if flag:
				if key in links: 
					del links[key]
				if key in fields:
					del fields[key]
				links[key] = h5py.ExternalLink(os.path.join(path, filename), group.name)
				fields[key] = h5py.SoftLink(os.path.join(links.name, key, key))
				fields[key].attrs['t'] = np.round(links[key]['time'][()])
				pbar.set_postfix_str(f'Added {group.name} to {key} library')
			elif key not in links:
				links[key] = h5py.ExternalLink(os.path.join(path, filename), '/ensemble')
				fields[key] = h5py.SoftLink(os.path.join(links.name, key, key))
				fields[key].attrs['t'] = np.round(links[key]['time'][()])
				pbar.set_postfix_str(f'Added ensemble to {key} library')
		
			sub_features = [ sf for sf in links[key][library] if key in links[key][library][sf].attrs]
			feature_names = feature_names.union(sub_features)

			pbar.update()
	feature_names = list(feature_names)
	
	return group, feature_names

def collect_library(data,
					libraries,
					feature_names, 
					extra_functions=[]):
	'''
	Populate library with links to relevant datasets
	Also compute any mixed or coupled features as necessary
	'''
	links = data['links']
	raw = data.require_group('X_raw')
	with tqdm(total=len(feature_names)+len(extra_functions), leave=False) as pbar:
		pbar.set_description('Collecting library features')
		for feature in feature_names:
			pbar.set_postfix(feature=feature)
			if not feature in raw:
				for key, _, group, _ in libraries:
					if feature in links[key][group]:
						path = os.path.join(links.name, key, group, feature)
						raw[feature] = h5py.SoftLink(path)
						raw[feature].attrs['t'] = np.round(links[key]['time'][()])
						break
			pbar.update()

		sleep(0.2)
		
		pbar.set_description('Collecting extra functions')
		for extra_function in extra_functions:
			pbar.set_postfix(function=extra_function)
			extra_function(data)
			pbar.update()

def take_time_derivatives(data,
						  libraries,
						  window_length=5):
	'''
	Compute the dynamics of live-imaged fields
	'''
	X_dot = data.require_group('X_dot')
	
	features = list(data['X_raw'].keys())
	
	with tqdm(total=len(libraries), leave=False) as pbar:
		for key, path, library, decomposition in libraries:
			pbar.set_postfix(key=key)
			if decomposition is None or data['fields'][key].shape[0] != data['fields/v'].shape[0]:
				pbar.set_postfix_str(f'Not computing dynamics for {key}')
				pbar.update()
				continue
			
			pbar.set_description('Time derivative')
			smoother_kws = {'window_length': window_length}
			
			diffT = ps.SmoothedFiniteDifference(d=1, axis=0, smoother_kws=smoother_kws)
			x = data['fields'][key]
			dot = diffT._differentiate(x, x.attrs['t'])[..., None]
			
			ds = X_dot.require_dataset(key, shape=dot.shape, dtype=dot.dtype)
			ds[:] = dot
			ds.attrs.update(smoother_kws)
			ds.attrs['t'] = x.attrs['t']

			pbar.update()

def decompose_library(data, 
					  libraries,
					  re_model=False):
	'''
	Create PCA libraries from the accumulated data
	Each PCA library should be of shape [T, P, L]
		T = time
		P = num PCA components
		L = library size
	
	Each library FOLDER should have an attribute ['t'] which
		indicates the timestamps for each element of that folder
	
	X_pca - the PCA libraries for fields live-imaged with each embryo (dynamic fields)
	X_dot_pca - the PCA libraries of live-imaged dynamics
	
	If the PCA for a given field has already been computed, we don't re-compute it to save time
	'''
	X_raw = data.require_group('X_raw')
	X_cpt = data.require_group('X_cpt')
	X_dot = data.require_group('X_dot')
	X_dot_cpt = data.require_group('X_dot_cpt')
	
	features = list(X_raw.keys())
	
	with tqdm(total=len(libraries), leave=False) as pbar:
		for key, path, library, decomposition in libraries:
			pbar.set_postfix(key=key)
			if decomposition is None or data['fields'][key].shape[0] != data['fields/v'].shape[0]:
				pbar.set_postfix_str(f'Not computing dynamics for {key}')
				pbar.update()
				continue
			
			pbar.set_description('Decomposing library')
			model = get_decomposition_model(data, key, decomposition)

			cpt_key = X_cpt.require_group(key)

			for feature in features:
				if re_model and feature in cpt_key:
					del cpt_key[feature]
				if feature not in cpt_key:
					raw = X_raw[feature]
					if model.can_transform(raw):
						cpt_key[feature] = model.transform(raw, remove_mean=True)
						cpt_key[feature].attrs['t'] = raw.attrs['t']

			dot_cpt = model.transform(X_dot[key], remove_mean=True)[..., None]
			ds = X_dot_cpt.require_dataset(key, shape=dot_cpt.shape, dtype=dot_cpt.dtype)
			ds[:] = dot_cpt
			ds.attrs.update(X_dot[key].attrs)

			pbar.update()

def library_to_tangent_space(data, libraries):
	'''
	Projects libraries onto the tangent space
	'''
	X_raw = data.require_group('X_raw')
	X_tan = data.require_group('X_tan')
	X_dot = data.require_group('X_dot')
	X_dot_tan = data.require_group('X_dot_tan')
	
	model = TangentSpaceTransformer().fit(None)

	features = list(X_raw.keys())
	
	with tqdm(total=len(libraries), leave=False) as pbar:
		for key, path, library, decomposition in libraries:
			pbar.set_postfix(key=key)
			if decomposition is None or data['fields'][key].shape[0] != data['fields/v'].shape[0]:
				pbar.set_postfix_str(f'Not computing dynamics for {key}')
				pbar.update()
				continue
			
			pbar.set_description('Projecting library')

			tan_key = X_tan.require_group(key)

			for feature in features:
				raw = X_raw[feature]
				if len(raw.shape) == 2: #Scalar
					tan_key.create_dataset(feature, data=raw[()])
				else: #Tensor
					tan_key.create_dataset(feature,
										   [raw.shape[0], 2, 2, raw.shape[-1]],
										   dtype=raw.dtype)
					for i in range(raw.shape[0]):
						tan_key[feature][i] = model.inverse_transform(raw[i][()])
				tan_key[feature].attrs['t'] = raw.attrs['t']

			ds = X_dot_tan.create_dataset(key, 
										  [*tan_key[key].shape, 1], 
										  dtype=tan_key[key].dtype)
			for i in range(ds.shape[0]):
				ds[i] = model.inverse_transform(X_dot[key][i][()])[..., None]
			ds.attrs.update(X_dot[key].attrs)

			pbar.update()
