import os
from time import sleep

import h5py
import numpy as np
import pysindy as ps
from tqdm.auto import tqdm

from ..decomposition.decomposition_utils import get_decomposition_model

def fill_group_info(h5f, embryo, libraries):
	'''
	Fill an embryo group with links, fields, and determine the feature names
	'''
	embryoID = embryo[:-3]
	group = h5f.require_group(embryoID)
	links = group.require_group('links')
	fields = group.require_group('fields')
	X = group.require_group('X')

	feature_names = set()
	with tqdm(total=len(libraries), leave=False) as pbar:
		pbar.set_description('Collecting links and metadata')
		for info in libraries:
			key, path, library = info

			if key in links:
				del links[key]
				del fields[key]

			# Is this embryo live-imaged in this library?
			if os.path.exists(f'{path}/{embryo}'):
				filename = f'{path}/{embryo}'
				pbar.set_postfix_str(f'Adding {embryo} to {key} library')
			else:
				filename = f'{path}/ensemble.h5'
				pbar.set_postfix_str(f'Adding ensemble to {key} library')
			
			with h5py.File(filename, 'r') as source:
				links[key] = h5py.ExternalLink(filename, '/')
				fields[key] = source[key][()]
				fields[key].attrs['t'] = np.round(source['time'][()])
		
			sub_features = [ sf for sf in links[key][library] if key in links[key][library][sf].attrs]
			feature_names = feature_names.union(sub_features)

			pbar.update()
	feature_names = list(feature_names)
	
	return group, feature_names

def collect_library(group,
					libraries,
					feature_names, 
					extra_functions=[]):
	'''
	Populate library with links to relevant datasets
	Also compute any mixed or coupled features as necessary
	'''
	links = group['links']
	features = group.require_group('features')
	with tqdm(total=len(feature_names)+len(extra_functions), leave=False) as pbar:
		pbar.set_description('Collecting library features')
		for feature in feature_names:
			pbar.set_postfix(feature=feature)
			if not feature in features:
				for key, _, library in libraries:
					if feature in links[key][library]:
						path = os.path.join(links.name, key, library, feature)
						features[feature] = h5py.SoftLink(path)
						features[feature].attrs['t'] = np.round(links[key]['time'][()])
						break
			pbar.update()

		sleep(0.2)
		
		pbar.set_description('Collecting extra functions')
		for extra_function in extra_functions:
			pbar.set_postfix(function=extra_function)
			extra_function(group)
			pbar.update()

def take_time_derivatives(data,
						  key='m_ij',
						  window_length=5):
	'''
	Compute the dynamics of live-imaged fields
	'''
	X_dot = data.require_group('X_dot')

	if window_length > 4:
		smoother_kws = {'window_length': window_length}
		diffT = ps.SmoothedFiniteDifference(d=1, axis=0, smoother_kws=smoother_kws)
	else:
		diffT = ps.FiniteDifference(d=1, axis=0)
	x = data['fields'][key]

	dot = diffT._differentiate(x[()], x.attrs['t'])[..., None]
	
	ds = X_dot.require_dataset(key, shape=dot.shape, dtype=dot.dtype)
	ds[:] = dot
	if isinstance(diffT, ps.SmoothedFiniteDifference):
		ds.attrs.update(smoother_kws)
	ds.attrs['t'] = x.attrs['t']

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
