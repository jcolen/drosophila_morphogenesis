from itertools import product, combinations
import numpy as np
from scipy.interpolate import interp1d

def combine_attrs(x1, x2):
	'''
	Combine the attributes of two fields
	Also return the time-aligned intersection of the two
	'''
	combined_attrs = {}
	for key in x1.attrs:
		if key == 't': 
			continue
		combined_attrs[key] = x1.attrs[key]
	for key in x2.attrs:
		if key == 't': 
			continue
		if key in combined_attrs:
			combined_attrs[key] += x2.attrs[key]
		else:
			combined_attrs[key] = x2.attrs[key]

	#Figure out time alignment
	t1, t2 = x1.attrs['t'], x2.attrs['t']
	tmin = max(np.min(t1), np.min(t2))
	tmax = min(np.max(t1), np.max(t2))

	t1_mask = np.logical_and(t1 >= tmin, t1 <= tmax)
	t2_mask = np.logical_and(t2 >= tmin, t2 <= tmax)
	combined_attrs['t'] = x1.attrs['t'][t1_mask]

	return combined_attrs, x1[t1_mask, ...], x2[t2_mask, ...]

def symmetric_tensor_couple(data, keys=['m_ij', 'E']):
	'''
	Generate the three symmetric tensor couplings of two tensor fields
	A Tr(B), B Tr(A), and {A.B + B.A}
	'''
	raw = data['X_raw']
	for key1, key2 in combinations(keys, 2):
		x1 = raw[key1]
		x2 = raw[key2]

		combined_attrs, x1, x2 = combine_attrs(x1, x2)

		feat = f'{key1} Tr({key2})'
		if feat not in raw:
			raw[feat] = np.einsum('bijyx,bkkyx->bijyx', x1, x2)

		raw[feat].attrs.update(combined_attrs)

		feat = f'{key2} Tr({key1})'
		if feat not in raw:
			raw[feat] = np.einsum('bijyx,bkkyx->bijyx', x2, x1)
		raw[feat].attrs.update(combined_attrs)

		feat = f'{{{key1}, {key2}}}'
		if feat not in raw:
			y = np.einsum('bikyx,bkjyx->bijyx', x1, x2) + np.einsum('bikyx,bkjyx->bijyx', x2, x1)
			raw[feat] = y
		raw[feat].attrs.update(combined_attrs)

def scalar_couple(data, scalars):
	'''
	Generate all order 2 couplings of scalar fields
	'''
	raw = data['X_raw']
	for key1, key2 in combinations(scalars, 2):
		x1 = raw[key1]
		x2 = raw[key2]

		combined_attrs, x1, x2 = combine_attrs(x1, x2)

		feat = f'{key1} {key2}'
		if feat not in raw:
			raw[feat] = x1 * x2
		raw[feat].attrs.update(combined_attrs)

	
def remove_terms(data, max_space_order=1):
	raw = data['X_raw']
	for key in raw:
		if raw[key].attrs['space'] > max_space_order:
			del raw[key]
			
	if 'O' in raw:
		del raw['O']
		
	if 'vv' in raw:
		del raw['vv']
		
def active_strain_decomposition(data, key='m_ij'):
	raw = data['X_raw']
	E = raw['E']
	x = raw[key]
	
	deviatoric = x - 0.5 * np.einsum('bkkyx,ij->bijyx', x, np.eye(2))
	deviatoric[deviatoric == 0] = 1e-5
	
	x_0 = np.linalg.norm(x, axis=(1, 2), keepdims=True).mean(axis=(3, 4), keepdims=True)
	dev_mag = np.linalg.norm(deviatoric, axis=(1, 2), keepdims=True)
		
	devE = np.einsum('bklyx,bklyx->byx', deviatoric, E)[:, None, None]
	
	E_active = E - np.sign(devE) * devE * deviatoric / dev_mag**2
	E_active = 0.5 * E_active * dev_mag / x_0
	
	E_passive = E - E_active

	attrs = dict(E.attrs)
	
	#Remove regular strain terms from library
	raw['E_full'] = E[()]
	raw['E_full'].attrs.update(attrs)
	del raw['E']
	
	raw['E_active'] = E_active
	raw['E_active'].attrs.update(attrs)
	
	raw['E_passive'] = E_passive
	raw['E_passive'].attrs.update(attrs)

def tensor_to_scalar(data, tensors):
	'''
	Convert tensors to scalars by taking the trace (order 1)
	'''
	raw = data['X_raw']
	for key in tensors:
		x = raw[key]
		feat = f'Tr({key})'
		if not feat in raw:
			raw[feat] = np.einsum('bkkyx->byx', x)
		raw[feat].attrs.update(x.attrs)

def scalar_tensor_couple(data, scalars, tensors):
	'''
	Order-1 couplings between a scalar and tensors
	Without adding gradients, the only thing we can really do is multiply scalar by the trace
	'''
	raw = data['X_raw']
	for key1, key2 in product(scalars, tensors):
		x1 = raw[key1]
		x2 = raw[key2]

		combined_attrs, x1, x2 = combine_attrs(x1, x2)

		feat = f'{key1} Tr({key2})'
		if feat not in raw:
			raw[feat] = np.einsum('byx,bkkyx->byx', x1, x2)
		raw[feat].attrs.update(combined_attrs)

def multiply_tensor_by_scalar(data, tensors, scalars):
	'''
	Order-1 couplings between scalars and tensors
	Without adding gradients, all we can do is multiply the two
	'''
	raw = data['X_raw']
	
	if tensors is None:
		tensors = []
		for key in raw:
			if len(raw[key].shape) == 5 and not 'O' in key:
				tensors.append(key)
	
	for key1, key2 in product(scalars, tensors):
		x1 = raw[key1]
		x2 = raw[key2]

		combined_attrs, x1, x2 = combine_attrs(x1, x2)

		feat = f'{key1} {key2}'
		if feat not in raw:
			raw[feat] = np.einsum('byx,bijyx->bijyx', x1, x2)
		raw[feat].attrs.update(combined_attrs)

	
def add_dorsal_source(data, couple='c', key='RECTANGLE',
					   x_path='Public/Masks/dorsal_mask_',
					   t_path='Public/Masks/dorsal_mask_time.npy'):
	'''
	Add the time-aligned and advected dorsal source to this dataset
	'''
	raw = data['X_raw']
	if not couple in raw:
		raise ValueError(f'{couple} not in X_raw')
	
	x = np.load(f'{x_path}{key}_advected_RESIZED.npy', mmap_mode='r')
	t0 = np.load(t_path, mmap_mode='r')
	
	t = raw[couple].attrs['t']
	
	x = interp1d(t0, x, axis=0, fill_value='extrapolate', bounds_error=False)(t)
	raw['Dorsal_Source'] = x
	raw['Dorsal_Source'].attrs.update({'space': 0})
	raw['Dorsal_Source'].attrs.update({'t': t})

def add_static_sources(data, couple='m_ij'):
	'''
	Add a static DV and AP source to the library
	'''
	raw = data['X_raw']
	if not couple in raw:
		raise ValueError(f'{couple} not in X_raw')
	
	x = np.zeros_like(raw[couple])
	x[:, 0, 0, :, :] = 1
	if not 'Static_DV' in raw:
		raw['Static_DV'] = x
	raw['Static_DV'].attrs.update({'space': 0, 't': raw[couple].attrs['t']})
	symmetric_tensor_couple(data, [couple, 'Static_DV'])
	del raw[f'{couple} Tr(Static_DV)']
	del raw[f'{{{couple}, Static_DV}}']

def add_constant_source(data, couple='c'):
	raw = data['X_raw']
	if not couple in raw:
		raise ValueError(f'{couple} not in X_raw')

	x = np.ones_like(raw[couple])
	if not '1' in raw:
		raw['1'] = x
	raw['1'].attrs.update({'space': 0, 't': raw[couple].attrs['t']})

def material_derivative_terms(data, keys=['c']):
	'''
	Add a material derivative term (v.grad) key 
	For tensor fields, also add the co-rotation [Omega.key - key.Omega]
	'''
	for key in keys:
		D1_x = data['links'][key][f'D1 {key}']
		v = data['fields']['v']

		#Implies that x is an ensemble-averaged field i.e. not time-aligned
		if v.shape[0] != D1_x.shape[0]:
			continue

		raw = data['X_raw']
		x = raw[key]
		O = raw['O']

		feat = f'v dot grad {key}'
		if feat not in raw:
			if len(x.shape) == 5:
				raw[feat] = np.einsum('bkyx,bijyxk->bijyx', v, D1_x)
			else:
				raw[feat] = np.einsum('bkyx,byxk->byx', v, D1_x)
		raw[feat].attrs.update({key: 1, 'v': 1, 'space': 1})
		raw[feat].attrs['t'] = x.attrs['t']

		#Co-rotation term
		#SINCE Omega is v_[i,j] and not D_[i v_j] put a minus sign
		if len(x.shape) == 5:
			feat = f'[O, {key}]'
			if feat not in raw:
				raw[feat] = -(np.einsum('bikyx,bkjyx->bijyx', O, x) - np.einsum('bikyx,bkjyx->bijyx', x, O))
			raw[feat].attrs.update({key: 1, 'v': 1, 'space': 1})
			raw[feat].attrs['t'] = x.attrs['t']
