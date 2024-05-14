from itertools import product, combinations
import numpy as np
from scipy.interpolate import interp1d

def is_tensor(x):
	if len(x.shape) < 3:
		return False

	return x.shape[1] == x.shape[2]

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

def material_derivative_terms(data, key='m_ij'):
	'''
	Add a material derivative term (v.grad) key 
	For tensor fields, also add the co-rotation [Omega.key - key.Omega]
	'''
	D1_x = data['links'][key][f'D1 {key}']
	v = data['fields']['v']

	raw = data['X_raw']
	x = raw[key]
	O = raw['O']

	feat = f'v dot grad {key}'
	if feat not in raw:
		if is_tensor(x):
			raw[feat] = np.einsum('bk...,bij...k->bij...', v, D1_x)
		else:
			raw[feat] = np.einsum('bk...,b...k->b...', v, D1_x)
	raw[feat].attrs.update({key: 1, 'v': 1, 'space': 1})
	raw[feat].attrs['t'] = x.attrs['t']

	#Co-rotation term
	#SINCE Omega is v_[i,j] and not D_[i v_j] put a minus sign
	if is_tensor(x):
		feat = f'[O, {key}]'
		raw[feat] = -(np.einsum('bik...,bkj...->bij...', O, x) - \
					  np.einsum('bik...,bkj...->bij...', x, O))
		raw[feat].attrs.update({key: 1, 'v': 1, 'space': 1})
		raw[feat].attrs['t'] = x.attrs['t']
		for f in raw: #Remove any Omega terms
			if 'O' in f and not f == feat:
				del raw[f]

'''
For myosin
'''

def symmetric_tensor_couple(data, keys=['m_ij', 'E'], max_space_order=1):
	'''
	Generate the three symmetric tensor couplings of two tensor fields
	A Tr(B), B Tr(A), and {A.B + B.A}
	'''
	raw = data['X_raw']
	for key1, key2 in combinations(keys, 2):
		x1 = raw[key1]
		x2 = raw[key2]

		combined_attrs, x1, x2 = combine_attrs(x1, x2)
		if combined_attrs['space'] > max_space_order:
			continue

		feat = f'{key1} Tr({key2})'
		raw[feat] = np.einsum('bij...,bkk...->bij...', x1, x2)
		raw[feat].attrs.update(combined_attrs)

		feat = f'{key2} Tr({key1})'
		raw[feat] = np.einsum('bij...,bkk...->bij...', x2, x1)
		raw[feat].attrs.update(combined_attrs)

		feat = f'{{{key1}, {key2}}}'
		raw[feat] = np.einsum('bik...,bkj...->bij...', x1, x2) + \
					np.einsum('bik...,bkj...->bij...', x2, x1)
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
			if 'O' in key:
				continue
			if is_tensor(raw[key]):
				tensors.append(key)
	
	for key1, key2 in product(scalars, tensors):
		x1 = raw[key1]
		x2 = raw[key2]

		combined_attrs, x1, x2 = combine_attrs(x1, x2)

		feat = f'{key1} {key2}'
		if feat not in raw:
			raw[feat] = np.einsum('b...,bij...->bij...', x1, x2)
		raw[feat].attrs.update(combined_attrs)

def add_static_sources(data, couple='m_ij'):
	'''
	Add a static DV source to the library
	'''
	raw = data['X_raw']
	if not couple in raw:
		raise ValueError(f'{couple} not in X_raw')
	
	dv = np.zeros_like(raw[couple])
	dv[:, 0, 0, ...] = 1
	if not 'Static_DV' in raw:
		raw['Static_DV'] = dv
	raw['Static_DV'].attrs.update({'space': 0, 't': raw[couple].attrs['t']})

	#Add a coupling to the key
	key = f'Static_DV Tr({couple})'
	raw[key] = np.einsum('bij...,bkk...->bij...', dv, raw[couple])
	raw[key].attrs.update(raw[couple].attrs)

'''
For E-cadherin
'''

def scalar_couple(data, keys=['c', 'Tr(E)', 'Tr(m_ij)'], max_space_order=1):
	'''
	Generate scalar couplings of scalar fields
	'''
	raw = data['X_raw']
	for ii in range(len(keys)):
		for jj in range(ii, len(keys)):
			key1, key2 = keys[ii], keys[jj]
			x1 = raw[key1]
			x2 = raw[key2]

			combined_attrs, x1, x2 = combine_attrs(x1, x2)
			if combined_attrs['space'] > max_space_order:
				continue

			feat = f'{key1} {key2}'
			raw[feat] = x1 * x2
			raw[feat].attrs.update(combined_attrs)


def add_v_squared(data):
	'''
	Include v^2 in the E-cadherin library
	'''
	raw = data['X_raw']
	v = data['fields']['v']
	
	feat = 'v v'
	raw[feat] = np.einsum('bk...,bk...->b...', v, v)
	raw[feat].attrs.update({'v': 2, 't': v.attrs['t'], 'space': 0})

def tensor_trace(data, keys=['m_ij', 'E']):
	'''
	Take the trace of a tensor and turn it into a scalar
	'''
	raw = data['X_raw']
	for key in keys:
		x = raw[key]
		feat = f'Tr({key})'
		raw[feat] = np.einsum('bkk...->b...', x)
		raw[feat].attrs.update(x.attrs)

def delete_high_order_scalars(data, max_space_order=1):
	'''
	Because for historical reasons we computed grad^2(c), we 
	have to remove it from the library
	'''
	raw = data['X_raw']
	for key in raw.keys():
		x = raw[key]
		if x.attrs['space'] > max_space_order:
			del raw[key]
