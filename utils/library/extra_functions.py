from itertools import product, combinations
import numpy as np
from scipy.interpolate import interp1d
from ..geometry.geometry_utils import TangentSpaceTransformer

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

def symmetric_tensor_powers(data, key='m_ij', max_order=3):
	raw = data['X_raw']
	x = raw[key]
	trx = np.einsum('bij...->b...', x)[:, None, None]
	attrs = dict(x.attrs).copy()
	
	if max_order < 2: 
		return
	
	attrs[key] = 2

	#feat = f'{key} Tr({key})'
	#raw[feat] = x * trx
	#raw[feat].attrs.update(attrs)

	feat = f'{key}^2'
	raw[feat] = np.einsum('bik...,bkj...->bij...', x, x)
	raw[feat].attrs.update(attrs)

	if max_order < 3:
		return

	attrs[key] = 3

	#feat = f'{key} Tr({key})^2'
	#raw[feat] = x * trx * trx
	#raw[feat].attrs.update(attrs)

	#feat = f'{key}^2 Tr({key})'
	#raw[feat] = np.einsum('bik...,bkj...->bij...', x, x) * trx
	#raw[feat].attrs.update(attrs)

	feat = f'{key}^3'
	raw[feat] = np.einsum('bik...,bkl...,blj...->bij...', x, x, x)
	raw[feat].attrs.update(attrs)

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
		
def active_strain_decomposition(data, key='m_ij'):
	raw = data['X_raw']
	E = raw['E']
	x = raw[key]
	
	deviatoric = x - 0.5 * np.einsum('bkk...,ij->bij...', x, np.eye(2))
	deviatoric[deviatoric == 0] = 1e-5
	
	x_0 = np.linalg.norm(x, axis=(1, 2), keepdims=True).mean(axis=(3, 4), keepdims=True)
	dev_mag = np.linalg.norm(deviatoric, axis=(1, 2), keepdims=True)
		
	devE = np.einsum('bkl...,bkl...->b...', deviatoric, E)[:, None, None]
	
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

def active_strain_decomposition_mesh(data, key='m_ij'):
	raw = data['X_raw']
	E = raw['E']
	x = raw[key]

	deviatoric = x - 0.5 * np.einsum('bkk...,ij->bij...', x, np.eye(3))
	deviatoric[deviatoric == 0] = 1e-5
	
	x_0 = np.linalg.norm(x, axis=(1, 2), keepdims=True)
	x_0 = np.mean(x_0, axis=-1, keepdims=True)
	dev_mag = np.linalg.norm(deviatoric, axis=(1, 2), keepdims=True)
		
	devE = np.einsum('bkl...,bkl...->b...', deviatoric, E)[:, None, None]
	
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
	Add a static DV and AP source to the library
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

def add_static_sources_mesh(data, couple='m_ij'):
	'''
	Add a static DV and AP
	'''
	raw = data['X_raw']
	if not couple in raw:
		raise ValueError(f'{couple} not in X_raw')

	dv = np.zeros([2, 2, raw[couple].shape[-1]])
	dv[0, 0] = 1
	dv = TangentSpaceTransformer().fit_transform(dv)

	#raw['Static_DV'] = np.zeros_like(raw[couple])
	#raw['Static_DV'][:] = dv
	#raw['Static_DV'].attrs.update({'space': 0, 't': raw[couple].attrs['t']})

	#Add a coupling to the key
	key = f'Static_DV Tr({couple})'
	raw[key] = np.einsum('ij...,bkk...->bij...', dv, raw[couple])
	raw[key].attrs.update(raw[couple].attrs)

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
