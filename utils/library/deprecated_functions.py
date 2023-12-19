import numpy as np

def symmetric_tensor_powers(data, key='m_ij', max_order=3):
	raw = data['X_raw']
	x = raw[key]
	trx = np.einsum('bij...->b...', x)[:, None, None]
	attrs = dict(x.attrs).copy()
	
	if max_order < 2: 
		return
	
	attrs[key] = 2

	feat = f'{key} Tr({key})'
	raw[feat] = x * trx
	raw[feat].attrs.update(attrs)

	feat = f'{key}^2'
	raw[feat] = np.einsum('bik...,bkj...->bij...', x, x)
	raw[feat].attrs.update(attrs)

	if max_order < 3:
		return

	attrs[key] = 3

	feat = f'{key} Tr({key})^2'
	raw[feat] = x * trx * trx
	raw[feat].attrs.update(attrs)

	#feat = f'{key}^2 Tr({key})'
	#raw[feat] = np.einsum('bik...,bkj...->bij...', x, x) * trx
	#raw[feat].attrs.update(attrs)

	feat = f'{key}^3'
	raw[feat] = np.einsum('bik...,bkl...,blj...->bij...', x, x, x)
	raw[feat].attrs.update(attrs)

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
