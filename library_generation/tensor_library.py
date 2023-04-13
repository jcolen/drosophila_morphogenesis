import numpy as np
import os
from ..utils.library.derivative_library_utils import validate_key_and_derivatives
from ..utils.library.derivative_library_utils import write_library_to_dataset
from ..utils.library.derivative_library_utils import project_embryo_data

def t2t_terms(x, group, YY, XX, key='m'):
	'''
	Compute terms mapping tensors to symmetric tensors
	'''
	d1_x = validate_key_and_derivatives(x, group, YY, XX, key, order=1)[0]
	
	lib = {}
	attrs = {}

	lib[key] = x
	attrs[key] = {key: 1, 'space': 0}

	lib['%s Tr(%s)' % (key, key)] = np.einsum('tkkyx,tijyx->tijyx', x, x)
	attrs['%s Tr(%s)' % (key, key)] = {key: 2, 'space': 0}

	lib['%s %s' % (key, key)] = np.einsum('tikyx,tkjyx->tijyx', x, x)
	attrs['%s %s' % (key, key)] = {key: 2, 'space': 0}
	
	write_library_to_dataset(lib, group.require_group('tensor_library'), attrs)
	return lib.keys()

def build_tensor_library(folder, embryoID, group, key='m',base='tensor',
							 project=True, threshold=0.95):
	'''
	Build a symmetric tensor library of velocities and tensors
	'''
	if project:
		try:
			T = project_embryo_data(folder, embryoID, base, threshold)
		except ValueError as e:
			print('Skipping ', embryoID, e)
			return
	else:
		embryoID = str(embryoID)
		T = np.load(os.path.join(folder, embryoID, base+'2D.npy'), mmap_mode='r')

	embryoID = str(embryoID)
	T = T.reshape([T.shape[0], 2, 2, *T.shape[-2:]])
	dv_coordinates = np.load(os.path.join(folder, embryoID, 'DV_coordinates.npy'), mmap_mode='r')
	ap_coordinates = np.load(os.path.join(folder, embryoID, 'AP_coordinates.npy'), mmap_mode='r')
	t2t_terms(T, group, dv_coordinates, ap_coordinates, key=key)
