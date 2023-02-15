import numpy as np
import os
import sys
basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))

from utils.derivative_library_utils import *

'''
SYMMETRIC Tensor libraries for anisotropy fields
'''
def t2s_terms(x, group, YY, XX, key='m'):
	d1_x, d2_x = validate_key_and_derivatives(x, group, YY, XX, key, order=2)
	lib = {}
	attrs = {}
	
	feat = 'Tr(%s)' % key
	lib[feat] = np.einsum('tiiyx->tyx', x)
	attrs[feat] = {key: 1, 'space': 0}
	
	feat = 'Tr(%s)^2' % key
	lib[feat] = np.einsum('tiiyx->tyx', x)**2
	attrs[feat] = {key: 2, 'space': 0}
	
	feat = 'grad(grad(%s))' % key
	lib[feat] = np.einsum('tijyxij->tyx', d2_x)
	attrs[feat] = {key: 1, 'space': 2}
	
	feat = 'grad^2 Tr(%s)' % key
	lib[feat] = np.einsum('tiiyxjj->tyx', d2_x)
	attrs[feat] = {key: 1, 'space': 2}
	
	feat = 'div(%s) div(%s)' % (key, key)
	lib[feat] = np.einsum('tijyxj,tikyxk->tyx', d1_x, d1_x)
	attrs[feat] = {key: 2, 'space': 2}
	
	feat = 'Tr(%s) grad^2 Tr(%s)' % (key, key)
	lib[feat] = np.einsum('tiiyx,tjjyxkk->tyx', x, d2_x)
	attrs[feat] = {key: 2, 'space': 2}
	
	feat = 'grad(Tr(%s))^2' % key
	lib[feat] = np.einsum('tiiyxj,tkkyxj->tyx', d1_x, d1_x)
	attrs[feat] = {key: 2, 'space': 2}
	
	write_library_to_dataset(lib, group.require_group('scalar_library'), attrs)

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
