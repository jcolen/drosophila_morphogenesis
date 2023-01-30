import numpy as np
import os
import sys
basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))

from utils.derivative_library_utils import *

'''
SYMMETRIC Tensor libraries for anisotropy fields
'''

def s2t_terms(x, group, key='Rnt'):
	'''
	Compute terms mapping scalars to symmetric tensors
	'''
	d1_x, d2_x = validate_key_and_derivatives(x, group, key, order=2)
	lib = {}
	attrs = {}
	
	lib['grad(grad(%s))' % key] = np.einsum('tyxij->tijyx', d2_x)
	attrs['grad(grad(%s))' % key] = {key: 1, 'space': 2}
	lib['%s grad(grad(%s))' % (key, key)] = np.einsum('tyx,tyxij->tijyx', x, d2_x)
	attrs['%s grad(grad(%s))' % (key, key)] = {key: 2, 'space': 2}
	lib['grad(%s)grad(%s)' % (key, key)] = np.einsum('tyxi,tyxj->tijyx', d1_x, d1_x)
	attrs['grad(%s)grad(%s)' % (key, key)] = {key: 2, 'space': 2}

	write_library_to_dataset(lib, group.require_group('symmetric_library'), attrs)

def t2t_terms(x, group, key='m'):
	'''
	Compute terms mapping tensors to symmetric tensors
	'''
	d1_x = validate_key_and_derivatives(x, group, key, order=1)[0]
	
	lib = {}
	attrs = {}

	lib[key] = x
	attrs[key] = {key: 1, 'space': 0}

	lib['%s Tr(%s)' % (key, key)] = np.einsum('tkkyx,tijyx->tijyx', x, x)
	attrs['%s Tr(%s)' % (key, key)] = {key: 2, 'space': 0}

	lib['%s %s' % (key, key)] = np.einsum('tikyx,tkjyx->tijyx', x, x)
	attrs['%s %s' % (key, key)] = {key: 2, 'space': 0}
	
	write_library_to_dataset(lib, group.require_group('symmetric_library'), attrs)
	return lib.keys()

def v2t_terms(u, group, key='v'):
	'''
	Compute terms mapping vectors to symmetric tensors
	'''
	d1_u = validate_key_and_derivatives(u, group, 'v', 1)[0]

	lib = {}
	attrs = {}
	Eij = 0.5 * (np.einsum('tiyxj->tijyx', d1_u) + np.einsum('tjyxi->tijyx', d1_u))
	Oij = 0.5 * (np.einsum('tiyxj->tijyx', d1_u) - np.einsum('tjyxi->tijyx', d1_u))
	
	#"Legacy" terms for use in additional computations
	#These should not enter into the physics-informed library though
	lib['vv'] = np.einsum('tiyx,tjyx->tijyx', u, u)
	attrs['vv'] = {'v': 2, 'space': 0}
	lib['O'] = Oij
	attrs['O'] = {'v': 1, 'space': 1}
	
	write_library_to_dataset(lib, group.require_group('symmetric_library'), attrs)

	#Since E is a symmetric tensor, just pass it forward
	keys = t2t_terms(Eij, group, key='E')
	for k in keys:
		glib = group['symmetric_library'][k]
		glib.attrs['v'] = glib.attrs['E']
		glib.attrs['space'] = glib.attrs['E'] + glib.attrs['space']

def veltensor2symtensor_library(folder, embryoID, group, key='m',
							 project=True, t_threshold=0.95, v_threshold=0.9):
	'''
	Build a symmetric tensor library of velocities and tensors
	'''
	if project:
		try:
			T = project_embryo_data(folder, embryoID, 'tensor', t_threshold)
			V = project_embryo_data(folder, embryoID, 'velocity', v_threshold)
		except Exception as e:
			print(e)
			return
	else:
		T = np.load(os.path.join(folder, embryoID, 'tensor2D.npy'), mmap_mode='r')
		V = np.load(os.path.join(folder, embryoID, 'velocity2D.npy'), mmap_mode='r')

	T = T.reshape([T.shape[0], 2, 2, *T.shape[-2:]])

	#We hold off on mixing terms since we're going to ultimately select a flow field to couple with after the fact
	t2t_terms(T, group, key=key)
	v2t_terms(V, group)

def scalar2symtensor_library(folder, embryoID, group, key='Rnt', 
						  project=True, s_threshold=0.95):
	'''
	Build a symmetric tensor library of just scalar information
	'''
	if project:
		S = project_embryo_data(folder, embryoID, 'raw', s_threshold)
	else:
		S = np.load(os.path.join(folder, embryoID, 'raw2D.npy'), mmap_mode='r')
	
	s2t_terms(S, group, key=key)
