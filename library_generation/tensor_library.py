import numpy as np
import os
import sys
basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))

from utils.derivative_library_utils import *


'''
Tensor libraries for anisotropy fields
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

	write_library_to_dataset(lib, group.require_group('tensor_library'), attrs)


def t2t_terms(x, group, key='m'):
	'''
	Compute terms mapping tensors to symmetric tensors
	'''
	d1_x, d2_x = validate_key_and_derivatives(x, group, key, order=2)
	
	lib = {}
	attrs = {}

	lib[key] = x
	attrs[key] = {key: 1, 'space': 0}

	lib['%s Tr(%s)' % (key, key)] = np.einsum('tkkyx,tijyx->tijyx', x, x)
	attrs['%s Tr(%s)' % (key, key)] = {key: 2, 'space': 0}
	
	lib['grad^2 %s' % key] = np.einsum('tijyxkk->tijyx', d2_x) # 2. d_{kk} x_{ij}
	attrs['grad^2 %s' % key] = {key: 1, 'space': 2}
	lib['grad(div(%s))' % key] = np.einsum('tikyxkj->tijyx', d2_x) # 3. d_{jk} x_{ik}
	attrs['grad(div(%s))' % key] = {key: 1, 'space': 2}
	lib['grad(grad(Tr(%s))' % key] = np.einsum('tkkyxij->tijyx', d2_x) # 4. d_{ij} x_{kk}
	attrs['grad(grad(Tr(%s))' % key] = {key: 1, 'space': 2}
	lib['Tr(%s) grad^2 %s' % (key, key)] = np.einsum('tijyxkk,tllyx->tijyx', d2_x, x) # 5. x_{ll} d_{kk} x_{ij}
	attrs['Tr(%s) grad^2 %s' % (key, key)] = {key: 2, 'space': 2}
	lib['Tr(%s) grad(div(%s))' % (key, key)] = np.einsum('tikyxkj,tllyx->tijyx', d2_x, x) # 6. x_{ll} d_{jk} x_{ik}
	attrs['Tr(%s) grad(div(%s))' % (key, key)] = {key: 2, 'space': 2}
	lib['Tr(%s) grad(grad(Tr(%s)))' % (key, key)] = np.einsum('tijyxkk,tllyx->tijyx', d2_x, x) # 7. x_{ll} d_{ij} x_{kk}
	attrs['Tr(%s) grad(grad(Tr(%s)))' % (key, key)] = {key: 2, 'space': 2}
	
	write_library_to_dataset(lib, group.require_group('tensor_library'), attrs)

def v2t_terms(u, group):
	'''
	Compute terms mapping vectors to symmetric tensors
	'''
	d1_u = validate_key_and_derivatives(u, group, 'v', 1)[0]

	lib = {}
	attrs = {}
	Eij = 0.5 * (np.einsum('tiyxj->tijyx', d1_u) + np.einsum('tjyxi->tijyx', d1_u))
	Oij = 0.5 * (np.einsum('tiyxj->tijyx', d1_u) - np.einsum('tjyxi->tijyx', d1_u))
	
	lib['vv'] = np.einsum('tiyx,tjyx->tijyx', u, u)
	attrs['vv'] = {'v': 2, 'space': 0}
	lib['E'] = Eij
	attrs['E'] = {'v': 1, 'space': 1}
	lib['O'] = Oij
	attrs['O'] = {'v': 1, 'space': 1}
	
	write_library_to_dataset(lib, group.require_group('tensor_library'), attrs)

def vt2t_terms(group, key='m'):
	'''
	Compute terms mapping vectors and tensors to symmetric tensors
	'''
	glib = group.require_group('tensor_library')
	x = group[key]
	u = group['v']
	
	lib = {}
	attrs = {}

	lib['vv Tr(%s)' % key] = np.einsum('tkkyx,tijyx->tijyx', x, glib['vv'])
	attrs['vv Tr(%s)' % key] = {key: 1, 'v': 2, 'space': 0}
	lib['v^2 %s' % key] = np.einsum('tkkyx,tijyx->tijyx', glib['vv'], x)
	attrs['v^2 %s' % key] = {key: 1, 'v': 2, 'space': 0}
	
	vvx = np.einsum('tikyx,tkjyx->tijyx', glib['vv'], x)
	xvv = np.einsum('tikyx,tkjyx->tijyx', x, glib['vv'])
	lib['(vv%s + %svv)' % (key, key)] = vvx + xvv
	attrs['(vv%s + %svv)' % (key, key)] = {key: 1, 'v': 2, 'space': 0}
	
	lib['Tr(%s) E' % key] = np.einsum('tkkyx,tijyx->tijyx', x, glib['E'])
	attrs['Tr(%s) E' % key] = {key: 1, 'v': 1, 'space': 1}
	
	xE = np.einsum('tikyx,tkjyx->tijyx', x, glib['E'])
	Ex = np.einsum('tikyx,tkjyx->tijyx', glib['E'], x)
	lib['(%sE + E%s)' % (key, key)] = xE + Ex
	attrs['(%sE + E%s)' % (key, key)] = {key: 1, 'v': 1, 'space': 1}
	
	xO = np.einsum('tikyx,tkjyx->tijyx', x, glib['O'])
	Ox = np.einsum('tikyx,tkjyx->tijyx', glib['O'], x)
	lib['(%sO - O%s)' % (key, key)] = xO - Ox
	attrs['(%sO - O%s)' % (key, key)] = {key: 1, 'v': 1, 'space': 1}
	
	
	lib['%s div v' % key] = np.einsum('tijyx,tkyxk->tijyx', x, group['D1 v'])
	attrs['%s div v' % key] = {key: 1, 'v': 1, 'space': 1}

	lib['v dot grad %s' % key] = np.einsum('tkyx,tijyxk->tijyx', u, group['D1 %s' % key])
	attrs['v dot grad %s' % key] = {key: 1, 'v': 1, 'space': 1}
	
	lib['%s (%s dot grad v)' % (key, key)] = np.einsum('tijyx,tklyx,tkyxl->tijyx', x, x, group['D1 v'])
	attrs['%s (%s dot grad v)' % (key, key)] = {key: 2, 'v': 1, 'space': 1}
	
	write_library_to_dataset(lib, group.require_group('tensor_library'), attrs)
	
def veltensor2tensor_library(folder, embryoID, group, key='m',
							 project=True, t_threshold=0.95, v_threshold=0.9):
	'''
	Build a mixed library of velocities and tensors
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

	t2t_terms(T, group, key=key)
	v2t_terms(V, group)
	vt2t_terms(group, key=key)

def scalar2tensor_library(folder, embryoID, group, key='Rnt', 
						  project=True, s_threshold=0.95):
	'''
	Build a library of just scalar information
	'''
	if project:
		S = project_embryo_data(folder, embryoID, 'raw', s_threshold)
	else:
		S = np.load(os.path.join(folder, embryoID, 'raw2D.npy'), mmap_mode='r')
	
	s2t_terms(S, group, key=key)
