import numpy as np
import h5py
import os

import sys

basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))
import warnings
from utils.translation_utils import *
from utils.derivative_library_utils import *

import pandas as pd
import pickle as pk
from tqdm import tqdm

import pysindy as ps
from scipy.io import loadmat

'''
Scalar libraries for intensity fields
'''

def s2s_terms(x, group, key='Rnt'):
	d1_x, d2_x = validate_key_and_derivatives(x, group, key, order=2)
	lib = {}
	attrs = {}

	feat = key
	lib[feat] = x
	attrs[feat] = {key: 1, 'space': 0}

	feat = '%s^2' % key
	lib[feat] = x**2
	attrs[feat] = {key: 2, 'space': 0}
	
	feat = 'grad(%s)^2' % key
	lib[feat] = np.einsum('tyxi,tyxi->tyx', d1_x, d1_x)
	attrs[feat] = {key: 2, 'space': 2}

	feat = 'grad^2 %s' % key
	lib[feat] = np.einsum('tyxii->tyx', d2_x)
	attrs[feat] = {key: 1, 'space': 2}
	
	feat = '%s grad^2 %s' % (key, key)
	lib[feat] = np.einsum('tyx,tyxii->tyx', x, d2_x)
	attrs[feat] = {key: 2, 'space': 2}

	write_library_to_dataset(lib, group.require_group('scalar_library'), attrs)

def t2s_terms(x, group, key='m'):
	d1_x, d2_x = validate_key_and_derivatives(x, group, key, order=2)
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

def v2s_terms(x, group, key='v'):
	'''
	Compute terms mapping vectors to symmetric tensors
	'''
	d1_x = validate_key_and_derivatives(x, group, key, 1)[0]

	lib = {}
	attrs = {}
	
	feat = '%s^2' % key
	lib[feat] = np.einsum('tiyx,tiyx->tyx', x, x)
	attrs[feat] = {key: 2, 'space': 0}

	feat = 'div %s' % key
	lib[feat] = np.einsum('tiyxi->tyx', d1_x)
	attrs[feat] = {key: 1, 'space': 1}

	feat = '(div %s)^2' % key
	lib[feat] = np.einsum('tiyxi,tjyxj->tyx', d1_x, d1_x)
	attrs[feat] = {key: 2, 'space': 2}

	feat = '(grad %s)^2' % key
	lib[feat] = np.einsum('tiyxj,tiyxj->tyx', d1_x, d1_x)
	attrs[feat] = {key: 2, 'space': 2}

	write_library_to_dataset(lib, group.require_group('scalar_library'), attrs)

def veltensor2scalar_library(folder, embryoID, group, key='m',
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

	t2s_terms(T, group, key=key)
	v2s_terms(V, group)

def scalar2scalar_library(folder, embryoID, group, key='Rnt', 
						  project=True, s_threshold=0.95):
	'''
	Build a library of just scalar information
	'''
	if project:
		S = project_embryo_data(folder, embryoID, 'raw', s_threshold)
	else:
		S = np.load(os.path.join(folder, embryoID, 'raw2D.npy'), mmap_mode='r')
	
	s2s_terms(S, group, key=key)
