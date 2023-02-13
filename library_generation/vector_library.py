import numpy as np
import os
import sys
basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))

from tensor_library import t2t_terms
from utils.derivative_library_utils import *


'''
Generate different library terms from vector fields
'''
def v2t_terms(u, group, key='v'):
	'''
	The velocity derivative terms we care about are VORTICITY and STRAIN RATE
	We will decompose them further and couple them with tensors later in the pipeline
	'''
	d1_u = validate_key_and_derivatives(u, group, 'v', 1)[0]

	lib = {}
	attrs = {}
	Eij = 0.5 * (np.einsum('tiyxj->tijyx', d1_u) + np.einsum('tjyxi->tijyx', d1_u))
	Oij = 0.5 * (np.einsum('tiyxj->tijyx', d1_u) - np.einsum('tjyxi->tijyx', d1_u))
	
	#"Legacy" terms for use in additional computations
	#These should not enter into the physics-informed library though
	#lib['vv'] = np.einsum('tiyx,tjyx->tijyx', u, u)
	#attrs['vv'] = {'v': 2, 'space': 0}
	lib['O'] = Oij
	attrs['O'] = {'v': 1, 'space': 1}

	lib['E'] = Eij
	attrs['E'] = {'v': 1, 'space': 1}
	
	write_library_to_dataset(lib, group.require_group('tensor_library'), attrs)

	#Since E is a symmetric tensor, just pass it forward
	'''
	keys = t2t_terms(Eij, group, key='E')
	for k in keys:
		glib = group['tensor_library'][k]
		glib.attrs['v'] = glib.attrs['E']
		glib.attrs['space'] = glib.attrs['E'] + glib.attrs['space']
	'''

def build_vector_library(folder, embryoID, group, key='v', base='velocity',
							project=True, threshold=0.95):
	'''
	Build a symmetric tensor library of velocities and tensors
	'''
	if project:
		try:
			V = project_embryo_data(folder, embryoID, base, threshold)
		except ValueError as e:
			print('Skipping ', embryoID, e)
			return
	else:
		V = np.load(os.path.join(folder, embryoID, base+'2D.npy'), mmap_mode='r')

	v2t_terms(V, group)
