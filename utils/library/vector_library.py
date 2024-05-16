import numpy as np
import os
from utils.library.derivative_library_utils import validate_key_and_derivatives
from utils.library.derivative_library_utils import write_library_to_dataset
from utils.library.derivative_library_utils import project_embryo_data

def v2t_terms(u, group, YY, XX, key='v'):
	'''
	The velocity derivative terms we care about are VORTICITY and STRAIN RATE
	We will decompose them further and couple them with tensors later in the pipeline
	'''
	d1_u = validate_key_and_derivatives(u, group, YY, XX, key, 1)[0]

	lib = {}
	attrs = {}
	Eij = 0.5 * (np.einsum('tiyxj->tijyx', d1_u) + np.einsum('tjyxi->tijyx', d1_u))
	Oij = 0.5 * (np.einsum('tiyxj->tijyx', d1_u) - np.einsum('tjyxi->tijyx', d1_u))
	
	lib['O'] = Oij
	attrs['O'] = {'v': 1, 'space': 1}

	lib['E'] = Eij
	attrs['E'] = {'v': 1, 'space': 1}
	
	write_library_to_dataset(lib, group.require_group('tensor_library'), attrs)

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
		embryoID = str(embryoID)
		V = np.load(os.path.join(folder, embryoID, base+'2D.npy'), mmap_mode='r')
	
	embryoID = str(embryoID)
	dv_coordinates = np.load(os.path.join(folder, embryoID, 'DV_coordinates.npy'), mmap_mode='r')
	ap_coordinates = np.load(os.path.join(folder, embryoID, 'AP_coordinates.npy'), mmap_mode='r')
	v2t_terms(V, group, dv_coordinates, ap_coordinates)
