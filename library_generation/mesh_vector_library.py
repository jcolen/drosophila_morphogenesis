import numpy as np
import os
from tqdm.auto import tqdm

import sys
sys.path.insert(0, '/project/vitelli/jonathan/REDO_fruitfly/src')

from utils.geometry.geometry_utils import MeshInterpolator, TangentSpaceTransformer

def build_vector_library(folder, group, mesh_grad, key='v', base='velocity'):
	'''
	Build a symmetric tensor library of vector fields
	'''
	mesh_int = MeshInterpolator().fit(None)
	tangent  = TangentSpaceTransformer().fit(None)

	print(f'Loading {base}2D.npy')
	embryoID = os.path.basename(group.name)
	V = np.load(os.path.join(folder, embryoID, base+'2D.npy'), mmap_mode='r').squeeze()

	x    = group.create_dataset(key, shape=[V.shape[0], 3, mesh_int.n_vertices_], dtype='float64')
	d1_x = group.create_dataset(f'D1 {key}', shape=[*x.shape, 3], dtype='float64')

	for i in tqdm(range(V.shape[0])):
		xi = mesh_int.transform(V[i]) #2, V
		x[i] = tangent.transform(xi) #3, V
		d1_x[i] = mesh_grad.transform(xi) #3, V, 3

	Eij = 0.5 * (np.einsum('tivj->tijv', d1_x) + np.einsum('tjvi->tijv', d1_x))
	Oij = 0.5 * (np.einsum('tivj->tijv', d1_x) - np.einsum('tjvi->tijv', d1_x))
	
	lib = group.require_group('tensor_library')
	
	lib.create_dataset('O', data=Oij)
	lib['O'].attrs.update({'v': 1, 'space': 1})

	lib.create_dataset('E', data=Eij)
	lib['E'].attrs.update({'v': 1, 'space': 1})
