import numpy as np
import os
from tqdm.auto import tqdm

import sys
sys.path.insert(0, '/project/vitelli/jonathan/REDO_fruitfly/src')

from utils.geometry.geometry_utils import MeshInterpolator, TangentSpaceTransformer

def build_tensor_library(folder, group, mesh_grad, key='m',base='tensor'):
	'''
	Build a symmetric tensor library of tensor fields
	'''
	mesh_int = MeshInterpolator().fit(None)
	tangent  = TangentSpaceTransformer().fit(None)

	print(f'Loading {base}2D.npy')
	embryoID = os.path.basename(group.name)
	T = np.load(os.path.join(folder, embryoID, base+'2D.npy'), mmap_mode='r').squeeze()
	T = T.reshape([T.shape[0], 2, 2, *T.shape[-2:]])

	x    = group.create_dataset(key, shape=[T.shape[0], 3, 3, mesh_int.n_vertices_], dtype='float64')
	d1_x = group.create_dataset(f'D1 {key}', shape=[*x.shape, 3], dtype='float64')

	for i in tqdm(range(T.shape[0])):
		xi = mesh_int.transform(T[i]) #2, 2, V
		x[i] = tangent.transform(xi) #3, 3, V
		d1_x[i] = mesh_grad.transform(xi) #3, 3, V, 3
	
	lib = group.require_group('tensor_library')

	lib.create_dataset(key, data=x)
	lib[key].attrs.update({key: 1, 'space': 0})
