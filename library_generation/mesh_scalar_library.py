import numpy as np
import os
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

import sys
sys.path.insert(0, '/project/vitelli/jonathan/REDO_fruitfly/src')

from utils.geometry.geometry_utils import MeshInterpolator, TangentSpaceTransformer

def build_scalar_library(folder, group, mesh_grad, key='c', base='raw', sigma=7):
	'''
	Build a library from scalar information
	Generate scalar terms from scalar fields
	'''
	mesh_int = MeshInterpolator().fit(None)
	tangent  = TangentSpaceTransformer().fit(None)

	print(f'Loading {base}2D.npy')
	embryoID = os.path.basename(group.name)
	S = np.load(os.path.join(folder, embryoID, base+'2D.npy'), mmap_mode='r').squeeze()

	x    = group.create_dataset(key, shape=[S.shape[0], mesh_int.n_vertices_], dtype='float64')
	d1_x = group.create_dataset(f'D1 {key}', shape=[*x.shape, 3], dtype='float64')
	d2_x = group.create_dataset(f'D2 {key}', shape=[*x.shape, 3, 3], dtype='float64')

	for i in tqdm(range(S.shape[0])):
		xi = gaussian_filter(S[i], sigma=sigma) #Smooth
		xi = mesh_int.transform(xi) #V
		
		x[i] = xi
		d1_xi = mesh_grad.transform(xi) #V, 3

		d1_xT = tangent.inverse_transform(d1_xi.transpose(1, 0)) #2, V
		d2_xi = mesh_grad.transform(d1_xT) #3, V, 3
		
		d1_x[i] = d1_xi
		d2_x[i] = d2_xi.transpose(1, 0, 2)

	lib = group.require_group('scalar_library')

	lib.create_dataset(key, data=x)
	lib[key].attrs.update({key: 1, 'space': 0})
