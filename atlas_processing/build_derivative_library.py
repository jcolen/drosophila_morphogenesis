import numpy as np
import h5py
import os

import sys

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.io import loadmat

basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))

from fenics import dx, Identity
from fenics import TrialFunction, TestFunction
from fenics import FunctionSpace, FiniteElement, VectorElement, TensorElement
from fenics import vertex_to_dof_map, assemble, solve
from ufl import indices

def rename_keys(L, old, new):
	for key in list(L.keys()):
		newkey = key.replace(old, new)
		L[newkey] = L.pop(key)
	return L

def t2t_library(tensor, test, label):	 
	i,j,k,l = indices(4)
	L = {
		r'tensor_{ij}': tensor[i,j] * test[i,j],
		r'tensor_{ij} Tr(tensor)': tensor[i,j] * tensor[k,k] * test[i,j], 
		r'\partial_{ij} Tr(tensor)': -tensor[k,k].dx(i) * test[i,j].dx(j),
		r'\nabla^2 tensor_{ij}': -tensor[i,j].dx(k) * test[i,j].dx(k), 
		r'\partial_{ik} tensor_{jk}': -tensor[j,k].dx(i) * test[i,j].dx(k),
		r'\nabla^2 tensor_{ij} Tr(tensor)': -(tensor[i,j]*tensor[l,l]).dx(k) * test[i,j].dx(k), 
		r'\partial_{ij} Tr(tensor)^2': -(tensor[k,k]*tensor[l,l]).dx(i) * test[i,j].dx(j),	  
		r'\partial_{ik} tensor_{jk} Tr(tensor)': -(tensor[j,k]*tensor[l,l]).dx(i) * test[i,j].dx(k), 
		r'\partial_{kl} tensor_{ik} tensor_{jl}': -(tensor[i,k]*tensor[j,l]).dx(k) * test[i,j].dx(l),
		r'\partial_{kl} tensor_{ij} tensor_{kl}': -(tensor[i,j]*tensor[k,l]).dx(k) * test[i,j].dx(l)
	}
	return rename_keys(L, 'tensor', label)	  

def get_vector_gradient(vector, test, A, FS):
	Dv = Function(FS)
	i,j = indices(2)
	L = -vector[i]*test[i,j].dx(j)
	solve(A*dx == L*dx, Dv, [])
	return Dv

def v2t_library(vector, Dv, test, label):
	i,j,k,l = indices(4)
	L = {
		r'vector_{i} vector_{j}': vector[i]*vector[j] * test[i,j],
		r'E_{ij}': (Dv[i,j]+Dv[j,i]) * test[i,j],
		r'\partial_{ij} vector^2': -(vector[k]*vector[k]).dx(i) * test[i,j].dx(j),
		r'\nabla^2 vector_{i} vector_{j}': -(vector[i]*vector[j]).dx(k) * test[i,j].dx(k),
		r'\nabla \cdot \mathbf{vector} E_{ij}': Dv[k,k]*(Dv[i,j]+Dv[j,i]) * test[i,j],
		r'\mathbf{vector} \cdot \nabla E_{ij}': vector[k] * (Dv[i,j]+Dv[j,i]).dx(k) * test[i,j],
		r'vector_{(i} \partial_{j)} \nabla \cdot \mathbf{vector}': (vector[i]*Dv[k,k].dx(j) + vector[j]*Dv[k,k].dx(i)) * test[i,j],
		r'\partial_{(i} vector_k \partial_k vector_{i)}': (Dv[i,k]*Dv[k,j] + Dv[j,k]*Dv[k,i]) * test[i,j], 
	}
	return rename_keys(L, 'vector', label)

def tv2t_library(tensor, vector, Dv, test, label_t, label_v):
	i,j,k,l = indices(4)
		
	L = {
		r'Tr(tensor) E_{ij}': tensor[k,k] * (Dv[i,j] + Dv[j,i]) * test[i,j],
		r'tensor_{ik} E_{jk}': tensor[i,k] * (Dv[k,j] + Dv[j,k]) * test[i,j],
		r'tensor_{ij} \nabla \cdot \mathbf{vector}': tensor[i,j] * Dv[k,k] * test[i,j],
		r'\mathbf{vector} \cdot \nabla tensor_{ij}': vector[k] * tensor[i,j].dx(k) * test[i,j],
	}
	
	L = rename_keys(L, 'vector', label_v)
	L = rename_keys(L, 'tensor', label_t)
	return L

def assemble_tensor_library(folder, 
							eIdx, 
							trials, 
							tests, 
							spaces,
							vector_label='v',
							tensor_label='tensor'):
	'''
	Generic signature for assemble_fem_library
	trials: tuple (Scalar, Vector, Tensor) functions
	tests:	tuple (Scalar, Vector, Tensor) functions
	spaces: tuple (Scalar, Vector, Tensor) function spaces
	'''
	
	#Load relevant data
	vel_flag = False
	tensor = np.load(os.path.join(folder, 'tensor3D.npy'), mmap_mode='r')[eIdx]
	tensor = interpolate_vertices(tensor.reshape([9, -1]), spaces[2])
	if os.path.exists(os.path.join(folder, 'velocity3D.npy')):
		velocity = np.load(os.path.join(folder, 'velocity3D.npy'), mmap_mode='r')[eIdx]
		velocity = interpolate_vertices(velocity, spaces[1])
		vel_flag=True
	
	_, _, trial = trials
	_, _, test = tests
	_, _, FS = spaces
		
	i,j = indices(2)
	A = trial[i,j]*test[i,j]
	
	library = {}
	
	if vel_flag:
		Dv = get_vector_gradient(velocity, test, A, FS)  #Precompute velocity gradients in weak form								  
		L = { 
			**t2t_library(tensor, test, tensor_label), 
			**v2t_library(velocity, Dv, test, vector_label),
			**tv2t_library(tensor, velocity, Dv, test, tensor_label, vector_label)
		}
	else:
		L = t2t_library(tensor, test, tensor_label)
	
	v2d = np.reshape(vertex_to_dof_map(FS), [-1, 9]).T.flatten()
	A = convert_bilinear_dof(assemble(A * dx), v2d, N=N_tensor)

	library = {}
	for key in L:
		l = assemble(L[key] * dx)
		l = convert_linear_dof(l, v2d, N=N_tensor)
		library[key] = spsolve(A, l)	
	
	return library

import pandas as pd
from tqdm import tqdm

from geometry_utils import *

def build_3D_library(directory,
					 filename,
					 assemble_fem_library,
					 fast_dev_run=False,
					 **afl_kwargs):
	S = FunctionSpace(mesh, FiniteElement('CG', mesh.ufl_cell(), degree=1))
	V = FunctionSpace(mesh, VectorElement('CG', mesh.ufl_cell(), degree=1))
	T = FunctionSpace(mesh, TensorElement('CG', mesh.ufl_cell(), degree=1))
	
	spaces = (S, V, T)
	trials = (TrialFunction(S), TrialFunction(V), TrialFunction(T))
	tests = (TestFunction(S), TestFunction(V), TestFunction(T))
	
	index = pd.read_csv(os.path.join(directory, 'dynamic_index.csv'))
	
	with h5py.File(os.path.join(directory, filename), 'w') as h5f:
		for i in tqdm(range(len(index))):
			
			folder = index.folder[i]
			embryo = str(index.embryoID[i])
			time = '%2f' % index.time[i]
			eIdx = index.eIdx[i]
			
			print(i, embryo, time, eIdx, folder)

			
			library = assemble_fem_library(
				folder, eIdx, trials, tests, spaces, **afl_kwargs)
			
			#Write embryo info to h5f
			if embryo in h5f.keys():
				gg = h5f[embryo]
			else:
				gg = h5f.create_group(embryo)
			
			gt = gg.create_group(time)
			for key in library:
				gt.create_dataset(key, data=library[key])
			
			if fast_dev_run:
				return

if __name__=='__main__':
	datadir = '/project/vitelli/jonathan/REDO_fruitfly/MLData'
	dirs = {
		'c': 'WT/ECad-GFP',
		'm': 'WT/sqh-mCherry',
	}
	for key in dirs:
		build_3D_library(os.path.join(datadir, dirs[key]),
						 'tensor_library_3D.h5',
						 assemble_tensor_library,
						 fast_dev_run=False,
						 tensor_label=key)
