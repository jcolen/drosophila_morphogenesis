import os
import numpy as np
import torch
from torch.nn import Parameter, ParameterList, Module

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, NotFittedError

from fenics import Mesh
from fenics import Function, FunctionSpace
from fenics import FiniteElement, VectorElement, TensorElement
from fenics import dof_to_vertex_map, vertex_to_dof_map
from fenics import TrialFunction, TestFunction
from fenics import assemble
from fenics import inner, grad, dx

from scipy import sparse

from .geometry_utils import geo_dir,TangentSpaceTransformer

def build_operator(expr, v2d):
	A = assemble(expr)
	A = np.array(A.array())
	A = sparse.csr_matrix(A)
	A = A[:, v2d] #Reorder degrees of freedom
	A = A[v2d, :]
	return A

class FenicsGradient(BaseEstimator, TransformerMixin, Module):
	'''
	Compute gradients on an embryo mesh using fenics
	'''
	def __init__(self, 
				 mesh_name='embryo_coarse_noll',
				 pixel_scale=0.2619):
		Module.__init__(self)
		self.mesh_name = mesh_name
		self.pixel_scale = pixel_scale #microns per pixel

	def build_gradient_operators(self, order, mesh, tangent):
		'''
		Returns the necessary assembed fenics operators for computing
		gradient operations on the mesh. Often this will be memory intensive
		'''
		raise NotImplementedError
	
	def fit(self, X, y0=None, tangent=None):

		if torch.is_tensor(X):
			self.mode = 'torch'
		else:
			self.mode = 'numpy'

		self.ops = {}
		mesh = Mesh(os.path.join(geo_dir, f'{self.mesh_name}.xml'))
		if tangent is None:
			tangent = TangentSpaceTransformer(mesh_name=self.mesh_name).fit(X)

		print('Building mesh gradient operators')
		for order in range(3):
			self.ops[order] = self.build_gradient_operators(order, mesh, tangent)
			#print(order, np.prod(self.ops[order].shape), self.ops[order].getnnz())

		return self

	def forward(self, x):
		return self.transform(x)



class FenicsGradient_v0(FenicsGradient):
	'''
	Compute gradients on an embryo mesh using fenics
	The inputs live in the tangent space but are defined on the mesh
	The outputs live in 3D space but are computed using the projected
		tangent operators
	The reason for this is that we can save some overhead by
		iteratively re-using the vector element operators,
		rather than using a tensor or higher-order element operator,
		to compute gradients in each direction
	'''
	def build_gradient_operators(self, order, mesh, tangent):
		'''
		Returns an inverse assembled fenics operator and 
		a series of gradient operators for computing gradients on a mesh

		The general procedure is that for a field defined on the mesh 
		tangent space, the gradient of that field in 3d is 
		np.stack([
			pull_from_tangent_space(
				Ainv.dot(grad[i].dot(f)
			) for i in range(3) ])

		For memory reasons, this transformer only operates in CPU mode

		Note that here Ainv = (N A N^T)^{-1} = N^T^{-1} A^{-1} N^{-1}
		So Ainv (N O N^T) = N^T^{-1} A^{-1} N^{-1} N O N^T
						  = N^T^{-1} A^{-1} O N^T


		'''
		#Check if it exists in file
		fname = f'{geo_dir}/gradient_operators/{self.mesh_name}_v0_order_{order}.npy'
		if os.path.exists(fname):
			print(f'Loading order {order} from file')
			grad = np.load(fname, mmap_mode='r')
		else:
			print(f'Building order {order} from scratch')
			if order == 2:
				fe = TensorElement('CG', mesh.ufl_cell(), 1)
			elif order == 1:
				fe = VectorElement('CG', mesh.ufl_cell(), 1)
			else:
				fe = FiniteElement('CG', mesh.ufl_cell(), 1)

			FS = FunctionSpace(mesh, fe)
			num_vertices = mesh.coordinates().shape[0]
			v2d = vertex_to_dof_map(FS).reshape([num_vertices, -1]).T.flatten()
			
			N = tangent.N[order]

			u = TrialFunction(FS)
			v = TestFunction(FS)
			A = build_operator( inner(u, v) * dx, v2d)
			A = N @ A @ N.T #Project to tangent space
			Ainv = sparse.linalg.inv(A).todense()

			grad = []
			for i in range(3):
				A = build_operator( -inner(u, v.dx(i)) * dx, v2d)
				A = N @ A @ N.T #Project to tangent space
				A = Ainv @ A
				A = N.T @ A #Project to 3D space
				grad.append(np.asarray(A))
			grad = np.vstack(grad)
				
			if not order+1 in tangent.N:
				tangent.build_N_matrix(order+1)
			N_out = tangent.N[order+1]

			grad = N_out @ grad

			np.save(fname, grad)

		if self.mode == 'torch':
			grad = torch.from_numpy(grad)

		return grad
	
	def transform(self, X):
		n_components = np.prod(X.shape[:-1])
		order = int(np.log2(n_components))

		grad = self.ops[order] @ X.flatten()
		grad = grad.reshape([2, *X.shape])

		if self.mode == 'numpy':
			grad = np.moveaxis(grad, 0, -1)
		else:
			grad = torch.moveaxis(grad, 0, -1)

		return grad / self.pixel_scale
		
class FenicsGradient_v1(FenicsGradient):
	'''
	Compute gradients on an embryo mesh using fenics
	The inputs live in the tangent space but are defined on the mesh
	The outputs also live in the tangent space and are computed using
		the projected gradient operators
	'''
	def build_gradient_operators(self, order, mesh, tangent):
		#Check if it exists in file
		fname = os.path.join(geo_dir, 'gradient_operators', f'{self.mesh_name}_order_{order}.npy')
		if os.path.exists(fname):
			print(f'Loading order {order} from file')
			grad = np.load(fname, mmap_mode='r')
		else:
			if order == 2:
				fe = TensorElement('CG', mesh.ufl_cell(), 1)
			elif order == 1:
				fe = VectorElement('CG', mesh.ufl_cell(), 1)
			else:
				fe = FiniteElement('CG', mesh.ufl_cell(), 1)

			FS = FunctionSpace(mesh, fe)
			num_vertices = mesh.coordinates().shape[0]
			v2d = vertex_to_dof_map(FS).reshape([num_vertices, -1]).T.flatten()
			
			u = TrialFunction(FS)
			v = TestFunction(FS)
			A = build_operator( inner(u, v) * dx, v2d)
			Ainv = sparse.linalg.inv(A).todense()

			grad = []
			for i in range(3):
				A = build_operator( - inner(u, v.dx(i)) * dx, v2d)
				A = Ainv @ A
				grad.append(np.asarray(A))
			grad = np.vstack(grad)
			
			if not order+1 in tangent.N:
				tangent.build_N_matrix(order+1)
			N = tangent.N[order]
			N_out = tangent.N[order+1]

			grad = N_out @ grad @ N.T

			np.save(fname, grad)

		if self.mode == 'torch':
			grad = torch.from_numpy(grad)

		return grad

	def transform(self, X):
		n_components = np.prod(X.shape[:-1])
		order = int(np.log2(n_components))

		grad = self.ops[order] @ X.flatten()
		grad = grad.reshape([2, *X.shape])

		if self.mode == 'numpy':
			grad = np.moveaxis(grad, 0, -1)
		else:
			grad = torch.moveaxis(grad, 0, -1)

		return grad / self.pixel_scale

class FenicsGradient_v2(FenicsGradient):
	'''
	This is an attempt using sparse spsolve instead of matrix operations
	It turns out that preserving sparsity does not make things significantly faster
	'''

	def build_gradient_operators(self, order, mesh, tangent):
		self.tangent = tangent

		if order == 2:
			fe = TensorElement('CG', mesh.ufl_cell(), 1)
		elif order == 1:
			fe = VectorElement('CG', mesh.ufl_cell(), 1)
		else:
			fe = FiniteElement('CG', mesh.ufl_cell(), 1)

		FS = FunctionSpace(mesh, fe)
		num_vertices = mesh.coordinates().shape[0]
		v2d = vertex_to_dof_map(FS).reshape([num_vertices, -1]).T.flatten()
		
		N = tangent.N[order]

		u = TrialFunction(FS)
		v = TestFunction(FS)
		A = build_operator( inner(u, v) * dx, v2d)
		A = N @ A @ N.T

		grad = []
		grad.append(A)

		for i in range(3):
			A = build_operator( -inner(u, v.dx(i)) * dx, v2d)
			A = N @ A @ N.T
			grad.append(A)
		
		#Make sure we have the tangent space matrix for this
		if not order+1 in tangent.N:
			tangent.build_N_matrix(order+1)
		N_out = tangent.N[order+1]

		return grad
	
	def transform(self, X):
		n_components = np.prod(X.shape[:-1])
		order = int(np.log2(n_components))

		ops = self.ops[order]
		grad = []

		for i in range(1, len(ops)):
			d = ops[i] @ X.flatten()
			d = sparse.linalg.spsolve(ops[0], d)
			d = self.tangent.N[order].T @ d #Push to 3D space
			grad.append(d)

		grad = np.stack(grad)
		grad = self.tangent.N[order+1] @ grad.flatten() #Push to tangent space
		grad = grad.reshape([2, *X.shape])

		if self.mode == 'numpy':
			grad = np.moveaxis(grad, 0, -1)
		else:
			grad = torch.moveaxis(grad, 0, -1)

		return grad / self.pixel_scale
