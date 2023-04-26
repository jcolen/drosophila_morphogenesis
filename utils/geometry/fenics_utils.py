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

from .geometry_utils import geo_dir, N_vector, N_tensor, embryo_mesh
from .geometry_utils import TangentSpaceTransformer

def build_operator(expr, v2d, N=None):
	A = assemble(expr)
	A = np.array(A.array())
	A = sparse.csr_matrix(A)
	A = A[:, v2d] #Reorder degrees of freedom
	A = A[v2d, :]
	if N is not None:
		A = N @ (A @ N.T) #Push A to tangent space
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
		for order in reversed(range(3)):
			self.ops[order] = self.build_gradient_operators(order, mesh, tangent)

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
		'''
		self.tangent = tangent

		#Check if it exists in file
		if order == 2:
			fe = TensorElement('CG', mesh.ufl_cell(), 1)
			key = 'tensor'
		elif order == 1:
			fe = VectorElement('CG', mesh.ufl_cell(), 1)
			key = 'vector'
		else:
			fe = FiniteElement('CG', mesh.ufl_cell(), 1)
			key = 'scalar'

		fname = f'{geo_dir}/gradient_operators/{self.mesh_name}_v0_order_{order}.npz'
		if os.path.exists(fname):
			print(f'Loading order {order} from file')
			grad = sparse.load_npz(fname)
		else:
			print(f'Building order {order} from scratch')
			FS = FunctionSpace(mesh, fe)
			num_vertices = mesh.coordinates().shape[0]
			v2d = vertex_to_dof_map(FS).reshape([num_vertices, -1]).T.flatten()
			
			N = tangent.N[order]
			NT = N.T

			u = TrialFunction(FS)
			v = TestFunction(FS)
			A = build_operator( inner(u, v) * dx, v2d, N)
			Ainv = sparse.linalg.inv(A)

			grad = [
				NT @ Ainv @ build_operator( -inner(u, v.dx(0)) * dx, v2d, N),
				NT @ Ainv @ build_operator( -inner(u, v.dx(1)) * dx, v2d, N),
				NT @ Ainv @ build_operator( -inner(u, v.dx(2)) * dx, v2d, N),
			]
			if not order+1 in tangent.N:
				tangent.build_N_matrix(order+1)
			N_out = tangent.N[order+1]
			
			grad = sparse.vstack(grad)
			grad = N_out @ grad

			sparse.save_npz(fname, grad)

		if self.mode == 'torch':
			grad = torch.from_numpy(grad[i].todense()).to_sparse()

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

		grad = grad / self.pixel_scale

		return grad
		
class FenicsGradient_v1(FenicsGradient):
	'''
	Compute gradients on an embryo mesh using fenics
	The inputs live in the tangent space but are defined on the mesh
	The outputs also live in the tangent space and are computed using
		the projected gradient operators
	'''
	def build_gradient_operators(self, order, mesh, tangent):
		#Check if it exists in file
		fname = os.path.join(geo_dir, 'gradient_operators', f'{self.mesh_name}_order_{order}.npz')
		if os.path.exists(fname):
			print(f'Loading order {order} from file')
			grad = sparse.load_npz(fname)

		else:
			if order == 2:
				fe = TensorElement('CG', mesh.ufl_cell(), 1)
			elif order == 1:
				fe = VectorElement('CG', mesh.ufl_cell(), 1)
			else:
				fe = FiniteElement('CG', mesh.ufl_cell(), 1)

			if not order in tangent.N:
				tangent.build_N_matrix(order)
			if not order+1 in tangent.N:
				tangent.build_N_matrix(order+1)
			N_in = tangent.N[order]
			N_out = tangent.N[order+1]

			print(f'{mesh.coordinates().shape[0]} vertices, N_in: {N_in.shape}, N_out: {N_out.shape}')

			FS = FunctionSpace(mesh, fe)
			num_vertices = mesh.coordinates().shape[0]
			v2d = vertex_to_dof_map(FS).reshape([num_vertices, -1]).T.flatten()
			
			u = TrialFunction(FS)
			v = TestFunction(FS)
			A = build_operator( inner(u, v) * dx, v2d)
			Ainv = sparse.linalg.inv(A)

			grad = sparse.vstack([
				Ainv @ build_operator( -inner(u, v.dx(0)) * dx, v2d),
				Ainv @ build_operator( -inner(u, v.dx(1)) * dx, v2d),
				Ainv @ build_operator( -inner(u, v.dx(2)) * dx, v2d),
			])

			grad = N_out @ grad @ N_in.T

			sparse.save_npz(fname, grad)

		if self.mode == 'torch':
			grad = torch.from_numpy(grad.todense()).to_sparse()

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

		grad = grad / self.pixel_scale

		return grad
