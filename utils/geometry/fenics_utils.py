import os
import numpy as np
import torch
from torch.nn import Parameter, ParameterList, Module

from sklearn.base import BaseEstimator, TransformerMixin

from fenics import Mesh
from fenics import Function, FunctionSpace
from fenics import FiniteElement, VectorElement, TensorElement
from fenics import vertex_to_dof_map
from fenics import TrialFunction, TestFunction
from fenics import assemble
from fenics import inner, grad, dx

from scipy import sparse
from scipy.io import loadmat

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
				 cutoff=0.0,
				 pixel_scale=0.2619):
		Module.__init__(self)
		self.mesh_name = mesh_name
		self.pixel_scale = pixel_scale #microns per pixel
		self.cutoff = cutoff #AP cutoff fraction

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
		tangent_space_data = loadmat(os.path.join(geo_dir, f'{self.mesh_name}.mat'))
		z_emb = tangent_space_data['z'].squeeze()

		self.mask = np.zeros(z_emb.shape, dtype=bool)
		self.mask[z_emb <= self.cutoff] = 1
		self.mask[z_emb >= (1 - self.cutoff)] = 1

		if tangent is None:
			if self.mode == 'torch':
				tangent = TangentSpaceTransformer(mesh_name=self.mesh_name).fit(X.cpu().numpy())
			else:
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
		#Check if it exists in file
		fname = os.path.join(geo_dir, 'gradient_operators', f'{self.mesh_name}_full_order_{order}.npz')
		if os.path.exists(fname):
			print(f'Loading order {order} from file')
			L = sparse.load_npz(fname)
		else:
			if order == 2:
				fe1 = TensorElement('CG', mesh.ufl_cell(), 1)
				fe2 = TensorElement('CG', mesh.ufl_cell(), 1, shape=(3,3,3,))
			elif order == 1:
				fe1 = VectorElement('CG', mesh.ufl_cell(), 1)
				fe2 = TensorElement('CG', mesh.ufl_cell(), 1)
			else:
				fe1 = FiniteElement('CG', mesh.ufl_cell(), 1)
				fe2 = VectorElement('CG', mesh.ufl_cell(), 1)

			FS1 = FunctionSpace(mesh, fe1)
			FS2 = FunctionSpace(mesh, fe2)
			num_vertices = mesh.coordinates().shape[0]
			v2d1 = vertex_to_dof_map(FS1).reshape([num_vertices, -1]).T.flatten()
			v2d2 = vertex_to_dof_map(FS2).reshape([num_vertices, -1]).T.flatten()
		
			A = assemble( inner(TrialFunction(FS2), TestFunction(FS2)) * dx)
			A = np.array(A.array())
			A = A @ np.ones(A.shape[1])
			A = 1. / A

			L = assemble( inner(grad(TrialFunction(FS1)), TestFunction(FS2)) * dx)
			L = np.array(L.array())

			L = L * A[:, None]
			L = sparse.csr_matrix(L)

			#Reorder degrees of freedom
			L = L[v2d2, :]
			L = L[:, v2d1]

			#Make sure we have the tangent space matrix for this
			if not order in tangent.N:
				tangent.build_N_matrix(order)
			if not order+1 in tangent.N:
				tangent.build_N_matrix(order+1)
			N1 = tangent.N[order]
			N2 = tangent.N[order+1]

			L = N2 @ L @ N1.T #Push operator to tangent space
			
			sparse.save_npz(fname, L)

		if self.mode == 'torch':
			L = torch.from_numpy(L.todense()).to_sparse()

		return L
	
	def transform(self, X):
		n_components = np.prod(X.shape[:-1])
		order = int(np.log2(n_components))

		grad = self.ops[order] @ X.flatten()
		grad = grad.reshape([2, *X.shape])

		#Note that higher order function space places gradient at last index
		if self.mode == 'numpy':
			grad = np.moveaxis(grad, -2, -1)
		else:
			grad = torch.moveaxis(grad, -2, -1)

		return grad / self.pixel_scale

class FenicsGradient_v3(FenicsGradient):
	'''
	This is an attempt using sparse spsolve instead of matrix operations
	It turns out that preserving sparsity does not make things significantly faster
	'''

	def build_gradient_operators(self, order, mesh, tangent):
		#Check if it exists in file
		fname = os.path.join(geo_dir, 'gradient_operators', f'{self.mesh_name}_full_order_{order}.npy')
		if os.path.exists(fname):
			print(f'Loading order {order} from file')
			L = np.load(fname, mmap_mode='r')
		else:
			if order == 2:
				fe1 = TensorElement('CG', mesh.ufl_cell(), 1)
				fe2 = TensorElement('CG', mesh.ufl_cell(), 1, shape=(3,3,3,))
			elif order == 1:
				fe1 = VectorElement('CG', mesh.ufl_cell(), 1)
				fe2 = TensorElement('CG', mesh.ufl_cell(), 1)
			else:
				fe1 = FiniteElement('CG', mesh.ufl_cell(), 1)
				fe2 = VectorElement('CG', mesh.ufl_cell(), 1)

			FS1 = FunctionSpace(mesh, fe1)
			FS2 = FunctionSpace(mesh, fe2)
			num_vertices = mesh.coordinates().shape[0]
			v2d1 = vertex_to_dof_map(FS1).reshape([num_vertices, -1]).T.flatten()
			v2d2 = vertex_to_dof_map(FS2).reshape([num_vertices, -1]).T.flatten()
		
			A = assemble( inner(TrialFunction(FS2), TestFunction(FS2)) * dx)
			A = sparse.csr_matrix(np.array(A.array()))

			L = assemble( inner(grad(TrialFunction(FS1)), TestFunction(FS2)) * dx)
			L = sparse.csr_matrix(np.array(L.array()))


			L = sparse.linalg.spsolve(A, L)

			#Reorder degrees of freedom
			L = L[v2d2, :]
			L = L[:, v2d1]

			#Make sure we have the tangent space matrix for this
			if not order in tangent.N:
				tangent.build_N_matrix(order)
			if not order+1 in tangent.N:
				tangent.build_N_matrix(order+1)
			N1 = tangent.N[order]
			N2 = tangent.N[order+1]

			L = N2 @ L @ N1.T #Push operator to tangent space
			
			L = np.asarray(L.todense())
			
			np.save(fname, L)

		if self.mode == 'torch':
			L = torch.from_numpy(L)

		return L
	
	def transform(self, X):
		n_components = np.prod(X.shape[:-1])
		order = int(np.log2(n_components))

		grad = self.ops[order] @ X.flatten()
		grad = grad.reshape([2, *X.shape])
		grad[..., self.mask] = 0 #Zero out gradients at the poles
		
		#Note that higher order function space places gradient at last index
		if self.mode == 'numpy':
			grad = np.moveaxis(grad, -2, -1)
		else:
			grad = torch.moveaxis(grad, -2, -1)

		return grad / self.pixel_scale
