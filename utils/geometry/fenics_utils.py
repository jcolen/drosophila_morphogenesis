import numpy as np
import torch
from torch.nn import Parameter, ParameterList

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, NotFittedError

from fenics import Function, FunctionSpace
from fenics import FiniteElement, VectorElement
from fenics import dof_to_vertex_map, vertex_to_dof_map
from fenics import TrialFunction, TestFunction
from fenics import assemble
from fenics import inner, dx

from scipy import sparse

from .geometry_utils import basedir, N_vector, N_tensor, embryo_mesh
from .geometry_utils import TangentSpaceTransformer

def interpolate_vertices_to_function(fun_3D, FS):
	fun = Function(FS)
	d2v = dof_to_vertex_map(FS)
	fun.vector().set_local(fun_3D.T.flatten()[d2v])
	return fun

def convert_bilinear_dof(a, v2d, N=None):
	a = np.array(a.array())
	a = a[:, v2d]
	a = a[v2d, :]
	a = sparse.csr_matrix(a)
	if N is not None:
		a = N.dot(a.dot(N.T)) #Push A to tangent space
	return a

def convert_linear_dof(l, v2d, N=None):
	l = np.array(l)
	l = l[v2d].T
	l = sparse.csr_matrix(l)
	if N is not None:
		l = N.dot(l.reshape([-1, 1])) #Push L to tangent space
	return l

def build_operator(expr, v2d, N):
	A = assemble(expr)
	A = convert_bilinear_dof(A, v2d, N)
	return A

def build_gradient_operators(fe, mesh=embryo_mesh):
	'''
	Returns an inverse assembled fenics operator and 
	a series of gradient operators for computing gradients on a mesh

	The general procedure is that for a field defined on the mesh 
	tangent space, the gradient of that field in 3d is 
	np.stack([
		pull_from_tangent_space(
			Ainv.dot(grad[i].dot(f)
		) for i in range(3) ])
	'''
	FS = FunctionSpace(mesh, fe)
	num_vertices = mesh.coordinates().shape[0]
	v2d = vertex_to_dof_map(FS).reshape([num_vertices, -1]).T.flatten()

	Nc = v2d.shape[0] // num_vertices

	if Nc == 1:
		N = None
	elif Nc == 3:
		N = N_vector
	elif Nc == 9:
		N = N_tensor
	
	u = TrialFunction(FS)
	v = TestFunction(FS)
	A = build_operator( inner(u, v) * dx, v2d, N)
	Ainv = sparse.linalg.inv(A)

	grad = [
		build_operator( -inner(u, v.dx(0)) * dx, v2d, N),
		build_operator( -inner(u, v.dx(1)) * dx, v2d, N),
		build_operator( -inner(u, v.dx(2)) * dx, v2d, N),
	]

	return Ainv, grad

class FenicsGradient(BaseEstimator, TransformerMixin, torch.nn.Module):
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
	def __init__(self, 
				 mesh=embryo_mesh,
				 tangent=TangentSpaceTransformer()):
		torch.nn.Module.__init__(self)
		self.mesh = mesh
		self.tangent = tangent
		self.pixel_scale = 0.2619 #microns per pixel

	def fit(self, X, y0=None):
		try:
			self.A_sca_ = sparse.load_npz(f'{basedir}/utils/geometry/A_scalar.npz')
			self.grad_sca_ = [sparse.load_npz(f'{basedir}/utils/geometry/grad_scalar_{i}.npz') \
							  for i in range(3)]

			self.A_vec_ = sparse.load_npz(f'{basedir}/utils/geometry/A_vector.npz')
			self.grad_vec_ = [sparse.load_npz(f'{basedir}/utils/geometry/grad_vector_{i}.npz') \
							  for i in range(3)]
			print('Loaded mesh gradient operators')

		except:
			print('Building mesh gradient operators')
			self.A_sca_, self.grad_sca_ = build_gradient_operators(
				FiniteElement('CG', embryo_mesh.ufl_cell(), 1))
			self.A_vec_, self.grad_vec_ = build_gradient_operators(
				VectorElement('CG', embryo_mesh.ufl_cell(), 1))

			print('Saving mesh gradient operators')
			sparse.save_npz(f'{basedir}/utils/geometry/A_scalar.npz', self.A_sca_)
			sparse.save_npz(f'{basedir}/utils/geometry/A_vector.npz', self.A_vec_)

			for i in range(3):
				sparse.save_npz(f'{basedir}/utils/geometry/grad_scalar_{i}.npz', self.grad_sca_[i])
				sparse.save_npz(f'{basedir}/utils/geometry/grad_vector_{i}.npz', self.grad_vec_[i])
			print('Done')

		try: 
			check_is_fitted(self.tangent)
		except NotFittedError:
			self.tangent.fit(X)

		if torch.is_tensor(X):
			self.mode = 'torch'
			convert = lambda x: Parameter(torch.from_numpy(x.todense()).to_sparse(), requires_grad=False)
			self.A_sca_ = convert(self.A_sca_)
			self.A_vec_ = convert(self.A_vec_)
			self.grad_sca_ = ParameterList([convert(gs) for gs in self.grad_sca_])
			self.grad_vec_ = ParameterList([convert(gs) for gs in self.grad_vec_])
		else:
			self.mode = 'numpy'

		return self

	def grad_scalar(self, X):
		if self.mode == 'numpy':
			grad = np.zeros([X.shape[-1], 3])
		else:
			grad = torch.zeros([X.shape[-1], 3], dtype=X.dtype, device=X.device)

		for i in range(3):
			L = self.grad_sca_[i] @ X.flatten()
			d = self.A_sca_ @ L
			grad[..., i] = d
		return grad

	def grad_vector(self, X):
		if self.mode == 'numpy':
			grad = np.zeros([3, X.shape[-1], 3])
		else:
			grad = torch.zeros([3, X.shape[-1], 3], dtype=X.dtype, device=X.device)

		for i in range(3):
			L = self.grad_vec_[i] @ X.flatten()
			d = self.A_vec_ @ L
			grad[..., i] = self.tangent.transform(d)
		return grad

	def grad_tensor(self, X):
		if self.mode == 'numpy':
			grad = np.zeros([3, 3, X.shape[-1], 3])
			d_i = np.zeros([*X.shape])
		else:
			grad = torch.zeros([3, 3, X.shape[-1], 3], dtype=X.dtype, device=X.device)
			d_i = torch.zeros([*X.shape], dtype=X.dtype, device=X.device)

		for i in range(3):
			for j in range(2):
				L = self.grad_vec_[i] @ X[j].flatten()
				d = self.A_vec_ @ L
				d_i[j] = d.reshape(X[j].shape)
			if self.mode == 'numpy': 
				d_i = 0.5 * (d_i + d_i.transpose(1, 0, 2))
			else:
				d_i = 0.5 * (d_i + d_i.permute(1, 0, 2))
			grad[..., i] = self.tangent.transform(d_i)
		return grad

	def transform(self, X):
		n_components = np.prod(X.shape[:-1])

		if n_components == 1:
			grad = self.grad_scalar(X)
		elif n_components == 2:
			grad = self.grad_vector(X)
		elif n_components == 4:
			grad = self.grad_tensor(X)

		grad = grad / self.pixel_scale

		return grad

	def forward(self, X):
		return self.transform(X)
