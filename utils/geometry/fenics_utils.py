import numpy as np

from fenics import Function, FunctionSpace
from fenics import dof_to_vertex_map, vertex_to_dof_map
from fenics import TrialFunction, TestFunction
from fenics import assemble
from fenics import inner, dx

from scipy import sparse

from .geometry_utils import N_vector, N_tensor, embryo_mesh

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
