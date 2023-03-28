from fenics import Function, dof_to_vertex_map, Mesh
from scipy.io import loadmat
from scipy import sparse
from sklearn.neighbors import KDTree

import numpy as np
import os

basedir = '/project/vitelli/jonathan/REDO_fruitfly/src'
geo_dir = '/project/vitelli/jonathan/REDO_fruitfly/flydrive.synology.me/minimalData/Atlas_Data/embryo_geometry'
from fenics import Mesh

mesh = Mesh(os.path.join(geo_dir, 'embryo_coarse_noll.xml'))
tangent_space_data = loadmat(os.path.join(geo_dir, 'embryo_3D_geometry.mat'))

e1 = tangent_space_data['e1']
e2 = tangent_space_data['e2']
Nv = e1.shape[0]
E11, E12, E21, E22 = [], [], [], []
E1, E2 = [], []
for i in range(3):
    E1.append(sparse.spdiags(e1[:, i], 0, Nv, Nv))
    E2.append(sparse.spdiags(e2[:, i], 0, Nv, Nv))
    for j in range(3):
        E11.append(sparse.spdiags(e1[:, i]*e1[:, j], 0, Nv, Nv))
        E12.append(sparse.spdiags(e1[:, i]*e2[:, j], 0, Nv, Nv))
        E21.append(sparse.spdiags(e2[:, i]*e1[:, j], 0, Nv, Nv))
        E22.append(sparse.spdiags(e2[:, i]*e2[:, j], 0, Nv, Nv))

N_vector = sparse.vstack([sparse.hstack(E1), sparse.hstack(E2)])
N_tensor = sparse.vstack([sparse.hstack(E11), sparse.hstack(E12), sparse.hstack(E21), sparse.hstack(E22)])

def left_right_symmetrize(pc, r_verts=mesh.coordinates()):
    #Build coordinate points for flipped mesh
    n_faces = r_verts.shape[0]
    l_faces = r_verts.copy()
    l_faces[..., 1] *= -1
    
    tree = KDTree(l_faces)
    dist, ind = tree.query(r_verts, k=1)
    ind = ind[:, 0]
    
    reflected = pc[:, ind]
    
    if reflected.shape[0] == 3:
        reflected[1] *= -1
    elif reflected.shape[0] == 9:
        reflected[1] *= -1
        reflected[3] *= -1
        reflected[5] *= -1
        reflected[7] *= -1
    
    return 0.5 * (pc + reflected)


def interpolate_vertices(fun_3D, FS):
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

def pull_vector_from_tangent_space(M, e1=e1, e2=e2):
    Nv = e1.shape[0]
    Mi = M.flatten()
    return np.einsum('Vi,V->iV', e1, Mi[:Nv]) + \
           np.einsum('Vi,V->iV', e2, Mi[Nv:])

def pull_tensor_from_tangent_space(M, e1=e1, e2=e2):
    Nv = e1.shape[0]
    Mi = M.flatten()
    return np.einsum('Vi,V,Vj->ijV', e1, Mi[:Nv], e1) + \
           np.einsum('Vi,V,Vj->ijV', e1, Mi[Nv:2*Nv], e2) + \
           np.einsum('Vi,V,Vj->ijV', e2, Mi[2*Nv:3*Nv], e1) + \
           np.einsum('Vi,V,Vj->ijV', e2, Mi[3*Nv:], e2)
