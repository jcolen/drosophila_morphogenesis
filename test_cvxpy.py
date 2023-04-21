import numpy as np
import pandas as pd
import h5py
import os
import glob
import warnings
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import sys
basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))


import pysindy as ps
from sklearn.decomposition import PCA
from utils.modeling.sindy_utils import *
from utils.modeling.fly_sindy import FlySINDy

tmin, tmax = 0, 20
n_components=5

with h5py.File('Public/mesh_dynamics_fitting.h5', 'r') as h5f:
    X, X_dot, feature_names = collect_data(h5f, 'm_ij', tmin, tmax, collect_mesh_data)
idx = feature_names.index('m_ij')
pca = PCA(n_components).fit(X[..., idx])

def transform(x, remove_mean=False):
    nfeats = x.shape[-1]
    x = x.transpose(0, 2, 1).reshape([-1, pca.n_features_])
    if remove_mean:
        x += pca.mean_
    x = pca.transform(x).reshape([-1, nfeats, pca.n_components_])
    x = x.transpose(0, 2, 1)
    return x

X = transform(X)
X_dot = transform(X_dot, remove_mean=True)

N = 2
constraint_lhs = np.zeros([N, len(feature_names)])
constraint_rhs = np.zeros(N)

#Detachment: (k_0 + k_1 c) m_ij
idx = feature_names.index('m_ij')
print(feature_names[idx:idx+2])
eps = 1e-3
constraint_lhs[0, idx] =  1
constraint_rhs[0] = -0.05 #k_0 is negative (detachment)
constraint_lhs[1, idx  ] = 1
constraint_lhs[1, idx+1] = 1
constraint_rhs[1] = -eps     #k_0 + k_1 < 0 (cadherin doesn't overrun)

optimizer = ps.ConstrainedSR3(constraint_lhs=constraint_lhs, 
							  constraint_rhs=constraint_rhs,
                              thresholder='l1',
                              threshold=1e-2, 
                              nu=1e-1, tol=1e-5, 
                              verbose_cvxpy=True, 
							  inequality_constraints=True)
print(optimizer.constraint_lhs)
print(optimizer.constraint_rhs)

sindy = FlySINDy(
    optimizer=optimizer,
    feature_names=feature_names,
    material_derivative=False,
    n_models=1)
sindy.fit(x=X, x_dot=X_dot)
sindy.print(lhs=['D_t m_ij'])

print(sindy.coefficients().flatten()[idx:idx+2])
