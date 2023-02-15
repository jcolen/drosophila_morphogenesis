import numpy as np
import h5py
import os

import sys

basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))
import warnings
from utils.translation_utils import *
from utils.derivative_library_utils import *

import pandas as pd
import pickle as pk
from tqdm import tqdm

import pysindy as ps
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

'''
Generate terms from scalar fields
'''

def s2s_terms(x, group, YY, XX, key='Rnt'):
	d1_x, d2_x = validate_key_and_derivatives(x, group, YY, XX, key, order=2)
	lib = {}
	attrs = {}

	feat = key
	lib[feat] = x
	attrs[feat] = {key: 1, 'space': 0}

	feat = '%s^2' % key
	lib[feat] = x**2
	attrs[feat] = {key: 2, 'space': 0}
	
	feat = 'grad(%s)^2' % key
	lib[feat] = np.einsum('tyxi,tyxi->tyx', d1_x, d1_x)
	attrs[feat] = {key: 2, 'space': 2}

	feat = 'grad^2 %s' % key
	lib[feat] = np.einsum('tyxii->tyx', d2_x)
	attrs[feat] = {key: 1, 'space': 2}
	
	feat = '%s grad^2 %s' % (key, key)
	lib[feat] = np.einsum('tyx,tyxii->tyx', x, d2_x)
	attrs[feat] = {key: 2, 'space': 2}

	write_library_to_dataset(lib, group.require_group('scalar_library'), attrs)


def s2t_terms(x, group, YY, XX, key='Rnt'):
	'''
	Compute terms mapping scalars to symmetric tensors
	'''
	d1_x, d2_x = validate_key_and_derivatives(x, group, YY, XX, key, order=2)
	lib = {}
	attrs = {}
	
	lib['grad(grad(%s))' % key] = np.einsum('tyxij->tijyx', d2_x)
	attrs['grad(grad(%s))' % key] = {key: 1, 'space': 2}

	lib['%s grad(grad(%s))' % (key, key)] = np.einsum('tyx,tyxij->tijyx', x, d2_x)
	attrs['%s grad(grad(%s))' % (key, key)] = {key: 2, 'space': 2}

	lib['grad(%s)grad(%s)' % (key, key)] = np.einsum('tyxi,tyxj->tijyx', d1_x, d1_x)
	attrs['grad(%s)grad(%s)' % (key, key)] = {key: 2, 'space': 2}

	write_library_to_dataset(lib, group.require_group('tensor_library'), attrs)

def build_scalar_library(folder, embryoID, group, key='c', base='cyt',
						 project=True,
						 threshold=0.95, sigma=8):
	'''
	Build a library from scalar information
	'''
	try:
		S = project_embryo_data(folder, embryoID, base, threshold)
	except Exception as e:
		embryoID = str(embryoID)
		print('Could not load projected components for', embryoID, e)
		print('Loading %s2D.npy' % base)
		S = np.load(os.path.join(folder, embryoID, base+'2D.npy'), mmap_mode='r')

		print('Smoothing S with cell size %s' % sigma)
		S = S.reshape([S.shape[0], *S.shape[-2:]])
		S = np.stack([gaussian_filter(S[i], sigma=sigma) for i in range(S.shape[0])])

	S = S.reshape([S.shape[0], *S.shape[-2:]])
	embryoID = str(embryoID)
	dv_coordinates = np.load(os.path.join(folder, embryoID, 'DV_coordinates.npy'), mmap_mode='r')
	ap_coordinates = np.load(os.path.join(folder, embryoID, 'AP_coordinates.npy'), mmap_mode='r')
	s2s_terms(S, group, dv_coordinates, ap_coordinates, key=key)
	#s2t_terms(S, group, key=key)
