'''
3D embryo geometry handling
'''

import numpy as np
import pandas as pd
import os
import glob
import warnings

from scipy.io import loadmat
from tqdm import tqdm

from atlas_processing import *

emb_geometry = loadmat('/project/vitelli/jonathan/REDO_fruitfly/flydrive.synology.me/minimalData/Atlas_Data/embryo_geometry/embryo_3D_geometry.mat')
AP_space = emb_geometry['X'][0]
DV_space = emb_geometry['Y'][:, 0]
Z_AP = emb_geometry['Z'][0]
Phi_AP = emb_geometry['Phi'][::-1, 0]

z_emb = emb_geometry['z'][:, 0]
phi_emb = emb_geometry['ph'][:, 0]

e1 = emb_geometry['e2']
e2 = emb_geometry['e1']
Ei = np.stack([e1, e2], axis=-1).transpose(1, 2, 0)

def push_to_embryo_surface(savedir):
	if not os.path.exists(os.path.join(savedir, 'dynamic_index.csv')):
		warnings.warn('Index does not exist, processing matstruct')
		convert_matstruct_to_csv(savedir)
		
	warnings.warn('Pushing fields to embryo surface')
	
	df =  pd.read_csv(os.path.join(savedir, 'dynamic_index.csv'))
	for folder in df.folder.unique():
		ss = df[df.folder == folder]
		eID = ss.embryoID.iloc[0]
		files = glob.glob(os.path.join(folder, '*2D.npy'))
		for file in files:
			data = np.load(file, mmap_mode='r')
			nTps = data.shape[0]
			data_emb = []
			for tt in tqdm(range(nTps)):
				frame = data[tt]
				header_shape = frame.shape[:-2]
				frame = frame.reshape([-1, *frame.shape[-2:]])

				f_APDV = np.stack([
					RectBivariateSpline(Ypix, Xpix, frame[i])(DV_space, AP_space)
					for i in range(frame.shape[0])])
				#We have to reverse the y axis here because RectBivariateSpline requires increasing order points
				f_emb = np.stack([
					RectBivariateSpline(Phi_AP, Z_AP, f_APDV[i, ::-1])(phi_emb, z_emb, grid=False)
					for i in range(frame.shape[0])])
				
				#Now convert using embryo surface basis vectors
				f_emb = f_emb.reshape([*header_shape, -1])
				
				if len(header_shape) == 1:
					#Transforms like a vector
					f_emb = np.einsum('ijv,jv->iv', Ei, f_emb)
				elif len(header_shape) == 2:
					#Transforms like a tensor
					f_emb = np.einsum('ikv,klv,jlv->ijv', Ei, f_emb, Ei)
				
				data_emb.append(f_emb)
					
			data_emb = np.stack(data_emb)
			np.save(file[:-6]+'3D.npy', data_emb)
