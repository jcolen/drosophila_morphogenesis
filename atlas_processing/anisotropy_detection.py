import numpy as np
import pandas as pd
import os
import glob
import warnings

from tqdm import tqdm
from scipy.io import loadmat

from scipy import ndimage
from skimage import transform
try:
	import atlas_processing.windowed_radon as wr
except:
	import windowed_radon as wr


from skimage.morphology import erosion, dilation, disk
from PIL import Image
from skimage.transform import resize

cell_size = 8
structure = disk(cell_size)

def cytosolic_normalize(frame):
	background = dilation(erosion(frame, structure), structure)
	normalized = (frame - background) / background
	normalized[background == 0] = 0.
	return normalized

size = 21
theta = np.linspace(0, 180, 45, endpoint=False)
radon_matrix = wr.get_radon_tf_matrix(size, theta=theta)

#sigma_space = 7 for sqh_ecad_dataset
#sigma_space = 12 otherwise (for some reason)
def anisotropy_tensor(frame, 
					  sigma_space=7,
					  threshold=1.75,
					  target_shape=(236, 200)):
	m_tensor = wr.windowed_radon(frame, radon_matrix, theta=theta*np.pi/180, threshold_mean=threshold,
								 method='global_maximum', return_lines=False)
	m_smooth = wr.filter_field(m_tensor, ndimage.gaussian_filter, kwargs={'sigma': sigma_space})
	m_smooth = wr.filter_field(m_smooth, transform.resize, kwargs={'output_shape': target_shape})
	#Correction to the radon transform code
	#Off-diagonal components are off by factor of -1 based on painting to 3d embryo
	#The reason is that we're using the origin='lower' convention
	#This is equivalent to the transform y->-y on the anisotropy tensor
	m_smooth[..., 0, 1] *= -1
	m_smooth[..., 1, 0] *= -1 
	
	#COMMIT to [M_YY, M_YX], [M_XY, M_XX] ordering
	tmatrix = np.array([[0, 1], [1, 0]])
	m_smooth = np.einsum('ik,...kl,lj->...ij', tmatrix, m_smooth, tmatrix)
	#Reshape
	m_smooth = m_smooth.transpose(2, 3, 0, 1)
	return m_smooth
		
def collect_anisotropy_tensor(savedir, threshold_sigma=None, **wr_kwargs):
	'''
	We are LOCKING IN to IJ ordering
	Spatial ordering is [ROWS, COLUMNS] or [Y, X]
	Channel ordering is [M_YY, M_YX, M_XY, M_XX], 
		which corresponds to the spatial ordering
	
	Remember this when visualizing and lifting to 3D space
	'''    
	if not os.path.exists(os.path.join(savedir, 'dynamic_index.csv')):
		warnings.warn('Index does not exist, processing matstruct')
		convert_matstruct_to_csv(savedir)
		
	warnings.warn('Collecting anisotropy tensors')
	
	df =  pd.read_csv(os.path.join(savedir, 'dynamic_index.csv'))
	for folder in df.folder.unique():
		ss = df[df.folder == folder]
		eID = ss.embryoID.iloc[0]
		tiff_fn = os.path.join(folder, ss.tiff.iloc[0])
		movie = Image.open(tiff_fn)
		frames = np.sort(ss.eIdx.values)
		print('Embryo: ', eID, movie.n_frames, len(frames))
		raws = []
		cyts = []
		tensors = []
		for fId in tqdm(frames):
			movie.seek(fId)
			raw = np.array(movie)
			
			#Clip fiduciary bead intensity
			if threshold_sigma is not None:
				threshold = raw.mean() + threshold_sigma * raw.std()
				raw[raw > threshold] = threshold
			
			cyt = cytosolic_normalize(raw)
			
			tensor = anisotropy_tensor(cyt, **wr_kwargs)
			
			raws.append(raw)
			cyts.append(cyt)
			tensors.append(tensor)
		
		raws = np.stack(raws).astype(float)
		cyts = np.stack(cyts)
		tensors = np.stack(tensors)
		
		raws /= np.median(raws, axis=(1, 2), keepdims=True)
		cyts /= np.median(cyts, axis=(1, 2), keepdims=True)
		
		raws = resize(raws, [raws.shape[0], *tensors.shape[-2:]])
		cyts = resize(cyts, [cyts.shape[0], *tensors.shape[-2:]])
		
		np.save(os.path.join(folder, 'raw2D'), raws)
		np.save(os.path.join(folder, 'cyt2D'), cyts)
		np.save(os.path.join(folder, 'tensor2D'), tensors)

def collect_thresholded_cytosolic_normalization(savedir, threshold_sigma=10):
	if not os.path.exists(os.path.join(savedir, 'dynamic_index.csv')):
		warnings.warn('Index does not exist, processing matstruct')
		convert_matstruct_to_csv(savedir)
		
	warnings.warn('Collecting anisotropy tensors')
	
	df =  pd.read_csv(os.path.join(savedir, 'dynamic_index.csv'))
	for folder in df.folder.unique():
		ss = df[df.folder == folder]
		eID = ss.embryoID.iloc[0]
		tiff_fn = os.path.join(folder, ss.tiff.iloc[0])
		movie = Image.open(tiff_fn)
		frames = np.sort(ss.eIdx.values)
		print('Embryo: ', eID, movie.n_frames, len(frames))
		raws = []
		cyts = []
		for fId in tqdm(frames):
			movie.seek(fId)
			raw = np.array(movie)
	
			#Clip fiduciary bead intensity
			threshold = raw.mean() + threshold_sigma * raw.std()
			raw[raw > threshold] = threshold

			cyt = cytosolic_normalize(raw)
			
			raws.append(raw)
			cyts.append(cyt)
		
		raws = np.stack(raws).astype(float)
		cyts = np.stack(cyts)
		
		raws /= np.median(raws, axis=(1, 2), keepdims=True)
		cyts /= np.median(cyts, axis=(1, 2), keepdims=True)
		
		raws = resize(raws, [raws.shape[0], 236, 200])
		cyts = resize(cyts, [cyts.shape[0], 236, 200])
		
		np.save(os.path.join(folder, 'raw2D'), raws)
		np.save(os.path.join(folder, 'cyt2D'), cyts)
