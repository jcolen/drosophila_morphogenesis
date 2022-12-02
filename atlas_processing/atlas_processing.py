'''
Metadata processing
'''
import numpy as np
import pandas as pd
import h5py
import os
import glob
import warnings

from tqdm import tqdm
from scipy.io import loadmat

basedir = '/project/vitelli/jonathan/REDO_fruitfly/MLData'

def convert_matstruct_to_csv(savedir):
	data = loadmat(os.path.join(savedir, 'dynamic_queried_sample.mat'))
	info = {}

	info['folder'] = [ df.astype('S') for df in data['folders'][0]]
	info['tiff'] = [ df.astype('S') for df in data['names'][0]]
	info['embryoID'] = [ df.astype('S') for df in data['embryoIDs'][0]]
	df = pd.DataFrame(info)
	df['folder'] = df['folder'].apply(lambda x: x[0].decode('utf-8'))
	df['tiff'] = df['tiff'].apply(lambda x: x[0].decode('utf-8'))
	df['embryoID'] = df['embryoID'].apply(lambda x: x[0].decode('utf-8'))

	times = [ dt.flatten()[:-1] for dt in data['times'][0]]
	df['time'] = times
	df = df.explode('time').reset_index(drop=True)
	df['eIdx'] = df.groupby(['embryoID']).cumcount()
	
	df.to_csv(os.path.join(savedir, 'dynamic_index.csv'), index=False)
	return df

'''
Pulling and reformatting data from Atlas
'''

from scipy.interpolate import RectBivariateSpline

piv_geometry = loadmat('/project/vitelli/jonathan/REDO_fruitfly/flydrive.synology.me/minimalData/Atlas_Data/embryo_geometry/embryo_rectPIVscale_fundamentalForms.mat')
pix_geometry = loadmat('/project/vitelli/jonathan/REDO_fruitfly/flydrive.synology.me/minimalData/vitelli_sharing/pixel_coordinates.mat')
Xpiv, Ypiv = piv_geometry['X0'][0], piv_geometry['Y0'][:, 0]
Xpix, Ypix = pix_geometry['XX'][0], pix_geometry['YY'][:, 0]

def collect_velocity_fields(savedir):
	'''
	We are LOCKING IN to IJ ordering
	Spatial ordering is [ROWS, COLUMNS] or [Y, X]
	Channel ordering is [VY, VX] corresponding to the spatial ordering
	
	Remember this when visualizing and lifting to 3D space
	'''
	if not os.path.exists(os.path.join(savedir, 'dynamic_index.csv')):
		warnings.warn('Index does not exist, processing matstruct')
		convert_matstruct_to_csv(savedir)
		
	warnings.warn('Collecting flow fields')
	
	df =  pd.read_csv(os.path.join(savedir, 'dynamic_index.csv'))
	for folder in df.folder.unique():
		ss = df[df.folder == folder]
		eID = ss.embryoID.iloc[0]
		vel_dir = os.path.join(folder, 'PIV_filtered')
		vels = []
		for eIdx in tqdm(sorted(ss.eIdx)):
			fn = os.path.join(vel_dir, 'VeloT_medfilt_%06d.mat' % (eIdx+1))
			vel = loadmat(fn)
			vx = RectBivariateSpline(Xpiv, Ypiv, vel['VX'])(Xpix, Ypix).T
			vy = RectBivariateSpline(Xpiv, Ypiv, vel['VY'])(Xpix, Ypix).T
			vels.append(np.stack([vx, vy]))
		vels = np.stack(vels)
		np.save(os.path.join(folder, 'velocity2D'), vels)

from skimage.morphology import erosion, dilation, disk
cell_size = 8
structure = disk(cell_size)

def cytosolic_normalize(frame):
	background = dilation(erosion(frame, structure), structure)
	normalized = (frame - background) / background
	normalized[background == 0] = 0.
	return normalized

from scipy import ndimage
from skimage import transform
import windowed_radon as wr

size = 21
theta = np.linspace(0, 180, 45, endpoint=False)
radon_matrix = wr.get_radon_tf_matrix(size, theta=theta)

#sigma_space = 7 for sqh_ecad_dataset
#sigma_space = 12 otherwise (for some reason)
def anisotropy_tensor(frame, 
					  sigma_space=7,
					  threshold=1.75,
					  target_shape=(len(Ypix), len(Xpix))):
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
		
from PIL import Image
from skimage.transform import resize

def collect_anisotropy_tensor(savedir, **wr_kwargs):
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
				
'''
3D embryo geometry handling
'''

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

savedirs = [
	#'WT/ECad-GFP',
	#'WT/sqh-mCherry',
	'WT/moesin-GFP',
	'WT/utr-mCherry',
	'WT/Sqh_RokK116A-GFP',
	'WT/histone-RFP',
	#'WT/Bazooka-GFP',
	#'WT/Runt',
	#'WT/Even_Skipped',
	#'Even-Skipped[r13]/Spaghetti_Squash-GFP',
	#'TollRM9/Spaghetti_Squash-GFP',
	#'optoRhoGEF2_sqhCherry/headIllumination',
	#'optoRhoGEF2_sqhCherry/singlePlaneIllumination'
]

for savedir in savedirs:
	fulldir = os.path.join(basedir, savedir)
	df = convert_matstruct_to_csv(fulldir)
	print(fulldir, len(df))
	#collect_velocity_fields(fulldir)
	collect_anisotropy_tensor(fulldir)
	#push_to_embryo_surface(fulldir)
