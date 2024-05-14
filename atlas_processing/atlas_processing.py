'''
Metadata processing
'''
import numpy as np
import pandas as pd
import h5py
import os
import glob
import warnings
from PIL import Image
from skimage.transform import resize

from tqdm import tqdm
from scipy.io import loadmat

from anisotropy_detection import collect_anisotropy_tensor, collect_thresholded_cytosolic_normalization


def convert_matstruct_to_csv(savedir, prefix='dynamic'):
	data = loadmat(os.path.join(savedir, '%s_queried_sample.mat' % prefix))
	info = {}

	info['folder'] = [ df.astype('S') for df in data['folders'][0]]
	info['tiff'] = [ df.astype('S') for df in data['names'][0]]
	info['embryoID'] = [ df.astype('S') for df in data['embryoIDs'][0]]
	df = pd.DataFrame(info)
	df['folder'] = df['folder'].apply(lambda x: x[0].decode('utf-8'))
	df['tiff'] = df['tiff'].apply(lambda x: x[0].decode('utf-8'))
	df['embryoID'] = df['embryoID'].apply(lambda x: x[0].decode('utf-8'))
	
	if prefix == 'dynamic':
		times = [ dt.flatten()[:-1] for dt in data['times'][0]]
	else:
		times = [ dt.flatten() for dt in data['times'][0]]
	df['time'] = times
	df = df.explode('time').reset_index(drop=True)
	df['eIdx'] = df.groupby(['embryoID']).cumcount()
	
	df.to_csv(os.path.join(savedir, '%s_index.csv' % prefix), index=False)
	return df

'''
Pulling and reformatting data from Atlas
'''

from scipy.interpolate import RectBivariateSpline
from plyfile import PlyData

atlas_dir = '/project/vitelli/jonathan/REDO_fruitfly/flydrive.synology.me/Public/dynamic_atlas'
basedir = os.path.join(atlas_dir, 'Atlas_Data')
geometry_dir = os.path.join(atlas_dir, 'embryo_geometry')

def collect_velocity_fields(savedir, subdir='PIV_filtered', prefix='VeloT_medfilt'):
	'''
	We are LOCKING IN to IJ ordering
	Spatial ordering is [ROWS, COLUMNS] or [Y, X]
	Channel ordering is [VY, VX] corresponding to the spatial ordering

	Note that PIV_Filtered folders ALREADY do this! The VX, VY are mislabeled
		They correspond to axes 1, 2, which are the Y, X axes
	
	Remember this when visualizing and lifting to 3D space

	Adjust coordinate systems! The PIV_filtered folder is in units 
		of PIV pixels / min
	The PIV image size is 0.4 x [original dimensions], so each PIV pixel is
		equivalent to 2.5 x original pixel size
	The original units are 1pix=0.2619 um, so 1 PIV pix = 0.65479 um

	The extent in PIV coordinates is [820, 696] 
	'''
	if not os.path.exists(os.path.join(savedir, 'dynamic_index.csv')):
		warnings.warn('Index does not exist, processing matstruct')
		convert_matstruct_to_csv(savedir)
		
	warnings.warn('Collecting flow fields')

	#PIV_filtered units conversion
	original_pixel_size = 0.2619
	piv_rescale_factor = 0.4
	piv_pixel_size = original_pixel_size / piv_rescale_factor

	#Original PIV_filtered coordinates
	piv_scale = loadmat(os.path.join(geometry_dir, 'embryo_rectPIVscale_fundamentalForms.mat'))
	Xpiv, Ypiv = piv_scale['X0'][0], piv_scale['Y0'][:, 0]

	#Establish target coordinate system in PIV coordinates
	rect_scale = PlyData.read(os.path.join(geometry_dir, 'rect_PIVImageScale.ply'))['vertex']
	xmin, xmax = np.min(rect_scale['x']), np.max(rect_scale['x'])
	ymin, ymax = np.min(rect_scale['y']), np.max(rect_scale['y'])
	nAP, nDV = 200, 236
	ap_space = np.linspace(xmin+xmax/nAP*0.5, xmax-xmin/nAP*0.5, nAP)
	dv_space = np.linspace(ymin+ymax/nDV*0.5, ymax-ymin/nDV*0.5, nDV)
	Ypix, Xpix = np.meshgrid(dv_space, ap_space, indexing='ij')
	
	df =  pd.read_csv(os.path.join(savedir, 'dynamic_index.csv'))
	for folder in df.folder.unique():
		ss = df[df.folder == folder]
		eID = ss.embryoID.iloc[0]

		#PIV_filtered folder
		vel_dir = os.path.join(folder, 'PIV_filtered')

		vels = []
		if os.path.exists(vel_dir):
			for eIdx in tqdm(sorted(ss.eIdx)):
				fn = os.path.join(vel_dir, 'VeloT_medfilt_%06d.mat' % (eIdx+1))
				vel = loadmat(fn)
				vx = RectBivariateSpline(Xpiv, Ypiv, vel['VX'])(ap_space, dv_space).T
				vy = RectBivariateSpline(Xpiv, Ypiv, vel['VY'])(ap_space, dv_space).T
				vels.append(np.stack([vx, vy]))
			vels = np.stack(vels)
			#vels = vels * piv_pixel_size #Dynamic Atlas code already corrects for flow units
			np.save(os.path.join(folder, 'velocity2D'), vels)
		elif os.path.exists(os.path.join(folder, '%s_velocity.mat' % eID)):
			pass

		#Save coordinate system in MICRONS
		np.save(os.path.join(folder, 'DV_coordinates'), Ypix * piv_pixel_size)
		np.save(os.path.join(folder, 'AP_coordinates'), Xpix * piv_pixel_size)

def downsample_raw_tiff(savedir, threshold_sigma=10):
	df = pd.DataFrame()
	if os.path.exists(os.path.join(savedir, 'static_index.csv')):
		df = pd.read_csv(os.path.join(savedir, 'static_index.csv'))
		print('Found static index')
	elif os.path.exists(os.path.join(savedir, 'dynamic_index.csv')):
		df = pd.read_csv(os.path.join(savedir, 'dynamic_index.csv'))
		print('Found dynamic index')

	for folder in df.folder.unique():
		ss = df[df.folder == folder]
		eID = ss.embryoID.iloc[0]
		tiff_fn = os.path.join(folder, ss.tiff.iloc[0])
		movie = Image.open(tiff_fn)
		frames = np.sort(ss.eIdx.values)
		print('Embryo: ', eID, movie.n_frames, len(frames))
		raws = []
		for fId in tqdm(frames):
			movie.seek(fId)
			raw = np.array(movie)
			
			#Clip fiduciary bead intensity
			if threshold_sigma is not None:
				threshold = raw.mean() + threshold_sigma * raw.std()
				raw[raw > threshold] = threshold
			
			raws.append(raw)
		
		raws = np.stack(raws).astype(float)
		raws /= np.median(raws, axis=(1, 2), keepdims=True)
		
		raws = resize(raws, [raws.shape[0], 236, 200])
		np.save(os.path.join(folder, 'raw2D'), raws)
	
'''
Build static ensembled timeline
'''
def build_ensemble_timeline(savedir, t_min=0, t_max=50, init_unc=3, sigma=3, drop_times=False):
	warnings.warn('Computing ensemble-averaged quantities for %s' % savedir)

	df = pd.DataFrame()
	if os.path.exists(os.path.join(savedir, 'static_index.csv')):
		df = pd.concat([df, pd.read_csv(os.path.join(savedir, 'static_index.csv'))]).drop_duplicates()
		print('Found static index')
	if os.path.exists(os.path.join(savedir, 'dynamic_index.csv')):
		df = pd.concat([df, pd.read_csv(os.path.join(savedir, 'dynamic_index.csv'))]).drop_duplicates()
		print('Found dynamic index')
	
	if drop_times:
		df.time = df.eIdx
	else:
		df = df.dropna(axis=0)

	if os.path.exists(os.path.join(savedir, 'morphodynamic_offsets.csv')):
			print('Correcting for offsets!')
			morpho = pd.read_csv(os.path.join(savedir, 'morphodynamic_offsets.csv'), index_col='embryoID')
			for eId in df.embryoID.unique():
				df.loc[df.embryoID == eId, 'time'] -= morpho.loc[eId, 'offset']


	print('Building ensemble movies from %d to %d' % (t_min, t_max))

	outdir = os.path.join(savedir, 'ensemble')
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	
	movies = {'t': []}
	ap_coordinates=None
	dv_coordinates=None

	for t in tqdm(range(t_min, t_max)):
		max_unc, flag = init_unc, True
		while flag:
			matches = df[np.abs(df.time - t) < max_unc]
			if len(matches) == 0:
				max_unc += 1
			else:
				flag = False
		if max_unc > init_unc:
			warnings.warn('t=%d\tIncreasing uncertainty to %d' % (t, max_unc))

		frame = {}
		total_weight = 0.
		for i, row in matches.iterrows():
			eframe = {}
			#Iterate over pre-computed movies 
			for npy in glob.glob(os.path.join(glob.escape(row.folder), '*2D.npy')):
				name = os.path.basename(npy)[:-4]
				mov = np.load(npy, mmap_mode='r')
				if row.eIdx < mov.shape[0]:
					eframe[name] = mov[row.eIdx]		
			
			if not 'raw2D' in eframe:
				try:
					tiff_fn = os.path.join(row.folder, row.tiff)
					movie = Image.open(tiff_fn)
					movie.seek(row.eIdx)

					mframe = np.array(movie)
					mframe = mframe / np.median(mframe)
					mframe = resize(mframe, [236, 200])
					eframe['raw2D'] = mframe
				except Exception as e:
					print(e)

			delta = (row.time - t) / sigma
			weight = np.exp(-0.5 * delta**2)
			total_weight += weight

			for key in eframe:
				if key in frame:
					frame[key] = frame[key] + weight*eframe[key]
				else:
					frame[key] = weight*eframe[key]
			
			try:
				ap_coordinates = np.load(os.path.join(row.folder, 'AP_coordinates.npy'), mmap_mode='r')
				dv_coordinates = np.load(os.path.join(row.folder, 'DV_coordinates.npy'), mmap_mode='r')
			except:
				pass
	
		for key in frame:
			frame[key] = frame[key] / total_weight
			if key in movies:
				movies[key].append(frame[key])
			else:
				movies[key] = [frame[key]]
		movies['t'].append(t)

	for key in movies:
		np.save(os.path.join(outdir, key), np.stack(movies[key]))
	
	if ap_coordinates is not None:
		np.save(os.path.join(outdir, 'AP_coordinates'), ap_coordinates)
		np.save(os.path.join(outdir, 'DV_coordinates'), dv_coordinates)

if __name__=='__main__':
	savedirs = [
		'toll[RM9]/Sqh-GFP',
		#'spaetzle[A]/Sqh-GFP',
		#'WT/ECad-GFP',
		#'Halo_Hetero_Twist[ey53]_Hetero/Sqh-GFP',
		#'WT/histone-RFP',
		#'WT/Runt',
		#'WT/Even_Skipped',
		#'WT/Tartan/',
		#'WT/Fushi_Tarazu',
		#'WT/Paired',
		#'WT/Sloppy_Paired',
		#'WT/Even_Skipped-YFP',
		#'Halo_twist[ey53]/Sqh-GFP',
		#'even-Skipped[r13]/Sqh-GFP',
		#'Eve_Mutants/Sqh-GFP',
		#'Dpp_Mutants/Sqh-GFP',
		#'Dpp_Controls/Sqh-GFP',
		#'WT/Moesin-GFP',
	]
	
	#savedir = 'spaetzle[A]/Sqh-GFP'
	#fulldir = os.path.join(basedir, savedir)
	#print(fulldir)
	#collect_anisotropy_tensor(fulldir, threshold_sigma=7)

	for savedir in savedirs:
		fulldir = os.path.join(basedir, savedir)
		print(savedir)
		'''
		Static datasets
		'''
		#df = convert_matstruct_to_csv(fulldir, prefix='static')
		#downsample_raw_tiff(fulldir)
		#We use a 20-minute morphodynamic offset for Runt
		#offsets = pd.DataFrame(columns=['embryoID', 'offset']).set_index('embryoID')
		#for eId in df.embryoID.unique():
		#	offsets.loc[eId, 'offset'] = 20
		#offsets.to_csv(os.path.join(fulldir, 'morphodynamic_offsets.csv'))
		#build_ensemble_timeline(fulldir, init_unc=1,
		#	t_min=-10, t_max=40)

		'''
		Dynamic datasets
		'''
		df = convert_matstruct_to_csv(fulldir, prefix='dynamic')
		print(fulldir)# len(df))
		collect_velocity_fields(fulldir)
		downsample_raw_tiff(fulldir)
		#collect_anisotropy_tensor(fulldir)
		collect_anisotropy_tensor(fulldir, threshold_sigma=7) #toll, spz
		#collect_thresholded_cytosolic_normalization(fulldir)
		#build_ensemble_timeline(fulldir, init_unc=1, #Only when we have a timeline
		#	t_min=-10, t_max=40,
		#	drop_times=np.any([a in savedir for a in ['Sqh', 'Eve', 'hist']]))
	
