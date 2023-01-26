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

basedir = '/project/vitelli/jonathan/REDO_fruitfly/MLData'

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


'''
Build static ensembled timeline
'''
def build_ensemble_timeline(savedir, t_min=0, t_max=50, init_unc=3, sigma=3):
	warnings.warn('Computing ensemble-averaged quantities for %s' % savedir)

	df = pd.DataFrame()
	if os.path.exists(os.path.join(savedir, 'static_index.csv')):
		df = pd.concat([df, pd.read_csv(os.path.join(savedir, 'static_index.csv'))]).drop_duplicates()
		print('Found static index')
	if os.path.exists(os.path.join(savedir, 'dynamic_index.csv')):
		df = pd.concat([df, pd.read_csv(os.path.join(savedir, 'dynamic_index.csv'))]).drop_duplicates()
		print('Found dynamic index')
	print('Building ensemble movies from %d to %d' % (t_min, t_max))

	outdir = os.path.join(savedir, 'ensemble')
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	
	movies = {'t': []}

	for t in tqdm(range(t_min, t_max)):
		max_unc, flag = init_unc, True
		while flag:
			matches = df[np.abs(df.time - t) < max_unc]
			if len(matches) == 0:
				max_unc += 1
				warnings.warn('t=%d\tIncreasing uncertainty to %d' % (t, max_unc))
			else:
				flag = False

		frame = {}
		total_weight = 0.
		for i, row in matches.iterrows():
			eframe = {}
			#Iterate over pre-computed movies 
			for npy in glob.glob(os.path.join(row.folder, '*2D.npy')):
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
	
		for key in frame:
			frame[key] = frame[key] / total_weight
			if key in movies:
				movies[key].append(frame[key])
			else:
				movies[key] = [frame[key]]
		movies['t'].append(t)

	for key in movies:
		np.save(os.path.join(outdir, key), np.stack(movies[key]))

savedirs = [
	#'WT/ECad-GFP',
	#'WT/sqh-mCherry',
	#'WT/moesin-GFP',
	#'WT/utr-mCherry',
	#'WT/Sqh_RokK116A-GFP',
	#'WT/histone-RFP',
	'WT/Tartan/',
	#'WT/Bazooka-GFP',
	#'WT/Runt',
	#'WT/Even_Skipped',
	#'WT/Fushi_Tarazu',
	#'WT/Hairy',
	#'WT/Paired',
	#'WT/Sloppy_Paired',
	#'Even-Skipped[r13]/Spaghetti_Squash-GFP',
	#'TollRM9/Spaghetti_Squash-GFP',
	#'optoRhoGEF2_sqhCherry/headIllumination',
	#'optoRhoGEF2_sqhCherry/singlePlaneIllumination'
]

for savedir in savedirs:
	fulldir = os.path.join(basedir, savedir)
	df = convert_matstruct_to_csv(fulldir, prefix='static')
	#print(fulldir, len(df))
	build_ensemble_timeline(fulldir, init_unc=1)
	#collect_velocity_fields(fulldir)
	#collect_anisotropy_tensor(fulldir)
	#push_to_embryo_surface(fulldir)
