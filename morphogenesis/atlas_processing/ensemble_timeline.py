
'''
Build static ensembled timeline
'''
import numpy as np
import pandas as pd
import h5py
import os
import glob
from PIL import Image
from skimage.transform import resize

from tqdm import tqdm
from scipy.io import loadmat

from scipy.interpolate import RectBivariateSpline

def build_ensemble_timeline(savedir, t_min=0, t_max=50, init_unc=3, sigma=3, drop_times=False):
	print('Computing ensemble-averaged quantities for %s' % savedir)

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
			print('t=%d\tIncreasing uncertainty to %d' % (t, max_unc))

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