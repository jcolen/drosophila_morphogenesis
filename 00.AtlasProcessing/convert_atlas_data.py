'''
Metadata processing
'''
import numpy as np
import os
import glob

from morphogenesis.atlas_processing.anisotropy_detection import *
from morphogenesis.atlas_processing.atlas_processing import *



if __name__=='__main__':
	basedir = '/Users/jcolen/Documents/drosophila_morphogenesis/Public/'

	'''
	Myosin anisotropy datasets
	'''
	savedirs = [
		'Halo_Hetero_Twist[ey53]_Hetero/Sqh-GFP',
		#'toll[RM9]/Sqh-GFP',
		#'spaetzle[A]/Sqh-GFP',
		#'Halo_twist[ey53]/Sqh-GFP',
	]
	for savedir in savedirs:
		fulldir = os.path.join(basedir, savedir)
		print(savedir)
		df = convert_matstruct_to_csv(fulldir, prefix='dynamic')
		print(fulldir)
		collect_velocity_fields(fulldir)
		if 'toll' in savedir or 'spaetzle' in savedir:
			collect_anisotropy_tensor(fulldir, threshold_sigma=7)
		else:
			collect_anisotropy_tensor(fulldir)

	'''
	Dynamic intensity datasets
	'''
	savedirs = [
		#'WT/ECad-GFP',
		#'WT/Moesin-GFP',
		#'WT/histone-RFP',
		#'WT/Runt',
		#'WT/Even_Skipped-YFP',
	]
	for savedir in savedirs:
		fulldir = os.path.join(basedir, savedir)
		print(savedir)
		df = convert_matstruct_to_csv(fulldir, prefix='dynamic')
		print(fulldir)# len(df))
		collect_velocity_fields(fulldir)
		downsample_raw_tiff(fulldir)