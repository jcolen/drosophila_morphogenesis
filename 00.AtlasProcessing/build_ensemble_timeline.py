'''
Metadata processing
'''
import numpy as np
import os
import glob

from morphogenesis.atlas_processing.ensemble_timeline import *

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
		'WT/ECad-GFP',
		#'WT/Moesin-GFP',
		#'WT/histone-RFP',
		#'WT/Runt',
		#'WT/Even_Skipped-YFP',
	]
	for savedir in savedirs:
		fulldir = os.path.join(basedir, savedir)
		print(savedir)
		build_ensemble_timeline(fulldir, init_unc=1,
						  t_min=-10, t_max=40,
						  drop_times=np.any([a in savedir for a in ['Sqh', 'Eve', 'hist']]))
