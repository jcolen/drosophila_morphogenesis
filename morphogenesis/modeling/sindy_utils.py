import numpy as np
from tqdm.auto import tqdm


'''
These are the features identified in LearnDynamicalModel.ipynb
'''
overleaf_feature_names = [
	'v dot grad m_ij', '[O, m_ij]',
	'm_ij', 'c m_ij',
	'm_ij Tr(m_ij)', 'c m_ij Tr(m_ij)',
	'm_ij Tr(E)', 'c m_ij Tr(E)',
	'Static_DV Tr(m_ij)', 'c Static_DV Tr(m_ij)',
]

def collect_library_terms(h5f, key, tmin, tmax, feature_names=None, keep_frac=0.2):
	'''
	Collect the data from a given h5f library and return X, X_dot, and the feature names
	'''
	X, X_dot = [], []

	with tqdm(total=len(h5f.keys())) as pbar:
		pbar.set_description('Collecting data')
		for eId in list(h5f.keys()):
			pbar.set_postfix(embryo=eId)
			pbar.update()
			if eId == 'ensemble':
				continue

			lib = h5f[eId]

			if feature_names is None:
				feature_names = list(lib['features'].keys())
				feature_names = [fn for fn in feature_names if not 'm_ij m_ij' in fn]

			#Pass 1 - get the proper time range
			data = lib['features'] #Only check terms that can be transformed/projected
			for feature in feature_names:
				tmin = max(tmin, np.min(data[feature].attrs['t']))
				tmax = min(tmax, np.max(data[feature].attrs['t']))

			#Pass 2 - collect points within that time range
			x = []
			data = lib['features'] #Get spatial data 
								   #Note fields have already been projected onto components in derivative library generation
								   #We used to get components directly but this created too little data for ensembling
			for feature in feature_names:
				t = data[feature].attrs['t']
				x.append(data[feature][np.logical_and(t >= tmin, t <= tmax), ...])
			x = np.stack(x, axis=-1)
			
			t = lib['X_dot'][key].attrs['t']
			x_dot = lib['X_dot'][key][np.logical_and(t >= tmin, t <= tmax), ...]
			
			if keep_frac < 1:   #Only keep a subset of the spatial data
				space_points = x.shape[-3] * x.shape[-2]
				keep_points = int(keep_frac * space_points)
				mask = np.zeros(space_points, dtype=bool)
				mask[np.random.choice(range(space_points), keep_points, replace=False)] = True
				mask = mask.reshape([x.shape[-3], x.shape[-2]])
				x = x[..., mask, :]
				x_dot = x_dot[..., mask, :]

			x = x.reshape([x.shape[0], -1, x.shape[-1]])
			x_dot = x_dot.reshape([x.shape[0], -1, 1])

			X.append(x)
			X_dot.append(x_dot)
	
	X = np.concatenate(X, axis=0)
	X_dot = np.concatenate(X_dot, axis=0)

	return X, X_dot, feature_names
