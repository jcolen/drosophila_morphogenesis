import warnings
import numpy as np

from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split

from pysindy.pysindy import SINDy
from pysindy.optimizers import SINDyOptimizer, EnsembleOptimizer
from pysindy.feature_library import IdentityLibrary

class ComponentConcatter(TransformerMixin):
	'''
	Similar to PySINDy's SampleConcatter but we need an extra function
	to transform a SVD component weight matrix

	Assuming an input X of size [N_samples, N_components, N_features]
		and component_weights of size N_components, create (and then transform)
		weight matrix by expanding like X 
	'''
	def fit(self, X, y):
		return self
	
	def __sklearn_is_fitted__(self):
		return True

	def transform(self, X):
		return X.reshape([-1, X.shape[-1]])

	def transform_component_weights(self, X, component_weight):
		#Equal weighting
		if component_weight is None:
			weight = np.ones_like(X)
			return self.transform(np.ones_like(X))

		assert component_weight.shape[0] == X.shape[1]
		weight = component_weight[None, :, None]
		weight = np.broadcast_to(weight, X.shape)
		return self.transform(weight)

class FlySINDy(SINDy):
	'''
	Extension to the SINDy object which implements a few assumptions
		1. Data has already been libraried
			We assume that we pre-compute the libraries to save time and overhead
			This also means that the only plausible library is IdentityLibrary, 
			so this is hardcoded in
		2. Data has already been transformed
			Rather than differentiate and compute the noisy dynamics of a data field,
			we assume that the data has been transformed via a linear dimensional 
			reduction technique to save time and overhead. This means that x_dot
			MUST BE PROVIDED during calls to SINDy.fit
		3. As the transformer and differentiator exists separately, this means
			it is not possible to call the simulate function, as this requires
			the ability to differentiate and transform the initial condition
	'''
	def __init__(self,
				 optimizer=None,
				 feature_names=None,
				 feature_library=IdentityLibrary(),
				 lhs=['m_ij'],
				 n_models=1,
				 n_candidates_to_drop=0,
				 subset_fraction=None,
				 material_derivative=True):
		self.material_derivative = material_derivative
		self.n_models = n_models
		self.n_candidates_to_drop = n_candidates_to_drop
		self.subset_fraction = subset_fraction
		self.lhs = lhs

		if self.subset_fraction is None:
			self.subset_fraction = 1 / self.n_models

		if self.n_candidates_to_drop == 0:
			self.n_candidates_to_drop = None
			self.library_ensemble = False
		else:
			self.library_ensemble = True

		super().__init__(
			optimizer=optimizer, 
			feature_names=feature_names.copy(),
			feature_library=feature_library,
		)
	
	def build_ensemble_optimizer(self, x, component_weight=None):
		#print(f'Building ensemble optimizer with {self.n_models} models')
		n_subset = x.shape[0] * self.subset_fraction
		if component_weight is None:
			n_subset *= x.shape[1] * self.subset_fraction

		n_subset = int(np.round(n_subset))

		optimizer = EnsembleOptimizer(
			opt=self.optimizer,
			bagging=True,
			library_ensemble=self.library_ensemble,
			n_models=self.n_models,
			n_subset=n_subset,
			n_candidates_to_drop=self.n_candidates_to_drop)

		return optimizer

	def shift_material_derivative(self, X, X_dot):
		for i, key in enumerate(self.lhs):
			for feat in [f'v dot grad {key}', f'[O, {key}]']:
				if feat in self.feature_names:
					loc = self.feature_names.index(feat)
					X_dot += X[..., loc:loc+1]
					X = np.delete(X, loc, axis=-1)
					self.feature_names.remove(feat)
				to_remove = []
				for i in range(len(self.feature_names)):
					if feat in self.feature_names[i]:
						X = np.delete(X, i, axis=-1)
						to_remove.append(i)
				self.feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if not i in to_remove]
				
			return X, X_dot

	def construct_sample_weight(self, x, x_dot, steps, component_weight=None):
		if component_weight is not None:
			mask = component_weight > 0
			x = x[:, mask]
			x_dot = x_dot[:, mask]
			sample_weight = steps[1][1].transform_component_weights(
				x, component_weight[mask])
		else:
			sample_weight=None

		return sample_weight
		

	def fit(self,
			x,
			x_dot,
			component_weight=None,
			unbias=True,
			quiet=True):
		if hasattr(self.optimizer, "unbias"):
			unbias = self.optimizer.unbias

		if self.material_derivative:
			x, x_dot = self.shift_material_derivative(x, x_dot)

		x, _, x_dot, __ = train_test_split(x, x_dot, test_size=0.2)

		if self.n_models > 1:
			self.optimizer = self.build_ensemble_optimizer(x, component_weight)

		optimizer = SINDyOptimizer(self.optimizer, unbias=unbias)
		steps = [
			("features", self.feature_library),
			("shaping", ComponentConcatter()),
			("model", optimizer),
		]

		sample_weight = self.construct_sample_weight(x, x_dot, steps, component_weight)
		x_dot = steps[1][1].transform(x_dot)
		steps[-1][1].ridge_kw = dict(sample_weight=sample_weight)

		self.model = Pipeline(steps)
		action = "ignore" if quiet else "default"
		with warnings.catch_warnings():
			warnings.filterwarnings(action, category=ConvergenceWarning)
			warnings.filterwarnings(action, category=LinAlgWarning)
			warnings.filterwarnings(action, category=UserWarning)
			self.model.fit(x, x_dot)

		self.n_features_in_ = self.model.steps[0][1].n_features_in_
		self.n_output_features_ = self.model.steps[0][1].n_output_features_
		self.n_control_features_ = 0

		return self

	def equations(self, precision=3):
		'''
		Override pysindy.utils.base.equations to group cadherin terms
		'''
		input_features = self.feature_names
		coef = self.model.steps[-1][1].coef_
		
		def joint_term(cm, cc, name):
			rounded_cm = np.round(cm, precision)
			rounded_cc = np.round(cc, precision)
			if rounded_cm == 0 and rounded_cc == 0:
				return ""
			elif rounded_cc == 0:
				return f"{cm:.{precision}f} {name}"
			elif rounded_cm == 0:
				return f"{cc:.{precision}f} c {name}"
			else:
				return f"({cm:.{precision}f} + {cc:.{precision}f} c) {name}"
		
		eqns = []
		for i in range(coef.shape[0]):
			components = []
			used = np.zeros(coef[i].shape, dtype=bool)
			
			for j, feat in enumerate(input_features):
				if used[j]: 
					continue
				if feat[0] == 'c': #Cadherin modulated
					mterm = feat[2:]
					cterm = feat
				else: #Not cadherin modulated
					mterm = feat
					cterm = 'c ' + feat
				try:	
					jj = input_features.index(mterm)
					kk = input_features.index(cterm)

					cm = coef[i, jj]
					cc = coef[i, kk]

					components.append(joint_term(cm, cc, mterm.replace('E_full', 'E_ij')))
					used[jj] = True
					used[kk] = True
				except:
					cj = coef[i, j]
					components.append(joint_term(cj, 0, feat))
					used[j] = True

			eq = " + ".join(filter(bool, components))
			if len(eq) == 0:
				eq = "0"
			eqns.append(eq)
		return eqns


