import warnings
import numpy as np

from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

from pysindy.pysindy import SINDy, _comprehend_and_validate_inputs
from pysindy.optimizers import SINDyOptimizer
from pysindy.feature_library import IdentityLibrary
from pysindy.utils import concat_sample_axis
from pysindy.utils import SampleConcatter

class ComponentConcatter(TransformerMixin):
	'''
	Similar to PySINDy's SampleConcatter but we need an extra function
	to transform a SVD component weight matrix

	Assuming an input X of size [N_samples, N_components, N_features]
		and component_weights of size N_components, create (and then transform)
		weight matrix by expanding like X 
	'''
	def fit(self, X, y0):
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
				 optimizer,
				 feature_names):
		super(FlySINDy, self).__init__(
			optimizer=optimizer, 
			feature_names=feature_names,
			feature_library=IdentityLibrary(),
		)

	def fit(self,
			x,
			x_dot,
			component_weight=None,
			unbias=True,
			quiet=False):
		if hasattr(self.optimizer, "unbias"):
			unbias = self.optimizer.unbias

		optimizer = SINDyOptimizer(self.optimizer, unbias=unbias)
		steps = [
			("features", self.feature_library),
			("shaping", ComponentConcatter()),
			("model", optimizer),
		]

		if component_weight is not None:
			mask = component_weight > 0
			x = x[:, mask]
			x_dot = x_dot[:, mask]
			sample_weight = steps[1][1].transform_component_weights(
				x, component_weight[mask])
		else:
			sample_weight=None

		x_dot = steps[1][1].transform(x_dot)
		steps[-1][1].ridge_kw = dict(sample_weight=sample_weight)

		fit_params = {
			'model__sample_weight': sample_weight
		}

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

	def predict(self, x):
		raise RuntimeError('FlySINDy object does not implement predict, try predict_h5f')

	def differentiate(self, x, t=None, multiple_trajectories=False):
		raise RuntimeError('FlySINDy object does not differentiate fields!')

	def simulate(self, x0, t):
		raise RuntimeError('FlySINDy object has no differentiators and cannot simulate!')
