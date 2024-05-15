import numpy as np
import h5py

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class StandardShaper(BaseEstimator, TransformerMixin):
	'''
	Standardizing transformer which maps all inputs to [n_samples, n_channels, H, W]
	'''
	def fit(self, X, y=None):
		'''
		Figure out how to reshape X to shape n_samples, _, Y, X
		'''
		self.data_shape_ = X.shape[-2:]
		self.n_samples_  = X.shape[0]

		X = X.reshape([self.n_samples_, -1, *self.data_shape_])
		self.n_vec_components_ = X.shape[1]

		return self

	@property
	def n_features_in_(self):
		'''
		Total number of input features, both spatial and vectorial
		'''
		return self.n_vec_components_ * np.prod(self.data_shape_)

	def transform(self, X):
		'''
		Apply the shape to the inputs
		'''
		n_samples = X.shape[0]
		return X.reshape([n_samples, self.n_vec_components_, *self.data_shape_])

	def inverse_transform(self, X):
		'''
		Don't keep track of the original shape so inverse transform does nothing
		We defer the work of fixing this to the user
		'''
		return X


class LeftRightSymmetrize(BaseEstimator, TransformerMixin):
	'''
	We assume that embryos are left-right symmetric and anything contrary is
	measurement error. This transformer makes everything explicitly symmetric
	and accounts for vector/tensor transformations on top of that
	'''
	def fit(self, X, y=None):
		'''
		Assume X has shape [n_samples, _, Y, X]
		'''
		self.n_vec_components_ = X.shape[1]
		return self

	def transform(self, X):
		'''
		Apply the transformation
		'''
		X_flip = X[..., ::-1, :].copy() #Flip DV axis

		#Invert DV component of vector field
		if self.n_vec_components_ == 2:
			X_flip[..., 0, :, :] *= -1

		#Invert off-diagonal components of tensor field
		elif self.n_vec_components_ == 4:
			X_flip[:, 1:3] *= -1

		return 0.5 * (X +  X_flip)

	def inverse_transform(self, X):
		'''
		We can't unsymmetrize an object, so this does nothing
		Included for compatiblity
		'''
		return X

class Masker(BaseEstimator, TransformerMixin):
	'''
	Apply a mask to the data
	'''
	def __init__(self, crop=0, mask=None):
		'''
		Crop defines a uniform edge crop
		Mask can be more granular spatially
		'''
		self.crop = crop
		self.mask = mask
		super().__init__()

	def fit(self, X, y=None):
		'''
		Define a boolean mask to apply to the input data
		'''
		self.mask_ = np.ones(X.shape[-2:], dtype=bool)

		if self.mask is not None:
			self.mask_ = np.logical_and(self.mask_, self.mask)

		if self.crop > 0:
			mask = np.zeros_like(self.mask_)
			mask[self.crop:-self.crop, self.crop:-self.crop] = True
			self.mask_ = np.logical_and(self.mask_, mask)

		self.n_features_out_ = np.count_nonzero(self.mask_)
		return self

	def transform(self, X):
		'''
		Apply the mask to the data and flatten it
		'''
		Xt = X.reshape([X.shape[0], -1, *self.mask_.shape])
		Xt = Xt[..., self.mask_]
		return Xt.reshape([Xt.shape[0], -1])

	def inverse_transform(self, X):
		'''
		Re-apply spatial structure and fill in masked regions with zeros
		'''
		X = X.reshape([X.shape[0], -1, self.n_features_out_])
		Xt = np.zeros([*X.shape[:2], *self.mask_.shape])
		Xt[..., self.mask_] = X

		return Xt

class SVDPipeline(Pipeline):
	'''
	Pipeline which shapes, masks, and scales the data before learning an SVD model
	'''
	def __init__(self, n_components=16, whiten=True, crop=0, mask=None, lrsym=True):
		'''
		Initialize the pipeline object
		'''
		self.n_components = n_components
		self.whiten = whiten
		self.crop = crop
		self.mask = mask
		self.lrsym = lrsym
		steps = [
			('shaper', StandardShaper()),
			('masker', Masker(crop=crop, mask=mask)),
			('scaler', StandardScaler(with_std=False)),
			('svd', TruncatedSVD(n_components=n_components)),
		]

		if lrsym:
			steps.insert(1, ('leftright', LeftRightSymmetrize()))

		super().__init__(steps)

	def __getitem__(self, key):
		'''
		Convenience method to access pipeline steps
		'''
		return self.named_steps[key]

	def can_transform(self, X):
		'''
		Check compatibility with input
		'''
		return self.n_features_in_ == np.prod(X.shape[1:])

	def fit(self, X, y=None, **fit_params):
		'''
		X is the data used for fitting the Truncated SVD
		y is the data used for fitting the Standard Scaler
		'''
		for step in self.steps:
			if step[0] == 'scaler':
				step[1].fit(y)
			else:
				step[1].fit(X, y)

			X = step[1].transform(X)
			y = step[1].transform(y)

		return self

	def transform(self, X, remove_mean=False):
		'''
		Apply the learned transformation to new data
		If we whiten and scale, there is a "mean" around which the components
			are centered. The argument "remove_mean" allows us to ignore this
			and get the component vectors as if they were deviations from that
			mean - use this when taking transformations of derivatives
		'''
		if isinstance(X, h5py.Dataset):
			X = X[()]

		if remove_mean:
			mean = self.inverse_transform(np.zeros([1, self['svd'].n_components]))
			mean = mean.reshape([-1, *X.shape[1:]])
			X += mean[0]

		Xt = super().transform(X)
		if self.whiten:
			return Xt / np.sqrt(self['svd'].explained_variance_)
		return Xt

	def inverse_transform(self, Xt, keep=None):
		'''
		Inverse the transformation. Using "Keep" allows us to use only some components
		in the inverse transformation
		'''
		if self.whiten:
			factor = np.sqrt(self['svd'].explained_variance_)
		else:
			factor = np.ones_like(self['svd'].explained_variance_)

		if keep is not None:
			X = np.zeros([Xt.shape[0], self['svd'].n_components])
			if Xt.shape[-1] == keep.shape[0]:
				X[:, keep] = Xt[:, keep]
			else:
				X[:, keep] = Xt
			Xt = X

		return super().inverse_transform(Xt * factor)

	def score(self, X, metric, multioutput='raw_values'):
		'''
		Score the quality of the dimensionality reduction
		'''
		y = self.inverse_transform(self.transform(X))

		#Only score in the masked regions
		X = X[..., self['masker'].mask].squeeze()
		y = y[..., self['masker'].mask].squeeze()

		score = metric(X, y, multioutput=multioutput)
		return score
