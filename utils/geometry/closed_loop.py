import torch
import numpy as np

from ..forecasting.closed_loop import ClosedFlyLoop
from ..forecasting.transforms import EmbryoSmoother
from .transforms import InputProcessor
from .geometry_utils import embryo_mesh, TangentSpaceTransformer, MeshInterpolator
from .fenics_utils import FenicsGradient


class ClosedLoopMesh(ClosedFlyLoop):
	'''
	A similar neural ODE model whose fields live on the mesh, rather than
	in the embryo grid
	
	However, to ensure we remain confined to the surface, all of the 
	dynamics will be projected onto the tangent space
	'''
	def __init__(self,
				 mesh=embryo_mesh,
				 v_model=None,
				 v_thresh=0,
				 sigma=3,
				 dv_mode='circular',
				 ap_mode='replicate'):
		super().__init__(v_model, v_thresh, sigma, dv_mode, ap_mode)
		self.mesh = mesh
		
	def fit(self, X, y0=None):
		self.inputs    = InputProcessor(self.mesh).fit(X)
		self.gradient  = FenicsGradient(self.mesh).fit(X)
		self.smoother  = EmbryoSmoother(sigma=self.sigma, 
									    dv_mode=self.dv_mode, 
									    ap_mode=self.ap_mode).fit(X)

		self.tangent   = TangentSpaceTransformer(mesh=self.mesh).fit(X)
		self.interp    = MeshInterpolator(mesh=self.mesh).fit(X)
		
		gamma_dv = np.zeros([2, 2, *self.inputs.data_shape_])
		gamma_dv[0, 0, ...] = 1.
		self.gamma_dv_ = self.tangent.transform(gamma_dv)
		
		if torch.is_tensor(X):
			self.mode_ = 'torch'
			self.einsum_ = torch.einsum
			self.gamma_dv_ = torch.nn.Parameter(
				torch.from_numpy(self.gamma_dv_),
				requires_grad=False)
		else:
			self.mode_ = 'numpy'
			self.einsum_ = np.einsum
		
	def get_velocity(self, t, y):
		v = self.v_model(t, y)
		return v

	def rhs(self, m, s, v, E):
		trm = self.einsum_('kkv->v', m)
		trE = self.einsum_('kkv->v', E)
		smean = s.mean()
		
		rhs  = -(0.066 - 0.061 * s) * m
		rhs +=	(0.489 + 0.318 * s) * m * trE
		rhs +=	(0.564 - 0.393 * s) * trm * m
		rhs +=	(0.047 - 0.037 * s) * trm * self.gamma_dv_
		return rhs

	def forward(self, t, y):
		#Get myosin and source
		m, s = self.inputs.transform(y)
		
		#Compute flow from myosin/cadherin
		v = self.get_velocity(t, y).squeeze()
		
		#Gradients are computed in tangent space and projected to 3D
		d1_m = self.gradient(m)
		d1_s = self.gradient(s)
		d1_v = self.gradient(v)
		
		#Project fields themselves to 3D
		s = self.tangent.transform(s)
		v = self.tangent.transform(v)
		m = self.tangent.transform(m)
				
		sdot = -self.einsum_('iv,vi->v', v, d1_s)
		
		O = -0.5 * (self.einsum_('ivj->ijv', d1_v) - \
					self.einsum_('ivj->jiv', d1_v))
		E =  0.5 * (self.einsum_('ivj->ijv', d1_v) + \
					self.einsum_('ivj->jiv', d1_v))
	
		lhs  = self.einsum_('kv,ijvk->ijv', v, d1_m)
		lhs += self.einsum_('ikv,kjv->ijv', O, m)
		lhs -= self.einsum_('ikv,kjv->ijv', m, O)
		
		#Myosin dynamics - right hand side
		rhs  = self.rhs(m, s, v, E)

		mdot = -lhs + rhs
		
		#Postprocess
		mdot = self.postprocess(mdot)
		sdot = self.postprocess(sdot)
				
		ydot = self.inputs.inverse_transform(mdot, sdot)
		return ydot
	
	def postprocess(self, f, nDV=54, nAP=46, dv_cut=2, ap_cut=3):
		#Project back to tangent space
		f = self.tangent.inverse_transform(f)

		#Interpolate to 2D for smoothing
		f = self.interp.inverse_transform(f, nDV=nDV, nAP=nAP)

		#Cut off poles
		f = f[..., dv_cut:-dv_cut, ap_cut:-ap_cut]
		
		#Smooth
		f = self.smoother.transform(f)
		
		#Interpolate back to mesh vertices
		f = self.interp.transform(f)
		return f
	
