import torch
import numpy as np

from ..forecasting.closed_loop import ClosedFlyLoop
from .geometry_utils import TangentSpaceTransformer, MeshInterpolator
from .fenics_utils import FenicsGradient, FenicsGradient_v0, FenicsGradient_v1


class HybridClosedLoop(ClosedFlyLoop):
	'''
	A neural ODE model whose derivatives live on the mesh and nothing else
	'''
	def __init__(self,
				 mesh_name='embryo_coarse_noll',
				 **kwargs):
		super().__init__(**kwargs)
		self.mesh_name = mesh_name
		
	def fit(self, X, y0=None):
		super().fit(X, y0)

		self.tangent   = TangentSpaceTransformer(self.mesh_name).fit(X)
		self.interp    = MeshInterpolator(self.mesh_name).fit(X)
		self.gradient  = FenicsGradient_v1(self.mesh_name).fit(X)

		del self.gamma_dv_
		self.gamma_dv_ = np.array([
				[1., 0.], 
				[0., 0.]
		])[..., None, None] #Constant over space
		if self.mode_ == 'torch':
			self.gamma_dv_ = torch.from_numpy(self.gamma_dv_)

	def forward(self, t, y):
		#Get myosin and source
		m, s = self.inputs.transform(y)

		#Compute flow from myosin/cadherin
		v = self.get_velocity(t, y).squeeze()

		if self.mode_ == 'torch':
			device = m.device
			m = m.cpu()
			s = s.cpu()
			v = v.cpu()
			self.gamma_dv_ = self.gamma_dv_.cpu()

		d1_m = self.gradient(self.interp.transform(m))
		d1_s = self.gradient(self.interp.transform(s))
		d1_v = self.gradient(self.interp.transform(v))

		v_verts = self.interp.transform(v)

		sdot = -self.einsum_('iv,vi->v', v_verts, d1_s)
		sdot = self.interp.inverse_transform(sdot)

		O = -0.5 * (self.einsum_('ivj->ijv', d1_v) - \
					self.einsum_('ivj->jiv', d1_v))
		E =  0.5 * (self.einsum_('ivj->ijv', d1_v) + \
					self.einsum_('ivj->jiv', d1_v))
		adv = self.einsum_('kv,ijvk->ijv', v_verts, d1_m)

		O = self.interp.inverse_transform(O)
		E = self.interp.inverse_transform(E)
		adv = self.interp.inverse_transform(adv)

		lhs  = adv
		lhs += self.einsum_('ik...,kj...->ij...', O, m)
		lhs -= self.einsum_('ik...,kj...->ij...', m, O)

		#Myosin dynamics - right hand side
		rhs  = self.rhs(m, s, v, E)

		mdot = -lhs + rhs

		if self.mode_ == 'torch':
			mdot = mdot.to(device)
			sdot = sdot.to(device)
		
		#Postprocess
		mdot = self.postprocess(mdot)
		sdot = self.postprocess(sdot)
				
		ydot = self.inputs.inverse_transform(mdot, sdot)

		return ydot
	
	def postprocess(self, f, dv_cut=10, ap_cut=10):
		#fill edges 
		f[..., :dv_cut, :] = f[..., dv_cut:dv_cut+1, :]
		f[..., -dv_cut:, :] = f[..., -dv_cut-1:-dv_cut, :]

		f[..., :ap_cut] = f[..., ap_cut:ap_cut+1]
		f[..., -ap_cut:] = f[..., -ap_cut-1:-ap_cut]
		
		#smooth 
		f = self.smoother.transform(f)

		return f
