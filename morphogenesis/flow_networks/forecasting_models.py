import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .translation_models import VAE


class VAE_Evolver(VAE):
	'''
	Translation model with a LSTM block at the latent bottleneck
	Forecast the initial condition for a specified amount of time
	'''
	def __init__(self, 
				 *args,
				 hidden_size=64,
				 lstm_layers=2,
				 **kwargs):
		super(VAE_Evolver, self).__init__(*args, **kwargs)
		
		self.evolver = nn.LSTM(input_size=self.num_latent,
							   proj_size=self.num_latent,
							   hidden_size=hidden_size,
							   num_layers=lstm_layers,
							   batch_first=True)

	def forward(self, x, lengths):
		#Encoder step identical to standard VAE
		b, c, h0, w0 = x.shape
		x = self.encoder(x)
				
		x = x.reshape([b, 1, -1])
		x = self.field_to_params(x)
		
		mu = x[..., :self.num_latent]
		logvar = x[..., self.num_latent:]

		#VAE Reparameterization trick
		params = mu
		if self.training:
			params = params + torch.randn_like(params) * (0.5 * logvar).exp()
		
		#Input sequence length = 1 (forecast from ICs)
		t = torch.max(lengths)
		params_list = torch.zeros([b, t, self.num_latent], device=x.device)
		params_list[:, 0:1] = params #Translate initial conditions

		outputs, hidden_state = self.evolver(params)
		params = params + outputs
		params_list[:, 1:2] = params
		
		for i in range(2, t):
			outputs, hidden_state = self.evolver(params, hidden_state)
			params = params + outputs
			params_list[:, i:i+1] = params
			
		x = self.params_to_field(params_list)
		x = F.gelu(x)
		x = x.reshape([b*t, -1, *self.bottleneck_size])
		
		x = self.decoder(x)
		if (x.shape[-2] != h0) or (x.shape[-1] != w0):
			x = torch.nn.functional.interpolate(x, size=[h0, w0], mode='bilinear')	

		x = x.reshape([b, t, -1, h0, w0])

		#Pad away excess predictions
		for i in range(b):
			x[i, lengths[i]:] *= 0.

		return x, (params, mu, logvar)

class MaskedVAE_Evolver(VAE_Evolver):
	def __init__(self, 
				 *args,
				 dv_min=15,
				 dv_max=-15,
				 ap_min=15, 
				 ap_max=-15,
				 **kwargs):
		super(MaskedVAE_Evolver, self).__init__(*args, **kwargs)
		
		self.mask = torch.ones(self.input_size, dtype=bool)
		self.mask[dv_min:dv_max, ap_min:ap_max] = 0
		self.mask = torch.nn.Parameter(self.mask, requires_grad=False)

	def forward(self, x, lengths):
		x[..., self.mask] = 0.
		return super(MaskedVAE_Evolver, self).forward(x, lengths)
