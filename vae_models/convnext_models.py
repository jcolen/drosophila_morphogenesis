import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

class LayerNorm2d(torch.nn.LayerNorm):
	r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
	"""

	def __init__(self, normalized_shape, eps=1e-6):
		super().__init__(normalized_shape, eps=eps)

	def forward(self, x) -> torch.Tensor:
		if x.is_contiguous(memory_format=torch.contiguous_format):
			return torch.nn.functional.layer_norm(
				x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
		else:
			s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
			x = (x - u) * torch.rsqrt(s + self.eps)
			x = x * self.weight[:, None, None] + self.bias[:, None, None]
			return x

class ConvNextBlock(nn.Module):
	def __init__(self, input_dim, output_dim):#, sd_prob=0.1):
		super(ConvNextBlock, self).__init__()
		self.depth_conv = nn.Conv2d(input_dim, input_dim, 
									kernel_size=7, padding='same', groups=input_dim)
		self.norm = LayerNorm2d(input_dim)
		
		self.conv1 = nn.Conv2d(input_dim, 4*input_dim, kernel_size=1)
		self.conv2 = nn.Conv2d(4*input_dim, output_dim, kernel_size=1)
		self.act = nn.GELU()
		self.dropout = nn.Dropout2d(p=0.2)
			
	def forward(self, x):
		x = self.depth_conv(x)
		x = self.norm(x)
		x = self.conv1(x)
		x = self.act(x)
		x = self.conv2(x)
		x = self.dropout(x)
		return x
class Encoder(nn.Module):
	def __init__(self, 
				 input_dims,
				 stage_dims):
		super(Encoder, self).__init__()
		self.downsample_blocks = nn.ModuleList()
	
		patchify_stem = nn.Sequential(
			nn.Conv2d(input_dims, stage_dims[0][0], kernel_size=4, stride=4),
			LayerNorm2d(stage_dims[0][0]),
		)
		self.downsample_blocks.append(patchify_stem)
		for i in range(len(stage_dims)-1):
			stage = nn.Sequential(
				LayerNorm2d(stage_dims[i][-1]),
				nn.Conv2d(stage_dims[i][-1], stage_dims[i+1][0], kernel_size=2, stride=2)
			)
			self.downsample_blocks.append(stage)
	
		self.stages = nn.ModuleList()
		for i in range(len(stage_dims)):
			stage = nn.Sequential(
				*[ConvNextBlock(stage_dims[i][j], stage_dims[i][j+1]) \
				  for j in range(len(stage_dims[i])-1)]
			)
			self.stages.append(stage)
	
	def forward(self, x): 
		encoder_outputs = []
		for i in range(len(self.stages)):
			encoder_outputs.append(x)
			x = self.downsample_blocks[i](x)
			x = self.stages[i](x)
		return x, encoder_outputs
	
class Decoder(nn.Module):
	def __init__(self,
				 latent_dims,
				 output_dims,
				 stage_dims):
		super(Decoder, self).__init__()
		self.upsample_blocks = nn.ModuleList()

		stage = nn.Sequential(
			LayerNorm2d(latent_dims),
			nn.ConvTranspose2d(latent_dims, stage_dims[-1][-1], kernel_size=2, stride=2)
		)
		self.upsample_blocks.append(stage)

		for i in range(1, len(stage_dims)-1):
			stage = nn.Sequential(
				LayerNorm2d(stage_dims[-i][0]),
				nn.ConvTranspose2d(stage_dims[-i][0], stage_dims[-(i+1)][-1], kernel_size=2, stride=2)
			)
			self.upsample_blocks.append(stage)

		unpatchify_stem = nn.Sequential(
			LayerNorm2d(stage_dims[1][0]),
			nn.ConvTranspose2d(stage_dims[1][0], stage_dims[0][-1], kernel_size=4, stride=4),
		)
		self.upsample_blocks.append(unpatchify_stem)

		self.stages = nn.ModuleList()
		for i in reversed(range(len(stage_dims))):
			stage = nn.Sequential(
				*[ConvNextBlock(stage_dims[i][j], stage_dims[i][j-1]) \
				  for j in reversed(range(1, len(stage_dims[i])))]
			)
			self.stages.append(stage)

		self.readout = nn.Conv2d(stage_dims[0][0], output_dims, kernel_size=1)

	def forward(self, x):
		for i in range(len(self.stages)):
			x = self.upsample_blocks[i](x)
			x = self.stages[i](x)
		x = self.readout(x)
		return x

class VAE(nn.Module):
	def __init__(self,
				 in_channels=4,
				 out_channels=2,
				 num_latent=16,
				 input_size=(236,200),
				 stage_dims=[[32,32],[64,64], [128,128]], 
				 **kwargs):
		super(VAE, self).__init__()
		
		if stage_dims[-1][-1] != num_latent:
			stage_dims[-1].append(num_latent)
		
		self.encoder = Encoder(in_channels, stage_dims)
		self.decoder = Decoder(num_latent, out_channels, stage_dims)
		
		self.num_latent = num_latent

		downsample_factor = 2**(len(stage_dims)-1+2)
		self.input_size = input_size
		self.bottleneck_size = (input_size[0] // downsample_factor,
								input_size[1] // downsample_factor)
		
		self.field_to_params = nn.Linear(
			num_latent*np.prod(self.bottleneck_size), 2*num_latent)
		self.params_to_field = nn.Linear(
			num_latent, num_latent*np.prod(self.bottleneck_size))

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			nn.init.xavier_uniform_(m.weight)
			nn.init.constant_(m.bias, 0.)

	def forward(self, x):
		b, c, h0, w0 = x.shape
		x, _ = self.encoder(x)
				
		x = x.reshape([b, -1])
		x = self.field_to_params(x)
		
		mu = x[:, :self.num_latent]
		logvar = x[:, self.num_latent:]

		params = mu
		if self.training:
			params = params + torch.randn_like(params) * (0.5 * logvar).exp()
			
		z = self.params_to_field(params)
		z = F.gelu(z)
		z = z.reshape([b, -1, *self.bottleneck_size])
		
		z = self.decoder(z)
		if (z.shape[-2] != h0) or (z.shape[-1] != w0):
			z = torch.nn.functional.interpolate(z, size=[h0, w0], mode='bilinear')	

		return z, (params, mu, logvar)
