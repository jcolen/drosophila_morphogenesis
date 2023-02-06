import numpy as np
import pandas as pd
import h5py
import sys
import os
import glob
import warnings
import gc

warnings.filterwarnings('ignore')

from tqdm import tqdm

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose

basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))

from utils.dataset import *
from convnext_models import *
	
def residual(u, v):
	umag = torch.linalg.norm(u, dim=-3)
	vmag = torch.linalg.norm(v, dim=-3)
	
	uavg = torch.sqrt(umag.pow(2).mean(dim=(-2, -1), keepdims=True))
	vavg = torch.sqrt(vmag.pow(2).mean(dim=(-2, -1), keepdims=True))

	res = uavg**2 * vmag**2 + vavg**2 * umag**2 - 2 * uavg * vavg * torch.einsum('...ijk,...ijk->...jk', u, v)
	res /= 2 * vavg**2 * uavg**2
	return res

def kld_loss(params, mu, logvar):
	kld = mu.pow(2) + logvar.exp() - logvar - 1
	kld = 0.5 * kld.sum(axis=-1).mean()
	return kld

class VAE_Evolver(VAE):
	'''
	Translation model with a LSTM block at the latent bottleneck
	'''
	def __init__(self, 
				 hidden_size=64,
				 lstm_layers=2,
				 *args, **kwargs):
		super(VAE_Evolver, self).__init__(*args, **kwargs)
		
		self.evolver = nn.LSTM(input_size=self.num_latent,
							   proj_size=self.num_latent,
							   hidden_size=hidden_size,
							   num_layers=lstm_layers,
							   batch_first=True)

	def forward(self, x, t=5):
		b, c, h0, w0 = x.shape
		x, _ = self.encoder(x)
				
		x = x.reshape([b, -1])
		x = self.field_to_params(x)
		
		mu = x[:, :self.num_latent]
		logvar = x[:, self.num_latent:]

		params = mu
		if self.training:
			params = params + torch.randn_like(params) * (0.5 * logvar).exp()
		
		params = params[:, None] #Input sequence length 1 - forecast from initial conditions
		
		params_list = []
		params_list.append(params)
		params, hidden_state = self.evolver(params)
		params_list.append(params)

		while len(params_list) < t:
			params1, hidden_state = self.evolver(params, hidden_state)
			params = params + params1
			params_list.append(params)

		params_list = torch.cat(params_list, dim=1)
		b, t, _ = params_list.shape
		params_list = params_list.reshape([b*t, -1])
			
		z = self.params_to_field(params_list)
		z = F.gelu(z)
		z = z.reshape([b*t, -1, *self.bottleneck_size])
		
		z = self.decoder(z)
		if (z.shape[-2] != h0) or (z.shape[-1] != w0):
			z = torch.nn.functional.interpolate(z, size=[h0, w0], mode='bilinear')	

		z = z.reshape([b, t, *z.shape[-3:]])

		return z, (params, mu, logvar)

class SequenceDataset(AlignedDataset):
	def __init__(self, 
				 max_len=5,
				 *args,
				 **kwargs):
		super(SequenceDataset, self).__init__(*args, **kwargs)

		self.max_len = max_len
		#Repackage dataset to account for sequences
		self.seq_df = pd.DataFrame()
		for eId in self.df.embryoID.unique():
			#Drop last max_len-1 from sequence
			sub = self.df[self.df.embryoID == eId]
			self.seq_df = self.seq_df.append(sub.iloc[:-max_len+1])
	
	def __len__(self):
		return len(self.seq_df)

	def __getitem__(self, idx):
		sample = {}
		index = self.seq_df.index[idx]
		for i in range(self.max_len):
			si = super(SequenceDataset, self).__getitem__(index+i)
			for key in si:
				if torch.is_tensor(si[key]):
					if key in sample: sample[key] = torch.cat([sample[key], si[key][None]])
					else: sample[key] = si[key][None]
				else:
					if key in sample: sample[key].append(si[key])
					else: sample[key] = [si[key]]

		return sample

def run_train(dataset,
			  model_kwargs,
			  dl_kwargs,
		      logdir='/project/vitelli/jonathan/REDO_fruitfly/tb_logs/',
			  grad_clip=0.5,
			  epochs=250):

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	val_size = len(dataset) // 5
	train, val = random_split(dataset, [len(dataset)-val_size, val_size])
	val_indices = val.indices
	val_df = dataset.df.iloc[val_indices]
	train_loader = DataLoader(train, **dl_kwargs)
	val_loader = DataLoader(val, **dl_kwargs)

	model = VAE(**model_kwargs)
	model.to(device)
	print(model_kwargs['input'], model_kwargs['output'])
	optimizer = torch.optim.Adam(
		filter(lambda p: p.requires_grad, model.parameters()), 
		lr=model_kwargs['lr'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

	model_logdir = os.path.join(logdir, 
		'_'.join([model.__class__.__name__,str(model_kwargs['input']), model_kwargs['output']]))
	if not os.path.exists(model_logdir):
		os.mkdir(model_logdir)

	best_res = 1e5
	
	for epoch in range(epochs):
		with tqdm(train_loader, unit='batch') as ttrain:
			for batch in ttrain:
				if isinstance(model_kwargs['input'], list):
					x = torch.cat([batch[i] for i in model_kwargs['input']], axis=-3).to(device)
				else:
					x = batch[model_kwargs['input']].to(device)
				y0 = batch[model_kwargs['output']].to(device)

				optimizer.zero_grad()
				y, pl = model.forward(x)
				res = residual(y, y0).mean()
				mag = (y.pow(2).sum(dim=(2,3)) - y0.pow(2).sum(dim=(2,3))).abs().mean()
				kld = kld_loss(*pl)
				loss = res + model_kwargs['alpha'] * mag + model_kwargs['beta'] * kld
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
				optimizer.step()

		val_loss = 0.
		res_val = 0.
		mag_val = 0.
		kld_val = 0.
		with torch.no_grad():
			with tqdm(val_loader, unit='batch') as tval:
				for batch in tval:
					if isinstance(model_kwargs['input'], list):
						x = torch.cat([batch[i] for i in model_kwargs['input']], axis=-3).to(device)
					else:
						x = batch[model_kwargs['input']].to(device)
					y0 = batch[model_kwargs['output']].to(device)
					
					y, pl = model(x)
					res = residual(y, y0).mean()
					mag = (y.pow(2).sum(dim=(2,3)) - y0.pow(2).sum(dim=(2,3))).abs().mean()
					kld = kld_loss(*pl)
					loss = res + model_kwargs['alpha'] * mag + model_kwargs['beta'] * kld
					val_loss += loss.item() / len(val_loader)
					res_val += res.item() / len(val_loader)
					mag_val += mag.item() / len(val_loader)
					kld_val += kld.item() / len(val_loader)

		scheduler.step(val_loss)
	
		outstr = 'Epoch %d\tVal Loss=%g' % (epoch, val_loss)
		outstr += '\tRes=%g\tKLD=%g' % (res_val, kld_val)
		print(outstr)
		if res_val < best_res:
			save_dict = {
				'state_dict': model.state_dict(),
				'hparams': model_kwargs,
				'epoch': epoch,
				'loss': val_loss,
				'val_df': val_df,
			}
			torch.save(
				save_dict, 
				os.path.join(model_logdir, 'beta=%.2g.ckpt' % model_kwargs['beta']))
			best_res = res_val

from argparse import ArgumentParser
if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--num_latent', type=int, default=32)
	parser.add_argument('--alpha', type=float, default=0)
	parser.add_argument('--beta', type=float, default=1e-3)
	parser.add_argument('--lr', type=float, default=1e-4)
	model_kwargs = vars(parser.parse_args())

	transform=Compose([Reshape2DField(), ToTensor()])
	dl_kwargs = dict(
		num_workers=2, 
		batch_size=8, 
		shuffle=True, 
		pin_memory=True
	)
	
	cad = AtlasDataset('WT', 'ECad-GFP', 'tensor2D', transform=transform)
	sqh = AtlasDataset('WT', 'sqh-mCherry', 'tensor2D', transform=transform)
	vel = AtlasDataset('WT', 'ECad-GFP', 'velocity2D', transform=transform)
	dataset = AlignedDataset(
		datasets=[sqh, cad, vel], 
		key_names=['sqh', 'cad', 'vel'])

	model_kwargs['stage_dims'] = [[32,32],[64,64],[128,128],[256,256]]
	model_kwargs['in_channels'] = 8
	model_kwargs['out_channels'] = 2
	model_kwargs['input'] = ['sqh', 'cad']
	model_kwargs['output'] = 'vel'
	run_train(dataset, model_kwargs, dl_kwargs)
	
	model_kwargs['in_channels'] = 4
	model_kwargs['input'] = 'sqh'
	run_train(dataset, model_kwargs, dl_kwargs)

	model_kwargs['input'] = 'cad'
	run_train(dataset, model_kwargs, dl_kwargs)

	rnt = AtlasDataset('WT', 'Runt', 'raw2D', transform=transform)
	vel = AtlasDataset('WT', 'Runt', 'velocity2D', transform=transform)
	dataset = AlignedDataset(
		datasets=[rnt, vel],
		key_names=['rnt', 'vel'])

	model_kwargs['in_channels'] = 1
	model_kwargs['input'] = 'rnt'
	run_train(dataset, model_kwargs, dl_kwargs)

	hst = AtlasDataset('WT', 'histone-RFP', 'raw2D', transform=transform, drop_no_time=False)
	vel = AtlasDataset('WT', 'histone-RFP', 'velocity2D', transform=transform, drop_no_time=False)
	dataset = AlignedDataset(
		datasets=[hst, vel],
		key_names=['hst', 'vel'])

	model_kwargs['input'] = 'hst'
	run_train(dataset, model_kwargs, dl_kwargs)
