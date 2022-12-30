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

from dataset import *
from convnext_models import *

from translation_vae import residual, kld_loss
import math

'''
Log PDF of Gaussian Distribution (mu, logvar) evaluated at x
'''
def log_density_gaussian(z, mu, logvar):
	norm = -0.5 * (logvar + math.log(2 * np.pi))
	inv_var = torch.exp(-logvar)
	log_density = norm - 0.5 * (inv_var * (z - mu)**2)
	return log_density
	

'''
Decompose KL into three terms

"Index-Code Mutual Information"
Mutual information between data and latent space
I[z; x] = KL[q(z, x) || q(x)q(z) ] = E_x [ KL[ q(z|x) || q(z) ] ]

"Total Correlation Loss"
Latent distribution should be factorizable
KL[ q(z) || \prod_i q(z_i) ]

"Dimension-wise KL"
Each latent distribution should conform to the prior
\sum_i KL[ q(z_i) || p(z_i) ] 
'''
def tc_vae_loss(z, mu, logvar, dataset_size):
	b, l = z.shape
	
	#Calculate log q(z|x)
	log_q_zx = log_density_gaussian(z, mu, logvar).sum(dim=1)

	#Calculate log p(z)
	zeros = torch.zeros_like(z)
	log_pz = log_density_gaussian(z, zeros, zeros).sum(dim=1)

	#Build a matrix of log q(z)
	mat_log_qz = log_density_gaussian(
		z.view(b, 1, l),
		mu.view(1, b, l),
		logvar.view(1, b, l)
	)

	#Importance weighting - minibatch stratified
	# E[log q(z)] ~ 1/M \sum_i [ log [ 1 / (NM) \sum_j q( z(n_i) | n_j )
	# Here M = batch size, N = dataset size
	N = dataset_size
	M = b - 1
	strat_weight = (N-M) / (N*M)
	importance_weights = torch.Tensor(b, b).fill_(1/M)
	importance_weights.view(-1)[::b] = 1/N
	importance_weights.view(-1)[1::b] = strat_weight
	importance_weights[b-2, 0] = strat_weight
	log_iw_mat = importance_weights.log().to(z.device)

	#Weight the log q(z) matrix by these importances
	#mat_log_qz = mat_log_qz + log_iw_mat.view(b, b, 1)


	log_qz = torch.logsumexp(mat_log_qz.sum(dim=2), dim=1, keepdim=False)
	log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(dim=1)

	mi_loss = (log_q_zx - log_qz).mean()
	tc_loss = (log_qz - log_prod_qzi).mean()
	dw_kl_loss = (log_prod_qzi - log_pz).mean()

	print(mi_loss.item(), tc_loss.item(), dw_kl_loss.item())

	return mi_loss, tc_loss, dw_kl_loss


'''
Sweeping on C didn't seem to work, so instead we'll sweep on beta

Following arxiv:1802.04942 we'll use the total-correlation VAE approach
This should enable learning disentangled representations in the latent space
Basically, the joint distribution should be factorizable into a product of marginals
'''
def run_beta_sweep(dataset, 
				   logdir, 
				   model_kwargs, 
				   dl_kwargs,
				   reconstruction_loss=residual,
				   vae_loss=tc_vae_loss,
				   grad_clip=0.5,
				   epochs=100,
				   save_every=25,
				   log_beta_max=0,
				   log_beta_min=-5):
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

	model_logdir = os.path.join(logdir, 
		'_'.join(['BetaTCVAE',model.__class__.__name__,model_kwargs['input'], model_kwargs['output']]))
	if not os.path.exists(model_logdir):
		os.mkdir(model_logdir)

	log_beta = log_beta_max
	log_beta_step = (log_beta_min-log_beta_max) / epochs / len(train_loader)

	best_rec = 10
	
	for epoch in range(epochs):
		for bb, batch in enumerate(train_loader):
			if isinstance(input, list):
				x = torch.cat([batch[i] for i in model_kwargs['input']], axis=-3).to(device)
			else:
				x = batch[model_kwargs['input']].to(device)
			y0 = batch[model_kwargs['output']].to(device)
			optimizer.zero_grad()
			y, pl = model.forward(x)
			rec = reconstruction_loss(y, y0).mean()
			mi, tc, dw_kl = vae_loss(*pl, dataset_size=len(train))
			vae = model_kwargs['alpha'] * mi + \
				  np.power(10., log_beta) * tc + \
				  model_kwargs['gamma'] * dw_kl
			loss = rec + vae
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			optimizer.step()

			log_beta = log_beta + log_beta_step

		val_loss = 0.
		rec_val = 0.
		mi_val = 0.
		tc_val = 0.
		dw_kl_val=0.
		with torch.no_grad():
			for bb, batch in enumerate(val_loader):
				if isinstance(input, list):
					x = torch.cat([batch[i] for i in model_kwargs['input']], axis=-3).to(device)
				else:
					x = batch[model_kwargs['input']].to(device)
				y0 = batch[model_kwargs['output']].to(device)
				
				y, pl = model(x)
				rec = reconstruction_loss(y, y0).mean()
				mi, tc, dw_kl = vae_loss(*pl, dataset_size=len(train))
				vae = model_kwargs['alpha'] * mi + \
					  np.power(10., log_beta) * tc + \
					  model_kwargs['gamma'] * dw_kl
				loss = rec + vae
				val_loss += loss.item() / len(val_loader)
				rec_val += rec.item() / len(val_loader)
				mi_val += mi.item() / len(val_loader)
				tc_val += tc.item() / len(val_loader)
				dw_kl_val += dw_kl.item() / len(val_loader)
	
		outstr = 'Epoch %d\t' % (epoch)
		outstr += '\tReconstruction=%g\tbeta=%g' % (rec_val, np.power(10, log_beta))
		outstr += '\tMI=%g\tTC=%g\tKL(dim)=%g' % (mi_val, tc_val, dw_kl_val)
		print(outstr)
		if epoch > 0 and epoch % save_every == 0:
			if rec_val < best_rec:	#If loosening the bottleneck improved reconstruction
				save_dict = {
					'state_dict': model.state_dict(),
					'hparams': model_kwargs,
					'epoch': epoch,
					'loss': val_loss,
					'val_df': val_df,
				}
				torch.save(
					save_dict, 
					os.path.join(model_logdir, 'log10beta=%.2g.ckpt' % log_beta))
				best_rec = rec_val
	
if __name__ == '__main__':
	transform=Compose([Reshape2DField(), ToTensor()])
	dl_kwargs = dict(
		num_workers=2, 
		batch_size=8, 
		shuffle=True, 
		pin_memory=True
	)

	
	model_kwargs = dict(
		lr=1e-4,
		num_latent=16,
		stage_dims=[[32,32],[64,64],[128,128],[256,256]],
		alpha=1.,
		gamma=1.
	)

	logdir = '/project/vitelli/jonathan/REDO_fruitfly/tb_logs/'
	
	cad = AtlasDataset('WT', 'ECad-GFP', 'tensor2D', transform=transform)
	sqh = AtlasDataset('WT', 'sqh-mCherry', 'tensor2D', transform=transform)
	vel = AtlasDataset('WT', 'ECad-GFP', 'velocity2D', transform=transform)
	dataset = AlignedDataset([sqh, cad, vel], ['sqh', 'cad', 'vel'])
	
	model_kwargs['in_channels'] = 2
	model_kwargs['out_channels'] = 2
	model_kwargs['input'] = 'vel'
	model_kwargs['output'] = 'vel'
	run_beta_sweep(dataset, logdir, model_kwargs, dl_kwargs, log_beta_max=0., log_beta_min=-4.)	
	
	'''	
	model_kwargs['in_channels'] = 4
	model_kwargs['out_channels'] = 4
	model_kwargs['input'] = 'sqh'
	model_kwargs['output'] = 'sqh'
	run_beta_sweep(dataset, logdir, model_kwargs, dl_kwargs, log_beta_max=-1., log_beta_min=-6.)

	model_kwargs['input'] = 'cad'
	model_kwargs['output'] = 'cad'
	run_beta_sweep(dataset, logdir, model_kwargs, dl_kwargs, log_beta_max=-1., log_beta_min=-6.)

	runt = AtlasDataset('WT', 'Runt', 'raw2D', transform=transform)
	dataset = AlignedDataset([runt], ['runt'])
	model_kwargs['input'] = 'runt'
	model_kwargs['output'] = 'runt'
	model_kwargs['in_channels'] = 1
	model_kwargs['out_channels'] = 1
	run_beta_sweep(dataset, logdir, model_kwargs, dl_kwargs, log_beta_max=-3., log_beta_min=-8.)
	'''	
