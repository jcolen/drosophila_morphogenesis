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

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose

basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))

from dataset import *
from convnext_models import *
	
def residual(u, v):
	umag = torch.linalg.norm(u, dim=-3)
	vmag = torch.linalg.norm(v, dim=-3)
	
	uavg = torch.sqrt(umag.pow(2).mean(dim=(-2, -1), keepdims=True))
	vavg = torch.sqrt(vmag.pow(2).mean(dim=(-2, -1), keepdims=True))

	res = uavg**2 * vmag**2 + vavg**2 * umag**2 -			2 * uavg * vavg * torch.einsum('...ijk,...ijk->...jk', u, v)
	res /= 2 * vavg**2 * uavg**2
	return res

def kld_loss(params, logvar):
	kld = params.pow(2) + logvar.exp() - logvar - 1
	kld = kld.sum(axis=-1).mean()
	return 0.5 * kld

class TranslationVAE(pl.LightningModule):
	def __init__(self,
				 input='sqh',
				 output='vel',
				 in_channels=4,
				 out_channels=2,
				 lr=1e-3,
				 weight_decay=1e-3,
				 beta=1e-2,
				 num_latent=16,
				 input_size=(236,200),
				 stage_dims=[[32,32],[64,64],[128,128]],
				 *args, **kwargs):
		super(TranslationVAE, self).__init__()
		self.model = VAE(
			num_latent=num_latent, 
			in_channels=in_channels,
			out_channels=out_channels,
			input_size=input_size,
			stage_dims=stage_dims,
			**kwargs)
		self.save_hyperparameters(
			'input', 'in_channels', 'output', 'out_channels', 'num_latent',
			'lr', 'weight_decay', 'beta', 'input_size', 'stage_dims')
	
	def getxy(self, batch):
		if isinstance(input, list):
			x = torch.cat([batch[i] for i in self.hparams['input']], axis=-3)
		else:
			x = batch[self.hparams['input']]
		y0 = batch[self.hparams['output']]
		return x, y0

	def compute_loss(self, batch):
		x, y0 = self.getxy(batch)
		y, pl = self(x)
		res_loss = residual(y, y0).mean()
		mse_loss = F.mse_loss(y, y0)
		vae_loss = kld_loss(*pl)
		return res_loss, mse_loss, vae_loss
		
	def forward(self, x):
		return self.model(x)
 
	def training_step(self, batch, batch_idx):
		res_loss, mse_loss, vae_loss = self.compute_loss(batch)
		self.log('train/res', res_loss)
		self.log('train/mse', mse_loss)
		self.log('train/kld', vae_loss)
		return {'loss': mse_loss + self.hparams['beta'] * vae_loss}
	
	def validation_step(self, batch, batch_idx):
		res_loss, mse_loss, vae_loss = self.compute_loss(batch)
		self.log('validation/res', res_loss)
		self.log('validation/mse', mse_loss)
		self.log('validation/kld', vae_loss)
		return {'loss': mse_loss + self.hparams['beta'] * vae_loss}
	
	def validation_epoch_end(self, outs):
		avg_loss = torch.stack([x['loss'] for x in outs]).mean()
		self.log('hp/val_loss', avg_loss)
	
	def test_step(self, batch, batch_idx):
		x, y0 = self.getxy(batch)
		y, _ = self(x)
		
		mag0 = torch.linalg.norm(y0, dim=-3).mean(dim=(-2, -1)).cpu().numpy()
		mag  = torch.linalg.norm(y, dim=-3).mean(dim=(-2, -1)).cpu().numpy()
		mse = (y0 - y).pow(2).mean(dim=(-3, -2, -1)).cpu().numpy()
		
		res = residual(y, y0).mean(dim=(-2, -1)).cpu().numpy()
		
		return {'df': pd.DataFrame({
			'res': res.flatten(),
			'mse': mse.flatten(),
			'mag0': mag0.flatten(),
			'mag': mag.flatten(),
			'time': batch['time'].cpu().numpy(),
			'embryoID': batch['embryoID'].cpu().numpy(),
			'genotype': batch['genotype'],
		})}

	def test_epoch_end(self, outs):
		self.test_df = pd.concat([x['df'] for x in outs], axis=0)
		
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams['lr'])
		return {'optimizer': optimizer}			  
	

if __name__ == '__main__':
	transform=Compose([Reshape2DField(), ToTensor()])
	#cad = AtlasDataset('WT', 'ECad-GFP', 'tensor2D', transform=transform)
	#sqh = AtlasDataset('WT', 'sqh-mCherry', 'tensor2D', transform=transform)
	#vel = AtlasDataset('WT', 'ECad-GFP', 'velocity2D', transform=transform)
	#dataset = AlignedDataset([sqh, cad, vel], ['sqh', 'cad', 'vel'])

	eve = AtlasDataset('WT', 'Even_Skipped', 'raw2D', transform=transform)
	dataset = AlignedDataset([eve], ['eve'])
	

	betas = np.power(10., np.arange(-4, -1))
	betas = np.power(10., np.arange(-1, 1))
	betas = np.power(10., np.arange(-6, -1))
	betas = np.power(10., np.arange(-10, -1))
	print(betas)

	for beta in betas:
		print(beta)
		test_size = len(dataset) // 5
		kwargs = dict(num_workers=0, batch_size=8, shuffle=True, pin_memory=True)
		train, test = random_split(dataset, [len(dataset)-test_size, test_size])
		train_loader = DataLoader(train, **kwargs)
		test_loader = DataLoader(test, **kwargs)

		model = TranslationVAE(
			lr=1e-4, 
			weight_decay=0., 
			beta=beta,
			num_latent=16,
			input='runt',
			in_channels=1,
			output='runt',
			out_channels=1,
			stage_dims=[[32,32],[64,64],[128,128],[256,256]],
		)

		kwargs = {
			'gpus': 1,
			'max_epochs': 1000,
			'accumulate_grad_batches': 1,
			'checkpoint_callback': True,
			'log_every_n_steps': 5,
			'flush_logs_every_n_steps': 50,
			'gradient_clip_val': 0.5,
			'fast_dev_run': False,
		}
		trainer = pl.Trainer(**kwargs, 
			callbacks=[pl.callbacks.early_stopping.EarlyStopping(monitor='hp/val_loss', patience=10, mode='min')]
		)
		logdir = '/project/vitelli/jonathan/REDO_fruitfly/tb_logs/'
		trainer.logger = pl.loggers.TensorBoardLogger(
			save_dir=logdir,
			name='_'.join([model.__class__.__name__,model.hparams['input'],model.hparams['output']]),
			default_hp_metric=False)
		trainer.fit(model, train_loader, test_loader)
		trainer.test(model, test_loader, verbose=False);
		model.test_df.to_csv(os.path.join(trainer.logger.log_dir, 'test_df.csv'));

		gc.collect()
