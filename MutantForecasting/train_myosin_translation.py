import os
import sys

from torchvision.transforms import Compose

sys.path.insert(0, '/project/vitelli/jonathan/REDO_fruitfly/release')
from mutant_datasets import *
from utils.vae.convnext_models import VAE

from sklearn.model_selection import train_test_split
from scipy.stats import sem
from copy import deepcopy
from torch.utils.data import DataLoader, random_split, Subset
from argparse import ArgumentParser

'''
Instantaneous translation model
'''
class MaskedVAE(VAE):
	def __init__(self, 
				 *args,
				 dv_min=15,
				 dv_max=-15,
				 ap_min=15, 
				 ap_max=-15,
				 **kwargs):
		super().__init__(*args, **kwargs)

		# Remove data wherever mask is True
		self.mask = torch.ones(self.input_size, dtype=bool)
		self.mask[dv_min:dv_max, ap_min:ap_max] = 0
		self.mask = torch.nn.Parameter(self.mask, requires_grad=False)
	
	def forward(self, x):
		x[..., self.mask] = 0. # Mask regions marked
		return super().forward(x)

def residual(u, v):
	'''
	We assume u is the INPUT and v is the TARGET
	Using residual metric from Sebastian's eLife paper

	'''
	umag = torch.linalg.norm(u, dim=-3)
	vmag = torch.linalg.norm(v, dim=-3)

	uavg = torch.sqrt(umag.pow(2).mean(dim=(-2, -1), keepdims=True))
	vavg = torch.sqrt(vmag.pow(2).mean(dim=(-2, -1), keepdims=True))

	res = uavg**2 * vmag**2 + vavg**2 * umag**2 - 2 * uavg * vavg * torch.einsum('...ijk,...ijk->...jk', u, v)
	denom = 2 * vavg**2 * uavg**2
	denom[denom == 0] += 1
	res /= denom

	return res.mean(dim=(-2, -1)) # Average over space

def mean_squared_error(u, v):
	sse = (u - v).pow(2).sum(dim=-3)
	return sse.mean(dim=(-2, -1)) #Average over space

def kld_loss(params, mu, logvar):
	kld = mu.pow(2) + logvar.exp() - logvar - 1
	kld = 0.5 * kld.sum(axis=-1).mean()
	return kld


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--num_latent', type=int, default=64)
	parser.add_argument('--beta', type=float, default=1e-4)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--input', type=str, default='sqh')
	parser.add_argument('--in_channels', type=int, default=4)
	parser.add_argument('--output', type=str, default='vel')
	parser.add_argument('--out_channels', type=int, default=2)
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--logdir', type=str, default='/project/vitelli/jonathan/REDO_fruitfly/tb_logs/May2024')
	args = parser.parse_args()

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	
	'''
	Build dataset
	'''
	print('Building datasets')
	transform = Compose([
		Reshape2DField(),
		RandomLR(),
		ToTensor()
	])

	#Base datasets
	dataset = torch.utils.data.ConcatDataset([
		WTDataset(transform=transform),
		TwistDataset(transform=transform),
		#TollDataset(transform=transform),
		#SpaetzleDataset(transform=transform),
	])

	# Split on embryos
	full_df = pd.concat([ds.df for ds in dataset.datasets], ignore_index=True)
	embryos = full_df.embryoID.unique()
	train, val = train_test_split(embryos, test_size=0.5, random_state=42)
	print('Train embryos: ', train)
	print('Val embryos: ', val)
	
	# Find dataset indices for each embryo
	train_idxs = full_df[full_df.embryoID.isin(train)].index.values
	val_idxs = full_df[full_df.embryoID.isin(val)].index.values
	train = Subset(deepcopy(dataset), train_idxs)
	val = Subset(deepcopy(dataset), val_idxs)
	print('Train size: ', len(train))
	print('Val size: ', len(val))
	

	train_loader = DataLoader(train, num_workers=4, batch_size=args.batch_size, shuffle=True, pin_memory=True)
	val_loader = DataLoader(val, num_workers=4, batch_size=args.batch_size, shuffle=True, pin_memory=True)

	'''
	Build the model
	'''
	model = MaskedVAE(in_channels=args.in_channels,
					  out_channels=args.out_channels,
					  num_latent=args.num_latent,
					  stage_dims=[[32,32],[64,64],[128,128],[256,256]])

	model.to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, min_lr=1e-5)
	
	'''
	Train the model
	'''
	
	savename = f'{model.__class__.__name__}_{args.input}_beta={args.beta:.2g}_split=embryo'
	savename = f'{model.__class__.__name__}_{args.input}_beta={args.beta:.2g}_split=embryo_WTTwist'
	print(savename)

	best_res = 1e5
	for epoch in range(args.epochs):
		model.train()
		train_loss = 0.
		for batch in train_loader:
			x = batch[args.input].to(device)
			y0 = batch[args.output].to(device)
			y, pl = model(x)
			
			kld = kld_loss(*pl)
			#res = residual(y, y0).mean()
			#loss = res + args.beta * kld
			
			mse = mean_squared_error(y, y0).mean()
			loss = mse + args.beta * kld
			
			train_loss += loss.item()
			
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
			optimizer.step()

		model.eval()
		val_loss = 0.
		res_val, mse_val, kld_val = 0., 0., 0.
		residuals, mses = [], []
		
		with torch.no_grad():
			for batch in val_loader:
				x = batch[args.input].to(device)
				y0 = batch[args.output].to(device)
				y, pl = model(x)

				kld = kld_loss(*pl)
				res = residual(y, y0).mean()
				mse = mean_squared_error(y, y0).mean()

				loss = mse + args.beta * kld
				val_loss += loss.item() / len(val_loader)
				res_val += res.item() / len(val_loader)
				mse_val += mse.item() / len(val_loader)
				kld_val += kld.item() / len(val_loader)

				residuals.append(res.item())
				mses.append(mse.item())

		scheduler.step(val_loss)

		res_val = np.mean(residuals)
		
		outstr	= f'Epoch {epoch:03d} Val Loss = {val_loss:.3f} '
		outstr += f'Res = {res_val:.3f} MSE = {mse_val:.3f} KLD = {kld_val:.3f}'
		print(outstr)
		if res_val < best_res:
			save_dict = {
				'state_dict': model.state_dict(),
				'hparams': vars(args),
				'epoch': epoch,
				'loss': val_loss,
				'mse': np.mean(mses),
				'mse_std': np.std(mses),
				'res': np.mean(residuals),
				'res_std': np.std(residuals),
				'val_df': full_df.iloc[val.indices],
			}
			torch.save(save_dict, f'{args.logdir}/{savename}')
			best_res = res_val
