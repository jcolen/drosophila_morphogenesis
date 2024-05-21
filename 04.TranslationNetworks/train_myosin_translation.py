import os
import sys
import torch
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

from flow_dataset import *
from morphogenesis.flow_networks.translation_models import VAE, MaskedVAE

def residual(u, v):
    '''
    We assume u is the INPUT and v is the TARGET
    Using residual metric from Sebastian's eLife paper to track how flow configuration is predicted
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
    parser.add_argument('--logdir', type=str, default='./tb_logs')
    parser.add_argument('--use_pmg_cf_mask', action='store_true')
    parser.add_argument('--edge_mask', type=int, default=15)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('Using device: ', device)

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
    dataset = FlowDataset(transform=transform)

    # Split on embryos
    df = dataset.df.copy()
    embryos = df.embryoID.unique()
    train, val = train_test_split(embryos, test_size=0.5, random_state=42)
    print('Train embryos: ', train)
    print('Val embryos: ', val)

    # Find dataset indices for each embryo
    train_idxs = df[df.embryoID.isin(train)].index.values
    val_idxs = df[df.embryoID.isin(val)].index.values
    train = Subset(deepcopy(dataset), train_idxs)
    val = Subset(deepcopy(dataset), val_idxs)
    print('Train size: ', len(train))
    print('Val size: ', len(val))


    train_loader = DataLoader(train, num_workers=2, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, num_workers=2, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    '''
    Build the model
    '''
    if args.use_pmg_cf_mask:
        mask = np.load('../flydrive/Masks/pmg_cf_mask.npy')
    else:
        mask = None
    
    model = MaskedVAE(in_channels=args.in_channels,
                      out_channels=args.out_channels,
                      num_latent=args.num_latent,
                      stage_dims=[[32,32],[64,64],[128,128],[256,256]],
                      dv_min=args.edge_mask,
                      dv_max=-args.edge_mask,
                      ap_min=args.edge_mask,
                      ap_max=-args.edge_mask,
                      mask=mask)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, min_lr=1e-5)

    '''
    Train the model
    '''

    savename = f'{model.__class__.__name__}_{args.input}_beta={args.beta:.2g}_split=embryo'
    print(savename)

    best_val = 1e5
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

        outstr	= f'Epoch {epoch:03d} Val Loss = {val_loss:.3f} '
        outstr += f'Res = {res_val:.3f} MSE = {mse_val:.3f} KLD = {kld_val:.3f}'
        print(outstr)
        if res_val < best_val:
            save_dict = {
                'state_dict': model.state_dict(),
                'hparams': vars(args),
                'epoch': epoch,
                'loss': val_loss,
                'mse': np.mean(mses),
                'mse_std': np.std(mses),
                'res': np.mean(residuals),
                'res_std': np.std(residuals),
            }
            torch.save(save_dict, f'{args.logdir}/{savename}')
            best_val = res_val