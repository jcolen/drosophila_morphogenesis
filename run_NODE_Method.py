import numpy as np
import pandas as pd
import h5py
import sys
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.transforms import Compose
from torchdiffeq import odeint_adjoint as odeint
from scipy.interpolate import interp1d

basedir = '/project/vitelli/jonathan/REDO_fruitfly/'
sys.path.insert(0, os.path.join(basedir, 'src'))

from utils.translation_utils import *
from utils.decomposition_utils import *
from utils.dataset import *
from atlas_processing.anisotropy_detection import cell_size

import warnings
warnings.filterwarnings('ignore')

class TrajectoryDataset(Dataset):
    def __init__(self,
                 datasets,
                 dorsal_path='Public/dorsal_mask_ELLIPSE_A=0.5_B=0.25_advected_RESIZED.npy',
                 dorsal_time='Public/dorsal_mask_time.npy',
                 dorsal_sigma=10):
        super(TrajectoryDataset, self).__init__()
        self.datasets = datasets
        
        df = pd.DataFrame(columns=['embryoID', 'time', 'eIdx', 'key'])
        for key in self.datasets:
            dataset = self.datasets[key]

            dfi = dataset.df
            dfi = dfi.drop(['folder', 'tiff'], axis=1)
            dfi['time'] = dfi['time'].astype(int)
            dfi['key'] = [[key] for _ in range(len(dfi))]
            
            df = df.merge(dfi, on=['embryoID', 'time', 'eIdx'], how='outer')
            df.loc[df.key_x.isnull(), 'key_x'] = df.loc[df.key_x.isnull(), 'key_x'].apply(lambda x: [])
            df.loc[df.key_y.isnull(), 'key_y'] = df.loc[df.key_y.isnull(), 'key_y'].apply(lambda x: [])
            df['key'] = df.key_x + df.key_y
            df = df.drop(['key_x', 'key_y'], axis=1)
        
        self.df = pd.DataFrame()
        for eId in df.embryoID.unique():
            sub = df[df.embryoID == eId]
            sub['max_len'] = np.max(sub.time) - sub.time
            self.df = self.df.append(sub[sub.max_len > 1], ignore_index=True)
        
        self.dorsal_source = interp1d(
            np.load(dorsal_time, mmap_mode='r'),
            gaussian_filter(np.load(dorsal_path, mmap_mode='r'), sigma=(0, dorsal_sigma, dorsal_sigma))[:, None],
            axis=0,
            fill_value='extrapolate',
        )
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        embryoID = row.embryoID
        
        seq_len = min(int(np.rint(np.abs(np.random.normal(scale=7)))), row.max_len)
        seq_len = max(seq_len, 2)
        times = np.linspace(row.time, row.time + seq_len, 5).astype(int)
        times = np.unique(times)
        eIdxs = times.copy() - np.min(times) + row.eIdx
        
        sample = {
            'times': torch.from_numpy(times).float(),
        }
        
        for key in self.datasets:
            transform = self.datasets[key].transform
            if key == 'v': #Include all flow data
                data = [transform(self.datasets[key].values[embryoID][eIdx]) for eIdx in range(eIdxs[0], eIdxs[-1]+1)]
            elif key in row.key: #Live-imaged data
                data = [transform(self.datasets[key].values[embryoID][eIdx]) for eIdx in eIdxs]
            else: #Ensemble-averaged data
                df = self.datasets[key].df
                data = []
                for time in times:
                    nearest = df[(df.time - time).abs() < 0.5]
                    if len(nearest) == 0:
                        nearest = df.iloc[(df.time - time).abs().argsort()[:3]]
                    frame = np.mean([transform(self.datasets[key].values[n.embryoID][n.eIdx]) for _, n in nearest.iterrows()], axis=0)
                    data.append(frame)
            sample[key] = torch.from_numpy(np.stack(data)).float()
        
        times = np.arange(times[0], times[-1]+1)
        sample['gamma_ds'] = torch.from_numpy(self.dorsal_source(times))
        
        return sample
                

def gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


class ClosedFlyLoop(nn.Module):
    def __init__(self, sigma=5):
        super(ClosedFlyLoop, self).__init__()
        self.dAP = 2.27
        self.dDV = 2.27
        
        self.gamma_dv = nn.Parameter(torch.FloatTensor([[1., 0.], [0., 0.]])[..., None, None], requires_grad=False)
            
        #D_t c = -0.025 c + 0.064 c trE + 0.261 Gamma^D
        self.cad_coefs = nn.Parameter(torch.FloatTensor([
            0.025, 0.064, 0.261]), requires_grad=True)
        #D_t m = -0.2 m + (1 - 0.4 c) {m, E} + Tr(m) (0.19 Gamma^{DV} + 0.1 m)
        self.myo_coefs = nn.Parameter(torch.FloatTensor([
            0.2, 1., 0.4, 0.19, 0.1]), requires_grad=True)
        
        diff_kernel = gaussian_kernel1d(sigma, 1, radius=int(4*sigma+0.5))[None, None]
        self.diff_kernel = nn.Parameter(torch.FloatTensor(diff_kernel), requires_grad=False)
        self.pad_size = self.diff_kernel.shape[-1] // 2
            
    def diffY(self, x):
        c, h, w = x.shape
        x = x.permute(0, 2, 1)
        x = F.pad(x, (self.pad_size, self.pad_size), mode='circular')
        x = x.view([c*w, 1, h+2*self.pad_size])
        x = F.conv1d(x, self.diff_kernel) / self.dDV
        x = x.view([c, w, h]).permute(0, 2, 1)
        return x
    
    def diffX(self, x):
        c, h, w = x.shape
        x = F.pad(x, (self.pad_size, self.pad_size), mode='reflect')
        x = x.view([c*h, 1, w+2*self.pad_size])
        x = F.conv1d(x, self.diff_kernel) / self.dAP
        x = x.view([c, h, w])
        return x
    
    def forward(self, t, y):
            m = y[:4]
            c = y[4:]

            d1_m = torch.stack([
                self.diffY(m),
                self.diffX(m),
            ], axis=-1).reshape([2, 2, *y.shape[-2:], 2])
            m = m.reshape([2, 2, *y.shape[-2:]])
            
            d1_c = torch.stack([
                self.diffY(c),
                self.diffX(c),
            ], axis=-1).reshape([*y.shape[-2:], 2])
            c = c[0]

            gamma_ds = torch.FloatTensor(self.ds_int(t.item())).to(m.device)
            v = torch.FloatTensor(self.v_int(t.item())).to(m.device)
            d1_v = torch.stack([
                self.diffY(v),
                self.diffX(v)
            ], axis=-1)

            O = -0.5 * (torch.einsum('iyxj->ijyx', d1_v) - \
                        torch.einsum('jyxi->ijyx', d1_v))
            E = 0.5 * (torch.einsum('iyxj->ijyx', d1_v) + \
                       torch.einsum('jyxi->ijyx', d1_v))

            deviatoric = m - 0.5 * torch.einsum('kkyx,ij->ijyx', m, torch.eye(2, device=m.device))

            m_0 = torch.linalg.norm(m, dim=(0, 1), keepdims=True).mean(dim=(2, 3), keepdims=True)
            dev_mag = torch.linalg.norm(deviatoric, dim=(0, 1), keepdims=True)

            devE = torch.einsum('klyx,klyx->yx', deviatoric, E)[None, None]

            E_active = E - torch.sign(devE) * devE * deviatoric / dev_mag**2
            E_active = 0.5 * E_active * dev_mag / m_0 
            E_passive = E - E_active

            mE = torch.einsum('ikyx,kjyx->ijyx', m, E_passive) + \
                 torch.einsum('ikyx,kjyx->ijyx', E_passive, m) 

            cdot =  -1.000 * torch.einsum('iyx,yxi->yx', v, d1_c)
            cdot += -F.relu(self.cad_coefs[0]) * c
            cdot +=  F.relu(self.cad_coefs[1]) * torch.einsum('yx,iyxi->yx', c, d1_v)
            cdot +=  F.relu(self.cad_coefs[2]) * gamma_ds

            trm = torch.einsum('kkyx->yx', m)[None, None]

            mdot =  -1.000 * torch.einsum('kyx,ijyxk->ijyx', v, d1_m)
            mdot -= +1.000 * torch.einsum('ikyx,kjyx->ijyx', O, m)
            mdot -= -1.000 * torch.einsum('ikyx,kjyx->ijyx', m, O)
            mdot += -F.relu(self.myo_coefs[0]) * m
            mdot +=  F.relu(self.myo_coefs[1]) * mE
            mdot += -F.relu(self.myo_coefs[2]) * c[None, None] * mE
            mdot +=  F.relu(self.myo_coefs[3]) * trm * self.gamma_dv
            mdot +=  F.relu(self.myo_coefs[4]) * trm * m

            mdot = mdot.reshape([4, *mdot.shape[-2:]])
            cdot = cdot.reshape([1, *cdot.shape[-2:]])
            ydot = torch.cat([mdot, cdot])
            return ydot
        
def integrate(model, m, c, v, gamma_ds, times):
    '''
    Integrate two fields m, c according to SINDy-identified equations
    Include a co-evolving velocity field and two source fields
    ''' 
    #Set up interpolators for control fields
    times = times.squeeze()
    v = v.squeeze()
    gamma_ds = gamma_ds.squeeze()
    t = np.arange(times[0].item(), times[-1].item()+1)
    v_int = interp1d(t, v.cpu().detach().numpy(), axis=0, fill_value='extrapolate')
    ds_int = interp1d(t, gamma_ds.cpu().detach().numpy(), axis=0, fill_value='extrapolate')
    
    model.v_int = v_int
    model.ds_int = ds_int
    model.t_int = t

    #Set up differentiators
    m = m.reshape([-1, *m.shape[-3:]])
    c = c.reshape([-1, *c.shape[-3:]])
    y0 = torch.cat([m[0], c[0]], dim=0)
    y = odeint(model, y0, times)
    
    m = y[:, :4].reshape([-1, 2, 2, *y.shape[-2:]])
    c = y[:, 4:]
    return m, c, times
    
    
#Train the model
def residual(input, target):
    '''
    Mean-squared error normalized by magnitude of target
    '''
    u = input.reshape([input.shape[0], -1, *input.shape[-2:]])
    v = target.reshape([target.shape[0], -1, *target.shape[-2:]])

    res = torch.pow(u - v, 2).sum(dim=-3)
    mean = torch.linalg.norm(v, axis=-3)
    res /= mean**2

    return res


def run_train(dataset,
              batch_size=8,
              lr=1e-3,
              logdir='/project/vitelli/jonathan/REDO_fruitfly/tb_logs/',
              grad_clip=0.5,
              epochs=100):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    val_size = len(dataset) // 5
    train, val = random_split(dataset, [len(dataset)-val_size, val_size])
    val_indices = val.indices
    val_df = dataset.df.iloc[val_indices]
    train_loader = DataLoader(train, num_workers=4, batch_size=1, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, num_workers=4, batch_size=1, shuffle=True, pin_memory=True)

    model = ClosedFlyLoop()
    model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    model_logdir = os.path.join(logdir, model.__class__.__name__)
    if not os.path.exists(model_logdir):
        os.mkdir(model_logdir)

    best_res = 1e5

    for epoch in range(epochs):
        i = 0
        with tqdm(train_loader, unit='batch') as ttrain:
            for batch in ttrain:
                for key in batch:
                    batch[key] = batch[key].to(device)
                m, c, times = integrate(model, **batch)
                
                if i % batch_size == 0:
                    optimizer.zero_grad()
                    
                res_m = residual(batch['m'][0], m).mean()
                res_c = residual(batch['c'][0], c).mean()
                loss = res_m + res_c
                loss.backward()
                
                if i % batch_size == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    
                i += 1

        val_loss = 0.
        res_m_val = 0.
        res_c_val = 0.
        with torch.no_grad():
            with tqdm(val_loader, unit='batch') as tval:
                for batch in tval:
                    for key in batch:
                        batch[key] = batch[key].to(device)
                    m, c, times = integrate(model, **batch)

                    res_m = residual(batch['m'][0], m).mean()
                    res_c = residual(batch['c'][0], c).mean()
                    loss = res_m + res_c
                    val_loss += loss.item() / len(val_loader)
                    res_m_val += res_m.item() / len(val_loader)
                    res_c_val += res_c.item() / len(val_loader)

        scheduler.step(val_loss)

        outstr = 'Epoch %d\tVal Loss=%g' % (epoch, val_loss)
        outstr += '\tRes Myo=%g\tRes Cad=%g' % (res_m_val, res_c_val)
        print(outstr)
        print('\t', model.cad_coefs.detach().cpu().numpy())
        print('\t', model.myo_coefs.detach().cpu().numpy())
        if val_loss < best_res:
            save_dict = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': val_loss,
                'val_df': val_df,
            }
            torch.save(
                save_dict, 
                os.path.join(model_logdir, 'checkpoint_Res=%g.ckpt' % val_loss))
            best_res = val_loss
            
    return model

if __name__ == '__main__':
	transform = Reshape2DField()   

	genotype, label = 'Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP'
	sqh_dataset = AtlasDataset(genotype, label, 'tensor2D', 
		transform=transform, drop_no_time=False)
	sqh_vel_dataset = AtlasDataset(genotype, label, 'velocity2D', 
		transform=transform, drop_no_time=False)

	genotype, label = 'WT', 'ECad-GFP'

	cad_dataset =  AtlasDataset(genotype, label, 'cyt2D', 
		transform=Compose([transform, Smooth2D(sigma=cell_size)]))
	cad_vel_dataset = AtlasDataset(genotype, label, 'velocity2D', 
		transform=transform)
	
	vel_dataset = sqh_vel_dataset + cad_vel_dataset
	dataset = TrajectoryDataset({
		'm': sqh_dataset,
		'c': cad_dataset,
		'v': vel_dataset,
	})
	model = run_train(dataset, epochs=50)
