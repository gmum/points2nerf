import numpy as np
import os
from os.path import join, exists
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from nerf_helpers import *
from itertools import chain
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import *

from dataset.dataset import NeRFShapeNetDataset

from models.encoder import Encoder
from models.nerf import NeRF
from models.resnet import resnet18
from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP

#Needed for workers for dataloader
from torch.multiprocessing import Pool, Process, set_start_method
set_start_method('spawn', force=True)

import argparse

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)

    parser = argparse.ArgumentParser(description='Start training')
    parser.add_argument('config_path', type=str,
                        help='Relative config path')

    args = parser.parse_args()

    config = None
    with open(args.config_path) as f:
        config = json.load(f)
    assert config is not None

    print(config)

    set_seed(config['seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset = NeRFShapeNetDataset(root_dir=config['data_dir'], classes=config['classes'])

    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                    shuffle=config['shuffle'],
                                    num_workers=8, drop_last=True,
                                    pin_memory=True, generator=torch.Generator(device='cuda'))

    embed_fn, config['model']['TN']['input_ch_embed'] = get_embedder(config['model']['TN']['multires'], config['model']['TN']['i_embed'])

    embeddirs_fn = None
    config['model']['TN']['input_ch_views_embed'] = 0
    if config['model']['TN']['use_viewdirs']:
        embeddirs_fn, config['model']['TN']['input_ch_views_embed']= get_embedder(config['model']['TN']['multires_views'], config['model']['TN']['i_embed'])


    # Create a NeRF network
    nerf = NeRF(config['model']['TN']['D'],config['model']['TN']['W'], 
                config['model']['TN']['input_ch_embed'], 
                config['model']['TN']['input_ch_views_embed'],
                config['model']['TN']['use_viewdirs']).to(device)

    #Hypernetwork
    hnet = ChunkedHMLP(nerf.param_shapes, uncond_in_size=config['z_size'], cond_in_size=0,
                layers=config['model']['HN']['arch'], chunk_size=config['model']['HN']['chunk_size'], cond_chunk_embs=False, use_bias=config['model']['HN']['use_bias']).to(device)

    print(hnet.param_shapes)
    
    #Create encoder: either Resnet or classic
    if config['resnet']==True:
        encoder = resnet18(num_classes=config['z_size']).to(device) 
    else:
        encoder = Encoder(config).to(device) 

    #RAdam because it might help with not collapsing to white background
    optimizer = torch.optim.RAdam(chain(encoder.parameters(), hnet.internal_params), **config['optimizer']['E_HN']['hyperparams'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])
    loss_fn = torch.nn.MSELoss()

    results_dir = config['results_dir']
    os.makedirs(join(dirname,results_dir), exist_ok=True)

    with open(join(results_dir, "config.json"), "w") as file:
        json.dump(config, file, indent=4)

    try:
        losses_r = np.load(join(results_dir, f'losses_r.npy')).tolist()
        print("Loaded reconstruction losses")
        losses_kld = np.load(join(results_dir, f'losses_kld.npy')).tolist()
        print("Loaded KLD losses")
        losses_total = np.load(join(results_dir, f'losses_total.npy')).tolist()
        print("Loaded total losses")
    except:
        print("Haven't found previous loss data. We are assuming that this is a new experiment.")
        losses_r = []
        losses_kld = []
        losses_total = []

    starting_epoch = len(losses_total)

    print("starting epoch:", starting_epoch)

    if(starting_epoch>0):
        print("Loading weights since previous losses were found")
        try:
            hnet.load_state_dict(torch.load(join(results_dir, f"model_hn_{starting_epoch-1}.pt"))) 
            print("Loaded HNet")
            encoder.load_state_dict(torch.load(join(results_dir, f"model_e_{starting_epoch-1}.pt")))
            print("Loaded Encoder")
            scheduler.load_state_dict(torch.load(join(results_dir, f"lr_{starting_epoch-1}.pt")))
            print("Loaded Scheduler")
        except:
            print("Haven't found all previous models.")


    hnet.train()
    encoder.train()

    os.makedirs(join(results_dir, 'samples'), exist_ok=True)

    for epoch in range(starting_epoch, starting_epoch+config['max_epochs'] + 1):
        start_epoch_time = datetime.now()
        
        total_loss = 0.0
        total_loss_r = 0.0
        total_loss_kld = 0.0
        
        for i, (entry, cat, obj_path) in enumerate(dataloader):
            x = []
            y = []
            
            if config['resnet']:
                nerf_Ws, mu, logvar = get_nerf_resnet(entry, encoder, hnet)
            else:
                nerf_Ws, mu, logvar = get_nerf(entry, encoder, hnet)

            #For batch size == 1 hnet doesn't return batch dimension...
            if config['batch_size'] == 1:
                nerf_Ws = [nerf_Ws]

            for j, target_w in enumerate(nerf_Ws):
                render_kwargs_train = get_render_kwargs(config, nerf, target_w, embed_fn, embeddirs_fn)
                
                for p in range(config["poses"]):
                    img_i = np.random.choice(len(entry['images'][j]), 1)
                    target = entry['images'][j][img_i][0].to(device)
                    target = torch.Tensor(target.float())
                    pose = entry['cam_poses'][j][img_i, :3,:4][0].to(device)

                    H = entry["images"][j].shape[1]
                    W = entry["images"][j].shape[2]
                    focal = .5 * W / np.tan(.5 * 0.6911112070083618) 

                    K = np.array([
                    [focal, 0, 0.5*W],
                    [0, focal, 0.5*H],
                    [0, 0, 1]
                    ])
                    
                    #Calculate rays from camera origin
                    rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose.float())) 
                    
                    #Create coordinates array (for ray selection)
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                    
                    #To 1D
                    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                    
                    #Select rays based on random coord selection 
                    select_inds = np.random.choice(coords.shape[0], size=[config['model']['TN']['N_rand'],], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()  # (N_rand, 2)
                    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays = torch.stack([rays_o, rays_d], 0)
                    target_s = target[select_coords[:, 0], select_coords[:, 1]] # (N_rand, 3)


                    img_r, _, _, _ = render(H, W, K, chunk=config['model']['TN']['netchunk'], rays=batch_rays.to(device),
                                                            verbose=True, retraw=True,
                                                            **render_kwargs_train)

                    x.append(target_s)
                    y.append(img_r)

            optimizer.zero_grad()
            x = torch.stack(x)
            y = torch.stack(y)

            loss_r = loss_fn(y, x)

            loss_kld = 0.5 * (torch.exp(logvar) + torch.pow(mu, 2) - 1 - logvar).sum()

            loss = loss_r + loss_kld

            loss.backward()
            optimizer.step()
            
            total_loss_r += loss_r.item()
            total_loss += loss.item()
            total_loss_kld += loss_kld.item()
            
        losses_r.append(total_loss_r)
        losses_kld.append(total_loss_kld)
        losses_total.append(total_loss)

        scheduler.step()

        #Log information, save models etc.
        if epoch % config['i_log'] == 0:
            print(f"Epoch {epoch}: took {round((datetime.now() - start_epoch_time).total_seconds(), 3)} seconds")
            print(f"Total loss: {total_loss}     Loss R: {total_loss_r}     Loss KLD: {total_loss_kld}")
        
        #Compare current reconstruction
        if epoch % config['i_sample'] == 0 or epoch == 0:
            with torch.no_grad():
                render_kwargs_test = {
                    k: render_kwargs_train[k] for k in render_kwargs_train}
                render_kwargs_test['perturb'] = False
                render_kwargs_test['raw_noise_std'] = 0.
                img, _, _, _ = render(H,W,K, chunk=config['model']['TN']['netchunk'], c2w=pose,
                                                    verbose=True, retraw=True,
                                                    **render_kwargs_test)
                f, axarr = plt.subplots(1,2)
                axarr[0].imshow(img.detach().cpu())
                axarr[1].imshow(target.detach().cpu())
                f.savefig(join(results_dir, 'samples', f"epoch_{epoch}.png"))
                plt.close(f)
                
                
        if epoch % config['i_save']==0:  
            torch.save(hnet.state_dict(), join(results_dir, f"model_hn_{epoch}.pt"))
            torch.save(encoder.state_dict(), join(results_dir, f"model_e_{epoch}.pt"))
            torch.save(scheduler.state_dict(), join(results_dir, f"lr_{epoch}.pt"))
            #torch.save(optimizer.state_dict(), join(results_dir, f"opt_{epoch}.pt"))
            
            np.save(join(results_dir, 'losses_r.npy'), np.array(losses_r))
            np.save(join(results_dir, 'losses_kld.npy'), np.array(losses_kld))
            np.save(join(results_dir, 'losses_total.npy'), np.array(losses_total))

            plt.plot(losses_r)
            plt.savefig(os.path.join(results_dir, f'loss_r_plot.png'))
            plt.close()

            plt.loglog(losses_r)
            plt.savefig(os.path.join(results_dir, f'loss_r_plot_log.png'))
            plt.close()

            plt.plot(losses_kld)
            plt.savefig(os.path.join(results_dir, f'loss_kld_plot.png'))
            plt.close()

            plt.plot(losses_total)
            plt.savefig(os.path.join(results_dir, f'loss_total_plot.png'))
            plt.close()
            
