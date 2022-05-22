import urllib
import shutil
import torch
import torch.nn as nn
from os import listdir, makedirs, remove
from os.path import exists, join
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from nerf_helpers import *

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_nerf(entry, encoder, hnet):
    points = entry["data"]
    points = points.to(device, dtype=torch.float)

    if points.size(-1) == 6: 
        points.transpose_(points.dim() - 2, points.dim() - 1)

    code, mu, logvar = encoder(points)
    nerf_W = hnet(uncond_input=code)
    
    return nerf_W, mu, logvar

def get_nerf_resnet(entry, encoder, hnet):
    img_i = np.random.choice(24, len(entry['images'])) #get 0..max_imgs random ids for each batch
    images = [imgs[i] for imgs, i in zip(entry["images"], img_i)] #get those images

    images = torch.stack(images)
    
    images = images.to(device, dtype=torch.float)
    images.transpose_(1, -1)
    code, mu, logvar = encoder(images)

    nerf_W = hnet(uncond_input=code)
    
    return nerf_W, mu, logvar

def get_code(entry, encoder):
    points = entry["data"]
    points = points.to(device, dtype=torch.float)

    if points.size(-1) == 6: 
        points.transpose_(points.dim() - 2, points.dim() - 1)

    code, mu, logvar = encoder(points)
    
    return code

def get_nerf_from_code(hnet, code):
    
    nerf_W = hnet(uncond_input=code)
    return nerf_W

def get_render_kwargs(config, nerf, nerf_w, embed_fn, embeddirs_fn):
    
    render_kwargs = {
            'network_query_fn' : lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                            embed_fn=embed_fn,
                                            embeddirs_fn=embeddirs_fn,
                                            netchunk=config['model']['TN']['netchunk']),
            'perturb' : config['model']['TN']['peturb'],
            'N_importance' : config['model']['TN']['N_importance'],
            'network_fine' : None,
            'N_samples' : config['model']['TN']['N_samples'],
            'network_fn' : lambda x: nerf(x,weights=nerf_w),
            'use_viewdirs' : config['model']['TN']['use_viewdirs'],
            'white_bkgd' : config['model']['TN']['white_bkgd'],
            'raw_noise_std' : config['model']['TN']['raw_noise_std'],
            'near': 2.,
            'far': 6.,
            'ndc': False
        }
    
    return render_kwargs
