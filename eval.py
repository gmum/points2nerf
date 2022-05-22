import numpy as np
import os
from os.path import join, exists
import matplotlib.pyplot as plt
import pandas as pd
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

import trimesh, mcubes

from dataset.dataset import NeRFShapeNetDataset

from models.encoder import Encoder
from models.nerf import NeRF
from models.resnet import resnet18
from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP

from ChamferDistancePytorch.fscore import fscore

import open3d as o3d

import argparse

import ChamferDistancePytorch.chamfer_python as chfp

#Needed for workers for dataloader
from torch.multiprocessing import Pool, Process, set_start_method
set_start_method('spawn', force=True)

def rot_x(angle):
    rx = torch.Tensor([ [1,0,0],
                        [0, math.cos(angle), -math.sin(angle)],
                        [0, math.sin(angle), math.cos(angle)]])
    return rx


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh

def calculate_best_mesh_metrics(obj_path, render_kwargs, save_pc=True, name="1", thresholds=[1,2,3]):

    fscores = [calculate_mesh_metrics(obj_path, render_kwargs, save_pc, name+f'_{t}', t) for t in thresholds]
    return max(fscores, key=lambda x: x[0].item())

def calculate_mesh_metrics(obj_path, render_kwargs, save_pc=True, name="1", threshold = 3):

    with torch.no_grad():
        N = 128
        t = torch.linspace(-1.1, 1.1, N+1)

        query_pts = torch.stack(torch.meshgrid(t, t, t), -1)
        sh = query_pts.shape
        flat = query_pts.reshape([-1,3])

        def batchify(fn, chunk):
            if chunk is None:
                return fn
            def ret(inputs):
                return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
            return ret

        fn = lambda i0, i1 : render_kwargs['network_query_fn'](flat[i0:i1,None,:], viewdirs=None, network_fn=render_kwargs['network_fn'])
        chunk = 1024*16
        raw = torch.cat([fn(i, i+chunk) for i in range(0, flat.shape[0], chunk)], 0)
        raw = torch.reshape(raw, list(sh[:-1]) + [-1])
        sigma = torch.maximum(raw[...,-1], torch.Tensor([0.]))


        vertices, triangles = mcubes.marching_cubes(sigma.cpu().numpy(), threshold)
        mesh = trimesh.Trimesh(vertices / N - .5, triangles)
        
        try:
            entry_mesh = trimesh.load_mesh(obj_path, force='mesh')
            entry_mesh = as_mesh(entry_mesh)

            entry_points = trimesh.sample.sample_surface(entry_mesh, 3000)
            entry_points = torch.from_numpy(entry_points[0]).to(device, dtype=torch.float)

            entry_points = rot_x(math.pi/2)@entry_points.T
            entry_points = entry_points.T
            entry_points = entry_points[None]

            sampled_points = trimesh.sample.sample_surface(mesh, 3000)
            sampled_points = torch.from_numpy(sampled_points[0])[None].to(device, dtype=torch.float)
            
            dist1, dist2, idx1, idx2 = chfp.distChamfer(entry_points, sampled_points)
            cd = (torch.mean(dist1)) + (torch.mean(dist2))
            f_score, precision, recall = fscore(dist1, dist2, 0.01)
        except Exception as e:
            print(e)
            f_score = torch.Tensor([0.0])
            cd = torch.Tensor([0.0])
        
        iou=0
        if save_pc:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(sampled_points.detach().cpu().numpy()[0])
                o3d.io.write_point_cloud(f"./results/pcs/{name}_sampled_points.ply", pcd)

                pcd = o3d.geometry.PointCloud()
                #print(entry_points.detach().cpu().numpy())
                #print(entry_points.detach().cpu().numpy().shape)
                pcd.points = o3d.utility.Vector3dVector(entry_points.detach().cpu().numpy()[0])
                o3d.io.write_point_cloud(f"./results/pcs/{name}_entry_points.ply", pcd)
            except Exception as e:
                print(e)
                print("something went wrong with saving point cloud!")
                raise e
        return f_score, cd #remember the threshold!


def calculate_image_metrics(entry, render_kwargs, metric_fn, count=5):
    x = []
    y = []
    with torch.no_grad():
        for c in range(count):
            img_i = np.random.choice(len(entry['images'][j]), 1)
            target = entry['images'][j][img_i][0].to(device)#entry['images'][j][img_i][0].to(device)
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

            img, _, _, _ = render(H, W, K, chunk=config['model']['TN']['netchunk'], c2w=pose,
                                                        verbose=True, retraw=True,
                                                        **render_kwargs)
            
            x.append(img)
            y.append(target)
        
        x = torch.stack(x)
        y = torch.stack(y)

        metric_val = metric_fn(y, x)

        return metric_val

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    dirname = os.path.dirname(__file__)

    parser = argparse.ArgumentParser(description='Start training HyperRF')
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

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    #config['classes'] = ['cars']

    dataset = NeRFShapeNetDataset(root_dir=config['data_dir'], classes=config['classes'], train=False)

    config['batch_size'] = 1

    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                    shuffle=config['shuffle'],
                                    num_workers=2, drop_last=True,
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

    #Create encoder: either Resnet or classic
    if config['resnet']==True:
        encoder = resnet18(num_classes=config['z_size']).to(device) 
    else:
        encoder = Encoder(config).to(device) 

    results_dir = config['results_dir']
    os.makedirs(join(dirname,results_dir), exist_ok=True)

    with open(join(results_dir, "config_eval.json"), "w") as file:
        json.dump(config, file, indent=4)

    try:
        losses_r = np.load(join(results_dir, f'losses_r.npy')).tolist()
        print("Loaded reconstruction losses")
        losses_kld = np.load(join(results_dir, f'losses_kld.npy')).tolist()
        print("Loaded KLD losses")
        losses_total = np.load(join(results_dir, f'losses_total.npy')).tolist()
        print("Loaded total losses")
    except:
        print("Haven't found previous losses. Is this a new experiment?")
        losses_r = []
        losses_kld = []
        losses_total = []

    if losses_total == []:
        print("Loading \'latest\' model without loaded losses")
        try:
            hnet.load_state_dict(torch.load(join(results_dir, f"model_hn_latest.pt"))) 
            print("Loaded HNet")
            encoder.load_state_dict(torch.load(join(results_dir, f"model_e_latest.pt")))
            print("Loaded Encoder")
            scheduler.load_state_dict(torch.load(join(results_dir, f"lr_latest.pt")))
            print("Loaded Scheduler")
        except:
            print("Haven't loaded all previous models.")
    else:
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
                print("Haven't loaded all previous models.")

    results_dir = join(results_dir, 'eval')
    os.makedirs(results_dir, exist_ok=True)

    encoder.eval()
    hnet.eval()

    mse = torch.nn.MSELoss()
    psnr_metric = lambda x,y: torch.mean(mse2psnr(mse(y,x)))

    eval_results = pd.DataFrame(columns=['class', 'fscore', 'cd', 'psnr'])

    for i, (entry, cat, obj_path) in enumerate(dataloader):
        start_time = datetime.now()

        if config['resnet']:
            nerf_Ws = get_nerf_resnet(entry, encoder, hnet)
        else:
            nerf_Ws, mu, logvar = get_nerf(entry, encoder, hnet)

        #For batch size == 1 hnet doesn't return batch dimension...
        if config['batch_size'] == 1:
            nerf_Ws = [nerf_Ws]
    
        for j, target_w in enumerate(nerf_Ws):
            render_kwargs = get_render_kwargs(config, nerf, target_w, embed_fn, embeddirs_fn)
            render_kwargs['perturb'] = False
            render_kwargs['raw_noise_std'] = 0.
            
            points = entry["data"][j]
            points = points.to(device, dtype=torch.float)

            #render_kwargs = get_render_kwargs(config, nerf, target_w, embed_fn, embeddirs_fn)

            #f_score,cd  = calculate_metrics(obj_path[0], render_kwargs,save_pc=i<=2, name=str(i))
            f_score, cd = calculate_best_mesh_metrics(obj_path[j], render_kwargs, save_pc=False, name=str(i), thresholds=[3])
            #f_score = torch.Tensor([0.])
            #cd = torch.Tensor([0.])
            psnr = calculate_image_metrics(entry, render_kwargs, psnr_metric, count=5)

            #print(i,' fscore-> ', f_score)
            #print(i, 'cd-> ', cd)
            #print(i,' psnr-> ', psnr)
            eval_results = eval_results.append({'class': cat[j], 'fscore': f_score.item(),'cd': cd.item(), 'psnr': psnr.item()}, ignore_index=True)
        #print("Time:", round((datetime.now() - start_time).total_seconds(), 2))
    print(eval_results.groupby("class").describe())
    print("---------------")
    print(eval_results[['fscore', 'cd', 'psnr']].describe())
