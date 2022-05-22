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

import open3d as o3d

import argparse

#Needed for workers for dataloader
from torch.multiprocessing import Pool, Process, set_start_method
set_start_method('spawn', force=True)

import math
def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = math.sqrt(XsqPlusYsq + z**2)               # r
    elev = math.atan2(z,math.sqrt(XsqPlusYsq))     # theta
    az = math.atan2(y,x)                           # phi
    return r, elev, az

def export_model(render_kwargs, focal, path, path_colored, N=256):
    width = 1.1
    with torch.no_grad():
        #Sample NeRF
        t = torch.linspace(-width, width, N+1)
        query_pts = torch.stack(torch.meshgrid(t, t, t), -1)
        print(query_pts.shape)
        sh = query_pts.shape
        flat = query_pts.reshape([-1,3])
        print(flat.shape)

        fn = lambda i0, i1 : render_kwargs['network_query_fn'](flat[i0:i1,None,:], viewdirs=None, network_fn=render_kwargs['network_fn'])
        chunk = 1024*16
        raw = torch.cat([fn(i, i+chunk) for i in range(0, flat.shape[0], chunk)], 0)
        raw = torch.reshape(raw, list(sh[:-1]) + [-1])
        sigma = torch.maximum(raw[...,-1], torch.Tensor([0.]))

        #Marching cubes
        threshold = 5
        vertices, triangles = mcubes.marching_cubes(sigma.cpu().numpy(), threshold)
        print('done', vertices.shape, triangles.shape)

        #Two meshes because colors tend to be misplaced on mesh_export
        mesh = trimesh.Trimesh((vertices / N) - 0.5, triangles)

        obj = trimesh.exchange.ply.export_ply(mesh)

        with open(path, "wb+") as f:
            f.write(obj)

        print("Saved uncolored model to", path)

        rgbs = []
        final = []
        vertex_colors = []
        radius = 0.05 # distance from camera to a vertex, theoretically it could be lower to properly capture its color

        H = 1
        W = 1
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

        for i, vert in enumerate(mesh.vertices): 
            coords = np.array(vert)

            coords = coords / np.linalg.norm(coords)
            r, phi, theta = cart2sph(*coords)
            theta += math.pi/2
            phi -= math.pi
            c2w = pose_spherical(theta * 180 / math.pi, phi * 180 / math.pi, r+radius)
            result = render(H, W, K, chunk=2048, c2w=c2w, **render_kwargs)
            rgb = np.clip(result[0].detach().cpu().numpy(),0,1).squeeze()
            rgbs.append(rgb)
            final.append([*vert, *rgb])
            mesh.visual.vertex_colors[i] = np.concatenate((rgb, [1]))*255

        obj = trimesh.exchange.ply.export_ply(mesh)

        with open(path_colored, "wb+") as f:
            f.write(obj)

        print("Saved colored model to", path_colored)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    dirname = os.path.dirname(__file__)

    parser = argparse.ArgumentParser(description='Start training HyperRF')
    parser.add_argument('config_path', type=str,
                        help='Relative config path')
    parser.add_argument('-o_anim_count', type=int, help='How many object animations')
    parser.add_argument('-g_anim_count', type=int, help='How many generated object animations')
    parser.add_argument('-i_anim_count', type=int, help='How many interpolation object animations')
    parser.add_argument('-train_ds', type=int, help="Use train dataset?", default=0)
    parser.add_argument('-epoch', type=int, help="Default epoch to use. Set 0 to use latest.", default=0)
    #TODO: dodac argumenty tutaj

    args = parser.parse_args()

    config = None
    with open(args.config_path) as f:
        config = json.load(f)
    assert config is not None

    print(config)

    set_seed(config['seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset = NeRFShapeNetDataset(root_dir=config['data_dir'], classes=config['classes'], train=args.train_ds != 0)

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


    print(args.epoch, "set as starting epoch")
    if args.epoch == 0:
        print("Loading \'latest\' models")
        try:
            hnet.load_state_dict(torch.load(join(results_dir, f"model_hn_latest.pt"))) 
            print("Loaded HNet")
            encoder.load_state_dict(torch.load(join(results_dir, f"model_e_latest.pt")))
            print("Loaded Encoder")
        except:
            print("Haven't loaded all previous models.")
    else:
        starting_epoch = args.epoch
        print("Starting epoch:", starting_epoch)

    if(starting_epoch>0):
        print("Loading weights")
        try:
            hnet.load_state_dict(torch.load(join(results_dir, f"model_hn_{starting_epoch}.pt"))) 
            print("Loaded HNet")
            encoder.load_state_dict(torch.load(join(results_dir, f"model_e_{starting_epoch}.pt")))
            print("Loaded Encoder")
        except:
            print("Haven't found all previous models.")

    results_dir = join(dirname, 'rendered_samples', config['classes'][0])
    os.makedirs(results_dir, exist_ok=True)
    results_dir_main = results_dir

    encoder.eval()
    hnet.eval()

    default_N = 256
    render_iterations = 60 + 1
    render_fps = 30

    for i, (entry, cat, obj_path) in enumerate(dataloader):
        if i > args.o_anim_count:
            break

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

            print("Animation", i, obj_path)
            H = entry["images"][j].shape[1]
            W = entry["images"][j].shape[2]
            focal = .5 * W / np.tan(.5 * 0.6911112070083618) 

            K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
            ])

            results_dir = join(results_dir_main, f'o{i}')
            os.makedirs(results_dir, exist_ok=True)
            torch.set_printoptions(threshold=100)
            
            #Render cloud of points
            """
            for el in [0,45,90,135, 180, 225, 270, 315]:
                for az in [0,45,90,135, 180, 225, 270, 315]:
                    fig = plt.figure(figsize=(8,8))
                    ax = fig.add_subplot(111, projection = '3d')
                    ax.view_init(elev=el, azim=az)
                    ax.scatter(entry['data'][j][:,0], entry['data'][j][:,1], entry['data'][j][:,2], c = entry['data'][j][:,3:])
                    ax.set_xlim3d(-1, 1)
                    ax.set_ylim3d(-1, 1)
                    ax.set_zlim3d(-1, 1)
                    plt.axis('off')
                    plt.grid(b=None)
                    plt.tight_layout()
                    plt.savefig(join(results_dir, f'pc_{el}_{az}.png'))
                    plt.close()
            """
            
            for gt in range(10):
                imageio.imsave(join(results_dir, f'ground_t_{gt}.png'), to8b(entry['images'][j][gt].detach().cpu().numpy()))

            with torch.no_grad():
                img_i = np.random.choice(len(entry['images'][j]), 1)
                target = entry['images'][j][img_i][0].to(device)
                target = torch.Tensor(target.float())
                pose = entry['cam_poses'][j][img_i, :3,:4][0].to(device)

                img_r, _, _, _ = render(H, W, K, chunk=config['model']['TN']['netchunk'], c2w = pose,
                                                            verbose=True, retraw=True,
                                                            **render_kwargs)

                frame = torch.cat([img_r,target], dim=1)

                imageio.imsave(join(results_dir, f'compare_{i}.png'), to8b(frame.detach().cpu().numpy()))

            with torch.no_grad():
                render_poses = torch.stack([pose_spherical(angle, -45, 3.2) for angle in np.linspace(-180,180,render_iterations)[:-1]], 0)
                frames = []
                for k, pose in enumerate(render_poses):

                    img, disp, acc, _ = render(H, W, K, chunk=config['model']['TN']['netchunk'], c2w=pose,
                                                        verbose=True, retraw=True,
                                                        **render_kwargs)
                    frames.append(to8b(img.detach().cpu().numpy()))

                    if k%4==0:
                        imageio.imsave(join(results_dir, f'o_{i}_{k}.png'), to8b(img.detach().cpu().numpy()))

            writer = imageio.get_writer(join(results_dir, f'an_{i}.gif'), fps=30)
            for frame in frames:
                writer.append_data(frame)
            writer.close()

            with torch.no_grad():
                render_poses = torch.stack([pose_spherical(angle, -45, 3.2) for angle in np.linspace(-180,180,9)[:-1]]+\
                                            [pose_spherical(angle, -30, 3.2) for angle in np.linspace(-180,180,9)[:-1]]+\
                                            [pose_spherical(angle, -15, 3.2) for angle in np.linspace(-180,180,9)[:-1]], 
                                            0)
                for k, pose in enumerate(render_poses):

                    img, disp, acc, _ = render(H, W, K, chunk=config['model']['TN']['netchunk'], c2w=pose,
                                                        verbose=True, retraw=True,
                                                        **render_kwargs)

                    
                    imageio.imsave(join(results_dir, f'o_other_{i}_{k}.png'), to8b(img.detach().cpu().numpy()))

            render_kwargs['near'] = 0.

            export_model(render_kwargs, focal, join(results_dir, f'o_model_{i}.ply'), join(results_dir, f'o_model_col_{i}.ply'), N=default_N)

        print("Time:", round((datetime.now() - start_time).total_seconds(), 2))

    for i in range(args.g_anim_count):
        start_time = datetime.now()
        sample = torch.normal(mean=torch.zeros(config["z_size"]), std=torch.full((config["z_size"],), fill_value=0.006))
        render_kwargs = get_render_kwargs(config, nerf, get_nerf_from_code(hnet, sample[None]), embed_fn, embeddirs_fn)
        render_kwargs['perturb'] = False
        render_kwargs['raw_noise_std'] = 0.
        
        results_dir = join(results_dir_main, f'g{i}')
        os.makedirs(results_dir, exist_ok=True)

        print("Generated Object Animation", i)
        with torch.no_grad():
            render_poses = torch.stack([pose_spherical(angle, -45, 3.2) for angle in np.linspace(-180,180,render_iterations)[:-1]], 0)
            frames = []
            for k, pose in enumerate(render_poses):

                img, disp, acc, _ = render(H, W, K, chunk=config['model']['TN']['netchunk'], c2w=pose,
                                                    verbose=True, retraw=True,
                                                    **render_kwargs)
                frames.append(to8b(img.detach().cpu().numpy()))

                if k%4==0:
                    imageio.imsave(join(results_dir, f'g_{i}_{k}.png'), to8b(img.detach().cpu().numpy()))

            writer = imageio.get_writer(join(results_dir, f'g_an_{i}.gif'), fps=render_fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()

            render_kwargs['near'] = 0. 

            export_model(render_kwargs, focal, join(results_dir, f'g_model_{i}.ply'), join(results_dir, f'g_model_col_{i}.ply'), N=default_N)
        print("Time:", round((datetime.now() - start_time).total_seconds(), 2))

    
    dl_iter = iter(dataloader)

    for i in range(args.i_anim_count):
        with torch.no_grad():

            results_dir = join(results_dir_main, f'i{i}')
            os.makedirs(results_dir, exist_ok=True)

            full_interpolations = None
            start_time = datetime.now()

            entry_1, cat_1, obj_path_1 = next(dl_iter)
            entry_2, cat_2, obj_path_2  = next(dl_iter)

            nerf_1_code = get_code(entry_1, encoder)
            nerf_2_code = get_code(entry_2, encoder)
            print("Generated Object Animation", i)
            print(obj_path_1)
            print(obj_path_2)
            
            kwargs_1 = get_render_kwargs(config, nerf, get_nerf_from_code(hnet, nerf_1_code), embed_fn, embeddirs_fn)
            kwargs_2 = get_render_kwargs(config, nerf, get_nerf_from_code(hnet, nerf_2_code), embed_fn, embeddirs_fn)

            kwargs_1['perturb'] = False
            kwargs_1['raw_noise_std'] = 0.

            kwargs_2['perturb'] = False
            kwargs_2['raw_noise_std'] = 0.

            steps = render_iterations + 1

            export_model(kwargs_1, focal, join(results_dir, f'i_1_model_{i}.ply'), join(results_dir, f'i_1_model_col_{i}.ply'), N=default_N)
            export_model(kwargs_2, focal, join(results_dir, f'i_2_model_{i}.ply'), join(results_dir, f'i_2_model_col_{i}.ply'), N=default_N)
        
            writer = imageio.get_writer(join(results_dir, f'i_an_{i}.gif'), fps=render_fps)
            render_poses = torch.stack([pose_spherical(angle, -45, 3.2) for angle in np.linspace(-180,180,steps)[:-1]], 0)
            for k, pose in enumerate(render_poses):
                
                #c2w=pose for rotation
                img1, disp, acc, _ = render(H, W, K, chunk=config['model']['TN']['netchunk'], c2w=render_poses[-36],
                                        verbose=True, retraw=True,**kwargs_1)
                img2, disp, acc, _ = render(H, W, K, chunk=config['model']['TN']['netchunk'], c2w=render_poses[-36],
                                        verbose=True, retraw=True,**kwargs_2)

                nerf_3_code=torch.lerp(nerf_1_code, nerf_2_code, k/steps)
                
                kwargs_3 = get_render_kwargs(config, nerf, get_nerf_from_code(hnet, nerf_3_code), embed_fn, embeddirs_fn)
                kwargs_3['perturb'] = False
                kwargs_3['raw_noise_std'] = 0.
                

                img3, disp, acc, _ = render(H, W, K, chunk=config['model']['TN']['netchunk'], c2w=render_poses[-36],
                                        verbose=True, retraw=True,**kwargs_3)
                
                frame = torch.cat([img1,img3,img2], dim=1)
                
                if k % 5==0:
                    kwargs_3['near'] = 0.
                    export_model(kwargs_3, focal, join(results_dir, f'interpolated_model_{i}_{k}.ply'), join(results_dir, f'interpolated_model_{i}_{k}.ply'), N=default_N)
                    imageio.imsave(join(results_dir, f'ii_{i}_{k}.png'), to8b(img3.detach().cpu().numpy()))
                
                writer.append_data(to8b(frame.detach().cpu().numpy()))
            writer.close()


            print("Time:", round((datetime.now() - start_time).total_seconds(), 2))

