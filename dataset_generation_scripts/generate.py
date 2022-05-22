import glob
import numpy
import os
import numpy as np
from pathlib import Path
import subprocess

from PIL import Image
import json
from numpy import asarray
import shutil


DEBUG = False
MAX = 99999
VIEW_COUNT = 50
RESOLUTION = 200



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render gltfs')
    parser.add_argument('path', type=str,
                        help='Glob string for model files (./shapenet/02958343/**/*.gltf)')
    parser.add_argument('cp_path', type=str,
                        help='Cloud of points files from cloud export script (./data/02958343/)')
    parser.add_argument('export_name', type=str,
                        help='Result folder name (./shapenet/02958343)')

    args = parser.parse_args()
    files = glob.glob(args.path, recursive=True)  
    count = len(files)

    RESULT_DIR = f"{args.export_name}_{VIEW_COUNT}_{RESOLUTION}x{RESOLUTION}"

    np.random.seed(1234)
    generated = 0
    for i, f in enumerate(files):
        os.makedirs(f"./{RESULT_DIR}", exist_ok=True)
        
        print(f"{i}/{count}")
        name = f.split('/')[-3]
        print(f"{name} opened at {f}")


        print("Rendering...")
        x=subprocess.run(f"blender --background --python render.py -- --output_folder ./tmp {f} --name {name} --views {VIEW_COUNT} --resolution {RESOLUTION}", capture_output=False) #render180.py for circle
        print("Images rendered") 
        
        images = []
        cam_poses = []

        for render in glob.glob(f'./tmp/{name}/*.png'):
            images.append(asarray(Image.open(render)))
            
            f = (render.split('/')[-1]).split('.')[0]
            with open(f"./tmp/{name}/{f}.json", "r") as file:
                cp = json.load(fp=file)
            cam_poses.append(np.array(cp))

        images = np.array(images, dtype="float16")/255

        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:]) #enforce white background

        cam_poses = np.array(cam_poses)

        if DEBUG:
            print(images.shape)
            print(cam_poses.shape)

        print("Sampling points...")

        vertices = []
        try:
            with open(f'{args.cp_path}/{name}_mesh_data.txt') as file:
                for line in file:
                    vertices.append([float(x) for x in line.split()])
            vertices = np.array(vertices)
        except Exception as e:
            continue
            #print(e)

        colors = []
        with open(f'{args.cp_path}/{name}_color_data.txt') as file:
            for line in file:
                colors.append([float(x) for x in line.split()])
        colors = np.array(colors)

        data = np.concatenate((vertices, colors[:,0:3]), axis=1)
        
        #uncomment if you want to take random number of points
        data = data[np.random.choice(data.shape[0], 2048, replace=False), :]

        if DEBUG:
            print(data.shape)
            print(data)

        print(f"Data for {name} was generated!")
        print(f"Saving {name}...")
        np.savez_compressed(f"./{RESULT_DIR}/{VIEW_COUNT}_{name}.npz", images=images, cam_poses=cam_poses, data=data)
        print(f"{name} was saved!")

        shutil.rmtree(f'./tmp/{name}')

        if DEBUG:
            print("DEBUG is TRUE: pausing rendering")
            break
        
        if generated >= MAX:
            break
        
        generated += 1
        
    print("Data was generated!")
