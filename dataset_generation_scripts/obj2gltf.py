import os
import glob
from tqdm import tqdm
from pathlib import Path

Path(__file__).parent

files = glob.glob('.\shapenet\02958343\**\*.obj', recursive=True)
for file in tqdm(files):
    os.system(f"obj2gltf -i {file} -o {Path(file).parent}\\model_normalized.gltf")