import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from os.path import join

synth_id_to_category = {
    '02691156': 'planes',  '02773838': 'bag',        '02801938': 'basket', #airplane = planes temporary
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',       '02958343': 'cars',        '03001627': 'chairs', #car=cars temporary chair=chairs
    '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
}

category_to_synth_id = {v: k for k, v in synth_id_to_category.items()}
synth_id_to_number = {k: i for i, k in enumerate(synth_id_to_category.keys())}


class NeRFShapeNetDataset(Dataset):
    def __init__(self, root_dir='/home/datasets/nerfdataset', shapenet_root_dir='/shared/sets/datasets/3D_points/ShapeNetCore.v2', classes=[],
                 transform=None, train=True):
        """
        Args:
            root_dir (string): Directory of structure: 
            >
                >classname1
                    >sampled
                        >count_{name}.npz
                >classname2
                    ...
            
            where sampled has all the .NPZ of format: images : (n, W, H, channels), cam_poses (n, 4, 4), data :(N, 6)
            and shapenet is a shapenet directory for this class (contains .obj files).

            classes: list of class names

            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.shapenet_root_dir = shapenet_root_dir
        self.transform = transform

        self.classes = classes
        self.train = train
    
        self.data = []

        self._load()

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


    def __getitem__(self, idx):
        if self.train:
            data_files = self.train_data
        else:
            data_files = self.test_data

        sample = np.load(data_files['sample_filename'][idx])
        class_name = data_files['class'][idx]
        if self.transform:
            sample = self.transform(sample)   

        #return self.data[idx]
        return sample, class_name, data_files['obj_filename'][idx]

    def _load(self):
        print("Loading dataset:")
        self.train_data = pd.DataFrame(columns=['class', 'name', 'sample_filename', 'obj_filename'])
        self.test_data = pd.DataFrame(columns=['class', 'name', 'sample_filename', 'obj_filename'])

        for data_class in self.classes:
            df = pd.DataFrame(columns=['class', 'name', 'sample_filename', 'obj_filename'])
            print(data_class)

            npz_glob =  glob.glob(join(self.root_dir,data_class,'sampled','*.npz'))
            print(len(npz_glob))
            for file in npz_glob:
                sample_name = file.split('_')[-1].split('.')[0]

                df = df.append({'class': data_class,
                                'name': sample_name,
                                'sample_filename':file, 
                                'obj_filename':join(self.shapenet_root_dir, category_to_synth_id[data_class], sample_name, 'models','model_normalized.obj')}, 
                                ignore_index=True)

                #with np.load(file) as data:
                #self.data.append({'data': np.array(data['data']), 'images':np.array(data['images']), 'cam_poses':np.array(data['cam_poses'])})

            #Sort and split, same like Atlasnet
            df = df.sort_values(by=['name'])
            df_train = df.head(max(1,int(len(df)*(0.8))))
            df_test = df.tail(max(1,int(len(df)*(0.2))))
        
            self.train_data = pd.concat([self.train_data, df_train])
            self.test_data = pd.concat([self.test_data, df_test])

            self.train_data = self.train_data.reset_index(drop=True)
            self.test_data = self.test_data.reset_index(drop=True)

        print("Loaded train data:", len(self.train_data), "samples")
        print("Loaded test data:", len(self.test_data), "samples")