import glob
from pathlib import Path

import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms.functional as F
from loguru import logger
from torch.utils.data import Dataset
import os

# only for render
class FlameDataset(Dataset):
    def __init__(self, image_folder, flame_param_folder):
        self.images = sorted(glob.glob((os.path.join(image_folder, '*.png'))))
        self.device = 'cuda:0'
        self.flame_param_folder = flame_param_folder

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        imagepath = self.images[index]
        imagename = os.path.basename(imagepath).split('.')[0]
        flame_param_path = os.path.join(self.flame_param_folder, f'{imagename}.frame')
        pil_image = Image.open(imagepath).convert("RGB")
        image = F.to_tensor(pil_image)
        
        payload = torch.load(flame_param_path)
        camera_params = payload['camera']
        flame_params = payload['flame']
        
        for k, v in camera_params.items():
            camera_params[k] = torch.from_numpy(v).squeeze(dim=0)
        
        for k, v in flame_params.items():
            flame_params[k] = torch.from_numpy(v).squeeze(dim=0)
        
        lmk_path = imagepath.replace('input', 'lmk_kpt').replace('.png', '.npy').replace('.jpg', '.npy')
        lmk_path_dense = imagepath.replace('input', 'lmk_kpt_dense').replace('.png', '.npy').replace('.jpg', '.npy')

        lmk = np.load(lmk_path, allow_pickle=True)
        dense_lmk = np.load(lmk_path_dense, allow_pickle=True)
        
        lmks = torch.from_numpy(lmk).float()
        dense_lmks = torch.from_numpy(dense_lmk).float()
        
        data = {
            'frame_id': imagename,
            'image': image,
            'lmk': lmks,
            'dense_lmk': dense_lmks,
            'camera': camera_params,
            'flame': flame_params
        }

        return data
