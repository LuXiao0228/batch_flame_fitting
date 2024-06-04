#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from avatar_utils.graphics_utils import getWorld2View2, getProjectionMatrix
from avatar_utils.general_utils import PILtoTorch
from copy import deepcopy
from PIL import Image
from typing import List
import glob
import os
from pytorch3d.transforms import matrix_to_rotation_6d
from avatar_utils.graphics_utils import focal2fov

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, K, tan_fovx_l, tan_fovx_r, tan_fovy_t, tan_fovy_b, bg, image_width, image, image_height, image_path,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 timestep=None, data_device = "cuda", mask=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.K = K
        # self.FoVx = FoVx
        # self.FoVy = FoVy
        self.tan_fovx_l = tan_fovx_l
        self.tan_fovx_r = tan_fovx_r
        self.tan_fovy_t = tan_fovy_t
        self.tan_fovy_b = tan_fovy_b
        self.bg = bg
        self.image = image
        self.mask = mask
        self.image_width = image_width
        self.image_height = image_height
        self.image_path = image_path
        self.image_name = image_name
        self.timestep = timestep

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)  #.cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, tanfovx_l=tan_fovx_l, tanfovx_r=tan_fovx_r, tanfovy_b=tan_fovy_b, tanfovy_t=tan_fovy_t).transpose(0,1)  #.cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, timestep):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.timestep = timestep

class CameraDataset(torch.utils.data.Dataset):
    def __init__(self, cameras: List[Camera]):
        self.cameras = cameras

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # ---- from readCamerasFromTransforms() ----
            camera = deepcopy(self.cameras[idx])

            if camera.image is None:
                image = Image.open(camera.image_path)
                im_data = np.array(image.convert("RGBA"))
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + camera.bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            else:
                image = camera.image

            # ---- from loadCam() and Camera.__init__() ----
            resized_image_rgb = PILtoTorch(image, (camera.image_width, camera.image_height))

            image = resized_image_rgb[:3, ...]

            if resized_image_rgb.shape[1] == 4:
                gt_alpha_mask = resized_image_rgb[3:4, ...]
                image *= gt_alpha_mask
            
            camera.original_image = image.clamp(0.0, 1.0)
            return camera
        elif isinstance(idx, slice):
            return CameraDataset(self.cameras[idx])
        else:
            raise TypeError("Invalid argument type")
        

class CameraDatasetDisk(torch.utils.data.Dataset):
    def __init__(self, image_data_folder, flame_param_folder, white_background=True, extension=".png"):
        self.image_data_folder = image_data_folder
        self.flame_param_folder = flame_param_folder
        self.images_paths = sorted(glob.glob(os.path.join(image_data_folder, f"*{extension}")))
        self.bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + self.bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
        
        
        # load flame params
        param_path = os.path.join(self.flame_param_folder, f'{image_name}.frame')
        I = matrix_to_rotation_6d(torch.cat([torch.eye(3)[None]], dim=0))
        param = torch.load(param_path)
        flame = param['flame']
        flame_param = {
            'shape': torch.from_numpy(flame['shape']),
            'expr': torch.from_numpy(flame['exp']),
            'rotation': I,
            'neck_pose': torch.from_numpy(flame['neck']),
            'jaw_pose': torch.from_numpy(flame['jaw']),
            'eyes_pose': torch.from_numpy(flame['eyes']),
            'translation': torch.zeros([1,3]),
            'static_offset': None,
            'eyelids': torch.from_numpy(flame['eyelids']),
        }
        
        H, W = int(param['img_size'][0]), int(param['img_size'][1]) 
        # load camera
        camera = param['opencv']    # opencv/colmap
        R = camera['R'][0]     # (1, 3, 3)
        t = camera['t'][0]     # (1, 3)
        K = camera['K'][0]     # (1, 3, 3)
        
        R = np.transpose(R)         # glm need!
        # fovx = focal2fov(K[0, 0], W)
        # fovy = focal2fov(K[1, 1], H)
        
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        tan_fovx_r = cx / fx            # NOTE：不知道为什么这里左右得反过来……
        tan_fovx_l = (W - cx) / fx
        tan_fovy_t = cy / fy
        tan_fovy_b = (H - cy) / fy  
        
        resized_image_rgb = PILtoTorch(image, (W, H))
        
        image = resized_image_rgb[:3, ...]

        if resized_image_rgb.shape[1] == 4:
            gt_alpha_mask = resized_image_rgb[3:4, ...]
            image *= gt_alpha_mask
        
        cam = Camera(
                colmap_id=idx, 
                R=R, T=t, K=K,
                # FoVx=fovx, 
                # FoVy=fovy, 
                tan_fovx_r=tan_fovx_r,
                tan_fovx_l=tan_fovx_l,
                tan_fovy_t=tan_fovy_t,
                tan_fovy_b=tan_fovy_b,
                image_width=W, 
                image_height=H,
                bg=self.bg, 
                image=image, 
                image_path=image_path,
                image_name=image_name, 
                uid=idx, 
                timestep=idx, 
            )

        cam.original_image = image.clamp(0.0, 1.0)
        
        return {'camera': cam, 'flame_param': flame_param}