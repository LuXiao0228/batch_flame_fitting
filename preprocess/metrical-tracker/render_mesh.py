import torch
import torch.nn as nn
import numpy as np
import os
from datasets.flame_dataset import FlameDataset
import torch.nn.functional as F
from enum import Enum
from loguru import logger
from pytorch3d.io import load_obj
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras, BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from torch.utils.data import DataLoader
from flame.FLAME import FLAME, FLAMETex
from renderer import Renderer
from tqdm import tqdm
import util
import cv2
import glob
from pathlib import Path
from configs.config import parse_args

mediapipe_idx = np.load('flame/mediapipe/mediapipe_landmark_embedding.npz', allow_pickle=True, encoding='latin1')['landmark_indices'].astype(int)

class View(Enum):
    GROUND_TRUTH = 1
    COLOR_OVERLAY = 2
    SHAPE_OVERLAY = 4
    SHAPE = 8
    LANDMARKS = 16
    HEATMAP = 32
    DEPTH = 64

class Render(nn.Module):
    def __init__(self, config, device="cuda:0"):
        super(Render, self).__init__()
        self.config = config
        self.device = device
        # input
        self.data_folder = config.save_folder
        self.image_folder = os.path.join(self.data_folder, 'input')
        self.ckpt_folder = os.path.join(self.data_folder, 'checkpoint')
        # output
        self.save_folder = os.path.join(self.data_folder, 'rendered')
        self.dataset = FlameDataset(self.image_folder, self.ckpt_folder)
        self.image_size = torch.tensor([[config.image_size[0], config.image_size[1]]]).cuda()
        self.setup_renderer()
        self.create_output_folder()
    
    def create_output_folder(self):
        os.makedirs(self.save_folder, exist_ok=True)
    
    def setup_renderer(self):
        mesh_file = 'data/head_template_mesh.obj'
        
        self.flame = FLAME(self.config).to(self.device)
        self.flametex = FLAMETex(self.config).to(self.device)
        self.diff_renderer = Renderer(self.image_size, obj_filename=mesh_file).to(self.device)
        self.faces = load_obj(mesh_file)[1]

        raster_settings = RasterizationSettings(
            image_size=self.get_image_size(),
            faces_per_pixel=1,
            cull_backfaces=True,
            perspective_correct=True
        )

        self.lights = PointLights(
            device=self.device,
            location=((0.0, 0.0, 5.0),),
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.5, 0.5, 0.5),)
        )

        self.mesh_rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.debug_renderer = MeshRenderer(
            rasterizer=self.mesh_rasterizer,
            shader=SoftPhongShader(device=self.device, lights=self.lights)
        )
    
    def get_image_size(self):
        return self.image_size[0][0].item(), self.image_size[0][1].item()
    
    def parse_batch(self, batch):
        frame_id = batch['frame_id'][0]
        
        image = batch['image'].to(self.device)
        
        camera_params = batch['camera']
        R = camera_params['R'].to(self.device)
        t = camera_params['t'].to(self.device)
        fl = camera_params['fl'].to(self.device)
        pp = camera_params['pp'].to(self.device)
        
        flame_params = batch['flame']
        shape = flame_params['shape'].to(self.device)
        exp = flame_params['exp'].to(self.device)
        eyes = flame_params['eyes'].to(self.device)
        eyelids = flame_params['eyelids'].to(self.device)
        jaw = flame_params['jaw'].to(self.device)
        neck = flame_params['neck'].to(self.device)
        
        lmk = batch['lmk'].to(self.device)
        dense_lmk = batch['dense_lmk'].to(self.device)
        
        return frame_id, image, R, t, fl, pp, shape, exp, eyes, eyelids, jaw, neck, lmk, dense_lmk[:, mediapipe_idx, :2]
        
    def images_to_video(self, video_name, extention='jpg', fps=25, video_format='DIVX'):
        logger.info(f"Generating {video_name}.avi from rendered results.")
        images_paths = sorted(glob.glob(os.path.join(self.save_folder, f'*.{extention}')))
        img0 = cv2.imread(images_paths[0])
        height, width, _ = img0.shape
        size = (width, height)
        out = cv2.VideoWriter(os.path.join(Path(self.save_folder).parent, f'{video_name}.avi'), cv2.VideoWriter_fourcc(*video_format), fps, size)
        
        for filename in tqdm(images_paths):
            img = cv2.imread(filename)
            out.write(img)
        out.release()
    
    def render_shape(self, vertices, faces=None, white=True):
        B = vertices.shape[0]
        V = vertices.shape[1]
        if faces is None:
            faces = self.faces.verts_idx.cuda()[None].repeat(B, 1, 1)
        if not white:
            verts_rgb = torch.from_numpy(np.array([80, 140, 200]) / 255.).cuda().float()[None, None, :].repeat(B, V, 1)
        else:
            verts_rgb = torch.from_numpy(np.array([1.0, 1.0, 1.0])).cuda().float()[None, None, :].repeat(B, V, 1)
        textures = TexturesVertex(verts_features=verts_rgb.cuda())
        meshes_world = Meshes(verts=[vertices[i] for i in range(B)], faces=[faces[i] for i in range(B)], textures=textures)

        blend = BlendParams(background_color=(1.0, 1.0, 1.0))

        fragments = self.mesh_rasterizer(meshes_world, cameras=self.cameras)
        rendering = self.debug_renderer.shader(fragments, meshes_world, cameras=self.cameras, blend_params=blend)
        rendering = rendering.permute(0, 3, 1, 2).detach()
        return rendering[:, 0:3, :, :]
        
    def run(self):
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
        
        for data in tqdm(dataloader):
            frame_id, images, R, t, fl, pp, shape, exp, eyes, eyelids, jaw, neck, lmks, dense_lmks = self.parse_batch(data)
            self.cameras = PerspectiveCameras(
                device=self.device,
                principal_point=pp,
                focal_length=fl,
                R=rotation_6d_to_matrix(R),
                T=t, image_size=self.image_size
            )
            
            self.diff_renderer.rasterizer.reset()
            self.diff_renderer.set_size(self.get_image_size())
            self.debug_renderer.rasterizer.raster_settings.image_size = self.get_image_size()

            vertices, lmk68, lmkMP = self.flame(
                cameras=torch.inverse(self.cameras.R),
                shape_params=shape,
                expression_params=exp,
                neck_pose_params=neck,
                eye_pose_params=eyes,
                jaw_pose_params=jaw,
                eyelid_params=eyelids
            )

            lmk68 = self.cameras.transform_points_screen(lmk68, image_size=self.image_size)
            lmkMP = self.cameras.transform_points_screen(lmkMP, image_size=self.image_size)


            final_views = []

            for views in [[View.GROUND_TRUTH, View.LANDMARKS, View.SHAPE]]:
                row = []
                for view in views:
                    if view == View.GROUND_TRUTH:
                        row.append(images[0].cpu().numpy())
                    if view == View.SHAPE:
                        shape = self.render_shape(vertices, white=False)
                        shape_lmks = shape.clone()
                        shape_lmks = util.tensor_vis_landmarks(shape_lmks, torch.cat([dense_lmks, lmks[:, :17, :]], dim=1), color='g')
                        shape_lmks = util.tensor_vis_landmarks(shape_lmks, torch.cat([lmkMP, lmk68[:, :17, :]], dim=1), color='r')
                        row.append(shape_lmks[0].cpu().numpy())
                    if view == View.LANDMARKS:
                        gt_lmks = images.clone()
                        gt_lmks = util.tensor_vis_landmarks(gt_lmks, torch.cat([dense_lmks, lmks[:, :17, :]], dim=1), color='g')
                        gt_lmks = util.tensor_vis_landmarks(gt_lmks, torch.cat([lmkMP, lmk68[:, :17, :]], dim=1), color='r')
                        row.append(gt_lmks[0].cpu().numpy())
                final_views.append(row)

            # VIDEO
            final_views = util.merge_views(final_views)

            cv2.imwrite(os.path.join(self.save_folder, f'{frame_id}.jpg'), final_views)
        
        self.images_to_video("render_mesh", 'jpg')


if __name__ == '__main__':
    config = parse_args()
    render = Render(config)
    render.run()