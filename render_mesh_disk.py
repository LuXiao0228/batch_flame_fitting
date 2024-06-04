from argparse import ArgumentParser
import os
from mesh_renderer import NVDiffRenderer
import glob
import torch
import numpy as np
from pytorch3d.transforms import matrix_to_rotation_6d
from PIL import Image
import cv2
import tqdm
from avatar_utils.graphics_utils import focal2fov, fov2focal
from flame_model.flame import FlameHead
from avatar_utils.cameras import CameraDatasetDisk

from torch.utils.data import DataLoader
import shutil
from tqdm import tqdm
import time 
import json
from pathlib import Path
from loguru import logger


def get_texture(img, verts_img, flame_model: FlameHead, tw=512, th=512):
    h, w, _ = img.shape
    texcoords = flame_model.verts_uvs.detach().cpu().numpy()
    
    texture = np.zeros((tw, th, 3)) + 255
    print(texture.shape)
    faces = flame_model.faces.detach().cpu().numpy()
    for fi in range(faces.shape[0]):
        face_vertices = faces[fi]
        face_vertices = [face_vertices[0],face_vertices[1],face_vertices[2]]
        tc = flame_model.textures_idx[fi]

        if max(abs(texcoords[tc[0] ][0] - texcoords[tc[1] ][0]),
               abs(texcoords[tc[0] ][0] - texcoords[tc[2] ][0]),
               abs(texcoords[tc[1] ][0] - texcoords[tc[2] ][0]),
               abs(texcoords[tc[0] ][1] - texcoords[tc[1] ][1]),
               abs(texcoords[tc[0] ][1] - texcoords[tc[2] ][1]),
               abs(texcoords[tc[1] ][1] - texcoords[tc[2] ][1])) > 0.3:
            continue

        tri1 = np.float32([[[(int(verts_img[face_vertices[0] , 1])),
                             int(verts_img[face_vertices[0] , 0])],
                            [(int(verts_img[face_vertices[1] , 1])),
                             int(verts_img[face_vertices[1] , 0])],
                            [(int(verts_img[face_vertices[2] , 1])),
                             int(verts_img[face_vertices[2] , 0])]]])
        tri2 = np.float32(
            [[[tw - texcoords[tc[0] ][1] * tw, texcoords[tc[0] ][0] * th],
              [tw - texcoords[tc[1] ][1] * tw, texcoords[tc[1] ][0] * th],
              [tw - texcoords[tc[2] ][1] * tw, texcoords[tc[2] ][0] * th]]])
        r1 = list(cv2.boundingRect(tri1))
        r2 = list(cv2.boundingRect(tri2))

        for si in range(len(r1)):
            r1[si] = max(r1[si], 0)
            r2[si] = max(r2[si], 0)

        if r2[1] + r2[3] > texture.shape[1]:
            r2[3] = texture.shape[1] - r2[1]

        tri1Cropped = []
        tri2Cropped = []

        for i in range(0, 3):
            tri1Cropped.append(((tri1[0][i][1] - r1[1]), (tri1[0][i][0] - r1[0])))  # 以矩形的左上角坐标点为原点，计算相应的点坐标
            tri2Cropped.append(((tri2[0][i][1] - r2[1]), (tri2[0][i][0] - r2[0])))

        # Apply warpImage to small rectangular patches
        img1Cropped = img[r1[0]:r1[0] + r1[2], r1[1]:r1[1] + r1[3]]
        warpMat = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))

        # Get mask by filling triangle
        mask = np.zeros((r2[2], r2[3], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)     # 将三角形填充到mask图像上

        try:
            # Apply the Affine Transform just found to the src image
            img2Cropped = cv2.warpAffine(img1Cropped, warpMat, (r2[3], r2[2]), None, flags=cv2.INTER_LINEAR,        # cv2.typing.Size: W,H
                                         borderMode=cv2.BORDER_REFLECT_101)         # 图像的patch变换到纹理的patch

            # Apply mask to cropped region
            img2Cropped = img2Cropped * mask    # 用mask去除三角形外面的矩形区域

            # Copy triangular region of the rectangular patch to the output image
            texture[r2[0]:r2[0] + r2[2], r2[1]:r2[1] + r2[3]] = texture[r2[0]:r2[0] + r2[2],
                                                                r2[1]:r2[1] + r2[3]] * ((1.0, 1.0, 1.0) - mask)

            texture[r2[0]:r2[0] + r2[2], r2[1]:r2[1] + r2[3]] = texture[r2[0]:r2[0] + r2[2],
                                                                r2[1]:r2[1] + r2[3]] + img2Cropped      # 填充回纹理图
        except Exception as e:
            print(e)
    texture = texture.astype(np.uint8)
    return texture

def plot_all_kpts(image, kpts, color='b'):
    if color == 'r':
        c = (0, 0, 255)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    elif color == 'p':
        c = (255, 100, 100)

    image = image.copy()
    kpts = kpts.copy()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        image = cv2.circle(image, (int(st[0]), int(st[1])), 1, c, 2)

    return image

def tensor_vis_landmarks(images, landmarks, color='g'):
    vis_landmarks = []
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    
    predicted_landmarks = landmarks.detach().cpu().numpy()

    for i in range(images.shape[0]):
        image = images[i]
        image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]].copy()
        image = (image * 255)
        predicted_landmark = predicted_landmarks[i]
        image_landmarks = plot_all_kpts(image, predicted_landmark, color)
        vis_landmarks.append(image_landmarks)

    vis_landmarks = np.stack(vis_landmarks)
    vis_landmarks = torch.from_numpy(
        vis_landmarks[:, :, :, [2, 1, 0]].transpose(0, 3, 1, 2)) / 255.  # , dtype=torch.float32)
    return vis_landmarks

def trans_point_screen(points, cam):
    R = torch.from_numpy(cam.R.T).cuda()
    T = torch.from_numpy(cam.T).cuda()[:,None]
    K = torch.from_numpy(cam.K).cuda()
    pt2d = (K @ (R @ points[..., None] + T)).squeeze(-1)  # B,n_lmks,3,1
    pt2d[..., 0] = pt2d[..., 0] / pt2d[..., 2]
    pt2d[..., 1] = pt2d[..., 1] / pt2d[..., 2]
    return pt2d[..., :2]

# def remove_neck(mask, faces, uv_idx):
#     boundary = mask.v.boundary
#     ears = mask.v.ears
#     scalp = mask.v.scalp
#     reserve_verts = torch.cat([
#         boundary,ears,scalp
#     ])
    
#     face_mask = torch.isin(faces, reserve_verts).any(dim=1)
#     return faces[~face_mask], uv_idx[~face_mask]

def reserve_(mask, faces, uv_idx):
    # lips = mask.v.lips
    face = mask.v.face

    reserve_verts = torch.cat([
        face
    ])
    
    face_mask = torch.isin(faces, reserve_verts).any(dim=1)
    
    return faces[face_mask], uv_idx[face_mask]

def remove_(mask, faces, uv_idx):
    boundary = mask.v.boundary
    ears = mask.v.ears
    hair = mask.v.hair
    irises = mask.v.irises
    eyeballs = mask.v.eyeballs
    eye_region = mask.v.eye_region
    left_eye = mask.v.left_eye
    right_eye = mask.v.right_eye
    left_eyeball = mask.v.left_eyeball
    left_eyelid = mask.v.left_eyelid

    right_eyeball = mask.v.right_eyeball
    right_eyelid = mask.v.right_eyelid

    left_eye_region = mask.v.left_eye_region
    right_eye_region = mask.v.right_eye_region
    right_ear = mask.v.right_ear
    left_ear = mask.v.left_ear

    eyelids = mask.v.eyelids
    forehead = mask.v.forehead
    scalp = mask.v.scalp
    sclerae = mask.v.sclerae

    bottomline = mask.v.bottomline
    nose = mask.v.nose
    lips = mask.v.lips
    front_middle_bottom_point_boundary = mask.v.front_middle_bottom_point_boundary

    # neck = mask.v.neck
    # neck_base = mask.v.neck_base
    neck_lower = mask.v.neck_lower

    neck_verts = torch.cat(
        [neck_lower, boundary, ears, eyeballs, eye_region, left_eye,
         right_eye,
         bottomline, front_middle_bottom_point_boundary, left_eyeball,
         right_eyeball, eyelids, left_ear, right_ear,
         hair, irises,
         scalp, sclerae])
    
    face_verts = mask.v.face
    
    face_in = torch.isin(faces, face_verts).any(dim=1)
    
    face_mask = torch.isin(faces, neck_verts).any(dim=1) 
    
    face_mask = torch.logical_or(~face_in, face_mask)
    
    return faces[~face_mask], uv_idx[~face_mask]
    # scalp = mask.v.scalp
    # boundary = mask.v.boundary
    # ears = mask.v.ears
    # neck = mask.v.neck
    # rm_verts = torch.cat([
    #     boundary, ears, neck
    # ])
    # face_mask = torch.isin(faces, rm_verts).any(dim=1)
    # return faces[~face_mask], uv_idx[~face_mask]

def merge_views(views):
    grid = []
    for view in views:
        grid.append(np.concatenate(view, axis=2))
    grid = np.concatenate(grid, axis=1)

    # tonemapping
    return to_image(grid)

def to_image(img):
    img = (img.transpose(1, 2, 0) * 255)[:, :, [2, 1, 0]]       # C,H,W->H,W,C   RBG->BGR
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def images_to_video(path, video_name, fps=25, video_format='DIVX'):
    logger.info(f"Generating {video_name}.avi from rendered results.")
    images_paths = sorted(glob.glob(os.path.join(path, '*.png')))
    img0 = cv2.imread(images_paths[0])
    height, width, _ = img0.shape
    size = (width, height)
    out = cv2.VideoWriter(os.path.join(Path(path).parent, f'{video_name}.avi'), cv2.VideoWriter_fourcc(*video_format), fps, size)
    
    for filename in tqdm(images_paths):
        img = cv2.imread(filename)
        out.write(img)
    out.release()

def parse_data(data, num_verts):
    flame_param = data['flame_param']
    flame_param['dynamic_offset'] = torch.zeros([1, num_verts, 3])
    flame_param['static_offset'] = torch.zeros([num_verts, 3])
    for k, v in flame_param.items():
        flame_param[k] = v.cuda()
    return data['camera'], flame_param

def render_mesh(dataloader, flame_model, mesh_render, save_folder, tex=None):
    num_timesteps = len(dataloader)
    merged_mesh_folder = os.path.join(save_folder, 'merged_mesh')

    if os.path.exists(merged_mesh_folder):
        shutil.rmtree(merged_mesh_folder)
    os.makedirs(merged_mesh_folder, exist_ok=True)
    
    flame_time = 0.0
    render_time = 0.0
    
    verts_uvs = flame_model.verts_uvs
    verts_uvs[:, 1] = 1 - verts_uvs[:, 1]
    
    faces, textures_idx = remove_(flame_model.mask, flame_model.faces, flame_model.textures_idx)
    
    logger.info('Rendering mesh using FLAME fitting results.')
    for data in tqdm(dataloader):
        cam, flame_param = parse_data(data, flame_model.v_template.shape[0])
        flame_start = time.time()
        
        verts, lmks3d, mp_lmks3d = flame_model(
            flame_param['shape'],
            flame_param['expr'],
            flame_param['rotation'],
            flame_param['neck_pose'],
            flame_param['jaw_pose'],
            flame_param['eyes_pose'],
            flame_param['translation'],
            zero_centered_at_root_node=False,
            return_landmarks=True,
            return_verts_cano=False,
            static_offset=flame_param['static_offset'],
            dynamic_offset=flame_param['dynamic_offset'],
            eyelids=flame_param['eyelids']
        )
        
        flame_end = time.time()
        
        image = np.array(cam.original_image[0:3, :, :])
        
        lmk2d = trans_point_screen(lmks3d, cam)
        mp_lmk2d = trans_point_screen(mp_lmks3d, cam)
        
        pred_lmks = image.copy()
        pred_lmks = tensor_vis_landmarks(pred_lmks[np.newaxis,...], torch.cat([mp_lmk2d, lmk2d[:, :17, :]], dim=1), 'g')
        
        render_start = time.time()
        out_dict = mesh_render.render_from_camera(verts, faces, cam, textures_idx=textures_idx.to(torch.int32), verts_uvs=verts_uvs, tex=tex)
        render_end = time.time()
        
        rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1).to("cpu").numpy() # (C, H, W)
        rgb_mesh = rgba_mesh[:3, :, :]
        alpha_mesh = rgba_mesh[3:, :, :]
        # mesh_opacity = 0.5
        # rendered_mesh = rgb_mesh * alpha_mesh * mesh_opacity  + image * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
        alpha_mesh[0] = cv2.erode(alpha_mesh[0], cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        rendered_mesh = rgb_mesh * alpha_mesh + image * (1 - alpha_mesh)
        
        merged = merge_views([[image, pred_lmks[0], rendered_mesh, rgb_mesh]])
        cv2.imwrite(os.path.join(merged_mesh_folder, f'{cam.image_name}.png'), merged)
        
        flame_time += (flame_end - flame_start)
        render_time += (render_end - render_start)
    
    images_to_video(merged_mesh_folder, 'merged_meshes')
    
    total_time = flame_time + render_time
    fps = num_timesteps / total_time
    
    time_data = [{
        'fps': fps,
        'flame time per frame': flame_time * 1000 / num_timesteps,      # ms
        'render time per frame': render_time * 1000 / num_timesteps,
        'total time': total_time * 1000 / num_timesteps
    }]
    
    with open(os.path.join(save_folder, 'render_time.json'), 'w') as fp:
        json.dump(time_data, fp)

def gen_tex(dataset, flame_model, index, save_folder):
    tex_path = os.path.join(save_folder, "texture.png")
    if os.path.exists(tex_path):
        texture_img = cv2.imread(tex_path)
        texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)
    else:
        cam, flame_param = parse_data(dataset[index], flame_model.v_template.shape[0])
        verts = flame_model(
                flame_param['shape'],
                flame_param['expr'],
                flame_param['rotation'],
                flame_param['neck_pose'],
                flame_param['jaw_pose'],
                flame_param['eyes_pose'],
                flame_param['translation'],
                zero_centered_at_root_node=False,
                return_landmarks=False,
                return_verts_cano=False,
                static_offset=flame_param['static_offset'],
                dynamic_offset=flame_param['dynamic_offset'],
                eyelids=flame_param['eyelids']
        )
        
        verts_screen = trans_point_screen(verts, cam)
        img = np.array(cam.original_image[0:3, :, :]).transpose((1, 2, 0)) * 255
        texture_img = get_texture(img, verts_screen[0].detach().cpu().numpy(),
                                    flame_model,
                                    tw=1024,
                                    th=1024)
        texture_img[texture_img < 0] = 0
        texture_img[texture_img > 255] = 255
        texture_img = texture_img.astype(np.uint8)
        cv2.imwrite(os.path.join(save_folder, "texture.png"), texture_img[:, :, ::-1])
    return torch.from_numpy(texture_img / 255.0).to(torch.float32).cuda().contiguous()
   
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/000")
    parser.add_argument("--frame_id_gen_tex", type=int, default=0)
    args = parser.parse_args()
    
    images_folder = os.path.join(args.data_path, "input")
    flame_param_folder = os.path.join(args.data_path, 'checkpoint')
    
    mesh_render = NVDiffRenderer(use_opengl=False)
    flame_model = FlameHead(shape_params=300, expr_params=100, add_teeth=False).cuda()
    cams = CameraDatasetDisk(images_folder, flame_param_folder)
    
    tex = gen_tex(cams, flame_model, args.frame_id_gen_tex, args.data_path)
    
    dataloder = DataLoader(cams, batch_size=None, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    
    render_mesh(dataloder, flame_model, mesh_render, args.data_path, tex)
    
    logger.info('Everything is OK!')