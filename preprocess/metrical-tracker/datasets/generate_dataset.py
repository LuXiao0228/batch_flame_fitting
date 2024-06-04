import os
from abc import ABC
from glob import glob
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm

from face_detector import FaceDetector
from image import crop_image_bbox, squarefiy, get_bbox


class GeneratorDataset(Dataset, ABC):
    def __init__(self, source, config):
        self.device = 'cuda:0'
        self.config = config
        self.source = Path(source)

        self.initialize()
        self.face_detector_mediapipe = FaceDetector('google')
        self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=self.device)

    def initialize(self):
        path = Path(self.source, 'source')
        if not path.exists() or len(os.listdir(str(path))) == 0:
            video_file = self.source / 'video.mp4'
            if not os.path.exists(video_file):
                logger.error(f'[ImagesDataset] Neither images nor a video was provided! Execution has stopped! {self.source}')
                exit(1)
            path.mkdir(parents=True, exist_ok=True)
            os.system(f'ffmpeg -i {video_file} -vf fps={self.config.fps} -q:v 1 {self.source}/source/%05d.png')

        self.images = sorted(glob(os.path.join(self.source, 'source', '*.png')))

    def process_face(self, image):
        lmks, scores, detected_faces = self.face_detector.get_landmarks_from_image(image, return_landmark_score=True, return_bboxes=True)
        if detected_faces is None:
            lmks = None
        else:
            lmks = lmks[0]
        dense_lmks = self.face_detector_mediapipe.dense(image)
        return lmks, dense_lmks

    def run(self):
        logger.info('Generating dataset...')
        bbox = None
        bbox_path = os.path.join(self.config.actor,"bbox.pt")

        if os.path.exists(bbox_path):
            bbox = torch.load(bbox_path)
        
        idx = 0
        lmk_folder = os.path.join(self.source, 'lmk_kpt')
        lmk_dense_folder = os.path.join(self.source, 'lmk_kpt_dense')
        image_folder = os.path.join(self.source, 'images')
        

        for imagepath in tqdm(self.images):     # 检测landmark部分, TODO: 移除无法识别landmark的帧
            need_save = True
            # lmk_path = imagepath.replace('source', 'lmk_kpt').replace('png', 'npy').replace('jpg', 'npy')
            # lmk_path_dense = imagepath.replace('source', 'lmk_kpt_dense').replace('png', 'npy').replace('jpg', 'npy')
            lmk_path = os.path.join(lmk_folder, f'{idx:05d}.npy')
            lmk_path_dense = os.path.join(lmk_dense_folder, f'{idx:05d}.npy')
            save_image_path = os.path.join(image_folder, f'{idx:05d}.png')

            if not os.path.exists(lmk_path) or not os.path.exists(lmk_path_dense):
                image = cv2.imread(imagepath)
                h, w, c = image.shape

                if bbox is None and self.config.crop_image:
                    lmk, _ = self.process_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # estimate initial bbox
                    bbox = get_bbox(image, lmk, bb_scale=self.config.bbox_scale)
                    torch.save(bbox, bbox_path)

                if self.config.crop_image:
                    image = crop_image_bbox(image, bbox)
                    if self.config.image_size[0] == self.config.image_size[1]:
                        image = squarefiy(image, size=self.config.image_size[0])
                else:
                    image = cv2.resize(image, (self.config.image_size[1], self.config.image_size[0]), interpolation=cv2.INTER_CUBIC)

                lmk, dense_lmk = self.process_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if lmk is None:
                    logger.info(f'Empty face_alignment lmks for path: ' + imagepath)
                    # lmk = np.zeros([68, 2])

                if dense_lmk is None:
                    logger.info(f'Empty mediapipe lmks for path: ' + imagepath)
                    # dense_lmk = np.zeros([478, 2])
                
                if lmk is None or dense_lmk is None:
                    need_save = False

                if need_save:
                    Path(lmk_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(lmk_path_dense).parent.mkdir(parents=True, exist_ok=True)
                    # Path(imagepath.replace('source', 'images')).parent.mkdir(parents=True, exist_ok=True)
                    Path(save_image_path).parent.mkdir(parents=True, exist_ok=True)

                    # cv2.imwrite(imagepath.replace('source', 'images'), image)
                    cv2.imwrite(save_image_path, image)
                    np.save(lmk_path_dense, dense_lmk)
                    np.save(lmk_path, lmk)
                    
                    idx += 1    # 跳过检测不到landmark的图片
            else:      
                idx += 1
