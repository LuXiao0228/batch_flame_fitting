import yaml
import argparse
import json
import os
import glob
from PIL import Image
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="/cfs/xiaolu/code/gs_avatar/IMavatar/preprocess/datasets/xk")
    parser.add_argument("--yml_name", type=str, default="config")
    parser.add_argument("--keyframe_ids", type=str, default="1,2")
    parser.add_argument("--begin_frames", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    dataset_folder = args.dataset_folder
    yml_file_path = os.path.join(dataset_folder, f'{args.yml_name}.yml')
    bbox_file = os.path.join(dataset_folder, "box_bound.json")
    
    # crop_img = False
    if os.path.exists(bbox_file):
        # get image size
        with open(bbox_file) as f:
            bbox = json.load(f)[0]
        image_size = [int(bbox['h_cropped']), int(bbox['w_cropped'])]
    else:
        images_path = sorted(glob.glob(os.path.join(dataset_folder, 'source', '*.png')))
        imag = Image.open(images_path[0])
        image_size = [imag.size[1], imag.size[0]]
        # crop_img = True
    
    kf = [int(s) for s in args.keyframe_ids.split(',')]
    
    data = {
        "actor": dataset_folder,
        "save_folder": dataset_folder,
        "optimize_shape": True,
        "optimize_jaw": True,
        "optimize_neck": False,
        "begin_frames": args.begin_frames,
        "keyframes": kf,
        "crop_image": False,
        "image_size": image_size,
        "batch_size": args.batch_size,
    }
    
    with open(yml_file_path, 'w') as fp:
        yaml.dump(data, fp, default_flow_style=False, allow_unicode=True)
    
    print(f'{args.yml_name}.yml writing completed!')