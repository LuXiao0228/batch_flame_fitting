export CUDA_VISIBLE_DEVICES=0
#############################################################################
# Things need to modify
subject_name='xk'
path='/cfs/xiaolu/code/gs_avatar/flame_fitting/data'
video_folder=$path/$subject_name
video_name='video.mp4'
fps=25

#############################################################################
pwd=$(pwd)
path_preprocess=$pwd/preprocess
path_MICA=$path_preprocess/MICA
path_tracker=$path_preprocess/metrical-tracker

#############################################################################
set -e
# extract images from video, saved in folder "source"

mkdir -p $video_folder/'source'
ffmpeg -i $video_folder/$video_name -vf fps=$fps -q:v 1 $video_folder/'source'/'%5d.png'

#############################################################################
# get identity code(shape) of subject using MICA shape predictor

cd $path_MICA
python my_demo.py \
    --ident_save_folder $video_folder \
    -i $video_folder/'source'/00001.png \
    -o $video_folder/'mica_output' \
    -a $video_folder/'arcface' \
    -m data/pretrained/mica.tar 

##############################################################################
# Create .yml script to prepare for head tracking

cd $path_preprocess
python touch_subject_yml.py \
    --dataset_folder $video_folder \
    --yml_name $subject_name \
    --keyframe_ids "0,1" \
    --batch_size 32

cat $video_folder/$subject_name.yml

##############################################################################
# tracking head using metrical-tracker

cd $path_tracker
python tracker_batch_wo_tex.py --cfg $video_folder/${subject_name}.yml
