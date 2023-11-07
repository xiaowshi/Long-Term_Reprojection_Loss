#### Pipeline
<p align="center">
<img src='assets/img/SL.png' width=600/> 
</p>

## Requirements
torchvision 0.12.0, CUDA 10.2, tensorboardX 1.4 

## Dataset
The dataset is available on the [SCARED official website](https://endovissub2019-scared.grand-challenge.org).

## Date preprocess
```shell
ffmpeg -i /path/to/rgb.mp4 -filter:v "crop=1280:1024:0:0" /path/to/crop_rgb.mp4
ffmpeg -i /path/to/crop_rgb.mp4 %6d.jpg
```

## Training

## Validation
```shell
CUDA_VISIBLE_DEVICES=1 python evaluate_depth.py --load_weights_folder <weights_path> --eval_mono  --eval_split endovis --data_path </path/to/SCARED> --max_depth 150.0
CUDA_VISIBLE_DEVICES=1 python evaluate_pose.py --data_path <path/to/SCARED/> --eval_split endovis --load_weights_folder <weights_path>
```

## Testing
```shell
CUDA_VISIBLE_DEVICES=1 python test_simple.py --model_path <model_path> --image_path <image_path>
```
