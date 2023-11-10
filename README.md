## Pipeline
<p align="center">
<img src='assets/img/SL.png' width=600/> 
</p>

## Requirements
torchvision 0.12.0, CUDA 10.2, tensorboardX 1.4 

If you're using older version of torchvision, you need to: 
- add 'align_corners=True' to 'F.interpolate' and 'F.grid_sample' when you train the network, to get a good camera trajectory.
- replace transforms.ColorJitter.get_params() with transforms.ColorJitter().

## Dataset
The dataset is available on the [SCARED official website](https://endovissub2019-scared.grand-challenge.org).

## Date preprocess
```shell
ffmpeg -i /path/to/rgb.mp4 -filter:v "crop=1280:1024:0:0" /path/to/crop_rgb.mp4
ffmpeg -i /path/to/crop_rgb.mp4 %6d.jpg
```

## Training
```shell
CUDA_VISIBLE_DEVICES=1 python train.py --data_path <path/to/SCARED/> --log_dir <path/to/save/weights>  --batch_size 20 --frames -2 -1 0 1 2
```

## Validation
```shell
CUDA_VISIBLE_DEVICES=1 python evaluate_depth.py --load_weights_folder <weights_path> --eval_mono  --eval_split endovis --data_path </path/to/SCARED> --max_depth 150.0
CUDA_VISIBLE_DEVICES=1 python evaluate_pose.py --data_path <path/to/SCARED/> --eval_split endovis --load_weights_folder <weights_path>
```

## Testing
| Model        | Abs Rel | Sq Rel | RMSE | RMSE log | Link |
| ------------ | ---------- | ------ | --------- | ---- | ---- |
| Results | 0.071 | 0.596 | 5.608 | 0.095 | [google drive](https://drive.google.com/file/d/1r6QRf7ZGconNBuBit9CK3Rr-3slQ9toq/view?usp=sharing)| 

```shell
CUDA_VISIBLE_DEVICES=1 python test_simple.py --model_path <model_path> --image_path <image_path>
```

<p float="left">
<img src="assets/img/input.gif" height="250">
<img src="assets/img/depth-min.gif" height="250">
</p>

## Colab Demos

- [Colab 1: Dataloader] (https://colab.research.google.com/drive/18TyEVJRmUQJ3Jd-Rc9bOQf3mgVHYLv08?usp=sharing): Visualise output of the dataloader.
- [Colab 2: Evaluation] (https://colab.research.google.com/drive/1q7y_f95LhsG40iXEFeQ98sG1Opd-85F4?usp=sharing): Calculate the depth and pose metrics and visualise trajectory using pretrained weights.