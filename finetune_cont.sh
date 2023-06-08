#!/usr/bin/bash
# Request a number of GPU cards, in this case 1 (the maximum is 2)
#$ -l gpu=true
# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=48:10:0
# Request a number of CPU threads
# -l
# tmem: GPU memory
#$ -l tmem=200G
# Set the name of the job.
#$ -N m
# Set the working directory
#$ -wd /home/xiaowshi/Mono


# Path variables
path_script_inference="/home/xiaowshi/Mono/test_simple.py"
path_script_train="/home/xiaowshi/Mono/train.py"

# Activate the venv
source /share/apps/source_files/python/python-3.8.5.source
source ~/Mono-Depth/bin/activate

# Exporting CUDA Paths. cuDNN included in cuda paths.
# Add the CUDA Path
export PATH=/share/apps/cuda-10.1/bin:/usr/local/cuda-10.1/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/cuda-10.1/lib64:/usr/local/cuda-10.1/lib:/lib64:${LD_LIBRARY_PATH}
export CUDA_INC_DIR=/share/apps/cuda-10.1/include
export LIBRARY_PATH=/share/apps/cuda-10.1/lib64:/usr/local/cuda-10.1/lib:/lib64:${LIBRARY_PATH}

# Run commands with Python
# Example to check that the modules are correctly installed and callable, especially pycuda and tensorflow.

#python3 ${path_script_inference} --image_path assets/test_image.jpg --model_name mono_640x192
python3 ${path_script_train} --model_name $1 --split endovis --data_path /SAN/medic/monodepth_laparoscope/SCARED --dataset endovis  --num_epochs 200 --log_dir models --load_weights_folder $2 --width 320 --height 256 --dpt

# --min_depth 0.1 --max_depth 100.0
#--dpt

# --batch_size 12
# models/mono_640x192
# tmp/finetuned_mono/models/weights_75
# tmp/mono_model/models/weights_19
#train_scared/mono_model/models/
#--learning_rate 5e-5
# models/cont_43/models/


