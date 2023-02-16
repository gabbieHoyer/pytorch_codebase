#!/bin/bash

#SBATCH -J baselines
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --partition=dgx
#SBATCH --gres=gpu:teslav100:8
#SBATCH --chdir=/data/VirtualAging/users/ghoyer/ssl/baselines/classifier/src/

#Load environment
export PATH=/netopt/rhel7/bin:$PATH
eval "$('/netopt/rhel7/versions/python/Anaconda3-edge/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
conda activate /data/VirtualAging/users/ghoyer/conda/envs/pytorch_env
cd /data/VirtualAging/users/ghoyer/ssl/baselines/classifier/src/

#Run my jobs
#########cifar10
python main.py --no_distributed --dataset 'cifar10' --n_classes 10 --learning_rate 0.01 --no_pretrained --use_net 'resnet' --experiment_desc 'cifar10 scratch w resnet' --visible_gpu 2       


python main.py --no_distributed --dataset 'cifar10'--n_classes 10 --learning_rate 0.01 --no_pretrained --use_net 'densenet' --experiment_desc 'cifar10 scratch w densenet' --visible_gpu 2    


python main.py --no_distributed --dataset 'fashionmnist' --n_classes 10 --learning_rate 0.01 --no_pretrained --use_net 'resnet' --experiment_desc 'fashionmnist scratch w resnet' --visible_gpu 2       


python main.py --no_distributed --dataset 'fashionmnist' --n_classes 10 --learning_rate 0.01 --no_pretrained --use_net 'densenet' --experiment_desc 'fashionmnist scratch w densenet' --visible_gpu 2 
