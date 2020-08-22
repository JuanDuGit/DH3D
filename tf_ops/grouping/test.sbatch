#!/bin/bash
#SBATCH --job-name="checkdg"
#SBATCH --qos=cvpr2019
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:titanxpascal:1,VRAM:6G
#SBATCH --time=00:05:00
#SBATCH --mail-type=None
#SBATCH --output=/usr/stud/duj/Desktop/proj/others/flexmine/check_group.out
source /usr/stud/duj/buildtf19/bin/activate
export PATH=/usr/local/cuda-9.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/stud/duj/.local/cuda/lib64:$LD_LIBRARY_PATH
srun python tf_grouping_op_test.py
