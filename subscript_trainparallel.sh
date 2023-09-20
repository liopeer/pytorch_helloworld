#!/bin/bash
#SBATCH  --output=log/%j.out
#SBATCH  --gres=gpu:2
#SBATCH  --mem=10G
#SBATCH  --job-name=double
#SBATCH  --constraint='geforce_gtx_titan_x|geforce_rtx_2080_ti|titan_xp|geforce_gtx_1080_ti|titan_x|a100'

source /scratch_net/biwidl311/peerli/conda/etc/profile.d/conda.sh
conda activate liotorch
mkdir log
python -u train_parallel.py "$@"