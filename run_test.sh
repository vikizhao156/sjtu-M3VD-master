#!/bin/bash
#SBATCH -J VIKI1
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH --gres=gpu:1
module load anaconda2/5.3.0
source activate viki
python3 cal_predict.py

