#!/bin/bash
#SBATCH --job-name=QA_fine_tune_ibd214
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ibd214/project/%j_%x.out
#SBATCH --error=/scratch/ibd214/project/%j_%x.err
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 4
#SBATCH --mail-user=<ibd214@nyu.edu>
#SBATCH --mail-type=ALL

singularity exec --nv --bind /scratch/ibd214 --overlay /scratch/ibd214/overlay-25GB-500K.ext3:ro /scratch/ibd214/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
cd /scratch/ibd214
python3 LRQA/run_lrqa.py \
    --model_name_or_path roberta-base \
    --model_mode mc \
    --task_name race \
    --output_dir /scratch/ibd214/project/proj_out \
    --overwrite_output_dir \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy no \
    --num_train_epochs 1
"
