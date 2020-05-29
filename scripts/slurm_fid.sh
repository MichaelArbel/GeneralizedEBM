#!/bin/bash
#
#SBATCH --job-name=kale_gan_train
#SBATCH --output=scripts/logs/test_%A.log
#SBATCH --time=2-12:00  # 2.5 days
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx2080:1

##echo "Loading CUDA 9.0"
##module add nvidia/9.0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "Using config file $1"
echo "Starting the job..."
##git commit -a -m "$SLURM_JOB_ID $3"
python main.py --config=$1 --latent_sampler=$2 --slurm_id=$SLURM_JOB_ID --temperature=$3

echo "Done"
exit