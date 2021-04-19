#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --mail-type=ALL
#SBATCH --mem=32GB
#SBATCH --job-name=transfer_learning_raccoons
#SBATCH --mail-user=alnah005@umn.edu
#SBATCH -p v100                                             
#SBATCH --gres=gpu:v100:1

cd $SLURM_SUBMIT_DIR
module load singularity
singularity exec --nv -i tfod_latest.sif transferLearning/commands.sh