#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mem=32GB
#SBATCH --job-name=triplet_loss_raccoons
#SBATCH --mail-user=alnah005@umn.edu
#SBATCH -p v100                                            
#SBATCH --gres=gpu:v100:1

cd $SLURM_SUBMIT_DIR
module load singularity
pwd
singularity exec --nv -i nv_od_local_v1.sif  raccoon_identification/Triplet_Loss_Framework/commands.sh && cd $SLURM_SUBMIT_DIR && cd ../Automatic_labeling_experiments && sbatch job.sh
