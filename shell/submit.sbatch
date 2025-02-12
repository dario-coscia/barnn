
#!/bin/bash

# SLURM job options
#SBATCH --partition=YOUR GPU here
#SBATCH --job-name=run-barnn
#SBATCH --nodes=2
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --mem=YOUR MEMORY (32000 in our experiments)
#SBATCH --time=12:00:00
#SBATCH --output=%x.o%j.%N
#SBATCH --error=%x.e%j.%N

# Print job details
NOW=`date +%H:%M-%a-%d/%b/%Y`
echo '------------------------------------------------------'
echo 'This job is allocated on '$SLURM_JOB_CPUS_PER_NODE' cpu(s)'
echo 'Job is running on node(s): '
echo  $SLURM_JOB_NODELIST
echo '------------------------------------------------------'
#
# ==== End of Info part (say things) ===== #
#

cd $SLURM_SUBMIT_DIR

export SLURM_NTASKS_PER_NODE=2  # need to export this (not all clusters need this)

# Load required modules (adjust according to your environment)
module load cuda/12.1  # Adjust for your CUDA version if needed
conda activate barnn_env

# Run the PyTorch Lightning script
srun python scripts/run_mol.py