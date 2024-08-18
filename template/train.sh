#!/bin/bash
#SBATCH --job-name=test_HF
#SBATCH --output=test_HF_output
#SBATCH --mail-user=tpan1@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=1-00:00:00             # Time limit hrs:min:sec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10gb                 # Job Memory
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --qos=woodard
# This script runs different tasks on a SLURM cluster

echo "Start Date : $(date)"
echo "Host       : $(hostname -s)"
echo "Directory  : $(pwd)"
echo "Running $SLURM_JOB_NAME on $SLURM_CPUS_ON_NODE CPU cores"; echo
start_time=$(date +%s)

module purge 
module load conda cuda
export PATH=/blue/woodard/share/Relation-Extraction/conda_envs/RE_LLM/bin:$PATH
conda activate /blue/woodard/share/Relation-Extraction/conda_envs/RE_LLM

export CUDA_VISIBLE_DEVICES=0,1

# Set dataset variable
dataset=$1
model=$2

for k in 1 5 10 20 30
do
    for seed in 13 42 100
    do
        echo "K: $k, seed: $seed, dataset: $dataset"
        echo "--------------------------------------------------------------------"; echo

        python3 /blue/woodard/share/Relation-Extraction/LLM_for_RE/template/main_HF.py \
        --task "$dataset" \
        --data_dir /blue/woodard/share/Relation-Extraction/Data \
        --k "$k" \
        --data_seed "$seed" \
        --out_path /blue/woodard/share/Relation-Extraction/LLM_for_RE/output/ \
        --model "$model" \
        --api_key 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
    done
done

echo "--------------------------------------------------------------------"; echo

end_time=$(date +%s)  # Record end time in seconds since epoch
elapsed_time=$((end_time - start_time))

# Calculate days, hours, minutes, and seconds
days=$((elapsed_time / 86400))  # 86400 seconds in a day
remaining_seconds=$((elapsed_time % 86400))
hours=$((remaining_seconds / 3600))  # 3600 seconds in an hour
remaining_seconds=$((remaining_seconds % 3600))
minutes=$((remaining_seconds / 60))
seconds=$((remaining_seconds % 60))
echo; echo "Elapsed time: $days-$hours:$minutes:$seconds"; echo

echo; echo "End Date : $(date)"
echo "Host       : $(hostname -s)"
echo "Directory  : $(pwd)"; echo