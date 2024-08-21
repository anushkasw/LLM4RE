#!/bin/bash
#SBATCH --job-name=test_selgen
#SBATCH --output=test_selgen_output
#SBATCH --mail-user=tpan1@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --time=2-00:00:00             # Time limit hrs:min:sec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=30gb                 # Job Memory
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
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

export CUDA_VISIBLE_DEVICES=0

python3 /blue/woodard/share/Relation-Extraction/bell/LLM4RE/SelfGen/SelfGen_RE/sg_icl.py \
  --task "dummy_tacred" \
  --data_dir "/blue/woodard/share/Relation-Extraction/Data" \
  --model_name_or_path "mistralai/Mistral-Large-Instruct-2407" \
  --output_dir "/blue/woodard/share/Relation-Extraction/bell/LLM4RE/SelfGen/SelfGen_RE/output" \
  --seed 42 \
  --label_token '[LABEL]' \
  --generation_max_length 30 \
  --generation_min_length 10 \
  --no_repeat_ngram_size 2 \
  --temperature 0.5 \
  --access_token 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 \
  --demo sel_gen

echo "Script execution complete!"