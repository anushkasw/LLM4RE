#!/bin/bash
#SBATCH --job-name=test_seladapt
#SBATCH --output=test_seladapt_output
#SBATCH --mail-user=tpan1@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --time=1-00:00:00             # Time limit hrs:min:sec
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 /blue/woodard/share/Relation-Extraction/bell/LLM4RE/SelfAdapt/SelfAdapt_RE/self_adapt_re.py \
  --data_dir "/blue/woodard/share/Relation-Extraction/Data" \
  --task "dummy_tacred" \
  --retriever_model "all-mpnet-base-v2" \
  --output_file "/blue/woodard/share/Relation-Extraction/bell/LLM4RE/SelfAdapt/SelfAdapt_RE/output/test.json" \
  --index_file "/blue/woodard/share/Relation-Extraction/bell/LLM4RE/SelfAdapt/SelfAdapt_RE/index/" \
  --batch_size 32 \
  --num_candidates 10 \
  --num_ice 8 \
  --method "topk" \
  --rand_seed 42 \
  --overwrite "True" \
  --model_name "gpt2-xl" \
  --field "token" \
  --dataset_split "validation"

echo "Script execution complete!"