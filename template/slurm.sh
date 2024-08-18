#!/bin/bash

# 'FewRel' 'crossRE' 'NYT10' 'tacred' 'retacred' 'WebNLG' 'semeval_nodir'
for data in 'FewRel' 'crossRE' 'NYT10' 'tacred' 'retacred' 'WebNLG' 'semeval_nodir'
do
    for model in 'meta-llama/Meta-Llama-3.1-8B-Instruct' 'mistralai/Mistral-Large-Instruct-2407' 'microsoft/Phi-3-mini-4k-instruct' 'lmsys/vicuna-7b-v1.5'
    do
        DATASET="$data"
        MODEL="$model"
        echo "==========${DATASET}==========="

        mkdir -p ./log/$DATASET

        JOBNAME="${DATASET}_${MODEL}_1"

        # k-shot slurm script
        sbatch \
        --job-name=$JOBNAME \
        --output=log/$DATASET/$JOBNAME.log \
        run.sh "$DATASET" "$MODEL"  # Pass dataset and model as arguments
    done
done