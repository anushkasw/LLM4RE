#!/bin/bash
set -x

echo "Submitting jobs..."

# List of datasets and models
datasets=('FewRel' 'crossRE')
models=('lmsys/vicuna-7b-v1.5')

# Loop through each dataset and model
for data in "${datasets[@]}"; do
    # Ensure the log directory exists for the current dataset
    mkdir -p ./log/$data

    for model in "${models[@]}"; do
        # Sanitize job name to avoid special characters
        JOBNAME="${data}_$(echo $model | tr '/' '_')"
        LOGFILE="log/$data/$JOBNAME.log"

        echo "Submitting job for dataset: $data, model: $model"

        # Submit the job with sbatch, passing both dataset and model as arguments
        sbatch --job-name=$JOBNAME --output=$LOGFILE run.sh "$data" "$model"

        echo "Job submitted for dataset: $data, model: $model"
    done
done

echo "All jobs submitted."