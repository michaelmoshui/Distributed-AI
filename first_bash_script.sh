#!/bin/bash

# Ensure the script receives exactly two arguments: the environment name and the file to execute
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <conda_environment_name> <python_file_to_execute>"
    exit 1
fi

# Assign arguments to variables
CONDA_ENV_NAME=$1
FILE_TO_EXECUTE=$2

# Activate the specified Conda environment
source ~/anaconda3/bin/activate "$CONDA_ENV_NAME"

# Check if the conda environment activation was successful
if [ $? -ne 0 ]; then
    echo "Failed to activate Conda environment: $CONDA_ENV_NAME"
    exit 1
fi

# Run the specified Python file
python "$FILE_TO_EXECUTE"

# Deactivate the Conda environment after execution
conda deactivate
