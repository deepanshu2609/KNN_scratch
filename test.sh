#                             Online Bash Shell.
#                 Code, Compile, Run and Debug Bash script online.
# Write your code in this editor and press "Run" button to execute it.

#!/bin/bash

# Check if the correct number of arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <dataset_file.npy> <python_script.py>"
    exit 1
fi

# Assign arguments to variables
dataset_file="$1"
python_script="$2"

# Check if the dataset file exists
if [ ! -f "$dataset_file" ]; then
    echo "Dataset file not found: $dataset_file"
    exit 1
fi

# Check if the python script exists
if [ ! -f "$python_script" ]; then
    echo "Python script not found: $python_script"
    exit 1
fi

# Run the python script with the dataset file as argument
python3 "$python_script" "$dataset_file"
