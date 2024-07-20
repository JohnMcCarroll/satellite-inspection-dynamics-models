#!/bin/bash

# Lists of arguments
# prediction_sizes=(10 20 30)
predict_deltas=(True False)
constrain_outputs=(True False)

# Iterate over all combinations of the lists

for predict_delta in "${predict_deltas[@]}"; do
    for constrain_output in "${constrain_outputs[@]}"; do
        # Call the Python script with the arguments
        python3 train.py --predict_delta "$predict_delta" --constrain_output "$constrain_output"
    done
done
