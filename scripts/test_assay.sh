#!/bin/bash

# Lists of arguments
# predict_deltas=(True False)
# constrain_outputs=(True False)
predict_deltas=(False True)
constrain_outputs=(False True)

for predict_delta in "${predict_deltas[@]}"; do
    for constrain_output in "${constrain_outputs[@]}"; do
        python scripts/train.py --predict_delta "$predict_delta" --constrain_output "$constrain_output"
    done
done

for predict_delta in "${predict_deltas[@]}"; do
    for constrain_output in "${constrain_outputs[@]}"; do
        python scripts/train_rnn.py --predict_delta "$predict_delta" --constrain_output "$constrain_output"
    done
done
