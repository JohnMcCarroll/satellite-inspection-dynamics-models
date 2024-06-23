"""
This script evaluates the model's prediction accuracy as a function of timesteps.
"""
import numpy as np
import torch
from load_dataset import load_test_dataset
from train import MLP256, MLP1024
import matplotlib.pyplot as plt
import copy
import pickle
import itertools
from typing import Optional
from pathlib import Path
from collections import defaultdict

# TODO - refactor this somewhere sensible
NAMED_STATE_RANGES = {
    "all": slice(0, 12),
    "position": slice(0, 3),
    "velocity": slice(3, 6),
    "inspected_points": slice(6, 7),
    "uninspected_points": slice(7, 10),
    "sun_angle": slice(10, 12),
}


# Calculate error between model output and target vector
def euclidean_distance(output, target):
    return np.sqrt(np.sum((output - target) ** 2))


def get_eval_data(models: dict, input_size=15, output_size=12, max_steps: int = 49,
                  save_file: Optional[Path] = None):
    # TODO - would be better to decouple this function from the model dict. Instead, pass in a
    #  single model at a time and save out one pkl file per model.
    if save_file is not None and save_file.exists():
        with open(str(save_file), 'rb') as file:
            return pickle.load(file)

    # Load test dataset from file
    test_df = load_test_dataset()

    # Initialize data structure to store evaluation data
    eval_data = {}

    # Evaluate model(s)
    for model_name, model_cfg in models.items():
        # Load the trained model
        model = model_cfg[0](input_size, output_size)
        model.load_state_dict(torch.load(model_cfg[1]))
        model.eval()

        # Evaluate the model on different slices of the data. Initialize errors so that
        # errors[state] is an array of size #trajectories x #steps. Pre-fill it with NaNs so that
        # quantiles will simply ignore missing data from jagged arrays.
        errors = defaultdict(lambda: np.full((len(test_df), max_steps), np.nan))

        with torch.no_grad():
            for trajectory_idx, trajectory in enumerate(test_df['Trajectory']):
                # Create multistep predictions for each trajectory in test dataset
                final_state_index = len(trajectory) - 1
                multistep_predictions = {}
                target_states = {}
                # final_state = trajectory[final_state_index][0:11]
                # TODO - vectorize over states by treating `trajectory` as a batch
                for start_t in range(final_state_index):
                    # predicted_trajectory = []
                    state_action = torch.tensor(trajectory[start_t], dtype=torch.float32)
                    future_actions = [state_action[-3:] for state_action in
                                      trajectory[start_t + 1:final_state_index]]

                    for delta_t in range(min(max_steps, final_state_index - start_t, len(future_actions))):
                        # Use the model to predict delta_t steps into the future
                        predicted_state = model(state_action)
                        # store model's prediction and ground truth target state
                        multistep_predictions[(start_t, start_t + delta_t + 1)] = predicted_state
                        target_states[(start_t, start_t + delta_t + 1)] = \
                            trajectory[start_t + delta_t + 1][0:12]
                        # Concatentate model's predicted state with next taken action
                        state_action = torch.tensor(
                            np.concatenate((predicted_state, future_actions[delta_t])),
                            dtype=torch.float32)

                for t1_t2, v in multistep_predictions.items():
                    num_steps = t1_t2[1] - t1_t2[0]
                    output = v.numpy()
                    for state_key, state_range in NAMED_STATE_RANGES.items():
                        errors[state_key][trajectory_idx, num_steps - 1] = euclidean_distance(
                            output[state_range],
                            target_states[t1_t2][state_range]
                        )

        # Calculate summary statistics of different slices of the data
        eval_data[model_name] = {"steps": np.arange(1, max_steps + 1)}
        for state_key in NAMED_STATE_RANGES.keys():
            eval_data[model_name][state_key] = {
                "median": np.nanmedian(errors[state_key], axis=0),
                "quantiles_25": np.nanquantile(errors[state_key], 0.25, axis=0),
                "quantiles_75": np.nanquantile(errors[state_key], 0.75, axis=0),
            }

    # Save out eval data
    with open(str(save_file), 'wb') as file:
        pickle.dump(eval_data, file)

    return eval_data


if __name__ == "__main__":
    models = {
        # model_name: (model_class, model_file_path)
        "linear_256": (MLP256, 'models/linear_model_256.pth'),
        "linear_1024": (MLP1024, 'models/linear_model_1024.pth'),
    }
    eval_save_file = Path("data") / "mlp_eval_data.pkl"
    eval_data = get_eval_data(models, save_file=eval_save_file)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for i, (model, state_stats) in enumerate(eval_data.items()):
        for stat in state_stats.keys():
            if stat == "steps":
                continue
            handle = ax[i].plot(state_stats["steps"], state_stats[stat]["median"], label=stat)
            ax[i].fill_between(state_stats["steps"], state_stats[stat]["quantiles_25"],
                               state_stats[stat]["quantiles_75"], alpha=0.3,
                               color=handle[0].get_color())
        ax[i].set_title(model)
        ax[i].set_xlabel("Steps")
        ax[i].set_ylabel("Prediction Error")
        ax[i].legend()
        ax[i].set_yscale("log")
    fig.tight_layout()
    plt.savefig("plots/mlp_error_by_steps.png")
    plt.show()
