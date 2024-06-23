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
from constants import NAMED_STATE_RANGES


# Calculate error between model output and target vector
def euclidean_distance(output, target):
    return np.sqrt(np.sum((output - target) ** 2))


def get_eval_data(model_name: str, model_cfg: tuple, input_size=15, output_size=12, max_steps: int = 50,
                  save_file: Optional[Path] = None):
    """
    Collects multistep prediction error of given model or loads evaluation data from file.
    """
    if save_file is not None and save_file.exists():
        with open(str(save_file), 'rb') as file:
            return pickle.load(file)

    # Load test dataset from file
    test_df = load_test_dataset()

    # Initialize data structures to store evaluation data
    eval_data = {}
    errors = {
        k: {num_steps: [] for num_steps in range(1,max_steps+1)} for k in NAMED_STATE_RANGES.keys()
    }

    # Load the trained model
    model = model_cfg[0](input_size, output_size)
    model.load_state_dict(torch.load(model_cfg[1]))
    model.eval()


    with torch.no_grad():
        for trajectory_idx, trajectory in enumerate(test_df['Trajectory']):
            # Create multistep predictions for each trajectory in test dataset
            final_state_index = len(trajectory)
            state_actions = torch.from_numpy(np.stack(np.array(trajectory))).to(torch.float32)
            states = state_actions[:,0:output_size]
            # prediction_size = 1
            for delta_t in range(1, max_steps+1):
                if delta_t > max_steps:
                    break

                # Compute next prediction step
                predicted_states = model(state_actions)
                predicted_states = predicted_states[0:final_state_index - delta_t]
                state_actions = torch.concat((predicted_states, state_actions[0:final_state_index-delta_t,output_size:input_size]), dim=1)
                # Compute and store error
                for i in range(0,final_state_index - delta_t + 1):
                    if i >= max_steps or i >= predicted_states.shape[0]:
                        break
                    for state_key, state_range in NAMED_STATE_RANGES.items():
                        errors[state_key][i+1].append(
                            euclidean_distance(
                                predicted_states[i][state_range].numpy(),
                                states[i+delta_t][state_range].numpy()
                            )
                        )

    # Calculate summary statistics of different slices of the data
    eval_data[model_name] = {"steps": np.arange(1, max_steps + 1)}
    for state_key in NAMED_STATE_RANGES.keys():
        medians = np.zeros(max_steps)
        quantiles_25 = np.zeros(max_steps)
        quantiles_75 = np.zeros(max_steps)
        for step, error in errors[state_key].items():
            medians[step-1] = np.median(error)
            quantiles_25[step-1] = np.quantile(error, 0.25)
            quantiles_75[step-1] = np.quantile(error, 0.75)

        eval_data[model_name][state_key] = {
            "median": medians,
            "quantiles_25": quantiles_25,
            "quantiles_75": quantiles_75,
        }

    # Save out eval data
    with open(str(save_file), 'wb') as file:
        pickle.dump(eval_data, file)

    return eval_data


if __name__ == "__main__":
    models = {
        # model_name: (model_class, model_file_path)
        "linear_256": (MLP256, 'models/linear_model_256.pth'),
        # "linear_1024": (MLP1024, 'models/linear_model_1024.pth'),
    }

    eval_data = {}

    for model_name, model_cfg in models.items():
        eval_save_file = Path("eval_data") / f"{model_name}_eval_data.pkl"
        # eval_save_file = None
        model_eval_data = get_eval_data(model_name, model_cfg, save_file=eval_save_file)
        eval_data = eval_data | model_eval_data

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
