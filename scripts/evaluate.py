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


# Plot given error and std dev
def error_plot(
        x: dict,
        y: dict,
        quantiles_75: dict,
        quantiles_25: dict,
        x_scale: int = 49,
        model_name: str = "",
        error_name: str = "",
        log_scale: bool = False
):
    # # Log scaling
    # if log_scale:
    #     y = copy.deepcopy(y)
    #     stddev = copy.deepcopy(stddev)
    #     for model_name, error in y.items():
    #         std_dev = stddev[model_name]
    #         y[model_name] = np.log10(error)
    #         stddev[model_name] = np.log10(std_dev)

    # Create plots
    for model_name, error in y.items():
        # std_dev = stddev[model_name]

        plt.plot(steps[model_name][0:x_scale], error[0:x_scale], label=model_name)
        # lower_bounds = np.subtract(error[0:x_scale], std_dev[0:x_scale])
        # upper_bounds = np.add(error[0:x_scale], std_dev[0:x_scale])
        lower_bounds = quantiles_25
        upper_bounds = quantiles_75
        plt.fill_between(steps[model_name][0:x_scale], lower_bounds[model_name][0:x_scale],
                         upper_bounds[model_name][0:x_scale], alpha=0.3)

    log_label = "Log " if log_scale else ""
    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title(f"{log_label}{error_name} Prediction Error by Steps")
    if log_scale:
        plt.yscale('log')
    plt.legend()

    # Save the plot
    log_label = "log_" if log_scale else ""
    plot_filename = f"{log_label}{error_name}_error_by_steps_{x_scale}.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.clf()


def get_eval_data(models: dict, input_size=15, output_size=12, save_file: Optional[Path] = None):
    eval_data = copy.deepcopy(models)

    if save_file is not None and save_file.exists():
        with open(str(save_file), 'rb') as file:
            return pickle.load(file)

    # Load test dataset from file
    test_df = load_test_dataset()

    # Evaluate model(s)
    for model_name, model_cfg in models.items():
        # Load the trained model
        model = model_cfg[0](input_size, output_size)
        model.load_state_dict(torch.load(model_cfg[1]))
        model.eval()

        # Evaluate the model on different slices of the data
        error_by_steps = defaultdict(lambda: defaultdict(list))

        with torch.no_grad():
            for trajectory in test_df['Trajectory']:
                # Create multistep predictions for each trajectory in test dataset
                final_state_index = len(trajectory) - 1
                multistep_predictions = {}
                target_states = {}
                # final_state = trajectory[final_state_index][0:11]
                for i in range(final_state_index):
                    # predicted_trajectory = []
                    state_action = torch.tensor(trajectory[i], dtype=torch.float32)
                    future_actions = [state_action[-3:] for state_action in
                                      trajectory[i + 1:final_state_index]]

                    n = final_state_index - i
                    for k in range(n):
                        if k > 49:
                            # Model error compounds exponentially, don't waste compute on long range
                            break
                        # Use the model to predict n steps into the future
                        # Where n is the number of timesteps between the input and target output in the trajectory
                        predicted_state = model(state_action)
                        # store model's prediction and ground truth target state
                        multistep_predictions[(i, i + k + 1)] = predicted_state
                        target_states[(i, i + k + 1)] = trajectory[i + k + 1][0:12]
                        # Concatentate model's predicted state with next taken action
                        if k < n - 1:
                            state_action = torch.tensor(
                                np.concatenate((predicted_state, future_actions[k])),
                                dtype=torch.float32)
                for k, v in multistep_predictions.items():
                    num_steps = k[1] - k[0]
                    output = v.numpy()
                    for state_key, state_range in NAMED_STATE_RANGES.items():
                        error_by_steps[state_key][num_steps].append(
                            euclidean_distance(output[state_range], target_states[k][state_range])
                        )

        # Calculate summary statistics of different slices of the data

        steps = []
        medians = defaultdict(list)
        quantiles_75 = defaultdict(list)
        quantiles_25 = defaultdict(list)
        for i in range(1, len(error_by_steps)):
            for state_key in NAMED_STATE_RANGES.keys():
                medians[state_key].append(np.quantile(error_by_steps[state_key][i], 0.50))
                quantiles_75[state_key].append(np.quantile(error_by_steps[state_key][i], 0.75))
                quantiles_25[state_key].append(np.quantile(error_by_steps[state_key][i], 0.25))

        # Store eval results
        eval_data[model_name] = {
            "steps": steps,
            "medians": medians,
            "quantiles_75": quantiles_75,
            "quantiles_25": quantiles_25
        }

    # Save out eval data
    with open(str(save_file), 'wb') as file:
        pickle.dump(eval_data, file)


if __name__ == "__main__":
    models = {
        # model_name: (model_class, model_file_path)
        "linear_256": (MLP256, 'models/linear_model_256.pth'),
        "linear_1024": (MLP1024, 'models/linear_model_1024.pth'),
    }
    eval_save_file = Path("data") / "mlp_eval_data.pkl"
    eval_data = get_eval_data(models, save_file=eval_save_file)

    # Configure plotting
    log_scale = [True, False]
    x_scale = [5, 10, 20]
    error_type = [
        "total",
        "position",
        "velocity",
        "inspected_points",
        "uninspected_points",
        "sun_angle",
    ]

    for error_type, x_scale, log_scale in itertools.product(error_type, x_scale, log_scale):
        # Parse eval data
        steps = {
            model_name: model_data["steps"] for model_name, model_data in eval_data.items()
        }
        error_key = "medians" if error_type == "total" else f"{error_type}_medians"
        quantile_75_key = "quantiles_75" if error_type == "total" else f"{error_type}_quantiles_75"
        quantile_25_key = "quantiles_25" if error_type == "total" else f"{error_type}_quantiles_25"
        error = {
            model_name: model_data[error_key] for model_name, model_data in eval_data.items()
        }
        quantiles_75 = {
            model_name: model_data[quantile_75_key] for model_name, model_data in eval_data.items()
        }
        quantiles_25 = {
            model_name: model_data[quantile_25_key] for model_name, model_data in eval_data.items()
        }

        error_plot(steps, error, quantiles_75, quantiles_25, x_scale=x_scale, error_name=error_type,
                   log_scale=log_scale)
