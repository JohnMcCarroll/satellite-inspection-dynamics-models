"""
This script evaluates the model's prediction accuracy as a function of timesteps.
"""
import numpy as np
import torch
from load_dataset import load_test_dataset
from train import MLP256, MLP1024, NonlinearMLP, RNN, apply_constraints
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import itertools
from typing import Optional
from pathlib import Path
from constants import NAMED_STATE_RANGES
import pandas as pd


# Calculate error between model output and target vector
def euclidean_distance(output, target):
    return np.sqrt(np.sum((output - target)**2))

def get_rnn_eval_data(
        test_df: pd.DataFrame,
        model_name: str,
        model_cfg: tuple,
        input_size: int = 15,
        output_size: int = 12,
        max_steps: int = 50,
        save_file: Optional[Path] = None,
        model: torch.nn.Module = None,
        prediction_size: int = 1,
        save_data: bool = True,
        validation: bool = False
    ):
    """
    Collects multistep prediction error of given model or loads evaluation data from file.
    """
    if save_file is not None and save_file.exists():
        with open(str(save_file), 'rb') as file:
            return pickle.load(file)

    # Initialize data structures to store evaluation data
    eval_data = {}
    errors = {
        k: {num_steps: [] for num_steps in range(prediction_size, max_steps+1, prediction_size)} for k in NAMED_STATE_RANGES.keys()
    }

    # Load the trained model
    if model is None:
        model = model_cfg[0](input_size, output_size)
        model.load_state_dict(torch.load(model_cfg[1]))
    model.eval()

    with torch.no_grad():
        for trajectory in test_df['Trajectory']:
            # Create multistep predictions for each trajectory in test dataset
            i = 0
            final_state_index = len(trajectory)-1
            multistep_predictions = {}
            target_states = {}
            hidden_state = None
            while i < final_state_index:
                # predicted_trajectory = []
                state_action = torch.tensor(trajectory[i], dtype=torch.float32)
                future_actions = [state_action[-3:] for state_action in trajectory[i+1:final_state_index]]

                n = final_state_index - i
                for k in range(n):
                    if k > 49:
                        # Model error compounds exponentially, don't waste compute on long range
                        break
                    # Use the model to predict n steps into the future
                    # Where n is the number of timesteps between the input and target output in the trajectory
                    predicted_state, hidden_state = model(state_action, hidden_state)
                    # store model's prediction and ground truth target state
                    multistep_predictions[(i,i+k+1)] = predicted_state
                    target_states[(i,i+k+1)] = trajectory[i+k+1][0:12]
                    # Concatentate model's predicted state with next taken action
                    if k < n - 1:
                        state_action = torch.tensor(np.concatenate((predicted_state,future_actions[k])), dtype=torch.float32)
                i += 1
                # Compute and store error
                for j in range(0,final_state_index - delta_t + 1):
                    # Iterate through predictions tensor (contains all predictions of depth `delta_t`)
                    if j >= predicted_states.shape[0]:
                        # Size of predictions within scope of trajectory decreases each delta_t increment
                        break
                    for state_key, state_range in NAMED_STATE_RANGES.items():
                        errors[state_key][delta_t * prediction_size].append(
                            euclidean_distance(
                                predicted_states[j][state_range].numpy(),
                                states[i+delta_t][state_range].numpy()
                            )
                        )
                if validation:
                    # no need to evaluate multiple steps for validation
                    break
        
        
        
        # # batched code
        # for trajectory_idx, trajectory in enumerate(test_df['Trajectory']):
        #     # Create multistep predictions for each trajectory in test dataset
        #     final_state_index = len(trajectory)
        #     state_actions = torch.from_numpy(np.stack(np.array(trajectory))).to(torch.float32)
        #     states = state_actions[:,0:output_size]
        #     for delta_t in range(1, max_steps+1):
        #         if delta_t * prediction_size > max_steps:
        #             # delta_t ~= number of subsequent model predictions
        #             # delta_t * prediction_size = num timesteps in the future model is predicting
        #             # Stop, if we are predicting past max_steps
        #             break
        #         # Compute next prediction step
        #         predicted_states = model(state_actions)
        #         predicted_states = predicted_states[0:final_state_index - delta_t]
        #         state_actions = torch.concat((predicted_states, state_actions[0:final_state_index-delta_t,output_size:input_size]), dim=1)
        #         # Compute and store error
        #         for i in range(0,final_state_index - delta_t + 1):
        #             # Iterate through predictions tensor (contains all predictions of depth `delta_t`)
        #             if i >= predicted_states.shape[0]:
        #                 # Size of predictions within scope of trajectory decreases each delta_t increment
        #                 break
        #             for state_key, state_range in NAMED_STATE_RANGES.items():
        #                 errors[state_key][delta_t * prediction_size].append(
        #                     euclidean_distance(
        #                         predicted_states[i][state_range].numpy(),
        #                         states[i+delta_t][state_range].numpy()
        #                     )
        #                 )
        #         if validation:
        #             # no need to evaluate multiple steps for validation
        #             break

    # Calculate summary statistics of different slices of the data
    eval_data[model_name] = {"steps": np.arange(prediction_size, max_steps + 1, prediction_size)}
    for state_key in NAMED_STATE_RANGES.keys():
        medians = np.full(len(eval_data[model_name]['steps']), np.nan)
        quantiles_25 = np.full(len(eval_data[model_name]['steps']), np.nan)
        quantiles_75 = np.full(len(eval_data[model_name]['steps']), np.nan)
        for index, (step, error) in enumerate(errors[state_key].items()):
            medians[index] = np.median(error)
            quantiles_25[index] = np.quantile(error, 0.25)
            quantiles_75[index] = np.quantile(error, 0.75)
            if validation:
                break

        eval_data[model_name][state_key] = {
            "median": medians,
            "quantiles_25": quantiles_25,
            "quantiles_75": quantiles_75,
        }

    # Save out eval data
    if save_data:
        with open(str(save_file), 'wb') as file:
            pickle.dump(eval_data, file)

    return eval_data


if __name__ == "__main__":
    # Configure Evaluation
    prediction_size = 1
    plot_save_path = "plots/rnn_error_by_steps.png"
    models = {
        # model_name: (model_class, model_file_path)
        "rnn": (RNN, 'models/rnn.pth'),
        # "linear_1024": (MLP1024, 'models/linear_model_1024.pth'),
    }
    input_size = 15
    output_size = 12
    # eval_data = copy.deepcopy(models)
    # eval_name = "LinearModelSize"

    # Load test dataset from file
    test_df = load_test_dataset()
    eval_data = {}

    # Evaluate model(s)
    for model_name, model_cfg in models.items():
        eval_save_file = Path("eval_data") / f"{model_name}_eval_data.pkl"
        model_eval_data = get_rnn_eval_data(test_df, model_name, model_cfg, save_file=eval_save_file, prediction_size=prediction_size)
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
    plt.savefig(plot_save_path)
    plt.show()



    # Evaluate the model
    error_by_steps = {}
    position_error_by_steps = {}
    velocity_error_by_steps = {}
    inspected_points_error_by_steps = {}
    uninspected_points_error_by_steps = {}
    sun_angle_error_by_steps = {}

    with torch.no_grad():
        for trajectory in test_df['Trajectory']:
            # Create multistep predictions for each trajectory in test dataset
            i = 0
            final_state_index = len(trajectory)-1
            multistep_predictions = {}
            target_states = {}
            # final_state = trajectory[final_state_index][0:11]
            while i < final_state_index:
                # predicted_trajectory = []
                state_action = torch.tensor(trajectory[i], dtype=torch.float32)
                future_actions = [state_action[-3:] for state_action in trajectory[i+1:final_state_index]]

                n = final_state_index - i
                for k in range(n):
                    if k > 49:
                        # Model error compounds exponentially, don't waste compute on long range
                        break
                    # Use the model to predict n steps into the future
                    # Where n is the number of timesteps between the input and target output in the trajectory
                    predicted_state = model(state_action)
                    # store model's prediction and ground truth target state
                    multistep_predictions[(i,i+k+1)] = predicted_state
                    target_states[(i,i+k+1)] = trajectory[i+k+1][0:12]
                    # Concatentate model's predicted state with next taken action
                    if k < n - 1:
                        state_action = torch.tensor(np.concatenate((predicted_state,future_actions[k])), dtype=torch.float32)
                i += 1
            for k,v in multistep_predictions.items():
                num_steps = k[1]-k[0]
                output = v.numpy()
                total_dist = euclidean_distance(output, target_states[k])
                position_dist = euclidean_distance(output[0:3], target_states[k][0:3])
                velocity_dist = euclidean_distance(output[3:6], target_states[k][3:6])
                inspected_points_dist = euclidean_distance(output[6], target_states[k][6])
                uninspected_pointstotal_dist = euclidean_distance(output[7:10], target_states[k][7:10])
                sun_angle_dist = euclidean_distance(output[10:], target_states[k][10:])
                if num_steps in error_by_steps:
                    error_by_steps[num_steps].append(total_dist)
                    position_error_by_steps[num_steps].append(position_dist)
                    velocity_error_by_steps[num_steps].append(velocity_dist)
                    inspected_points_error_by_steps[num_steps].append(inspected_points_dist)
                    uninspected_points_error_by_steps[num_steps].append(uninspected_pointstotal_dist)
                    sun_angle_error_by_steps[num_steps].append(sun_angle_dist)
                else:
                    error_by_steps[num_steps] = [total_dist]
                    position_error_by_steps[num_steps] = [position_dist]
                    velocity_error_by_steps[num_steps] = [velocity_dist]
                    inspected_points_error_by_steps[num_steps] = [inspected_points_dist]
                    uninspected_points_error_by_steps[num_steps] = [uninspected_pointstotal_dist]
                    sun_angle_error_by_steps[num_steps] = [sun_angle_dist]

    # Calculate the mean and standard deviation
    steps = []
    means = []
    std_devs = []
    position_means = []
    position_std_devs = []
    velocity_means = []
    velocity_std_devs = []
    inspected_points_means = []
    inspected_points_std_devs = []
    uninspected_points_means = []
    uninspected_points_std_devs = []
    sun_angle_means = []
    sun_angle_std_devs = []
    for i in range(1, len(error_by_steps)):
        means.append(np.mean(error_by_steps[i]))
        std_devs.append(np.std(error_by_steps[i]))
        steps.append(i)
        position_means.append(np.mean(position_error_by_steps[i]))
        position_std_devs.append(np.std(position_error_by_steps[i]))
        velocity_means.append(np.mean(velocity_error_by_steps[i]))
        velocity_std_devs.append(np.std(velocity_error_by_steps[i]))
        inspected_points_means.append(np.mean(inspected_points_error_by_steps[i]))
        inspected_points_std_devs.append(np.std(inspected_points_error_by_steps[i]))
        uninspected_points_means.append(np.mean(uninspected_points_error_by_steps[i]))
        uninspected_points_std_devs.append(np.std(uninspected_points_error_by_steps[i]))
        sun_angle_means.append(np.mean(sun_angle_error_by_steps[i]))
        sun_angle_std_devs.append(np.std(sun_angle_error_by_steps[i]))

    # Store eval results
    eval_data[model_name] = {
        "steps": steps,
        "means": means,
        "std_devs": std_devs,
        "position_means": position_means,
        "position_std_devs": position_std_devs,
        "velocity_means": velocity_means,
        "velocity_std_devs": velocity_std_devs,
        "inspected_points_means": inspected_points_means,
        "inspected_points_std_devs": inspected_points_std_devs,
        "uninspected_points_means": uninspected_points_means,
        "uninspected_points_std_devs": uninspected_points_std_devs,
        "sun_angle_means": sun_angle_means,
        "sun_angle_std_devs": sun_angle_std_devs,
    }

# Save out eval data
pickle.dump(eval_data, open(f'data/{eval_name}_eval_data.pkl', 'wb'))

# # Read data from file
# with open(f'data/{eval_name}_eval_data.pkl', 'rb') as file:
#     eval_data = pickle.load(file)

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
    error_key = "means" if error_type == "total" else f"{error_type}_means"
    std_devs_key = "std_devs" if error_type == "total" else f"{error_type}_std_devs"
    error = {
        model_name: model_data[error_key] for model_name, model_data in eval_data.items()
    }
    std_devs = {
        model_name: model_data[std_devs_key] for model_name, model_data in eval_data.items()
    }

    error_plot(steps, error, std_devs, x_scale=x_scale, error_name=error_type, log_scale=log_scale)
