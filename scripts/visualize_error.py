"""
This script evaluates the model's prediction accuracy as a function of timesteps.
"""
import numpy as np
import torch
from load_dataset import load_test_dataset
from models import MLP256, MLP1024
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from typing import Optional
from pathlib import Path
from constants import NAMED_STATE_RANGES


# Calculate error between model output and target vector
def euclidean_distance(output, target):
    return np.sqrt(np.sum((output - target) ** 2))

def viz_predictions(
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

    # Evaluate the model
    error_by_steps = {}
    position_error_by_steps = {}
    velocity_error_by_steps = {}
    inspected_points_error_by_steps = {}
    uninspected_points_error_by_steps = {}
    sun_angle_error_by_steps = {}

    with torch.no_grad():
        trajectories = [70, 23, 100]
        initial_states = [0, 17]
        for trajectory_row in trajectories:
            trajectory = test_df["Trajectory"][trajectory_row]
            for initial_state in initial_states:
            # Create multistep predictions for each trajectory in test dataset
        
                final_state_index = len(trajectory)-1
                multistep_predictions = {}
                target_states = {}
                i = initial_state
                while i < final_state_index:
                    # predicted_trajectory = []
                    state_action = torch.tensor(trajectory[i], dtype=torch.float32)
                    future_actions = [state_action[-3:] for state_action in trajectory[i+1:final_state_index]]

                    n = final_state_index - i
                    for k in range(n):
                        if k > 10:
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
                    break

                fig, axes = plt.subplots(4, 3, figsize=(12, 6))

                actual = {
                    "x": [],
                    "y": [],
                    "z": [],
                    "x_vel": [],
                    "y_vel": [],
                    "z_vel": [],
                    "points": [],
                    "x_cluster": [],
                    "y_cluster": [],
                    "z_cluster": [],
                    "x_sun": [],
                    "y_sun": [],
                }
                predicted = {
                    "x": [],
                    "y": [],
                    "z": [],
                    "x_vel": [],
                    "y_vel": [],
                    "z_vel": [],
                    "points": [],
                    "x_cluster": [],
                    "y_cluster": [],
                    "z_cluster": [],
                    "x_sun": [],
                    "y_sun": [],
                }

                for k,v in multistep_predictions.items():
                    output = v.numpy()

                    actual['x'].append(target_states[k][0])
                    actual['y'].append(target_states[k][1])
                    actual['z'].append(target_states[k][2])
                    actual['x_vel'].append(target_states[k][3])
                    actual['y_vel'].append(target_states[k][4])
                    actual['z_vel'].append(target_states[k][5])
                    actual['points'].append(target_states[k][6])
                    actual['x_cluster'].append(target_states[k][7])
                    actual['y_cluster'].append(target_states[k][8])
                    actual['z_cluster'].append(target_states[k][9])
                    actual['x_sun'].append(target_states[k][10])
                    actual['y_sun'].append(target_states[k][11])

                    predicted['x'].append(output[0])
                    predicted['y'].append(output[1])
                    predicted['z'].append(output[2])
                    predicted['x_vel'].append(output[3])
                    predicted['y_vel'].append(output[4])
                    predicted['z_vel'].append(output[5])
                    predicted['points'].append(output[6])
                    predicted['x_cluster'].append(output[7])
                    predicted['y_cluster'].append(output[8])
                    predicted['z_cluster'].append(output[9])
                    predicted['x_sun'].append(output[10])
                    predicted['y_sun'].append(output[11])

                # Convert to numpy
                for k,v in predicted.items():
                    actual[k] = np.array(actual[k])
                    predicted[k] = np.array(predicted[k])

                # Plot differences
                axes[0][0].plot(actual["x"], label="real")
                axes[0][0].plot(predicted["x"], label="predicted")
                axes[0][0].set_ylabel("x")
                axes[0][0].set_xlabel("steps")
                axes[0][1].plot(actual["y"], label="real")
                axes[0][1].plot(predicted["y"], label="predicted")
                axes[0][1].set_ylabel("y")
                axes[0][1].set_xlabel("steps")
                axes[0][2].plot(actual["z"], label="real")
                axes[0][2].plot(predicted["z"], label="predicted")
                axes[0][2].set_ylabel("z")
                axes[0][2].set_xlabel("steps")
                axes[1][0].plot(actual["x_vel"], label="real")
                axes[1][0].plot(predicted["x_vel"], label="predicted")
                axes[1][0].set_ylabel("x_vel")
                axes[1][0].set_xlabel("steps")
                axes[1][1].plot(actual["y_vel"], label="real")
                axes[1][1].plot(predicted["y_vel"], label="predicted")
                axes[1][1].set_ylabel("y_vel")
                axes[1][1].set_xlabel("steps")
                axes[1][2].plot(actual["z_vel"], label="real")
                axes[1][2].plot(predicted["z_vel"], label="predicted")
                axes[1][2].set_ylabel("z_vel")
                axes[1][2].set_xlabel("steps")
                axes[1][2].plot(actual["z_vel"], label="real")
                axes[1][2].plot(predicted["z_vel"], label="predicted")
                axes[1][2].set_ylabel("z_vel")
                axes[1][2].set_xlabel("steps")
                axes[2][0].plot(actual["x_cluster"], label="real")
                axes[2][0].plot(predicted["x_cluster"], label="predicted")
                axes[2][0].set_ylabel("x_cluster")
                axes[2][0].set_xlabel("steps")
                axes[2][1].plot(actual["y_cluster"], label="real")
                axes[2][1].plot(predicted["y_cluster"], label="predicted")
                axes[2][1].set_ylabel("y_cluster")
                axes[2][1].set_xlabel("steps")
                axes[2][2].plot(actual["z_cluster"], label="real")
                axes[2][2].plot(predicted["z_cluster"], label="predicted")
                axes[2][2].set_ylabel("z_cluster")
                axes[2][2].set_xlabel("steps")
                axes[3][0].plot(actual["points"], label="real")
                axes[3][0].plot(predicted["points"], label="predicted")
                axes[3][0].set_ylabel("points")
                axes[3][0].set_xlabel("steps")
                axes[3][1].plot(actual["x_sun"], label="real")
                axes[3][1].plot(predicted["x_sun"], label="predicted")
                axes[3][1].set_ylabel("x_sun")
                axes[3][1].set_xlabel("steps")
                axes[3][2].plot(actual["y_sun"], label="real")
                axes[3][2].plot(predicted["y_sun"], label="predicted")
                axes[3][2].set_ylabel("y_sun")
                axes[3][2].set_xlabel("steps")

                fig.tight_layout()
                for row in axes:
                    for ax in row:
                        ax.set_ylim([-5, 5])
                plot_save_path = f"plots/compare_trajectories_row{trajectory_row}_step{initial_state}.png"
                plt.savefig(plot_save_path)


if __name__ == "__main__":
    prediction_size = 1
    models = {
        # model_name: (model_class, model_file_path)
        "linear_256": (MLP256, 'models/linear_model_256.pth'),
        "linear_1024": (MLP1024, 'models/linear_model_1024.pth'),
    }

    # Load test dataset from file
    test_df = load_test_dataset()
    eval_data = {}

    for model_name, model_cfg in models.items():
        eval_save_file = Path("eval_data") / f"{model_name}_eval_data.pkl"
        viz_predictions(test_df, model_name, model_cfg, save_file=eval_save_file, prediction_size=prediction_size)
