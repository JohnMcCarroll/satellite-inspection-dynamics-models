"""
This script evaluates the model's prediction accuracy as a function of timesteps.
"""
import numpy as np
import torch
from load_dataset import load_test_dataset
from train import MLP256, MLP1024
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import itertools


# Calculate error between model output and target vector
def euclidean_distance(output, target):
    return np.sqrt(np.sum((output - target)**2))

# Plot given error and std dev
def error_plot(
        x: dict, 
        y: dict, 
        stddev: dict, 
        x_scale: int = 49, 
        model_name: str = "", 
        error_name: str = "", 
        log_scale: bool = False
    ):

    # Log scaling
    if log_scale:
        y = copy.deepcopy(y)
        stddev = copy.deepcopy(stddev)
        for model_name, error in y.items():
            std_dev = stddev[model_name]
            y[model_name] = np.log10(error)
            stddev[model_name] = np.log10(std_dev)

    # Create plots
    for model_name, error in y.items():
        std_dev = stddev[model_name]

        plt.plot(steps[model_name][0:x_scale], error[0:x_scale], label=model_name)
        lower_bounds = np.subtract(error[0:x_scale], std_dev[0:x_scale])
        upper_bounds = np.add(error[0:x_scale], std_dev[0:x_scale])
        plt.fill_between(steps[model_name][0:x_scale], lower_bounds, upper_bounds, alpha=0.3)

    log_label = "Log " if log_scale else ""
    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title(f"{log_label}{error_name} Prediction Error by Steps")
    plt.legend()

    # Save the plot
    log_label = "log_" if log_scale else ""
    plot_filename = f"{log_label}{error_name}_error_by_steps.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.clf()


if __name__ == "__main__":
    # Configure Evaluation
    models = {
        # model_name: (model_class, model_file_path)
        "linear_256": (MLP256, 'models/linear_model_256.pth'),
        "linear_1024": (MLP1024, 'models/linear_model_1024.pth'),
    }
    input_size = 15
    output_size = 12
    eval_data = copy.deepcopy(models)
    eval_name = "LinearModelSize"

    # Load test dataset from file
    test_df = load_test_dataset()

    # Evaluate model(s)
    for model_name, model_cfg in models.items():
        # Load the trained model
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
