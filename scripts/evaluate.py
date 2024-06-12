"""
This script evaluates the model's prediction accuracy as a function of timesteps.
"""
import numpy as np
import torch
from load_dataset import load_test_dataset
from train import MLP
import numpy as np
import matplotlib.pyplot as plt
import pickle


def euclidean_distance(output, target):
    return np.sqrt(np.sum((output - target)**2))


if __name__ == "__main__":
    # Load the trained model
    model_path = 'models/linear_model_256.pth'
    input_size = 14
    output_size = 11

    model = MLP(input_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load test dataset from file
    test_df = load_test_dataset()

    # Evaluate the model
    error_by_steps = {}

    # with torch.no_grad():
    #     for trajectory in test_df['Trajectory']:
    #         # Parse input-output pairs from trajectory
    #         i = 0
    #         j = 1
    #         while i < len(trajectory) - 1:
    #             state_action = torch.tensor(trajectory[i], dtype=torch.float32)
    #             target_state = trajectory[j][0:11]
    #             actions = [state_action[-3:] for state_action in trajectory[i+1:j]]
    #             n = j - i
    #             for k in range(n):
    #                 # Use the model to predict n steps into the future
    #                 # Where n is the number of timesteps between the input and target output in the trajectory
    #                 predicted_state = model(state_action)
    #                 # Concatentate model's predicted state with next taken action
    #                 if k < n - 1:
    #                     state_action = torch.tensor(np.concatenate((predicted_state,actions[k])), dtype=torch.float32)
    #             output = predicted_state.numpy()
    #             dist = euclidean_distance(output, target_state)
    #             if n in error_by_steps:
    #                 error_by_steps[n].append(dist)
    #             else:
    #                 error_by_steps[n] = [dist]
    #             # Increment input-output pair along trajectory
    #             if j < len(trajectory) - 1:
    #                 j += 1
    #             else:
    #                 i += 1
    #                 j = i + 1

    # TODO: remove redundant calculation of multistep predictions
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
                    target_states[(i,i+k+1)] = trajectory[i+k+1][0:11]

                    # Concatentate model's predicted state with next taken action
                    if k < n - 1:
                        state_action = torch.tensor(np.concatenate((predicted_state,future_actions[k])), dtype=torch.float32)
                i += 1
            for k,v in multistep_predictions.items():
                num_steps = k[1]-k[0]
                output = v.numpy()
                dist = euclidean_distance(output, target_states[k])
                if num_steps in error_by_steps:
                    error_by_steps[num_steps].append(dist)
                else:
                    error_by_steps[num_steps] = [dist]


    # Calculate the mean and standard deviation
    means = []
    std_devs = []

    steps = []
    means = []
    std_devs = []
    for i in range(1, len(error_by_steps)):
        means.append(np.mean(error_by_steps[i]))
        std_devs.append(np.std(error_by_steps[i]))
        steps.append(i)

    # Create plots
    plt.plot(steps, means, label="Error")
    lower_bounds = np.subtract(means, std_devs)
    upper_bounds = np.add(means, std_devs)
    plt.fill_between(steps, lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'error_by_steps.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    # plot 20 timesteps
    plt.clf()
    first_20_steps = steps[0:20]
    first_20_means = means[0:20]
    first_20_std_devs = std_devs[0:20]
    plt.plot(first_20_steps, first_20_means, label="Error")
    lower_bounds = np.subtract(first_20_means, first_20_std_devs)
    upper_bounds = np.add(first_20_means, first_20_std_devs)
    plt.fill_between(first_20_steps, lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'error_by_steps_20.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    # plot 10 timesteps
    plt.clf()
    first_10_steps = steps[0:10]
    first_10_means = means[0:10]
    first_10_std_devs = std_devs[0:10]
    plt.plot(first_10_steps, first_10_means, label="Error")
    lower_bounds = np.subtract(first_10_means, first_10_std_devs)
    upper_bounds = np.add(first_10_means, first_10_std_devs)
    plt.fill_between(first_10_steps, lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Prediction Error by Steps")
    plt.legend()


    # Save the plot
    plot_filename = 'error_by_steps_10.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    # plot 5 timesteps
    plt.clf()
    first_5_steps = steps[0:5]
    first_5_means = means[0:5]
    first_5_std_devs = std_devs[0:5]
    plt.plot(first_5_steps, first_5_means, label="Error")
    lower_bounds = np.subtract(first_5_means, first_5_std_devs)
    upper_bounds = np.add(first_5_means, first_5_std_devs)
    plt.fill_between(first_5_steps, lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Prediction Error by Steps")
    plt.legend()


    # Save the plot
    plot_filename = 'error_by_steps_5.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    dict_filename = 'error_by_steps.pkl'
    with open(dict_filename, 'wb') as f:
        pickle.dump(error_by_steps, f)
    print(f"Dictionary saved to {dict_filename}")

    # calculate single step error
    avg_error = means[0]
    std_error = std_devs[0]
    print("Single Step Error Mean and Std")
    print(avg_error)
    print(std_error)
