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
    model_path = 'models/linear_model.pth'
    input_size = 14
    output_size = 11

    model = MLP(input_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load test dataset from file
    test_df = load_test_dataset()

    # Evaluate the model
    error_by_steps = {}

    with torch.no_grad():
        for trajectory in test_df['Trajectory']:
            # Parse input-output pairs from trajectory
            i = 0
            j = 1
            while i < len(trajectory) - 1:
                state_action = torch.tensor(trajectory[i], dtype=torch.float32)
                target_state = trajectory[j][0:11]
                actions = [state_action[-3:] for state_action in trajectory[i+1:j]]
                n = j - i
                for k in range(n):
                    # Use the model to predict n steps into the future
                    # Where n is the number of timesteps between the input and target output in the trajectory
                    predicted_state = model(state_action)
                    # Concatentate model's predicted state with next taken action
                    if k < n - 1:
                        state_action = torch.tensor(np.concatenate((predicted_state,actions[k])), dtype=torch.float32)
                output = predicted_state.numpy()
                dist = euclidean_distance(output, target_state)
                if n in error_by_steps:
                    error_by_steps[n].append(dist)
                else:
                    error_by_steps[n] = [dist]
                # Increment input-output pair along trajectory
                if j < len(trajectory) - 1:
                    j += 1
                else:
                    i += 1
                    j = i + 1

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
    first_20_steps = steps[0:21]
    first_20_means = means[0:21]
    first_20_std_devs = std_devs[0:21]
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
    first_10_steps = steps[0:11]
    first_10_means = means[0:11]
    first_10_std_devs = std_devs[0:11]
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

    dict_filename = 'error_by_steps.pkl'
    with open(dict_filename, 'wb') as f:
        pickle.dump(error_by_steps, f)
    print(f"Dictionary saved to {dict_filename}")

    plt.show()
