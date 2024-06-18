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
    input_size = 15
    output_size = 12

    model = MLP(input_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load test dataset from file
    test_df = load_test_dataset()

    # Evaluate the model
    error_by_steps = {}
    position_error_by_steps = {}
    velocity_error_by_steps = {}
    inspected_points_error_by_steps = {}
    uninspected_points_error_by_steps = {}
    sun_angle_error_by_steps = {}

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

    plt.clf()
    plt.plot(steps, position_means, label="Error")
    lower_bounds = np.subtract(position_means, position_std_devs)
    upper_bounds = np.add(position_means, position_std_devs)
    plt.fill_between(steps, lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Position Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'position_error_by_steps.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps, velocity_means, label="Error")
    lower_bounds = np.subtract(velocity_means, velocity_std_devs)
    upper_bounds = np.add(velocity_means, velocity_std_devs)
    plt.fill_between(steps, lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Velocity Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'velocity_error_by_steps.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps, inspected_points_means, label="Error")
    lower_bounds = np.subtract(inspected_points_means, inspected_points_std_devs)
    upper_bounds = np.add(inspected_points_means, inspected_points_std_devs)
    plt.fill_between(steps, lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Inspected Points Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'inspected_points_error_by_steps.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps, uninspected_points_means, label="Error")
    lower_bounds = np.subtract(uninspected_points_means, uninspected_points_std_devs)
    upper_bounds = np.add(uninspected_points_means, uninspected_points_std_devs)
    plt.fill_between(steps, lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Uninspected Points Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'uninspected_points_error_by_steps.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps, sun_angle_means, label="Error")
    lower_bounds = np.subtract(sun_angle_means, sun_angle_std_devs)
    upper_bounds = np.add(sun_angle_means, sun_angle_std_devs)
    plt.fill_between(steps, lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Sun Angle Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'sun_angle_error_by_steps.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")





    # plot 20 timesteps
    plt.clf()
    plt.plot(steps[0:20], means[0:20], label="Error")
    lower_bounds = np.subtract(means[0:20], std_devs[0:20])
    upper_bounds = np.add(means[0:20], std_devs[0:20])
    plt.fill_between(steps[0:20], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'error_by_steps_20.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps[0:20], position_means[0:20], label="Error")
    lower_bounds = np.subtract(position_means[0:20], position_std_devs[0:20])
    upper_bounds = np.add(position_means[0:20], position_std_devs[0:20])
    plt.fill_between(steps[0:20], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Position Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'position_error_by_steps_20.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps[0:20], velocity_means[0:20], label="Error")
    lower_bounds = np.subtract(velocity_means[0:20], velocity_std_devs[0:20])
    upper_bounds = np.add(velocity_means[0:20], velocity_std_devs[0:20])
    plt.fill_between(steps[0:20], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Velocity Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'velocity_error_by_steps_20.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps[0:20], inspected_points_means[0:20], label="Error")
    lower_bounds = np.subtract(inspected_points_means[0:20], inspected_points_std_devs[0:20])
    upper_bounds = np.add(inspected_points_means[0:20], inspected_points_std_devs[0:20])
    plt.fill_between(steps[0:20], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Inspected Points Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'inspected_points_error_by_steps_20.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps[0:20], uninspected_points_means[0:20], label="Error")
    lower_bounds = np.subtract(uninspected_points_means[0:20], uninspected_points_std_devs[0:20])
    upper_bounds = np.add(uninspected_points_means[0:20], uninspected_points_std_devs[0:20])
    plt.fill_between(steps[0:20], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Uninspected Points Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'uninspected_points_error_by_steps_20.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps[0:20], sun_angle_means[0:20], label="Error")
    lower_bounds = np.subtract(sun_angle_means[0:20], sun_angle_std_devs[0:20])
    upper_bounds = np.add(sun_angle_means[0:20], sun_angle_std_devs[0:20])
    plt.fill_between(steps[0:20], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Sun Angle Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'sun_angle_error_by_steps_20.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")



    # Plot 10 timesteps
    plt.clf()
    plt.plot(steps[0:10], means[0:10], label="Error")
    lower_bounds = np.subtract(means[0:10], std_devs[0:10])
    upper_bounds = np.add(means[0:10], std_devs[0:10])
    plt.fill_between(steps[0:10], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'error_by_steps_10.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps[0:10], position_means[0:10], label="Error")
    lower_bounds = np.subtract(position_means[0:10], position_std_devs[0:10])
    upper_bounds = np.add(position_means[0:10], position_std_devs[0:10])
    plt.fill_between(steps[0:10], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Position Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'position_error_by_steps_10.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps[0:10], velocity_means[0:10], label="Error")
    lower_bounds = np.subtract(velocity_means[0:10], velocity_std_devs[0:10])
    upper_bounds = np.add(velocity_means[0:10], velocity_std_devs[0:10])
    plt.fill_between(steps[0:10], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Velocity Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'velocity_error_by_steps_10.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps[0:10], inspected_points_means[0:10], label="Error")
    lower_bounds = np.subtract(inspected_points_means[0:10], inspected_points_std_devs[0:10])
    upper_bounds = np.add(inspected_points_means[0:10], inspected_points_std_devs[0:10])
    plt.fill_between(steps[0:10], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Inspected Points Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'inspected_points_error_by_steps_10.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps[0:10], uninspected_points_means[0:10], label="Error")
    lower_bounds = np.subtract(uninspected_points_means[0:10], uninspected_points_std_devs[0:10])
    upper_bounds = np.add(uninspected_points_means[0:10], uninspected_points_std_devs[0:10])
    plt.fill_between(steps[0:10], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Uninspected Points Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'uninspected_points_error_by_steps_10.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps[0:10], sun_angle_means[0:10], label="Error")
    lower_bounds = np.subtract(sun_angle_means[0:10], sun_angle_std_devs[0:10])
    upper_bounds = np.add(sun_angle_means[0:10], sun_angle_std_devs[0:10])
    plt.fill_between(steps[0:10], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Sun Angle Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'sun_angle_error_by_steps_10.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")





    # Plot 5 Timesteps
    plt.clf()
    plt.plot(steps[0:5], means[0:5], label="Error")
    lower_bounds = np.subtract(means[0:5], std_devs[0:5])
    upper_bounds = np.add(means[0:5], std_devs[0:5])
    plt.fill_between(steps[0:5], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'error_by_steps_5.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps[0:5], position_means[0:5], label="Error")
    lower_bounds = np.subtract(position_means[0:5], position_std_devs[0:5])
    upper_bounds = np.add(position_means[0:5], position_std_devs[0:5])
    plt.fill_between(steps[0:5], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Position Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'position_error_by_steps_5.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps[0:5], velocity_means[0:5], label="Error")
    lower_bounds = np.subtract(velocity_means[0:5], velocity_std_devs[0:5])
    upper_bounds = np.add(velocity_means[0:5], velocity_std_devs[0:5])
    plt.fill_between(steps[0:5], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Velocity Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'velocity_error_by_steps_5.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps[0:5], inspected_points_means[0:5], label="Error")
    lower_bounds = np.subtract(inspected_points_means[0:5], inspected_points_std_devs[0:5])
    upper_bounds = np.add(inspected_points_means[0:5], inspected_points_std_devs[0:5])
    plt.fill_between(steps[0:5], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Inspected Points Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'inspected_points_error_by_steps_5.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps[0:5], uninspected_points_means[0:5], label="Error")
    lower_bounds = np.subtract(uninspected_points_means[0:5], uninspected_points_std_devs[0:5])
    upper_bounds = np.add(uninspected_points_means[0:5], uninspected_points_std_devs[0:5])
    plt.fill_between(steps[0:5], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Uninspected Points Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'uninspected_points_error_by_steps_5.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    plt.clf()
    plt.plot(steps[0:5], sun_angle_means[0:5], label="Error")
    lower_bounds = np.subtract(sun_angle_means[0:5], sun_angle_std_devs[0:5])
    upper_bounds = np.add(sun_angle_means[0:5], sun_angle_std_devs[0:5])
    plt.fill_between(steps[0:5], lower_bounds, upper_bounds, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.title("Sun Angle Prediction Error by Steps")
    plt.legend()

    # Save the plot
    plot_filename = 'sun_angle_error_by_steps_5.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

