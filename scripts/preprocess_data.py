import pickle
import pandas as pd
import numpy as np
import math


# load the DataFrame
def load_dataframe(file_path):
    return pd.read_pickle(file_path)

# calculate mean and standard deviation across all elements
def calculate_stats(df):
    all_obs = None
    for traj in df['Trajectory']:
        for state in traj:
            if all_obs is None:
                all_obs = np.array([state[0]])
            else:
                all_obs = np.concatenate((all_obs, [state[0]]), axis=0)
    mean = np.mean(all_obs, axis=0)
    std = np.std(all_obs, axis=0)
    return mean, std

# z-score the arrays
def z_score_array(array, mean, std):
    return (array - mean) / std

# z-score the entire DataFrame
def z_score_dataframe(df, mean, std):
    z_scored_trajectories = []
    for traj in df['Trajectory']:
        processed_traj = []
        for state in traj:
            processed_obs = z_score_array(state[0], mean, std)
            # add sun angle conversion
            sun_angle = state[0][-1]
            x = math.cos(sun_angle)
            y = math.sin(sun_angle)
            processed_obs[-1] = x
            processed_obs = np.append(processed_obs, y)
            state = (processed_obs, state[1], state[2], state[3], state[4])

            processed_traj.append(state)
        z_scored_trajectories.append(processed_traj)
    return pd.DataFrame({'Trajectory': z_scored_trajectories})

# load data
file_path1 = "datasets/ppo_dataset3.pkl"
file_path2 = "datasets/ppo_dataset4.pkl"
# file_path3 = "datasets/random_dataset.pkl"
# file_path4 = "datasets/random_dataset3.pkl"
# file_path5 = "datasets/random_dataset4.pkl"
file_path5 = "datasets/ppo_test_dataset.pkl"
file_path6 = "datasets/ppo_val_dataset.pkl"

data_files = [
    file_path1,
    file_path2,
    # file_path3,
    # file_path4,
    file_path5,
    file_path6
]

for file_path in data_files:
    df = load_dataframe(file_path6)

    mean, std = calculate_stats(df)

    z_scored_df = z_score_dataframe(df, mean, std)

    new_file = file_path.split('/')[-1]
    z_scored_df.to_pickle(f'processed_{new_file}')

