import pickle
import pandas as pd
import numpy as np

def load_df_from_file(file_path):
    """
    Load and return a dictionary from a pickle file.

    :param file_path: Path to the file containing the pickled dictionary.
    :return: The unpickled dictionary.
    """
    try:
        with open(file_path, 'rb') as file:
            # Load the dictionary from the file
            dictionary = pickle.load(file)
            return dictionary
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Load Dataframes
# file_path1 = "/tmp/MBRL/eval_data_df.pkl"
# file_path2 = "../random_dataset2.pkl"
file_path2 = "ppo_eval_data_df2.pkl"
file_path2 = "/home/john/RIT CS Masters/MBRL/satellite-inspection-dynamics-models/ppo_eval_data_df2.pkl"
# policy_df = load_df_from_file(file_path1)
policy_df = load_df_from_file(file_path2)


# Remove extra columns + add labels
policy_df['Policy'] = "PPO"
# random_df['Policy'] = "Random"
# policy_df = policy_df[['Trajectory', 'Policy']]

# Combine random and policy datasets
# dataset = pd.concat([policy_df, random_df], ignore_index=True)
dataset = policy_df

# Parse action and observation dicts
for i in range(dataset.shape[0]):
    # get trajectory and prepare new list
    new_traj = []
    traj = dataset['Trajectory'][i]
    for j in range(len(traj)):
        state = traj[j]
        obs = state[0]
        actions = state[1]
        reward = state[3]
        done = state[4]

        obs_array = np.array([0.0]*11, dtype=np.float32)
        actions_array = np.array([0.0]*3, dtype=np.float32)

        if 'blue0_ctrl' in obs:
            # handle random policy data format + undo normalization
            obs_array[0] = obs['blue0_ctrl']['Obs_Sensor_Position']['direct_observation'][0] * 100
            obs_array[1] = obs['blue0_ctrl']['Obs_Sensor_Position']['direct_observation'][1] * 100
            obs_array[2] = obs['blue0_ctrl']['Obs_Sensor_Position']['direct_observation'][2] * 100
            obs_array[3] = obs['blue0_ctrl']['Obs_Sensor_Velocity']['direct_observation'][0] * 0.5
            obs_array[4] = obs['blue0_ctrl']['Obs_Sensor_Velocity']['direct_observation'][1] * 0.5
            obs_array[5] = obs['blue0_ctrl']['Obs_Sensor_Velocity']['direct_observation'][2] * 0.5
            obs_array[6] = float(int(obs['blue0_ctrl']['Obs_Sensor_InspectedPoints']['direct_observation'][0] * 100))
            obs_array[7] = obs['blue0_ctrl']['Obs_Sensor_UninspectedPoints']['direct_observation'][0]
            obs_array[8] = obs['blue0_ctrl']['Obs_Sensor_UninspectedPoints']['direct_observation'][1]
            obs_array[9] = obs['blue0_ctrl']['Obs_Sensor_UninspectedPoints']['direct_observation'][2]
            obs_array[10] = obs['blue0_ctrl']['Obs_Sensor_SunAngle']['direct_observation'][0]

            actions_array[0] = actions['blue0_ctrl']['RTAModule'][0]['x_thrust'][0]
            actions_array[1] = actions['blue0_ctrl']['RTAModule'][1]['y_thrust'][0]
            actions_array[2] = actions['blue0_ctrl']['RTAModule'][2]['z_thrust'][0]

            reward = reward['blue0_ctrl']
            done = done['__all__']

        else:
            # handle on policy data format
            obs_array[0] = obs['Obs_Sensor_Position']['direct_observation'][0].m
            obs_array[1] = obs['Obs_Sensor_Position']['direct_observation'][1].m
            obs_array[2] = obs['Obs_Sensor_Position']['direct_observation'][2].m
            obs_array[3] = obs['Obs_Sensor_Velocity']['direct_observation'][0].m
            obs_array[4] = obs['Obs_Sensor_Velocity']['direct_observation'][1].m
            obs_array[5] = obs['Obs_Sensor_Velocity']['direct_observation'][2].m
            obs_array[6] = obs['Obs_Sensor_InspectedPoints']['direct_observation'][0].m
            obs_array[7] = obs['Obs_Sensor_UninspectedPoints']['direct_observation'][0].m
            obs_array[8] = obs['Obs_Sensor_UninspectedPoints']['direct_observation'][1].m
            obs_array[9] = obs['Obs_Sensor_UninspectedPoints']['direct_observation'][2].m
            obs_array[10] = obs['Obs_Sensor_SunAngle']['direct_observation'][0].m

            actions_array[0] = actions['RTAModule.x_thrust'][0]
            actions_array[1] = actions['RTAModule.y_thrust'][0]
            actions_array[2] = actions['RTAModule.z_thrust'][0]

        new_state = (obs_array, actions_array, state[2], reward, done)
        new_traj.append(new_state)

    dataset['Trajectory'][i] = new_traj


split_index = len(dataset) // 2
test_split_index = (len(dataset) * 6) // 8
val_split_index = (len(dataset) * 7) // 8
df1 = dataset.iloc[:split_index].copy()
df2 = dataset.iloc[split_index:test_split_index].copy()
df_test = dataset.iloc[test_split_index:val_split_index].copy()
df_val = dataset.iloc[val_split_index:].copy()

df1.to_pickle('ppo_dataset3.pkl')
df2.to_pickle('ppo_dataset4.pkl')
df_test.to_pickle('ppo_test_dataset.pkl')
df_val.to_pickle('ppo_val_dataset.pkl')

