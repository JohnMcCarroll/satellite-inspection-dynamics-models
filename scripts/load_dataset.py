"""
This script loads the dataset from file(s) and organizes it into transition pairs.
"""

import os
import pickle
import pandas as pd
import numpy as np


def load_dataset(prediction_size: int = 1):
    # Load in DataFrames from file
    ppo1_path = os.path.join(os.path.dirname(__file__), "../datasets/processed_ppo_dataset1.pkl")
    ppo2_path = os.path.join(os.path.dirname(__file__), "../datasets/processed_ppo_dataset2.pkl")
    ppo3_path = os.path.join(os.path.dirname(__file__), "../datasets/processed_ppo_dataset3.pkl")
    ppo4_path = os.path.join(os.path.dirname(__file__), "../datasets/processed_ppo_dataset4.pkl")
    # random1_path = os.path.join(os.path.dirname(__file__), "../datasets/random_dataset.pkl")
    # random2_path = os.path.join(os.path.dirname(__file__), "../datasets/random_dataset4.pkl")
    # random3_path = os.path.join(os.path.dirname(__file__), "../datasets/random_dataset3.pkl")

    with open(ppo1_path, 'rb') as file:
        ppo1_data = pickle.load(file)

    with open(ppo2_path, 'rb') as file:
        ppo2_data = pickle.load(file)

    with open(ppo3_path, 'rb') as file:
        ppo3_data = pickle.load(file)

    with open(ppo4_path, 'rb') as file:
        ppo4_data = pickle.load(file)

    # with open(random3_path, 'rb') as file:
    #     random3_data = pickle.load(file)

    datasets = [
        ppo1_data,
        ppo2_data,
        ppo3_data,
        ppo4_data,
    ]

    # Parse data
    data_dict = {
        "sa": [],
        "s*": []
    }
    num_ppo_data = 0
    num_random_data = 0

    for dataset in datasets:
        # Collect state-action pairs from each dataset
        for row in dataset["Trajectory"]:
            if len(row) <= prediction_size:
                break
            for i in range(len(row)):
                if row[i+prediction_size] is row[-1]:
                    break
                else:
                    data_dict['sa'].append(np.concatenate((row[i][0], row[i][1])))
                    data_dict['s*'].append(row[i+prediction_size][0])
                    num_ppo_data += 1

    return pd.DataFrame(data_dict)

def load_test_dataset():
    # Load in DataFrames from file
    test_path = os.path.join(os.path.dirname(__file__), "../datasets/processed_ppo_test_dataset.pkl")

    with open(test_path, 'rb') as file:
        test_data = pickle.load(file)

    # Parse data
    data_dict = {
        "Trajectory": [],
    }

    for row in test_data["Trajectory"]:
        traj = [np.concatenate((mdp[0],mdp[1])) for mdp in row]
        data_dict['Trajectory'].append(traj)

    return pd.DataFrame(data_dict)

def load_validation_dataset():
    # Load in DataFrames from file
    test_path = os.path.join(os.path.dirname(__file__), "../datasets/processed_ppo_val_dataset.pkl")

    with open(test_path, 'rb') as file:
        test_data = pickle.load(file)

    # Parse data
    data_dict = {
        "Trajectory": [],
    }

    for row in test_data["Trajectory"]:
        traj = [np.concatenate((mdp[0],mdp[1])) for mdp in row]
        data_dict['Trajectory'].append(traj)

    return pd.DataFrame(data_dict)
