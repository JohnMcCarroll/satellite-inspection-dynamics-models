import pandas as pd
import os

# Load the DataFrame from a pickle file
ppo2_path = os.path.join(os.path.dirname(__file__), "../data/ppo_dataset2.pkl")
random1_path = os.path.join(os.path.dirname(__file__), "../data/random_dataset.pkl")
df1 = pd.read_pickle(ppo2_path)
df2 = pd.read_pickle(random1_path)

df1 = df1.reset_index(drop=True)
df2 = df2.reset_index(drop=True)

# Split the DataFrame into two roughly equal parts
split_index1 = 0
transitions = 0
for index, traj in df1["Trajectory"].items():
    t = len(traj) - 1
    transitions += t
    if transitions > 50000:
        split_index1 = index
        break

split_index2 = 0
transitions = 0
for index, traj in df2["Trajectory"].items():
    t = len(traj) - 1
    transitions += t
    if transitions > 50000:
        split_index2 = index
        break

df1a = df1.iloc[:split_index1].copy()
df1b = df1.iloc[split_index1:].copy()

df2a = df2.iloc[:split_index2].copy()
df2b = df2.iloc[split_index2:].copy()

result = pd.concat([df1a, df2a])

# Save the two DataFrames to separate pickle files
output_pickle_file1 = 'ppo_dataset2.pkl'
# output_pickle_file2 = 'ppo_dataset_test.pkl'
output_pickle_file3 = 'random_dataset.pkl'
output_pickle_file4 = 'test_dataset.pkl'

df1b.to_pickle(output_pickle_file1)
# df2.to_pickle(output_pickle_file2)
df2b.to_pickle(output_pickle_file3)
result.to_pickle(output_pickle_file4)

