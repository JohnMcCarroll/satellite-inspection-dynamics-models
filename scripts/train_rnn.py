"""
This script loads in the dataset and trains a linear NN.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import math
from load_dataset import load_dataset, load_validation_dataset, load_sequence_dataset
from evaluate import get_eval_data
from evaluate_rnn import get_rnn_eval_data
from models import MLP256, MLP1024, NonlinearMLP, RNN, apply_constraints
import numpy as np


class DataFrameDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = self.data[idx, :-1][0]
        output_data = self.data[idx, -1]
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(output_data, dtype=torch.float32)


# Function to pad lists with NaNs
def pad_list(lst, max_length):
    return lst + [[np.nan]*15] * (max_length - len(lst))


if __name__ == "__main__":
    # Define training configuration
    prediction_size = 1  # Define number of steps model will be trained to predict
    predict_delta = False  # Model's prediction of state change or absolute next state
    constrain_output = True  # Constrain model's output to not violate environment constraints
    input_size = 15  # Define input and output sizes
    output_size = 12
    hidden_layer_size = 256
    num_epochs = 100
    model = RNN(input_size, hidden_layer_size, output_size)
    model_save_path = 'models/constrained_rnn_model_lr0.001_bs128.pth'
    batch_size = 128

    # Load in training data
    # df = load_dataset(prediction_size=prediction_size)
    df = load_sequence_dataset(prediction_size=prediction_size)
    val_df = load_validation_dataset()

    # Create dataset and dataloader
    # dataset = DataFrameDataset(df)
    # dataset = SequenceDataset(df)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the network, loss function, and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    torch.autograd.set_detect_anomaly(True)
    best_error = math.inf
    for epoch in range(num_epochs):
        shuffled_df = df.sample(frac=1).reset_index(drop=True)
        for j in range(0, len(shuffled_df), batch_size):
            if j+batch_size in shuffled_df.index:
                batch = shuffled_df.iloc[j:j+batch_size]
            else:
                batch = shuffled_df.iloc[j:-1]

            max_length = batch.map(len).max().max()
            # Apply padding function to each element in the DataFrame
            padded_df = batch.map(lambda x: pad_list(x, max_length))
            padded_array = np.array(padded_df.values.tolist(), dtype=np.float32)
            trajectories = torch.tensor(padded_array, dtype=torch.float32, device='cuda').squeeze(1)
            hidden_state = torch.zeros(1, len(batch), model.hidden_size).to('cuda')
            # Zero the parameter gradients
            optimizer.zero_grad()
            outputs = torch.zeros_like(trajectories[:,1:,0:12], dtype=torch.float32, device='cuda')
            targets = torch.zeros_like(trajectories[:,1:,0:12], dtype=torch.float32, device='cuda')

            for i in range(0, trajectories.shape[1]-1):
                model_input = trajectories[:,i:i+1,:]
                target_output = trajectories[:,i+prediction_size:i+prediction_size+1,0:12]

                # Handle NaNs
                mask = ~torch.isnan(target_output)[:,0,0]

                # Forward pass
                output, hidden_state = model(model_input, hidden_state, mask=mask)
                if constrain_output:
                    output = apply_constraints(output.view(-1,output_size), model_input[mask].view(-1,input_size)).view(-1, 1, output_size)

                # store outputs
                outputs[:,i:i+1,:][mask] = output
                targets[:,i:i+1,:][mask] = target_output[mask]

            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Calculate validation set error
        model_val_data = get_rnn_eval_data(val_df, "model_in_training", (), model=model, prediction_size=prediction_size, save_data=False, validation=True, batch_size=batch_size, constrain_output=constrain_output)
        model.train()
        val_error = model_val_data['model_in_training']['all']['median'][0]
        if val_error < best_error:
            best_error = val_error
            # Save the trained model
            torch.save(model.state_dict(), model_save_path)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'Val Error: {val_error:.4f}, Best Val Error: {best_error:.4f}')

    print("Training completed.")
