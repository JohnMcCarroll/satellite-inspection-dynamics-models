"""
This script loads in the dataset and trains a linear NN.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import math
from load_dataset import load_dataset, load_validation_dataset, load_sequence_dataset
from evaluate_rnn import get_rnn_eval_data
from models import RNN, apply_constraints
import numpy as np
import argparse
import pickle


# class DataFrameDataset(Dataset):
#     def __init__(self, dataframe):
#         self.data = dataframe.values

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         input_data = self.data[idx, :-1][0]
#         output_data = self.data[idx, -1]
#         return torch.tensor(input_data, dtype=torch.float32), torch.tensor(output_data, dtype=torch.float32)


# Function to pad lists with NaNs
def pad_list(lst, max_length):
    return lst + [[np.nan]*15] * (max_length - len(lst))


if __name__ == "__main__":
    # Parse cmd line args
    parser = argparse.ArgumentParser(description='Example script.')
    parser.add_argument('--prediction_size', type=int, default=1, help='The number of steps into the future the dynamics model will predict')
    parser.add_argument('--predict_delta', type=bool, default=False, help='If the dynamics model will predict the change in state')
    parser.add_argument('--constrain_output', type=bool, default=False, help="If the dynamics model's output will be constrained to only possible states")
    parser.add_argument('--input_size', type=int, default=15, help="The size of the input vector")
    parser.add_argument('--output_size', type=int, default=12, help="The size of the output vector")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs of training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Scale of optimization steps")
    parser.add_argument('--batch_size', type=int, default=128, help="Size of training data batches")
    parser.add_argument('--seed', type=int, default=105, help="The training seed")
    parser.add_argument('--hidden_layer_size', type=int, default=256, help="The size of the RNN's hidden layer")
    parser.add_argument('--model', type=str, default="RNN", help="The class name of dynamics model to train")

    args = parser.parse_args()

    # Define training configuration
    prediction_size = args.prediction_size  # Define number of steps model will be trained to predict
    predict_delta = args.predict_delta  # Model's prediction of state change or absolute next state
    constrain_output = args.constrain_output  # Constrain model's output to not violate environment constraints
    input_size = args.input_size  # Define input and output sizes
    output_size = args.output_size
    num_epochs = args.num_epochs
    seed = args.seed
    learning_rate = args.learning_rate
    hidden_layer_size = args.hidden_layer_size
    batch_size = args.batch_size
    model_name = args.model
    model_config_save_path = f'models/{model_name}_pred_size={prediction_size}_constrained={str(constrain_output)}_delta={str(predict_delta)}_lr{learning_rate}_bs{batch_size}.pkl'

    # Instantiate Model
    torch.manual_seed(seed)
    model = eval(model_name)(input_size, hidden_layer_size, output_size)

    # Load in training data
    df = load_sequence_dataset(prediction_size=prediction_size)
    val_df = load_validation_dataset()

    # Initialize the network, loss function, and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
            hidden_state = None
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
            model_config = {
                "model": model_name,
                "model_params": model.state_dict(),
                "prediction_size": prediction_size,
                "predict_delta": predict_delta,
                "constrain_output": constrain_output,
                "input_size": input_size,
                "output_size": output_size,
                "num_epochs": num_epochs,
                "seed": seed,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }
            with open(model_config_save_path, "wb") as f:
                pickle.dump(model_config, f)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'Val Error: {val_error:.4f}, Best Val Error: {best_error:.4f}')

    print("Training completed.")
