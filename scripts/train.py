"""
This script loads in the dataset and trains a linear NN.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import math
from load_dataset import load_dataset, load_validation_dataset
from evaluate import get_eval_data
from models import MLP256, MLP1024, NonlinearMLP, apply_constraints


class DataFrameDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = self.data[idx, :-1][0]
        output_data = self.data[idx, -1]
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(output_data, dtype=torch.float32)


if __name__ == "__main__":
    # Define training configuration
    prediction_size = 1  # Define number of steps model will be trained to predict
    predict_delta = True  # Model's prediction of state change or absolute next state
    constrain_output = False  # Constrain model's output to not violate environment constraints
    input_size = 15  # Define input and output sizes
    output_size = 12
    num_epochs = 100
    seed = 105
    model_save_path = 'models/delta_nonlinear_MLP_lr0.001_bs128.pth'
    
    # Instantiate Model
    torch.manual_seed(seed)
    model = NonlinearMLP(input_size, output_size, predict_delta=predict_delta)
    

    # Load in training data
    df = load_dataset(prediction_size=prediction_size)
    val_df = load_validation_dataset()

    # Create dataset and dataloader
    dataset = DataFrameDataset(df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize the network, loss function, and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    torch.autograd.set_detect_anomaly(True)
    best_error = math.inf
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            if constrain_output:
                outputs = apply_constraints(outputs, inputs)

            loss = criterion(outputs.squeeze(), targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Calculate validation set error
        model_val_data = get_eval_data(val_df, "model_in_training", (), model=model, prediction_size=prediction_size, save_data=False, validation=True)
        model.train()
        val_error = model_val_data['model_in_training']['all']['median'][0]
        if val_error < best_error:
            best_error = val_error
            # Save the trained model
            torch.save(model.state_dict(), model_save_path)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'Val Error: {val_error:.4f}, Best Val Error: {best_error:.4f}')

    print("Training completed.")
