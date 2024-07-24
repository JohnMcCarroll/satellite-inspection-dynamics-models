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
from models import MLP256, MLP1024, apply_constraints
import argparse
import pickle
import gc
from utils import str2bool


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
    # Parse cmd line args
    parser = argparse.ArgumentParser(description='Example script.')
    parser.add_argument('--prediction_size', type=int, default=1, help='The number of steps into the future the dynamics model will predict')
    parser.add_argument('--predict_delta', type=str2bool, default=False, help='If the dynamics model will predict the change in state')
    parser.add_argument('--constrain_output', type=str2bool, default=False, help="If the dynamics model's output will be constrained to only possible states")
    parser.add_argument('--input_size', type=int, default=15, help="The size of the input vector")
    parser.add_argument('--output_size', type=int, default=12, help="The size of the output vector")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs of training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Scale of optimization steps")
    parser.add_argument('--batch_size', type=int, default=128, help="Size of training data batches")
    parser.add_argument('--seed', type=int, default=105, help="The training seed")
    parser.add_argument('--model', type=str, default="MLP256", help="The class name of dynamics model to train")

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
    batch_size = args.batch_size
    model_name = args.model
    model_config_save_path = f'models/{model_name}_pred_size={prediction_size}_constrained={str(constrain_output)}_delta={str(predict_delta)}_lr{learning_rate}_bs{batch_size}.pkl'

    # Instantiate Model
    torch.manual_seed(seed)
    model = eval(model_name)(input_size, output_size, predict_delta=predict_delta)

    # Load in training data
    df = load_dataset(prediction_size=prediction_size)
    val_df = load_validation_dataset()

    # Create dataset and dataloader
    dataset = DataFrameDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the network, loss function, and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping hyperparameters
    plateu_length = 5
    stop_length = 15
    steps_since_improvement = 0

    # Training loop
    torch.autograd.set_detect_anomaly(True)
    best_error = math.inf
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            inputs = inputs.to('cuda')
            targets = targets.to('cuda')

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
            steps_since_improvement = 0
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
        else:
            steps_since_improvement += 1

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'Val Error: {val_error:.4f}, Best Val Error: {best_error:.4f}')

        # Enforce Early Stopping
        if steps_since_improvement >= stop_length:
            break
        if steps_since_improvement >= plateu_length:
            learning_rate = learning_rate / 2
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Training completed.")

    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
