"""
This script loads in the dataset and trains a linear NN.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from load_dataset import load_dataset


class DataFrameDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = self.data[idx, :-1][0]
        output_data = self.data[idx, -1]
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(output_data, dtype=torch.float32)


class MLP256(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP256, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP1024(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP1024, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # Define number of steps model will be trained to predict
    prediction_size = 5
    
    # Load in training data
    df = load_dataset(prediction_size=prediction_size)

    # Create dataset and dataloader
    dataset = DataFrameDataset(df)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define input and output sizes
    input_size = 15
    output_size = 12

    # Initialize the network, loss function, and optimizer
    model = MLP256(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training completed.")

    # Save the trained model
    model_path = 'models/linear_model_1024.pth'
    torch.save(model.state_dict(), model_path)
