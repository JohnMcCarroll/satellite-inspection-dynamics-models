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


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    df = load_dataset()

    # Create dataset and dataloader
    dataset = DataFrameDataset(df)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define input and output sizes
    input_size = 14
    output_size = 11

    # Initialize the network, loss function, and optimizer
    model = MLP(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
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
    model_path = 'models/linear_model.pth'
    torch.save(model.state_dict(), model_path)