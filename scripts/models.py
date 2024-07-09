"""
Define world models for training.
"""

import torch.nn as nn
import torch
import numpy as np
from torch import tanh, add, divide


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


class ConstrainedDeltaMLP(nn.Module):
    """
    Linear MLP with 256-dim hidden layer.
    This output of this model is constrained so that impossible predictions do not occur.
    NN Model outputs are used to predict the delta between current state and next state.
    Constraints:
    - minimum and maximum values of xyz position enforced
    - number of inspected points can never decrease
    - uninspected points cluster will never move towards the agent or beyond the radius of chief satellite
    - sun angle prediction restricted to point on unit circle
    Note: data will be normalized between [-1,1], rather than z scored.
    """

    def __init__(self, input_size, output_size):
        super(MLP256, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, input):
        # Predict state Delta
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # Apply delta constraints
        # inspected points delta must not be negative
        x[6] = divide(add(tanh(x[6]), torch.tensor(np.array([1])), torch.tensor(np.array([2]))))
        # Add model's output delta to input
        output = add(input, x)
        # Apply state constraints
        output = torch.clamp(output, min=-1, max=1)
        return output


class ConstrainedMLP(nn.Module):
    """
    Linear MLP with 256-dim hidden layer.
    This output of this model is constrained so that impossible predictions do not occur.
    Constraints:
    - minimum and maximum values of xyz position enforced
    - number of inspected points can never decrease
    - uninspected points cluster will never move towards the agent or beyond the radius of chief satellite
    - sun angle prediction restricted to point on unit circle
    Note: data will be normalized between [-1,1], rather than z scored.
    """

    def __init__(self, input_size, output_size):
        super(MLP256, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, input):
        # Predict next state
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # Apply constraints
        # position must not exceed range [-1.0, 1.0] (-800 m, 800 m)
        # max distance constraint
        x[0:2] = tanh(x[0:2])

        # Apply collision constraint: distance from chief (origin) cannot fall below [0.01875] (15 m)
        collision_threshold = 0.01875
        deputy_distance = torch.norm(x[0:2])
        if deputy_distance <= collision_threshold:
            # Normalize the point to lie on the sphere of the given collision_threshold radius
            x[0:2] = x[0:2] / deputy_distance * collision_threshold
        
        # No velocity constraint to apply

        # Inspected points must not exceed 1.0 (100 points inspected)
        predicted_inspected_points = divide(add(tanh(x[6]), torch.tensor(np.array([1])), torch.tensor(np.array([2]))))
        # Inspected points must not decrease
        x[6] = torch.max(predicted_inspected_points, x[6])

        # Uninspected points cluster must not exceed [0.0125] (10 m)
        # Looks like cluster location is already normalized, meaning it won't be outside of a unit sphere
        chief_radius = 1
        cluster_distance = torch.norm(x[7:9])
        if cluster_distance >= chief_radius:
            # Normalize the point to lie on the sphere of the given chief_radius radius
            x[7:9] = x[7:9] / cluster_distance * chief_radius

        # uninspected points location does not move towards agent
        cur_dist_from_deputy = torch.norm(input[7:9] - input[0:2])
        pred_dist_from_deputy = torch.norm(x[7:9] - input[0:2])
        if pred_dist_from_deputy < cur_dist_from_deputy:
            # Return previous cluster location
            x[7:9] = input[7:9]

        # Sun angle prediction must be point on unit circle
        normalized_sun_angle = torch.norm(x[10:11])
        x[10:11] = x[10:11] / (normalized_sun_angle + 1e-8)

        return x
