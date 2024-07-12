"""
Define world models for training.
"""

import torch.nn as nn
import torch
import numpy as np
from torch import tanh, add, divide, sigmoid, subtract


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
        super(ConstrainedDeltaMLP, self).__init__()
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
        super(ConstrainedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, input):
        # Predict next state
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        x_constrained = x.clone()

        # transform model output to state ranges
        predicted_positions = tanh(x[:,0:3])
        predicted_velocities = tanh(x[:,3:6])
        predicted_inspected_points = sigmoid(x[:,6:7])
        predicted_cluster_positions = tanh(x[:,7:10])
        predicted_sun_angles = tanh(x[:,10:12])
        x_constrained[:,0:3] = predicted_positions
        x_constrained[:,3:6] = predicted_velocities
        x_constrained[:,6:7] = predicted_inspected_points
        x_constrained[:,7:10] = predicted_cluster_positions
        x_constrained[:,10:12] = predicted_sun_angles

        # return x_constrained # change var name

        # Apply constraints
        # Position must not exceed range [-1.0, 1.0] (-800 m, 800 m)
        max_distance = 1
        distances = torch.norm(predicted_positions, dim=1)
        out_of_bounds = distances > max_distance
        x_constrained[:,0:3][out_of_bounds] = predicted_positions[out_of_bounds] / distances[out_of_bounds].view(-1, 1) * max_distance

        # No velocity constraint to apply

        # Inspected points must not exceed 1.0 (100 points inspected) or decrease
        previous_num_inspected_points = input[:,6:7]
        over_estimates = predicted_inspected_points > 1
        under_estimates = predicted_inspected_points < previous_num_inspected_points
        ones = torch.ones_like(predicted_inspected_points[over_estimates])
        over_differences = subtract(predicted_inspected_points[over_estimates], ones)
        x_constrained[:,6:7][over_estimates] = subtract(predicted_inspected_points[over_estimates], over_differences)
        under_differences = subtract(previous_num_inspected_points[under_estimates], predicted_inspected_points[under_estimates])
        x_constrained[:,6:7][under_estimates] = add(predicted_inspected_points[under_estimates], under_differences)

        # Uninspected points location does not move towards agent
        offset_predicted_cluster_positions = predicted_cluster_positions.clone()
        delta_vector = subtract(predicted_cluster_positions, input[:,7:10]) # can't use predicted_cluster_pos
        deputy_vector = subtract(input[:,0:3], input[:,7:10])
        normalized_deputy_vector = deputy_vector / torch.norm(deputy_vector, dim=1).view(-1,1)
        deputy_direction_component = torch.einsum('ij,ij->i', delta_vector, deputy_vector)
        violations = deputy_direction_component > 0
        offset_predicted_cluster_positions[violations] = subtract(predicted_cluster_positions[violations], normalized_deputy_vector[violations])

        # Uninspected points cluster must not exceed [1] (10 m radius, but normalized already)
        cluster_distances = torch.norm(offset_predicted_cluster_positions, dim=1)
        exceedences = cluster_distances > 1
        # Normalize the point to lie on the unit sphere
        x_constrained[:,7:10][exceedences] = offset_predicted_cluster_positions[exceedences] / cluster_distances[exceedences].view(-1, 1)

        # Sun angle prediction must be point on unit circle
        normalized_sun_angle = torch.norm(predicted_sun_angles, dim=1)
        x_constrained[:,10:12] = divide(predicted_sun_angles, normalized_sun_angle.view(-1, 1))

        return x_constrained
