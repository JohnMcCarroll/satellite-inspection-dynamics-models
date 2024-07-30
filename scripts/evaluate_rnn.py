"""
This script evaluates the model's prediction accuracy as a function of timesteps.
"""
import numpy as np
import torch
from load_dataset import load_test_dataset
from models import RNN, ProbRNN, apply_constraints
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import itertools
from typing import Optional
from pathlib import Path
from constants import NAMED_STATE_RANGES
import pandas as pd


# Calculate error between model output and target vector
def euclidean_distance(output, target):
    return np.sqrt(np.sum((output - target)**2))

# Function to pad lists with NaNs
def pad_list(lst, max_length):
    return lst + [[np.nan]*15] * (max_length - len(lst))

def get_rnn_eval_data(
        test_df: pd.DataFrame,
        model_name: str,
        model_cfg: tuple,
        input_size: int = 15,
        output_size: int = 12,
        max_steps: int = 50,
        save_file: Optional[Path] = None,
        model: torch.nn.Module = None,
        prediction_size: int = 1,
        save_data: bool = True,
        validation: bool = False,
        batch_size: int = 128,
        constrain_output: bool = False,
    ):
    """
    Collects multistep prediction error of given model or loads evaluation data from file.
    """
    if save_file is not None and save_file.exists():
        with open(str(save_file), 'rb') as file:
            return pickle.load(file)

    # Initialize data structures to store evaluation data
    eval_data = {}
    errors = {
        k: {num_steps: [] for num_steps in range(prediction_size, max_steps+1, prediction_size)} for k in NAMED_STATE_RANGES.keys()
    }

    # Load the trained model
    if model is None:
        # TODO: add hidden layer size (get from model config)
        model = model_cfg[0](input_size, 256, output_size)
        model.load_state_dict(model_cfg[1])
        model.predict_delta = model_cfg[2]
    model.eval()

    is_probabilistic = False
    if isinstance(model, ProbRNN):
        is_probabilistic = True

    with torch.no_grad():
        for j in range(0, len(test_df), batch_size):
            if j+batch_size in test_df.index:
                batch = test_df.iloc[j:j+batch_size]
            else:
                batch = test_df.iloc[j:-1]
            max_length = batch.map(len).max().max()
            # Apply padding function to each element in the DataFrame
            padded_df = batch.map(lambda x: pad_list(x, max_length))
            padded_array = np.array(padded_df.values.tolist(), dtype=np.float32)
            trajectories = torch.tensor(padded_array, dtype=torch.float32, device='cuda').squeeze(1)
            hidden_state = torch.zeros(1, len(batch), model.hidden_size).to('cuda')
            actions = trajectories[:,:,-3:]
            multistep_predictions = {}
            target_states = {}
            final_state_index = trajectories.shape[1]-1

            for i in range(0, trajectories.shape[1]-1):
                # compute first prediction step + store hidden state
                state_action = trajectories[:,i:i+1,:]
                target_output = trajectories[:,i+prediction_size:i+prediction_size+1,0:12]

                # Handle NaNs
                mask = ~torch.isnan(target_output)[:,0,0]

                # Forward pass
                output, hidden_state = model(state_action, hidden_state, mask=mask)

                # Sample distributions (for probabilistic models only)
                if is_probabilistic:
                    means = output[:, :, :12]
                    log_vars = output[:, :, 12:]
                    # Convert log variances to standard deviations
                    std_devs = torch.exp(0.5 * log_vars)
                    # Create multivariate normal distributions
                    mvn = torch.distributions.Normal(loc=means, scale=std_devs)
                    # Sample from each distribution
                    output = mvn.sample()

                if constrain_output:
                    output = apply_constraints(output.view(-1,output_size), state_action[mask].view(-1,input_size)).view(-1, 1, output_size)

                multistep_predictions[(i,i+prediction_size)] = output
                target_states[(i,i+prediction_size)] = target_output[mask]

                # compute multistep predictions from initial step i+prediction_size
                if not validation:
                    remaining_steps = final_state_index - (i+prediction_size)
                    multistep_prediction_hidden_state = hidden_state[:,mask,:].clone()
                    multistep_target_outputs = trajectories[mask,:,0:12].clone()
                    multistep_actions = actions[mask].clone()
                    for k in range(prediction_size, remaining_steps, prediction_size):
                        if k > max_steps:
                            # Model error compounds exponentially, don't waste compute on long range
                            break
                        # Concatentate model's predicted state with next taken action
                        if k+i+prediction_size < final_state_index-1:
                            multistep_target_output = multistep_target_outputs[:,k+i+prediction_size:k+i+prediction_size+1,:]
                            # Handle NaNs
                            multistep_mask = ~torch.isnan(multistep_target_output)[:,0,0]
                            state_action = torch.concatenate((output[multistep_mask],multistep_actions[multistep_mask,k+i+prediction_size:k+i+prediction_size+1,:]), dim=2)
                            # Forward pass
                            output, multistep_prediction_hidden_state = model(state_action, multistep_prediction_hidden_state[:,multistep_mask,:])

                            # Sample distributions (for probabilistic models only)
                            if is_probabilistic:
                                means = output[:, :, :12]
                                log_vars = output[:, :, 12:]
                                # Convert log variances to standard deviations
                                std_devs = torch.exp(0.5 * log_vars)
                                # Create multivariate normal distributions
                                mvn = torch.distributions.Normal(loc=means, scale=std_devs)
                                # Sample from each distribution
                                output = mvn.sample()

                            if constrain_output:
                                output = apply_constraints(output.view(-1,output_size), state_action.view(-1,input_size)).view(-1, 1, output_size)

                            # store model's prediction and ground truth target state
                            multistep_predictions[(i+prediction_size,i+prediction_size+k)] = output
                            target_states[(i+prediction_size,i+prediction_size+k)] = multistep_target_output[multistep_mask]

                            # remove ended trajectories, to keep output + target dims aligned
                            multistep_target_outputs = multistep_target_outputs[multistep_mask]
                            multistep_actions = multistep_actions[multistep_mask]
                        else:
                            break

            # Compute and store error
            for key, predicted_states in multistep_predictions.items():
                for l in range(predicted_states.shape[0]):
                    actual_state = target_states[key][l]
                    predicted_state = predicted_states[l]
                    num_steps = key[1] - key[0]
                    for state_key, state_range in NAMED_STATE_RANGES.items():
                        errors[state_key][num_steps].append(
                            euclidean_distance(
                                predicted_state[0][state_range].cpu().numpy(),
                                actual_state[0][state_range].cpu().numpy()
                            )
                        )

        # Calculate summary statistics of different slices of the data
        eval_data[model_name] = {"steps": np.arange(prediction_size, max_steps + 1, prediction_size)}
        for state_key in NAMED_STATE_RANGES.keys():
            medians = np.full(len(eval_data[model_name]['steps']), np.nan)
            quantiles_25 = np.full(len(eval_data[model_name]['steps']), np.nan)
            quantiles_75 = np.full(len(eval_data[model_name]['steps']), np.nan)
            for index, (step, error) in enumerate(errors[state_key].items()):
                medians[index] = np.median(error)
                quantiles_25[index] = np.quantile(error, 0.25)
                quantiles_75[index] = np.quantile(error, 0.75)
                if validation:
                    break

            eval_data[model_name][state_key] = {
                "median": medians,
                "quantiles_25": quantiles_25,
                "quantiles_75": quantiles_75,
            }

        # Save out eval data
        if save_data:
            with open(str(save_file), 'wb') as file:
                pickle.dump(eval_data, file)

        return eval_data


if __name__ == "__main__":
    # Configure Evaluation
    plot_save_path = "plots/rnn_error_by_steps.png"
    models = {
        # model_name: model_config_file_path
        "rnn": 'models/RNN_pred_size=1_constrained=False_delta=False_lr0.001_bs128.pkl',
        # "linear_1024": (MLP1024, 'models/linear_model_1024.pth'),
    }
    # input_size = 15
    # output_size = 12
    # eval_data = copy.deepcopy(models)
    # eval_name = "LinearModelSize"

    # Load test dataset from file
    test_df = load_test_dataset()
    eval_data = {}

    # Evaluate model(s)
    for model_name, model_cfg_path in models.items():
        # load model from model config
        with open(model_cfg_path, 'rb') as f:
            model_config = pickle.load(f)
        model_cfg = (globals()[model_config['model']], model_config['model_params'], model_config['predict_delta'])
        prediction_size = model_config['prediction_size']
        constrain_output = model_config['constrain_output']
        eval_save_file = Path("eval_data") / f"{model_name}_eval_data.pkl"
        model_eval_data = get_rnn_eval_data(test_df, model_name, model_cfg, save_file=eval_save_file, prediction_size=prediction_size, constrain_output=constrain_output, max_steps=20)
        eval_data = eval_data | model_eval_data

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for i, (model, state_stats) in enumerate(eval_data.items()):
        for stat in state_stats.keys():
            if stat == "steps":
                continue
            handle = ax[i].plot(state_stats["steps"], state_stats[stat]["median"], label=stat)
            ax[i].fill_between(state_stats["steps"], state_stats[stat]["quantiles_25"],
                               state_stats[stat]["quantiles_75"], alpha=0.3,
                               color=handle[0].get_color())
        ax[i].set_title(model)
        ax[i].set_xlabel("Steps")
        ax[i].set_ylabel("Prediction Error")
        ax[i].legend()
        ax[i].set_yscale("log")

    fig.tight_layout()
    plt.savefig(plot_save_path)
    plt.show()
