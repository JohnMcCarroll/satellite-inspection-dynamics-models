"""
This script evaluates the model's prediction accuracy as a function of timesteps.
"""
import numpy as np
import torch
from load_dataset import load_test_dataset
from models import MLP256, MLP1024, apply_constraints
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from typing import Optional
from pathlib import Path
from constants import NAMED_STATE_RANGES


# Calculate error between model output and target vector
def euclidean_distance(output, target):
    return np.sqrt(np.sum((output - target) ** 2))

def get_eval_data(
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
        constrain_output: bool = False
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
        model = model_cfg[0](input_size, output_size)
        model.load_state_dict(model_cfg[1])
    model.eval()

    with torch.no_grad():
        for trajectory_idx, trajectory in enumerate(test_df['Trajectory']):
            # Create multistep predictions for each trajectory in test dataset
            final_state_index = len(trajectory)
            state_actions = torch.from_numpy(np.stack(np.array(trajectory))).to(torch.float32).to('cuda')
            states = state_actions[:,0:output_size]
            for delta_t in range(1, max_steps+1):
                if delta_t * prediction_size > max_steps:
                    # delta_t ~= number of subsequent model predictions
                    # delta_t * prediction_size = num timesteps in the future model is predicting
                    # Stop, if we are predicting past max_steps
                    break
                # Compute next prediction step
                predicted_states = model(state_actions)
                if constrain_output:
                    predicted_states = apply_constraints(predicted_states, state_actions)
                predicted_states = predicted_states[0:final_state_index - delta_t]
                state_actions = torch.concat((predicted_states, state_actions[0:final_state_index-delta_t,output_size:input_size]), dim=1).to('cuda')
                # Compute and store error
                for i in range(0,final_state_index - delta_t + 1):
                    # Iterate through predictions tensor (contains all predictions of depth `delta_t`)
                    if i >= predicted_states.shape[0]:
                        # Size of predictions within scope of trajectory decreases each delta_t increment
                        break
                    for state_key, state_range in NAMED_STATE_RANGES.items():
                        errors[state_key][delta_t * prediction_size].append(
                            euclidean_distance(
                                predicted_states[i][state_range].cpu().numpy(),
                                states[i+delta_t][state_range].cpu().numpy()
                            )
                        )
                if validation:
                    # no need to evaluate multiple steps for validation
                    break

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
    # prediction_size = 5
    plot_save_path = "plots/5_step_mlp_error_by_steps.png"
    models = {
        # model_name: model_config_file_path
        "linear_256": '/home/john/RIT CS Masters/MBRL/satellite-inspection-dynamics-models/models/MLP256_pred_size=1_constrained=False_delta=False_lr0.001_bs128.pkl',
        # "5_step_linear_1024": (MLP1024, 'models/5_step_linear_model_1024.pth'),
    }

    # Load test dataset from file
    test_df = load_test_dataset()
    eval_data = {}

    for model_name, model_cfg_path in models.items():
        # load model from model config
        with open(model_cfg_path, 'rb') as f:
            model_config = pickle.load(f)
        model_cfg = (globals()[model_config['model']], model_config['model_params'])
        prediction_size = model_config['prediction_size']
        constrain_output = model_config['constrain_output']
        eval_save_file = Path("eval_data") / f"{model_name}_eval_data_TEST.pkl"
        # eval_save_file = None
        model_eval_data = get_eval_data(test_df, model_name, model_cfg, save_file=eval_save_file, prediction_size=prediction_size, constrain_output=constrain_output,  max_steps=20)
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
