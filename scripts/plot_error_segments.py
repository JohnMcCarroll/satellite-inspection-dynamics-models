"""
Module responsible for generating plots of each model for each error segment.
"""
from load_dataset import load_test_dataset
from evaluate import get_eval_data
from evaluate_rnn import get_rnn_eval_data
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from models import MLP256, ProbMLP, RNN, ProbRNN


if __name__ == "__main__":
    # prediction_size = 5
    plot_save_path = "plots/position_error_by_steps.png"
    # plot_save_path = "plots/prob_mlp_log_prob_error_by_steps.png"
    models = {
        # model_name: model_config_file_path
        "mlp": 'models/MLP256_pred_size=1_constrained=False_delta=False_lr0.001_bs128.pkl',
        ## "delta_mlp": 'models/MLP256_pred_size=1_constrained=False_delta=True_lr0.001_bs128.pkl',
        "rnn": 'models/RNN_pred_size=1_constrained=False_delta=False_lr0.001_bs128.pkl',
        # "delta_rnn": 'models/RNN_pred_size=1_constrained=False_delta=True_lr0.001_bs128.pkl',
        # "constrained_mlp": 'models/MLP256_pred_size=1_constrained=False_delta=False_lr0.001_bs128.pkl',
        # "constrained_delta_mlp": 'models/MLP256_pred_size=1_constrained=False_delta=True_lr0.001_bs128.pkl',
        # "constrained_rnn": 'models/RNN_pred_size=1_constrained=False_delta=False_lr0.001_bs128.pkl',
        # "constrained_delta_rnn": 'models/RNN_pred_size=1_constrained=False_delta=True_lr0.001_bs128.pkl',
        # "prob_mlp": 'models/MLP256_pred_size=1_constrained=False_delta=False_lr0.001_bs128.pkl',
        ## "delta_prob_mlp": 'models/MLP256_pred_size=1_constrained=False_delta=True_lr0.001_bs128.pkl',
        # "prob_rnn": 'models/RNN_pred_size=1_constrained=False_delta=False_lr0.001_bs128.pkl',
        # "delta_prob_rnn": 'models/RNN_pred_size=1_constrained=False_delta=True_lr0.001_bs128.pkl',
    }
    log_prob_error = False
    max_steps = 20
    state_key = "position"
    # state_key = "velocity"
    # state_key = "inspected_points"
    # state_key = "uninspected_points"
    # state_key = "sun_angle"
    # state_key = "all"

    # Load test dataset from file
    test_df = load_test_dataset()
    eval_data = {}

    for model_name, model_cfg_path in models.items():
        # load model from model config
        with open(model_cfg_path, 'rb') as f:
            model_config = pickle.load(f)
        model_cfg = (globals()[model_config['model']], model_config['model_params'], model_config['predict_delta'])
        prediction_size = model_config['prediction_size']
        constrain_output = model_config['constrain_output']
        eval_save_file = Path("eval_data") / f"{model_name}_eval_data.pkl"
        # if model_name.split("_")[-1] == "mlp":
        #     model_eval_data, _ = get_eval_data(test_df, model_name, model_cfg, save_file=eval_save_file, prediction_size=prediction_size, constrain_output=constrain_output,  max_steps=max_steps, log_prob_error=log_prob_error)
        # else:
        #     model_eval_data, _ = get_rnn_eval_data(test_df, model_name, model_cfg, save_file=eval_save_file, prediction_size=prediction_size, constrain_output=constrain_output,  max_steps=max_steps, log_prob_error=log_prob_error)
        with open(str(eval_save_file), 'rb') as file:
            model_eval_data = pickle.load(file)

        eval_data = eval_data | model_eval_data

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for i, (model, state_stats) in enumerate(eval_data.items()):
        data = state_stats[state_key]
        handle = ax.plot(state_stats["steps"], state_stats[state_key]["median"], label=model)
        ax.fill_between(state_stats["steps"], state_stats[state_key]["quantiles_25"],
                            state_stats[state_key]["quantiles_75"], alpha=0.3,
                            color=handle[0].get_color())
        ax.set_title(state_key)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Prediction Error")
        ax.legend()
        ax.set_yscale("log")

    fig.tight_layout()
    plt.savefig(plot_save_path)
    plt.show()
