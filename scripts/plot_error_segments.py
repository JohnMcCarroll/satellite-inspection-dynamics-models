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
        # model_name: model_eval_data_path
        #   log MSE plots
        "mlp": 'eval_data/mlp_eval_data.pkl',
        "delta_mlp": 'eval_data/delta_mlp_eval_data_NUMPY.pkl',
        "rnn": 'eval_data/rnn_eval_data.pkl',
        "delta_rnn": 'eval_data/delta_rnn_eval_data.pkl',
        "constrained_mlp": 'eval_data/constrained_mlp_eval_data.pkl',
        "constrained_delta_mlp": 'eval_data/constrained_delta_mlp_eval_data.pkl',
        "constrained_rnn": 'eval_data/constrained_rnn_eval_data.pkl',
        "constrained_delta_rnn": 'eval_data/constrained_delta_rnn_eval_data.pkl',

        #  log prob plots:
        # "prob_mlp": 'eval_data/prob_mlp_log_prob_eval_data.pkl',
        # "delta_prob_mlp": 'eval_data/delta_prob_mlp_log_prob_eval_data.pkl',
        # "prob_rnn": 'eval_data/prob_rnn_log_prob_eval_data.pkl',
        # "delta_prob_rnn": 'eval_data/delta_prob_rnn_log_prob_eval_data.pkl',
        # "mlp": 'eval_data/mlp_log_prob_eval_data.pkl',
        # "delta_mlp": 'eval_data/delta_mlp_log_prob_eval_data.pkl',
        # "rnn": 'eval_data/rnn_log_prob_eval_data.pkl',
        # "delta_rnn": 'eval_data/delta_rnn_log_prob_eval_data.pkl',
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

    for model_name, model_eval_data_path in models.items():
        # load model from model config
        # with open(model_cfg_path, 'rb') as f:
        #     model_config = pickle.load(f)
        # model_cfg = (globals()[model_config['model']], model_config['model_params'], model_config['predict_delta'])
        # prediction_size = model_config['prediction_size']
        # constrain_output = model_config['constrain_output']
        # eval_save_file = Path("eval_data") / f"{model_name}_eval_data.pkl"
        # if model_name.split("_")[-1] == "mlp":
        #     model_eval_data, _ = get_eval_data(test_df, model_name, model_cfg, save_file=eval_save_file, prediction_size=prediction_size, constrain_output=constrain_output,  max_steps=max_steps, log_prob_error=log_prob_error)
        # else:
        #     model_eval_data, _ = get_rnn_eval_data(test_df, model_name, model_cfg, save_file=eval_save_file, prediction_size=prediction_size, constrain_output=constrain_output,  max_steps=max_steps, log_prob_error=log_prob_error)
        eval_save_file = f"{model_eval_data_path}"
        with open(eval_save_file, 'rb') as file:
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
