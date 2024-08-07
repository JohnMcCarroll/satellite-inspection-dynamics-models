import pickle
import numpy as np

# # load in pickle file and save out to different format
# save_file = "eval_data/delta_mlp_eval_data.pkl"
# # save_file = "eval_data/delta_prob_mlp_eval_data.pkl"

# dump_file = "eval_data/delta_mlp_eval_data_NUMPYLESS.pkl"
# # dump_file = "eval_data/delta_prob_mlp_eval_data_NUMPYLESS.pkl"

# with open(str(save_file), 'rb') as file:
#     data = pickle.load(file)
#     # convert numpy arrays to another format (lists?)
#     x = 1

# # remove numpy arrays
# for model in data.keys():
#     model_data = data[model]
#     new_model_data = {}
#     new_model_data["steps"] = list(model_data['steps'])
#     new_model_data['all'] = {
#         "median": list(model_data['all']['median']),
#         "quantiles_25": list(model_data['all']['quantiles_25']),
#         "quantiles_75": list(model_data['all']['quantiles_75'])
#     } 
#     new_model_data['position'] = {
#         "median": list(model_data['position']['median']),
#         "quantiles_25": list(model_data['position']['quantiles_25']),
#         "quantiles_75": list(model_data['position']['quantiles_75'])
#     }
#     new_model_data['velocity'] = {
#         "median": list(model_data['velocity']['median']),
#         "quantiles_25": list(model_data['velocity']['quantiles_25']),
#         "quantiles_75": list(model_data['velocity']['quantiles_75'])
#     }
#     new_model_data['inspected_points'] = {
#         "median": list(model_data['inspected_points']['median']),
#         "quantiles_25": list(model_data['inspected_points']['quantiles_25']),
#         "quantiles_75": list(model_data['inspected_points']['quantiles_75'])
#     }
#     new_model_data['uninspected_points'] = {
#         "median": list(model_data['uninspected_points']['median']),
#         "quantiles_25": list(model_data['uninspected_points']['quantiles_25']),
#         "quantiles_75": list(model_data['uninspected_points']['quantiles_75'])
#     }
#     new_model_data['sun_angle'] = {
#         "median": list(model_data['sun_angle']['median']),
#         "quantiles_25": list(model_data['sun_angle']['quantiles_25']),
#         "quantiles_75": list(model_data['sun_angle']['quantiles_75'])
#     }


# with open(str(dump_file), 'wb') as file:
#     pickle.dump(new_model_data, file)



# Inverse
import pickle
import numpy as np

# Load the pickle file and save it out in a different format
# save_file = "eval_data/delta_prob_mlp_eval_data_NUMPYLESS.pkl"
save_file = "eval_data/delta_mlp_eval_data_NUMPYLESS.pkl"

# dump_file = "eval_data/delta_prob_mlp_eval_data_NUMPY.pkl"
dump_file = "eval_data/delta_mlp_eval_data_NUMPY.pkl"

with open(save_file, 'rb') as file:
    data = pickle.load(file)

# Convert lists back to numpy arrays (float32)
data = {
    # "delta_prob_mlp": data
    "delta_mlp": data
}
for model in data.keys():
    model_data = data[model]
    new_model_data = {}
    new_model_data["steps"] = np.array(model_data['steps'], dtype=np.float32)
    new_model_data['all'] = {
        "median": np.array(model_data['all']['median'], dtype=np.float32),
        "quantiles_25": np.array(model_data['all']['quantiles_25'], dtype=np.float32),
        "quantiles_75": np.array(model_data['all']['quantiles_75'], dtype=np.float32)
    }
    new_model_data['position'] = {
        "median": np.array(model_data['position']['median'], dtype=np.float32),
        "quantiles_25": np.array(model_data['position']['quantiles_25'], dtype=np.float32),
        "quantiles_75": np.array(model_data['position']['quantiles_75'], dtype=np.float32),
    }
    new_model_data['velocity'] = {
        "median": np.array(model_data['velocity']['median'], dtype=np.float32),
        "quantiles_25": np.array(model_data['velocity']['quantiles_25'], dtype=np.float32),
        "quantiles_75": np.array(model_data['velocity']['quantiles_75'], dtype=np.float32),
    }
    new_model_data['inspected_points'] = {
        "median": np.array(model_data['inspected_points']['median'], dtype=np.float32),
        "quantiles_25": np.array(model_data['inspected_points']['quantiles_25'], dtype=np.float32),
        "quantiles_75": np.array(model_data['inspected_points']['quantiles_75'], dtype=np.float32),
    }
    new_model_data['uninspected_points'] = {
        "median": np.array(model_data['uninspected_points']['median'], dtype=np.float32),
        "quantiles_25": np.array(model_data['uninspected_points']['quantiles_25'], dtype=np.float32),
        "quantiles_75": np.array(model_data['uninspected_points']['quantiles_75'], dtype=np.float32),
    }
    new_model_data['sun_angle'] = {
        "median": np.array(model_data['sun_angle']['median'], dtype=np.float32),
        "quantiles_25": np.array(model_data['sun_angle']['quantiles_25'], dtype=np.float32),
        "quantiles_75": np.array(model_data['sun_angle']['quantiles_75'], dtype=np.float32),
    }
new_model_data = {
    # "delta_prob_mlp": data
    "delta_mlp": new_model_data
}

# Save the updated data to a new pickle file
with open(dump_file, 'wb') as file:
    pickle.dump(new_model_data, file)
