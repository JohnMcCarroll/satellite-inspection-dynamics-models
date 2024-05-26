import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from corl.evaluation.api import evaluate
from corl.evaluation.api_utils import get_training_iter_from_ckpt_path
from safe_autonomy_sims.evaluation.launch.serialize_cwh3d import SerializeCWH3D
from corl.evaluation.runners.section_factories.test_cases.default_strategy import DefaultStrategy
from safe_autonomy_sims.evaluation.inspection_metrics import DeltaV, InspectedPoints, Success
from corl.evaluation.metrics.metric import Metric
from corl.libraries.units import Quantity
from corl.evaluation.metrics.generator import MetricGeneratorTerminalEventScope
from corl.evaluation.episode_artifact import EpisodeArtifact


class Trajectory(MetricGeneratorTerminalEventScope):
    """
    """

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:  # noqa: PLR6301
        if "agent_id" not in kwargs:
            raise RuntimeError('Expecting "agent_id" to be provided')

        trajectories = []

        for step_info in params.steps:
            # collect state info for each timestep
            obs = step_info.agents['blue0_ctrl'].observations
            actions = step_info.agents['blue0_ctrl'].actions
            reward_components = step_info.agents['blue0_ctrl'].rewards
            reward = step_info.agents['blue0_ctrl'].total_reward
            done = True if step_info is params.steps[-1] else False

            trajectories.append((obs, actions, reward_components, reward, done))
        
        return trajectories


# Define variables
experiment_name = "RllibPPOInspectionTranslation"
checkpoint_paths = ["/tmp/safe-autonomy-sims/output/tune/TRANSLATIONAL-INSPECTION/TRANSLATIONAL-INSPECTION-PPO_CorlMultiAgentEnv_8f57d_00000_0_2024-05-14_15-24-19/checkpoint_000011"]
expr_config = "/home/john/AFRL/dle/safe-autonomy-sims/configs/translational-inspection-no-rejection-sampler/experiment.yml"
launch_dir_of_experiment = "/home/john/AFRL/sas/safe-autonomy-sims"
task_config_path = "/home/john/AFRL/dle/safe-autonomy-sims/configs/translational-inspection-no-rejection-sampler/task.yml"
test_case_manager_config = {
    "type": f"{DefaultStrategy.__module__}.{DefaultStrategy.__name__}",
    "config": {
        "num_test_cases": 1500
    }
}
output_path = "/tmp/MBRL/"
# parallel eval episodes
num_workers = 10

output_paths = []

for ckpt_path in checkpoint_paths:

    ckpt_num = get_training_iter_from_ckpt_path(ckpt_path)
    ckpt_output_path = output_path + str(ckpt_num)
    os.makedirs(ckpt_output_path, exist_ok=True)

    # launch evaluate
    evaluate(
        task_config_path,
        ckpt_path,
        ckpt_output_path,
        expr_config,
        launch_dir_of_experiment,
        SerializeCWH3D,
        test_case_manager_config,
        num_workers=num_workers,
    )

    output_paths.append(ckpt_output_path)

# calculate metrics
metrics = {
    "Trajectory": Trajectory(name="Trajectory"),
    # "DeltaV": DeltaV(name="DeltaV"),
    # "InspectedPoints": InspectedPoints(name="InspectedPoints"),
    # "InspectionSuccess": Success(name="InspectionSuccess"),
    # "Timesteps": Timesteps(), #TODO
}
df_columns = [key for key in metrics.keys()]
df_columns = ["ckpt", "test_case"] + df_columns
data = {key: [] for key in df_columns}

agent_id="blue0_ctrl"

for ckpt_output_path in output_paths:
    test_cases_glob = list(glob.glob(ckpt_output_path + "/test_case_*"))
    for test_case_path in test_cases_glob:
        data["ckpt"].append(get_training_iter_from_ckpt_path(ckpt_output_path))
        data["test_case"].append(get_training_iter_from_ckpt_path(test_case_path))
        # load episode artifact
        episode_artifact_file_path = glob.glob(test_case_path + "/*episode_artifact.pkl")[0]
        with open(episode_artifact_file_path, "rb") as episode_artifact_file:
            episode_artifact = pickle.load(episode_artifact_file)
        # process metrics
        for name, metric in metrics.items():
            result = metric.generate_metric(episode_artifact, agent_id=agent_id)
            if isinstance(result, Metric):
                result = result.value
            if isinstance(result, Quantity):
                result = result.value
            data[name].append(result)

df = pd.DataFrame(data)
df.to_pickle(output_path + '/eval_data_df.pkl')

