from corl.evaluation.metrics.metric import Metric
from corl.evaluation.metrics.generator import MetricGeneratorTerminalEventScope
from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.environment.multi_agent_env import ACT3MultiAgentEnv
from corl.train_rl import parse_corl_args, build_experiment
import random
import ray
from ray.rllib.algorithms.ppo import PPO
import gymnasium as gym
import pandas as pd


class Trajectory(MetricGeneratorTerminalEventScope):
    """Save lists of (obs, actions, reward_components, reward, done) tuples representing trajectories.
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


def sample_nested_action_space(action_space):
    """
    Randomly sample each action space in a nested gymnasium.spaces.Dict or gymnasium.spaces.Tuple of Box spaces.
    
    :param action_space: A gymnasium.spaces.Dict or gymnasium.spaces.Tuple of Box spaces.
    :return: A nested dictionary or tuple of NumPy array actions.
    """
    def recursive_sample(space):
        if isinstance(space, gym.spaces.Dict):
            return {key: recursive_sample(subspace) for key, subspace in space.spaces.items()}
        elif isinstance(space, gym.spaces.Tuple):
            return tuple(recursive_sample(subspace) for subspace in space.spaces)
        elif isinstance(space, gym.spaces.Box):
            return space.sample()
        else:
            raise ValueError("Unsupported space type")

    return recursive_sample(action_space)


# Define variables
checkpoint_paths = [
    "/tmp/safe-autonomy-sims/output/tune/TRANSLATIONAL-INSPECTION/TRANSLATIONAL-INSPECTION-PPO_CorlMultiAgentEnv_8f57d_00000_0_2024-05-14_15-24-19/checkpoint_000011",
]
expr_config = "/home/john/AFRL/dle/safe-autonomy-sims/configs/translational-inspection-no-rejection-sampler/experiment.yml"
output_path = "/tmp/MBRL/"

# initialize ray, parse experiment args, initialize env, and load PPO policy from ckpt
ray.init(ignore_reinit_error=True)
args = parse_corl_args(["--cfg", expr_config])
experiment_class, experiment_file_validated = build_experiment(args)
experiment_class.config.env_config["agents"], experiment_class.config.env_config["agent_platforms"] = experiment_class.create_agents(
    experiment_file_validated.platform_config, experiment_file_validated.agent_config
)
env = ACT3MultiAgentEnv(experiment_class.config.env_config)
env_config = experiment_class.config.env_config
observation_space = env.observation_space
action_space = env.action_space
config = {
        "env": ACT3MultiAgentEnv,
        "env_config": env_config,
        "observation_space": observation_space,
        "action_space": action_space,
    }

agent = PPO(config=config)
agent.restore(checkpoint_paths[0])
trajectories = {'Trajectory': []}
num_steps = 0
# run random agent eval
# initialize episode with trained policy for random num timesteps
# then substitute for random policy + collect data until episode end
# store random actor transistions for dataset
# track num stored transitions. terminate at 1500
while num_steps < 552925:
    seed = random.randrange(2000)
    obs, reward = env.reset(seed=seed)
    num_policy_steps = random.randrange(337)
    trajectory = []
    terminated = {'blue0_ctrl': False}
    ep_len = 0
    for _ in range(num_policy_steps):
        action_dict = {'blue0_ctrl': agent.compute_single_action(obs['blue0_ctrl'], policy_id='blue0_ctrl')}
        obs, reward, dones, terminated, info = env.step(action_dict)
        ep_len += 1
        if any(terminated.values()):
            break
    while not any(terminated.values()):
        action = sample_nested_action_space(action_space)
        obs, reward, dones, terminated, info = env.step(action)
        trajectory.append((obs, action, env.reward_info['blue0_ctrl'], reward, terminated))
        num_steps += 1
        ep_len += 1
    print(ep_len)
    if len(trajectory) > 0:
        trajectories['Trajectory'].append(trajectory)
        num_steps += len(trajectory)

df = pd.DataFrame(trajectories)
df.to_pickle(output_path + '/random_data.pkl')

