# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example of collecting offline data with OmniSafe."""

import sys
import os
cwd = os.getcwd()
sys.path.append(cwd.replace('/examples', ''))

from omnisafe.common.offline.data_collector import OfflineDataCollector
from tqdm import tqdm
import numpy as np
import pickle5 as pickle
import os
import safety_gymnasium as safety_gym
import yaml

# please change agent path and env name
# also, please make sure you have run:
# python train_policy.py --algo PPO --env ENVID
# where ENVID is the environment from which you want to collect data.
# The `PATH_TO_AGENT` is the directory path containing the `torch_save`.

def collect_random_pickle5(env_name: str, save_dir: str, num_episodes: int) -> None:
    """Collect data from the registered agents.

    Args:
        save_dir (str): The directory to save the collected data.
    """
    # check each agent's size
    # total_size = 0
    # for agent in self.agents:
    #     assert agent.size <= self._size, f'Agent {agent} size is larger than collector size.'
    #     total_size += agent.size
    # assert total_size == self._size, 'Sum of agent size is not equal to collector size.'

    # collect data
    ptx = 0
    save_str = 'random'
    progress_bar = tqdm(total=num_episodes, desc='Collecting data...')

    total_dicts = []
    total_transitions = 0
    cost_threshold = 25.0

    episode_rewards, episode_costs, episode_lengths, episode_ret_until_violation = [], [], [], []
    violations = []
    env = safety_gym.make(env_name, render_mode=None)
    
    for ep_idx in range(num_episodes):
        ep_ret, ep_cost, ep_len, ep_ret_until_violation = 0.0, 0.0, 0.0, 0.0
        
        violation = False

        obs, _ = env.reset()
        done = False

        save_dict = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'costs': [],
            'next_observations': [],
            'terminals': []
        }

        while not done:
            action = env.action_space.sample()
            next_obs, reward, cost, terminate, truncated, _ = env.step(action)
            done = terminate or truncated

            save_dict['observations'].append(obs)
            save_dict['actions'].append(action)
            save_dict['rewards'].append(reward)
            save_dict['costs'].append(cost)
            save_dict['next_observations'].append(next_obs)
            save_dict['terminals'].append(done)

            ep_ret += reward
            ep_cost += cost
            ep_len += 1
            total_transitions += 1

            if cost > cost_threshold:
                violation = True
            
            if not violation:
                ep_ret_until_violation += reward

            obs = next_obs
        
        save_dict['observations'] = np.array(save_dict['observations'])
        save_dict['actions'] = np.array(save_dict['actions'])
        save_dict['rewards'] = np.array(save_dict['rewards'])
        save_dict['costs'] = np.array(save_dict['costs'])
        save_dict['next_observations'] = np.array(save_dict['next_observations'])
        save_dict['terminals'] = np.array(save_dict['terminals'])

        episode_ret_until_violation.append(ep_ret_until_violation)
        violations.append(violation)
        
        print(f'episode{ep_idx}: return={ep_ret}, cost={ep_cost}, episode_length={ep_len}, return_until_violation={ep_ret_until_violation}, violation={violation}')

        episode_rewards.append(ep_ret)
        episode_costs.append(ep_cost)
        episode_lengths.append(ep_len)
        total_dicts.append(save_dict)

        progress_bar.update(ep_idx + 1)

    print(f'Agent {env_name} collected {total_transitions} data points.')
    print(f'Average return: {np.mean(episode_rewards)}')
    print(f'Average cost: {np.mean(episode_costs)}')
    print(f'Average episode length: {np.mean(episode_lengths)}')
    print(f'Average return until violation: {np.mean(episode_ret_until_violation)}')
    print(f'Violation rate: {np.mean(violations)}')
    print()

    # Create stats dictionary for YAML output
    stats = {
        'env_name': env_name,
        'num_episodes': num_episodes,
        'total_transitions': total_transitions,
        'average_return': float(np.mean(episode_rewards))   ,
        'std_return': float(np.std(episode_rewards)),
        'average_cost': float(np.mean(episode_costs)),
        'std_cost': float(np.std(episode_costs)),
        'average_episode_length': float(np.mean(episode_lengths)),
        'std_episode_length': float(np.std(episode_lengths)),
        'min_return': float(np.min(episode_rewards)),
        'max_return': float(np.max(episode_rewards)),
        'min_cost': float(np.min(episode_costs)),
        'max_cost': float(np.max(episode_costs)),
        'min_episode_length': float(np.min(episode_lengths)),
        'max_episode_length': float(np.max(episode_lengths)),
        'average_return_until_violation': float(np.mean(episode_ret_until_violation)),
        'std_return_until_violation': float(np.std(episode_ret_until_violation)),
        'min_return_until_violation': float(np.min(episode_ret_until_violation)),
        'max_return_until_violation': float(np.max(episode_ret_until_violation)),
        'violation_rate': float(np.mean(violations)),
        'collection_timestamp': str(np.datetime64('now')),
    }

    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save stats to YAML file
    stats_path = os.path.join(save_dir, f'{env_name}-{save_str}-stats.yaml')
    with open(stats_path, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
    
    print(f'Statistics saved to: {stats_path}')

    # save data
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f'{env_name}-{save_str}-n{num_episodes}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(total_dicts, f)
                

if __name__ == '__main__':
    # env_name_list = ['SafetyPointCircle1-v0', 'SafetyPointCircle2-v0']
    env_name_list = ['SafetyHalfCheetahVelocity-v0']
    save_dir = './data'
    num_episodes = 1000

    for env_name in env_name_list:
        collect_random_pickle5(env_name, save_dir, num_episodes)
