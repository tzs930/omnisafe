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

import pickle5 as pickle
import yaml
import numpy as np
import safety_gymnasium as safety_gym
from tqdm import tqdm
from omnisafe.common.offline.data_collector import OfflineDataCollector

# from omnisafe.common.offline.data_collector import OfflineDataCollector
# please change agent path and env name
# also, please make sure you have run:
# python train_policy.py --algo PPO --env ENVID
# where ENVID is the environment from which you want to collect data.
# The `PATH_TO_AGENT` is the directory path containing the `torch_save`.

if __name__ == '__main__':
    num_episodes = 1000

    policy_type_dict = {
        'safe-expert-v1': ('PPOLag','epoch-100.pt'),
        'unsafe-expert-v1': ('PPO','epoch-100.pt'),
        'safe-medium-v1': ('PPOLag','epoch-40.pt'),
        'unsafe-medium-v1': ('PPO','epoch-40.pt'),
    }
    env_name_list = ['SafetyPointCircle1-v0'] #, 'SafetyPointCircle2-v0']

    size = 1_000_000
    path_dict = {
        'SafetyPointCircle1-v0': {
            'PPO': '/home/siseo/icil-data/omnisafe/runs/PPO-{SafetyPointCircle1-v0}/seed-000-2025-08-07-09-34-49',
            'PPOLag': '/home/siseo/icil-data/omnisafe/runs/PPOLag-{SafetyPointCircle1-v0}/seed-000-2025-08-07-02-01-16',
        },
        'SafetyPointCircle2-v0': {
            'PPO': '/home/siseo/icil-data/omnisafe/runs/PPO-{SafetyPointCircle2-v0}/seed-000-2025-08-07-09-34-49',
            'PPOLag': '/home/siseo/icil-data/omnisafe/runs/PPOLag-{SafetyPointCircle2-v0}/seed-000-2025-08-08-08-45-15',
        },
    }

    save_dir = './data'
    for env_name in env_name_list:
        for policy_type in policy_type_dict.keys():
        # agents = agents_dict[env_name]
            policy, iteration = policy_type_dict[policy_type]
            env_path = path_dict[env_name][policy]
            # agents = (env_path, iteration, size)

            col = OfflineDataCollector(size, env_name)
            agent, model_name, num = env_path, iteration, size
            col.register_agent(agent, model_name, num)
            col.collect_agent_pickle5_with_stats(save_dir, policy_type, num_episodes)
