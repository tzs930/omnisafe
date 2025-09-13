import os
import pickle
import numpy as np
import yaml
import wandb
import torch

def accumulate_replay_buffer(prev_trajectories, buffer, agent, epoch, max_trajectories: int = 1000, env=None):
    """Accumulate replay buffer data in memory without saving to disk."""
    
    # Get save path from agent's logger
    save_path = agent.agent.logger.log_dir
    os.makedirs(save_path, exist_ok=True)
    
    # Handle VectorOnPolicyBuffer - collect data from ALL environments
    if hasattr(buffer, 'buffers') and len(buffer.buffers) > 0:
        # VectorOnPolicyBuffer contains multiple OnPolicyBuffer instances
        print(f"Accumulating data from VectorOnPolicyBuffer with {len(buffer.buffers)} buffers")
        
        # Collect all trajectories from all environments
        all_trajectories = []
        global_trajectory_id = 0
        
        for env_idx, env_buffer in enumerate(buffer.buffers):
            # Create a copy of the data to avoid modifying the original buffer
            env_data = {}
            for key, value in env_buffer.data.items():
                if hasattr(value, 'clone'):
                    env_data[key] = value.clone()
                elif hasattr(value, 'copy'):
                    env_data[key] = value.copy()
                else:
                    env_data[key] = value
            
            # Extract trajectory IDs for this environment
            trajectory_ids = _extract_trajectory_ids(env_data)
            
            # Convert to trajectory-unit format for this environment
            env_trajectories = _convert_to_trajectory_format(env_data, trajectory_ids, env)
            
            # Assign global trajectory IDs (0, 1, 2, ...) and remove env_id
            for traj in env_trajectories:
                # traj['trajectory_id'] = global_trajectory_id  # Commented out for dataset compatibility
                if 'env_id' in traj:
                    del traj['env_id']  # Remove env_id
                global_trajectory_id += 1
            
            all_trajectories.extend(env_trajectories)
            # print(f"    - Environment {env_idx + 1}: {len(env_trajectories)} trajectories")
        
        new_trajectories = all_trajectories
        # print(f"New trajectories from all environments: {len(new_trajectories)}")
        
    else:
        # Direct OnPolicyBuffer - create a copy to avoid modifying original
        data = {}
        for key, value in buffer.data.items():
            if hasattr(value, 'clone'):
                data[key] = value.clone()
            elif hasattr(value, 'copy'):
                data[key] = value.copy()
            else:
                data[key] = value
        
        # Extract trajectory IDs
        trajectory_ids = _extract_trajectory_ids(data)
        
        # Convert to trajectory-unit format
        new_trajectories = _convert_to_trajectory_format(data, trajectory_ids, env)
        # print(f"New trajectories: {len(new_trajectories)}")
    
    # Filter out incomplete trajectories (those cut off by epoch end)
    complete_new_trajectories = []
    for traj in new_trajectories:
        # Check if trajectory has any episode end (terminal or timeout)
        has_episode_end = np.any(traj['terminals']) or np.any(traj['timeouts'])
        if has_episode_end:
            complete_new_trajectories.append(traj)
        else:
            # Only discard if trajectory is truly incomplete (no terminal, no timeout, but cut off)
            # This means the trajectory was cut off by epoch end without natural episode termination
            print(f"  Discarding incomplete trajectory (cut off by epoch end)")
    
    new_trajectories = complete_new_trajectories
    print(f"Complete new trajectories after filtering: {len(new_trajectories)}")
    
    if len(new_trajectories) == 0:
        print("No complete new trajectories to accumulate")
        return prev_trajectories
    
    # Load existing trajectories if file exists
    # existing_trajectories = prev_trajectories
    # pkl_path = os.path.join(save_path, 'replay_buffer.pkl')
    # if os.path.exists(pkl_path):
    #     try:
    #         with open(pkl_path, 'rb') as f:
    #             existing_trajectories = pickle.load(f)
    #         print(f"Loaded {len(existing_trajectories)} existing trajectories")
    #     except Exception as e:
    #         print(f"Warning: Could not load existing trajectories: {e}")
    #         existing_trajectories = []
    
    # Combine existing and new trajectories
    all_trajectories = prev_trajectories + new_trajectories
    print(f"Total trajectories (prev + new): {len(all_trajectories)}")
    
    # Keep only the latest max_trajectories
    if len(all_trajectories) > max_trajectories:
        all_trajectories = all_trajectories[-max_trajectories:]
        print(f"Kept latest {max_trajectories} trajectories")
    
    # Reassign trajectory IDs to be continuous (0, 1, 2, ...)
    # for i, traj in enumerate(all_trajectories):  # Commented out for dataset compatibility
    #     traj['trajectory_id'] = i
    
    # Save trajectories (accumulated) - this is the key difference from save_replay_buffer
    pkl_path = os.path.join(save_path, 'replay_buffer.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_trajectories, f)
    
    print(f"Replay buffer accumulated in memory at epoch {epoch}")
    print(f"Total trajectories: {len(all_trajectories)}")
    
    return all_trajectories


def save_replay_buffer(full_trajectories, agent, epoch, max_trajectories: int = 1000, 
                      cost_threshold: float = 25.0, return_threshold: float = 0.0, env_name: str = None):
    """Save replay buffer statistics and create backup files.
    
    Args:
        buffer: The replay buffer to save
        agent: The agent instance
        epoch: Current epoch number
        max_trajectories: Maximum number of trajectories to keep
        cost_threshold: Cost threshold for safe trajectory classification
        return_threshold: Return threshold for high-return trajectory classification
    """
    import pickle
    
    # Get save path from agent's logger
    save_path = agent.agent.logger.log_dir
    os.makedirs(save_path, exist_ok=True)
    
    # Load accumulated trajectories
    # pkl_path = os.path.join(save_path, 'replay_buffer.pkl')
    # if not os.path.exists(pkl_path):
    #     print("No accumulated replay buffer found to save")
    #     return
    
    # try:
    #     with open(pkl_path, 'rb') as f:
    #         all_trajectories = pickle.load(f)
    #     print(f"Loaded {len(all_trajectories)} accumulated trajectories")
    # except Exception as e:
    #     print(f"Error loading accumulated trajectories: {e}")
    #     return
    
    # Calculate statistics
    if len(full_trajectories) > max_trajectories:
        full_trajectories = full_trajectories[-max_trajectories:]
    else:
        full_trajectories = full_trajectories

    stats = _calculate_trajectory_statistics(full_trajectories, cost_threshold, return_threshold, env_name)
    
    # Create backup files with epoch suffix
    backup_pkl_path = os.path.join(save_path, f'replay_buffer_epoch_{epoch}.pkl')
    backup_yaml_path = os.path.join(save_path, f'replay_buffer_epoch_{epoch}_stats.yaml')
    
    # Save backup files
    with open(backup_pkl_path, 'wb') as f:
        pickle.dump(full_trajectories, f)
    
    with open(backup_yaml_path, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    # Log to wandb
    print(f"\n=== Saving replay buffer at epoch {epoch} ===")
    log_buffer_statistics_to_wandb(stats, epoch)
    
    print(f"Replay buffer backup saved to {backup_pkl_path}")
    print(f"Statistics saved to {backup_yaml_path}")
    print(f"Total trajectories: {len(full_trajectories)}")
    print(f"Safe trajectories: {stats['safe_trajectories']} ({stats['safe_ratio']:.2%})")
    print(f"High return trajectories: {stats['high_return_trajectories']} ({stats['high_return_ratio']:.2%})")
    print(f"Safe & high return: {stats['safe_high_return_trajectories']} ({stats['safe_high_return_ratio']:.2%})")
    print(f"Periodic replay buffer saved at epoch {epoch}")


def _extract_trajectory_ids(data):
    """Extract trajectory IDs from buffer data."""
    # Get terminals (done) and timeout flags
    if hasattr(data, 'get'):
        terminals_flags = data.get('terminals', None)
        timeout_flags = data.get('timeout', None)  # Re-enabled for trajectory separation
    else:
        terminals_flags = data.get('terminals', None) if 'terminals' in data else None
        timeout_flags = data.get('timeout', None) if 'timeout' in data else None  # Re-enabled for trajectory separation
    
    # if terminals_flags is None and timeout_flags is None:
    #     print("Warning: No 'terminals' or 'timeout' flags found, treating as single trajectory")
    #     # Fallback: treat all data as a single trajectory
    #     data_length = len(data['obs'])
    #     return [0] * data_length  # All steps belong to trajectory 0
    
    # Convert to numpy if tensor
    if terminals_flags is not None:
        if hasattr(terminals_flags, 'cpu'):
            terminals_flags = terminals_flags.cpu().numpy()
        elif hasattr(terminals_flags, 'numpy'):
            terminals_flags = terminals_flags.numpy()
        # Ensure boolean type
        terminals_flags = terminals_flags.astype(bool)
    else:
        terminals_flags = np.zeros(len(data['obs']), dtype=bool)
    
    if timeout_flags is not None:
        if hasattr(timeout_flags, 'cpu'):
            timeout_flags = timeout_flags.cpu().numpy()
        elif hasattr(timeout_flags, 'numpy'):
            timeout_flags = timeout_flags.numpy()
        # Ensure boolean type
        timeout_flags = timeout_flags.astype(bool)
    else:
        timeout_flags = np.zeros(len(data['obs']), dtype=bool)
    
    # Combine terminals and timeout flags (episode ends if either is True)
    episode_ends = terminals_flags | timeout_flags
    
    # Check if any episodes actually completed
    if np.sum(episode_ends) == 0:
        print("Warning: No episodes completed (all terminals/timeout flags are False), treating as single trajectory")
        # All data belongs to one trajectory
        data_length = len(episode_ends)
        return [0] * data_length
    
    # Create trajectory IDs based on episode end flags
    trajectory_ids = []
    current_id = 0
    
    for i, episode_end in enumerate(episode_ends):
        trajectory_ids.append(current_id)
        if episode_end:
            current_id += 1
    
    # Note: We don't modify the original data here to avoid buffer corruption
    # Instead, we just identify which trajectories are complete and let the caller handle filtering
    if len(episode_ends) > 0 and not episode_ends[-1]:
        # Check if the last trajectory has any terminal or timeout signals
        last_trajectory_id = trajectory_ids[-1] if trajectory_ids else 0
        last_trajectory_has_termination = False
        
        # Check if any step in the last trajectory has terminal or timeout
        for i, (traj_id, term, timeout) in enumerate(zip(trajectory_ids, terminals_flags, timeout_flags)):
            if traj_id == last_trajectory_id and (term or timeout):
                last_trajectory_has_termination = True
                break
        
        # Only mark as incomplete if the last trajectory has no termination signals
        if not last_trajectory_has_termination:
            # Find the last complete trajectory
            last_complete_idx = np.where(episode_ends)[0]
            if len(last_complete_idx) > 0:
                last_complete_end = last_complete_idx[-1] + 1
                # Truncate trajectory_ids to only include complete trajectories
                trajectory_ids = trajectory_ids[:last_complete_end]
                print(f"Identified incomplete trajectory, will keep {last_complete_end} steps from {len(episode_ends)} total steps")
                current_id = len(last_complete_idx)  # Update current_id to reflect only complete trajectories
        else:
            print(f"Last trajectory {last_trajectory_id} has termination signals, keeping it")
    
    # print(f"Extracted {current_id} trajectories from episode end flags")
    # print(f"  - Terminals flags: {np.sum(terminals_flags)} episodes")
    # print(f"  - Timeout flags: {np.sum(timeout_flags)} episodes")
    # print(f"  - Total episode ends: {np.sum(episode_ends)} episodes")
    return trajectory_ids


def _convert_to_trajectory_format(data, trajectory_ids, env=None):
    """Convert buffer data to trajectory-unit format."""
    trajectories = []
    
    # Group data by trajectory ID
    trajectory_data = {}
    
    # Only process up to the length of trajectory_ids to avoid index errors
    max_idx = len(trajectory_ids)
    
    for i, traj_id in enumerate(trajectory_ids):
        if traj_id not in trajectory_data:
            trajectory_data[traj_id] = {
                'observations': [],  # Will store original observations
                'next_observations': [],  # Will store original next observations
                'actions': [],
                'rewards': [],
                'costs': [],
                'terminals': [],
                # 'timeouts': [],        # Added back for trajectory separation
                # 'values_r': [],        # Commented out for dataset compatibility
                # 'values_c': [],        # Commented out for dataset compatibility
                # 'log_probs': [],       # Commented out for dataset compatibility
                # 'trajectory_id': traj_id  # Commented out for dataset compatibility
            }
        
        # Extract data for this step
        for key in ['obs_original', 'next_obs_original', 'act', 'reward', 'cost', 'terminals']:  # Added original observations
            if key in data:
                # Ensure we don't go beyond the available data
                if i < len(data[key]):
                    value = data[key][i]
                    if hasattr(value, 'cpu'):
                        value = value.cpu().numpy()
                    elif hasattr(value, 'numpy'):
                        value = value.numpy()
                    
                    traj_key = {
                        'obs_original': 'observations',  # Store normalized obs in normalized field
                        'next_obs_original': 'next_observations',  # Store normalized next_obs in normalized field
                        'act': 'actions', 
                        'reward': 'rewards',
                        'cost': 'costs',
                        'terminals': 'terminals',
                        # 'value_r': 'values_r',      # Commented out for dataset compatibility
                        # 'value_c': 'values_c',      # Commented out for dataset compatibility
                        # 'logp': 'log_probs',        # Commented out for dataset compatibility
                        # 'timeout': 'timeouts'       # Commented out for dataset compatibility
                    }[key]
                    
                    trajectory_data[traj_id][traj_key].append(value)
    
    # Convert to list of trajectories
    for traj_id, traj_data in trajectory_data.items():
        if len(traj_data['observations']) > 0:  # Only include non-empty trajectories
            # Convert lists to numpy arrays
            for key in traj_data:
                if key != 'trajectory_id' and isinstance(traj_data[key], list):
                    traj_data[key] = np.array(traj_data[key])
            
            trajectories.append(traj_data)
    
    return trajectories


def _calculate_trajectory_statistics(trajectories, cost_threshold, return_threshold, env_name=None):
    """Calculate statistics for trajectories."""
    if not trajectories:
        return {
            'total_trajectories': 0,
            'safe_trajectories': 0,
            'high_return_trajectories': 0,
            'safe_high_return_trajectories': 0,
            'safe_ratio': 0.0,
            'high_return_ratio': 0.0,
            'safe_high_return_ratio': 0.0,
            'mean_return': 0.0,
            'mean_cost': 0.0,
            'mean_length': 0.0,
            'feasible_return': 0.0,
            'violation_rate': 0.0
        }
    
    returns = []
    costs = []
    lengths = []
    feasible_returns = []
    safe_trajectories = 0
    high_return_trajectories = 0
    safe_high_return_trajectories = 0
    total_transitions = 0
    
    for traj in trajectories:
        # Calculate return and cost
        traj_return = np.sum(traj['rewards'])
        traj_cost = np.sum(traj['costs'])
        traj_length = len(traj['observations'])
        
        returns.append(traj_return)
        costs.append(traj_cost)
        lengths.append(traj_length)
        total_transitions += traj_length
        
        # Calculate feasible return (reward accumulated before cost violation)
        feasible_return = 0.0
        cumulative_cost = 0.0
        
        for i, (reward, cost) in enumerate(zip(traj['rewards'], traj['costs'])):
            cumulative_cost += cost
            if cumulative_cost <= cost_threshold:
                feasible_return += reward
            else:
                break  # Stop accumulating reward after cost violation
        
        feasible_returns.append(feasible_return)
        
        # Check thresholds
        is_safe = traj_cost <= cost_threshold
        is_high_return = traj_return >= return_threshold
        
        if is_safe:
            safe_trajectories += 1
        
        if is_high_return:
            high_return_trajectories += 1
        if is_safe and is_high_return:
            safe_high_return_trajectories += 1
    
    total_trajectories = len(trajectories)
    violation_rate = (total_trajectories - safe_trajectories) / total_trajectories if total_trajectories > 0 else 0.0
    
    return {
        # Basic info (similar to safe-expert-v0-stats.yaml)
        'env_name': env_name if env_name is not None else 'Unknown',
        'num_episodes': int(total_trajectories),
        'total_transitions': int(total_transitions),
        
        # Return statistics
        'average_return': float(np.mean(returns) if returns else 0.0),
        'std_return': float(np.std(returns) if returns else 0.0),
        'min_return': float(np.min(returns) if returns else 0.0),
        'max_return': float(np.max(returns) if returns else 0.0),
        
        # Cost statistics
        'average_cost': float(np.mean(costs) if costs else 0.0),
        'std_cost': float(np.std(costs) if costs else 0.0),
        'min_cost': float(np.min(costs) if costs else 0.0),
        'max_cost': float(np.max(costs) if costs else 0.0),
        
        # Episode length statistics
        'average_episode_length': float(np.mean(lengths) if lengths else 0.0),
        'std_episode_length': float(np.std(lengths) if lengths else 0.0),
        'min_episode_length': float(np.min(lengths) if lengths else 0.0),
        'max_episode_length': float(np.max(lengths) if lengths else 0.0),
        
        # Feasible return statistics (return until violation)
        'average_return_until_violation': float(np.mean(feasible_returns) if feasible_returns else 0.0),
        'std_return_until_violation': float(np.std(feasible_returns) if feasible_returns else 0.0),
        'min_return_until_violation': float(np.min(feasible_returns) if feasible_returns else 0.0),
        'max_return_until_violation': float(np.max(feasible_returns) if feasible_returns else 0.0),
        
        # Safety statistics
        'violation_rate': float(violation_rate),
        'safe_trajectories': int(safe_trajectories),
        'safe_ratio': float(safe_trajectories / total_trajectories if total_trajectories > 0 else 0.0),
        
        # Additional statistics
        'high_return_trajectories': int(high_return_trajectories),
        'high_return_ratio': float(high_return_trajectories / total_trajectories if total_trajectories > 0 else 0.0),
        'safe_high_return_trajectories': int(safe_high_return_trajectories),
        'safe_high_return_ratio': float(safe_high_return_trajectories / total_trajectories if total_trajectories > 0 else 0.0),
        
        # Thresholds
        'cost_threshold': float(cost_threshold),
        'return_threshold': float(return_threshold),
        
        # Legacy fields for compatibility
        'total_trajectories': int(total_trajectories),
        'mean_return': float(np.mean(returns) if returns else 0.0),
        'mean_cost': float(np.mean(costs) if costs else 0.0),
        'mean_length': float(np.mean(lengths) if lengths else 0.0),
        'feasible_return': float(np.mean(feasible_returns) if feasible_returns else 0.0)
    }


def log_buffer_statistics_to_wandb(stats, epoch):
    """Log buffer statistics to wandb."""
    try:
        # Check if wandb is initialized
        if wandb.run is None:
            print("  ⚠️  wandb not initialized, skipping logging")
            return
            
        # print(f"Logging to wandb at epoch {epoch}:")
        # print(f"  Total trajectories: {stats['total_trajectories']}")
        # print(f"  Safe trajectories: {stats['safe_trajectories']}")
        # print(f"  High return trajectories: {stats['high_return_trajectories']}")
        # print(f"  Mean return: {stats['mean_return']:.4f}")
        # print(f"  Mean cost: {stats['mean_cost']:.4f}")
        
        wandb.log({
            # Basic info
            'Buffer/NumEpisodes': stats['num_episodes'],
            'Buffer/TotalTransitions': stats['total_transitions'],
            
            # Return statistics
            'Buffer/AverageReturn': stats['average_return'],
            'Buffer/StdReturn': stats['std_return'],
            'Buffer/MinReturn': stats['min_return'],
            'Buffer/MaxReturn': stats['max_return'],
            
            # Cost statistics
            'Buffer/AverageCost': stats['average_cost'],
            'Buffer/StdCost': stats['std_cost'],
            'Buffer/MinCost': stats['min_cost'],
            'Buffer/MaxCost': stats['max_cost'],
            
            # Episode length statistics
            'Buffer/AverageEpisodeLength': stats['average_episode_length'],
            'Buffer/StdEpisodeLength': stats['std_episode_length'],
            'Buffer/MinEpisodeLength': stats['min_episode_length'],
            'Buffer/MaxEpisodeLength': stats['max_episode_length'],
            
            # Feasible return statistics
            'Buffer/AverageReturnUntilViolation': stats['average_return_until_violation'],
            'Buffer/StdReturnUntilViolation': stats['std_return_until_violation'],
            'Buffer/MinReturnUntilViolation': stats['min_return_until_violation'],
            'Buffer/MaxReturnUntilViolation': stats['max_return_until_violation'],
            
            # Safety statistics
            'Buffer/ViolationRate': stats['violation_rate'],
            'Buffer/SafeTrajectories': stats['safe_trajectories'],
            'Buffer/SafeRatio': stats['safe_ratio'],
            
            # Additional statistics
            'Buffer/HighReturnTrajectories': stats['high_return_trajectories'],
            'Buffer/HighReturnRatio': stats['high_return_ratio'],
            'Buffer/SafeHighReturnTrajectories': stats['safe_high_return_trajectories'],
            'Buffer/SafeHighReturnRatio': stats['safe_high_return_ratio'],
            
            # Thresholds
            'Buffer/CostThreshold': stats['cost_threshold'],
            'Buffer/ReturnThreshold': stats['return_threshold'],
            
            'Epoch': epoch
        })
        print("  ✅ Successfully logged to wandb")
    except Exception as e:
        print(f"  ❌ Error logging to wandb: {e}")


def load_reward_threshold(env_name: str, task_info_path: str = "examples/task_info.yaml") -> float:
    """Load reward threshold for a specific environment from task_info.yaml."""
    try:
        with open(task_info_path, 'r') as f:
            task_info = yaml.safe_load(f)
        
        if 'environments' in task_info and env_name in task_info['environments']:
            return task_info['environments'][env_name]['reward_threshold']
        else:
            print(f"Warning: Environment {env_name} not found in {task_info_path}")
            return 0.0
    except Exception as e:
        print(f"Warning: Could not load reward threshold from {task_info_path}: {e}")
        return 0.0


def load_cost_threshold(env_name: str, task_info_path: str = "examples/task_info.yaml") -> float:
    """Load cost threshold for a specific environment from task_info.yaml."""
    try:
        with open(task_info_path, 'r') as f:
            task_info = yaml.safe_load(f)
        
        if 'environments' in task_info and env_name in task_info['environments']:
            return task_info['environments'][env_name]['cost_threshold']
        else:
            print(f"Warning: Environment {env_name} not found in {task_info_path}")
            return 25.0
    except Exception as e:
        print(f"Warning: Could not load cost threshold from {task_info_path}: {e}")
        return 25.0
