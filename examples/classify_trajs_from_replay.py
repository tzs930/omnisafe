#!/usr/bin/env python3
"""
Classify trajectories from replay buffer files into safe and performant sets.

This script loads replay buffer pickle files and classifies trajectories based on:
- Safe trajectories: cost <= safety_threshold (cost_threshold)
- Performant trajectories: return >= return_threshold

For trajectories that are both safe and performant, they are randomly split 50/50 
between the two sets. The script then shuffles each set and places safe & performant 
demo trajectories at the beginning of each set.
"""

import os
import pickle
import numpy as np
import yaml
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse


def load_replay_buffer(file_path: str) -> List[Dict[str, Any]]:
    """Load trajectories from a replay buffer pickle file."""
    print(f"Loading replay buffer from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Replay buffer file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"Loaded {len(trajectories)} trajectories")
    return trajectories


def load_thresholds(env_name: str, task_info_path: str = "examples/task_info.yaml") -> Tuple[float, float]:
    """Load safety and return thresholds for a specific environment."""
    try:
        with open(task_info_path, 'r') as f:
            task_info = yaml.safe_load(f)
        
        if 'environments' in task_info and env_name in task_info['environments']:
            env_config = task_info['environments'][env_name]
            cost_threshold = env_config.get('cost_threshold', 25.0)
            return_threshold = env_config.get('reward_threshold', 0.0)
            return cost_threshold, return_threshold
        else:
            print(f"Warning: Environment {env_name} not found in {task_info_path}")
            return 25.0, 0.0
    except Exception as e:
        print(f"Warning: Could not load thresholds from {task_info_path}: {e}")
        return 25.0, 0.0


def calculate_trajectory_metrics(trajectory: Dict[str, Any]) -> Tuple[float, float]:
    """Calculate return and cost for a trajectory."""
    return_val = np.sum(trajectory['rewards'])
    cost_val = np.sum(trajectory['costs'])
    return return_val, cost_val


def classify_trajectories(trajectories: List[Dict[str, Any]], 
                        cost_threshold: float, 
                        return_threshold: float) -> Dict[str, List[int]]:
    """
    Classify trajectories into safe and performant sets.
    
    Returns:
        Dictionary with trajectory indices for each category:
        - 'safe_only': trajectories that are safe but not performant
        - 'performant_only': trajectories that are performant but not safe  
        - 'both': trajectories that are both safe and performant
        - 'neither': trajectories that are neither safe nor performant
    """
    print(f"Classifying trajectories with cost_threshold={cost_threshold}, return_threshold={return_threshold}")
    
    safe_only = []
    performant_only = []
    both = []
    neither = []
    
    for idx, trajectory in enumerate(trajectories):
        return_val, cost_val = calculate_trajectory_metrics(trajectory)
        
        is_safe = cost_val <= cost_threshold
        is_performant = return_val >= return_threshold
        
        if is_safe and is_performant:
            both.append(idx)
        elif is_safe and not is_performant:
            safe_only.append(idx)
        elif not is_safe and is_performant:
            performant_only.append(idx)
        else:
            neither.append(idx)
    
    print(f"Classification results:")
    print(f"  Safe only: {len(safe_only)}")
    print(f"  Performant only: {len(performant_only)}")
    print(f"  Both safe & performant: {len(both)}")
    print(f"  Neither: {len(neither)}")
    
    return {
        'safe_only': safe_only,
        'performant_only': performant_only,
        'both': both,
        'neither': neither
    }


def split_both_trajectories(both_indices: List[int], random_seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Randomly split trajectories that are both safe and performant 50/50.
    
    Returns:
        Tuple of (safe_half, performant_half)
    """
    if not both_indices:
        return [], []
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Shuffle the indices
    shuffled_indices = both_indices.copy()
    random.shuffle(shuffled_indices)
    
    # Split in half
    mid_point = len(shuffled_indices) // 2
    safe_half = shuffled_indices[:mid_point]
    performant_half = shuffled_indices[mid_point:]
    
    print(f"Split {len(both_indices)} both-trajectories:")
    print(f"  Safe set gets: {len(safe_half)}")
    print(f"  Performant set gets: {len(performant_half)}")
    
    return safe_half, performant_half


def create_final_sets(classified: Dict[str, List[int]], 
                     both_split: Tuple[List[int], List[int]],
                     random_seed: int = 42) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """
    Create final safe and performant trajectory sets.
    
    Returns:
        Tuple of (safe_indices, performant_indices, tracking_info)
    """
    safe_half, performant_half = both_split
    
    # Combine trajectories for each set
    safe_indices = classified['safe_only'] + safe_half
    performant_indices = classified['performant_only'] + performant_half
    
    # Create tracking information
    tracking_info = {
        'safe_only_count': len(classified['safe_only']),
        'performant_only_count': len(classified['performant_only']),
        'both_safe_half_count': len(safe_half),
        'both_performant_half_count': len(performant_half),
        'safe_only_indices': classified['safe_only'],
        'performant_only_indices': classified['performant_only'],
        'both_safe_half_indices': safe_half,
        'both_performant_half_indices': performant_half,
        'total_safe': len(safe_indices),
        'total_performant': len(performant_indices)
    }
    
    print(f"Final set sizes:")
    print(f"  Safe set: {len(safe_indices)} trajectories")
    print(f"  Performant set: {len(performant_indices)} trajectories")
    
    return safe_indices, performant_indices, tracking_info


def shuffle_with_demo_first(trajectory_indices: List[int], 
                           trajectories: List[Dict[str, Any]],
                           cost_threshold: float,
                           return_threshold: float,
                           is_safe_set: bool,
                           random_seed: int = 42) -> List[int]:
    """
    Shuffle trajectory indices with ONLY ONE safe & performant demo first.
    
    Args:
        trajectory_indices: List of trajectory indices to shuffle
        trajectories: All trajectories
        cost_threshold: Cost threshold for safety
        return_threshold: Return threshold for performance
        is_safe_set: True if this is the safe set, False if performant set
        random_seed: Random seed for reproducibility
    
    Returns:
        Shuffled list of indices with ONE demo first, rest randomly shuffled
    """
    if not trajectory_indices:
        return []
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Find demo trajectories (safe & performant) in this set
    demo_indices = []
    regular_indices = []
    
    for idx in trajectory_indices:
        trajectory = trajectories[idx]
        return_val, cost_val = calculate_trajectory_metrics(trajectory)
        
        is_safe = cost_val <= cost_threshold
        is_performant = return_val >= return_threshold
        
        if is_safe and is_performant:
            demo_indices.append(idx)
        else:
            regular_indices.append(idx)
    
    # Shuffle ALL remaining trajectories (including other demos)
    all_other_indices = demo_indices[1:] + regular_indices  # Skip first demo
    random.shuffle(all_other_indices)
    
    # Combine: ONLY FIRST demo, then all other trajectories randomly shuffled
    if demo_indices:
        final_indices = [demo_indices[0]] + all_other_indices
        print(f"  {'Safe' if is_safe_set else 'Performant'} set shuffling:")
        print(f"    First demo trajectory (safe & performant): 1")
        print(f"    Other trajectories (randomly shuffled): {len(all_other_indices)}")
    else:
        final_indices = all_other_indices
        print(f"  {'Safe' if is_safe_set else 'Performant'} set shuffling:")
        print(f"    No demo trajectories found")
        print(f"    All trajectories (randomly shuffled): {len(all_other_indices)}")
    
    return final_indices


def calculate_trajectory_statistics(trajectories: List[Dict[str, Any]], 
                                   cost_threshold: float, 
                                   return_threshold: float, 
                                   env_name: str = None) -> Dict[str, Any]:
    """Calculate comprehensive statistics for trajectories."""
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
        # Basic info (similar to safe-expert-v1-stats.yaml)
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


def add_trajectory_ids(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add trajectory IDs based on original order to each trajectory."""
    for i, traj in enumerate(trajectories):
        traj['trajectory_id'] = i
    return trajectories


def save_trajectories(trajectories: List[Dict[str, Any]], 
                     indices: List[int], 
                     output_path: str,
                     tracking_info: Dict[str, Any] = None) -> None:
    """Save selected trajectories to a pickle file."""
    selected_trajectories = [trajectories[i] for i in indices]
    
    # Add tracking information to the first trajectory if provided
    if tracking_info and selected_trajectories:
        selected_trajectories[0]['_tracking_info'] = tracking_info
    
    with open(output_path, 'wb') as f:
        pickle.dump(selected_trajectories, f)
    
    print(f"Saved {len(selected_trajectories)} trajectories to: {output_path}")




def save_statistics(stats: Dict[str, Any], 
                   output_dir: str, 
                   dataset_name: str) -> None:
    """Save trajectory statistics to a YAML file."""
    # Ensure stats directory exists
    stats_dir = os.path.join(output_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    
    # Create stats file path
    stats_file = os.path.join(stats_dir, f'{dataset_name}-stats.yaml')
    
    # Save statistics
    with open(stats_file, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    print(f"Saved statistics to: {stats_file}")


def shuffle_all_trajectories(trajectories: List[Dict[str, Any]], 
                           random_seed: int = 42) -> List[Dict[str, Any]]:
    """Shuffle all trajectories randomly."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    shuffled = trajectories.copy()
    random.shuffle(shuffled)
    
    print(f"Shuffled all {len(shuffled)} trajectories")
    return shuffled


def main():
    parser = argparse.ArgumentParser(description='Classify trajectories from replay buffer')
    parser.add_argument('--input_file', type=str, 
                       default='datasets/SafetyPointCircle2-v0/replay-buffer-v0.pkl',
                       help='Path to input replay buffer file')
    parser.add_argument('--class_type', type=str, default='both',
                       choices=['both', 'safe', 'performant', 'shuffle'],
                       help='Type of classification: both, safe, performant, shuffle')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (defaults to same directory as input file)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--task_info', type=str, default='examples/task_info.yaml',
                       help='Path to task_info.yaml file')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input_file)
    
    # Extract environment name from input file path
    env_name = None
    for part in Path(args.input_file).parts:
        if part.startswith('Safety'):
            env_name = part
            break
    
    if env_name is None:
        print("Warning: Could not determine environment name from input file path")
        env_name = "SafetyPointCircle2-v1"  # Default fallback
    
    print(f"Environment: {env_name}")
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Classification type: {args.class_type}")
    print(f"Random seed: {args.random_seed}")
    print("=" * 60)
    
    # Load trajectories and thresholds
    trajectories = load_replay_buffer(args.input_file)
    
    # Add trajectory IDs based on original order
    trajectories = add_trajectory_ids(trajectories)
    
    # Process based on class_type
    if args.class_type == 'shuffle':
        # Simply shuffle all trajectories
        print("Shuffling all trajectories...")
        all_shuffled = shuffle_all_trajectories(trajectories, args.random_seed)
        shuffled_output = os.path.join(args.output_dir, 'replay-shuffled-v1.pkl')
        
        with open(shuffled_output, 'wb') as f:
            pickle.dump(all_shuffled, f)
        
        print(f"Saved {len(all_shuffled)} shuffled trajectories to: {shuffled_output}")
        
        # Calculate and save statistics for shuffled set
        cost_threshold, return_threshold = load_thresholds(env_name, args.task_info)
        shuffled_stats = calculate_trajectory_statistics(all_shuffled, cost_threshold, return_threshold, env_name)
        save_statistics(shuffled_stats, args.output_dir, 'replay-shuffled-v1')
        
        print("=" * 60)
        print("SUMMARY:")
        print(f"Total trajectories processed: {len(trajectories)}")
        print(f"Shuffled set size: {len(all_shuffled)}")
        print(f"Output files saved in: {args.output_dir}")
        print("  - replay-shuffled-v1.pkl")
        print("Statistics saved in: stats/")
        print("  - replay-shuffled-v1-stats.yaml")
        
    else:
        # Load thresholds for classification
        cost_threshold, return_threshold = load_thresholds(env_name, args.task_info)
        print(f"Using thresholds: cost_threshold={cost_threshold}, return_threshold={return_threshold}")
        print("=" * 60)
        
        # Classify trajectories
        classified = classify_trajectories(trajectories, cost_threshold, return_threshold)
        
        if args.class_type == 'safe':
            # Include all safe trajectories (safe_only + both)
            print("Creating safe trajectory set (includes all safe trajectories)...")
            safe_indices = classified['safe_only'] + classified['both']
            
            # Shuffle with demo first
            safe_shuffled = shuffle_with_demo_first(safe_indices, trajectories, cost_threshold, return_threshold, 
                                                  is_safe_set=True, random_seed=args.random_seed)
            
            # Save safe set
            safe_output = os.path.join(args.output_dir, 'replay-safe-v1.pkl')
            save_trajectories(trajectories, safe_shuffled, safe_output)
            
            # Calculate and save statistics
            safe_trajectories = [trajectories[i] for i in safe_shuffled]
            safe_stats = calculate_trajectory_statistics(safe_trajectories, cost_threshold, return_threshold, env_name)
            save_statistics(safe_stats, args.output_dir, 'replay-safe-v1')
            
            print("=" * 60)
            print("SUMMARY:")
            print(f"Total trajectories processed: {len(trajectories)}")
            print(f"Safe set size: {len(safe_shuffled)}")
            print(f"  - Safe only: {len(classified['safe_only'])}")
            print(f"  - Both safe & performant: {len(classified['both'])}")
            print(f"Output files saved in: {args.output_dir}")
            print("  - replay-safe-v1.pkl")
            print("Statistics saved in: stats/")
            print("  - replay-safe-v1-stats.yaml")
            
        elif args.class_type == 'performant':
            # Include all performant trajectories (performant_only + both)
            print("Creating performant trajectory set (includes all performant trajectories)...")
            performant_indices = classified['performant_only'] + classified['both']
            
            # Shuffle with demo first
            performant_shuffled = shuffle_with_demo_first(performant_indices, trajectories, cost_threshold, return_threshold,
                                                        is_safe_set=False, random_seed=args.random_seed)
            
            # Save performant set
            performant_output = os.path.join(args.output_dir, 'replay-performance-v1.pkl')
            save_trajectories(trajectories, performant_shuffled, performant_output)
            
            # Calculate and save statistics
            performant_trajectories = [trajectories[i] for i in performant_shuffled]
            performant_stats = calculate_trajectory_statistics(performant_trajectories, cost_threshold, return_threshold, env_name)
            save_statistics(performant_stats, args.output_dir, 'replay-performance-v1')
            
            print("=" * 60)
            print("SUMMARY:")
            print(f"Total trajectories processed: {len(trajectories)}")
            print(f"Performant set size: {len(performant_shuffled)}")
            print(f"  - Performant only: {len(classified['performant_only'])}")
            print(f"  - Both safe & performant: {len(classified['both'])}")
            print(f"Output files saved in: {args.output_dir}")
            print("  - replay-performance-v1.pkl")
            print("Statistics saved in: stats/")
            print("  - replay-performance-v1-stats.yaml")
            
        elif args.class_type == 'both':
            # Original logic: split 'both' trajectories 50/50
            print("Creating both safe and performant trajectory sets...")
            
            # Split trajectories that are both safe and performant
            both_split = split_both_trajectories(classified['both'], args.random_seed)
            
            # Create final sets
            safe_indices, performant_indices, tracking_info = create_final_sets(classified, both_split, args.random_seed)
            
            # Shuffle each set with demos first
            print("=" * 60)
            print("Shuffling sets with demos first...")
            
            safe_shuffled = shuffle_with_demo_first(safe_indices, trajectories, cost_threshold, return_threshold, 
                                                  is_safe_set=True, random_seed=args.random_seed)
            performant_shuffled = shuffle_with_demo_first(performant_indices, trajectories, cost_threshold, return_threshold,
                                                        is_safe_set=False, random_seed=args.random_seed)
            
            # Save classified sets
            print("=" * 60)
            print("Saving classified trajectories...")
            
            safe_output = os.path.join(args.output_dir, 'replay-safe-v1.pkl')
            performant_output = os.path.join(args.output_dir, 'replay-performance-v1.pkl')
            
            save_trajectories(trajectories, safe_shuffled, safe_output, tracking_info)
            save_trajectories(trajectories, performant_shuffled, performant_output, tracking_info)
            
            # Shuffle and save all trajectories
            print("=" * 60)
            print("Shuffling and saving all trajectories...")
            
            all_shuffled = shuffle_all_trajectories(trajectories, args.random_seed)
            shuffled_output = os.path.join(args.output_dir, 'replay-shuffled-v1.pkl')
            
            with open(shuffled_output, 'wb') as f:
                pickle.dump(all_shuffled, f)
            
            print(f"Saved {len(all_shuffled)} shuffled trajectories to: {shuffled_output}")
            
            # Calculate and save statistics for each output file
            print("=" * 60)
            print("Calculating and saving statistics...")
            
            # Calculate statistics for safe set
            safe_trajectories = [trajectories[i] for i in safe_shuffled]
            safe_stats = calculate_trajectory_statistics(safe_trajectories, cost_threshold, return_threshold, env_name)
            save_statistics(safe_stats, args.output_dir, 'replay-safe-v1')
            
            # Calculate statistics for performant set
            performant_trajectories = [trajectories[i] for i in performant_shuffled]
            performant_stats = calculate_trajectory_statistics(performant_trajectories, cost_threshold, return_threshold, env_name)
            save_statistics(performant_stats, args.output_dir, 'replay-performance-v1')
            
            # Calculate statistics for shuffled set
            shuffled_stats = calculate_trajectory_statistics(all_shuffled, cost_threshold, return_threshold, env_name)
            save_statistics(shuffled_stats, args.output_dir, 'replay-shuffled-v1')
            
            # Print summary
            print("=" * 60)
            print("SUMMARY:")
            print(f"Total trajectories processed: {len(trajectories)}")
            print(f"Safe set size: {len(safe_shuffled)}")
            print(f"Performant set size: {len(performant_shuffled)}")
            print(f"Shuffled set size: {len(all_shuffled)}")
            print(f"Output files saved in: {args.output_dir}")
            print("  - replay-safe-v1.pkl")
            print("  - replay-performance-v1.pkl") 
            print("  - replay-shuffled-v1.pkl")
            print("Statistics saved in: stats/")
            print("  - replay-safe-v1-stats.yaml")
            print("  - replay-performance-v1-stats.yaml")
            print("  - replay-shuffled-v1-stats.yaml")


if __name__ == "__main__":
    main()
