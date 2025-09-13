#!/usr/bin/env python3
"""
Calculate reward thresholds for each environment based on dataset statistics.

For each environment:
- min_return: from random-v0 stats
- max_return: from safe-expert-v0 stats  
- normalized_return: (return - min_return) / (max_return - min_return)
- reward_threshold: 0.9 * (max_return - min_return) + min_return
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

def load_stats(env_path: Path, dataset_type: str) -> Dict[str, Any]:
    """Load statistics from a specific dataset type."""
    stats_file = env_path / "stats" / f"{dataset_type}-stats.yaml"
    if not stats_file.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_file}")
    
    with open(stats_file, 'r') as f:
        return yaml.safe_load(f)

def calculate_reward_thresholds(datasets_dir: str = "datasets") -> Dict[str, Dict[str, float]]:
    """Calculate reward thresholds for all environments."""
    datasets_path = Path(datasets_dir)
    if not datasets_path.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")
    
    thresholds = {}
    
    # Get all environment directories
    env_dirs = [d for d in datasets_path.iterdir() if d.is_dir() and d.name.startswith("Safety")]
    
    for env_dir in sorted(env_dirs):
        env_name = env_dir.name
        print(f"Processing {env_name}...")
        
        try:
            # Load random-v0 stats for min_return
            random_stats = load_stats(env_dir, "random-v0")
            min_return = random_stats["min_return"]
            
            # Load safe-expert-v0 stats for max_return
            safe_expert_stats = load_stats(env_dir, "safe-expert-v0")
            max_return = safe_expert_stats["max_return"]
            
            # Calculate normalized return range
            return_range = max_return - min_return
            
            # Calculate reward threshold (0.9 normalized return)
            reward_threshold = 0.9 * return_range + min_return
            
            # Store results
            thresholds[env_name] = {
                "min_return": min_return,
                "max_return": max_return,
                "return_range": return_range,
                "reward_threshold": reward_threshold,
                "normalized_threshold": 0.9
            }
            
            print(f"  min_return: {min_return:.4f}")
            print(f"  max_return: {max_return:.4f}")
            print(f"  return_range: {return_range:.4f}")
            print(f"  reward_threshold: {reward_threshold:.4f}")
            print()
            
        except Exception as e:
            print(f"  Error processing {env_name}: {e}")
            continue
    
    return thresholds

def save_task_info(thresholds: Dict[str, Dict[str, float]], output_file: str = "examples/task_info.yaml"):
    """Save calculated thresholds to task_info.yaml."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the task info structure
    task_info = {
        "environments": {}
    }
    
    for env_name, data in thresholds.items():
        task_info["environments"][env_name] = {
            "reward_threshold": data["reward_threshold"],
            "min_return": data["min_return"],
            "max_return": data["max_return"],
            "return_range": data["return_range"],
            "normalized_threshold": data["normalized_threshold"],
            "description": f"Reward threshold calculated as 0.9 normalized return for {env_name}"
        }
    
    # Add metadata
    task_info["metadata"] = {
        "calculation_method": "0.9 normalized return based on random-v0 (min) and safe-expert-v0 (max)",
        "normalized_formula": "(return - min_return) / (max_return - min_return)",
        "threshold_formula": "0.9 * (max_return - min_return) + min_return",
        "total_environments": len(thresholds)
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        yaml.dump(task_info, f, default_flow_style=False, sort_keys=False)
    
    print(f"Task info saved to: {output_path}")
    return output_path

def main():
    """Main function to calculate and save reward thresholds."""
    print("Calculating reward thresholds for all environments...")
    print("=" * 60)
    
    try:
        # Calculate thresholds
        thresholds = calculate_reward_thresholds()
        
        if not thresholds:
            print("No thresholds calculated!")
            return
        
        # Save to file
        output_file = save_task_info(thresholds)
        
        print("=" * 60)
        print(f"Successfully calculated thresholds for {len(thresholds)} environments")
        print(f"Results saved to: {output_file}")
        
        # Print summary
        print("\nSummary:")
        for env_name, data in thresholds.items():
            print(f"  {env_name}: {data['reward_threshold']:.4f}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
