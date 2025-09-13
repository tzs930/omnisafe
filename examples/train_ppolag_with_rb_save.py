"""
#!/usr/bin/env python3
PPOLag with Periodic Replay Buffer Saving

This script demonstrates how to implement periodic replay buffer saving
by monkey-patching the learn method at the epoch level.
"""

import os
import sys
import time
import pickle
import yaml
import numpy as np
import torch
import wandb
from omnisafe import Agent

from omnisafe.algorithms import ALGORITHM2TYPE
from omnisafe.utils.config import get_default_kwargs_yaml
from replay_buffer_helpers import accumulate_replay_buffer, save_replay_buffer, load_cost_threshold, load_reward_threshold

def main():
    """Main function with periodic buffer saving."""
    # Configuration
    algo = 'PPOLag'
    env_id = 'SafetyHopperVelocity-v1'  # 'SafetyPointCircle1-v0'
    algo_type = ALGORITHM2TYPE[algo]

    custom_cfgs = get_default_kwargs_yaml(algo, env_id, algo_type)
    custom_cfgs['train_cfgs']['total_steps'] = 1000000 # total epochs = total_steps / steps_per_epoch
    custom_cfgs['train_cfgs']['vector_env_nums'] = 10
    custom_cfgs['algo_cfgs']['obs_normalize'] = True
    custom_cfgs['algo_cfgs']['steps_per_epoch'] = 10000 
    custom_cfgs['logger_cfgs']['save_model_freq'] = 100
    custom_cfgs['logger_cfgs']['use_wandb'] = True

    replay_buffer_save_freq = 10
    replay_buffer_max_trajsize = 5000

    # Load thresholds
    cost_threshold = load_cost_threshold(env_id)
    return_threshold = load_reward_threshold(env_id)
    
    print(f"Environment: {env_id}")
    print(f"Using cost threshold: {cost_threshold}")
    print(f"Using return threshold: {return_threshold:.4f}")
    
    # Create agent
    agent = Agent(algo, env_id, custom_cfgs=custom_cfgs)
    
    # Get the algorithm instance
    algo_instance = agent.agent
    
    def learn_with_periodic_saving(save_freq):
        """Modified learn method with periodic buffer saving."""
        import time
        
        # Calculate epochs from total_steps and steps_per_epoch
        total_steps = algo_instance._cfgs.train_cfgs.total_steps
        steps_per_epoch = algo_instance._cfgs.algo_cfgs.steps_per_epoch
        epochs = total_steps // steps_per_epoch
        
        print(f"Training for {epochs} epochs with periodic buffer saving every {save_freq} epochs")
        print(f"Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}")
        print(f"Buffer size: {algo_instance._buf.size if hasattr(algo_instance._buf, 'size') else 'Unknown'}")

        # Custom rollout function that saves both original and normalized observations
        def custom_rollout_with_original_obs(steps_per_epoch, agent, buffer, logger):
            """Custom rollout that saves both original and normalized observations."""
            algo_instance._env._reset_log()
            
            obs, info = algo_instance._env.reset()
            # Get original observation from info (reset also normalizes)
            original_obs = info.get('original_obs', obs)
            
            for step in range(steps_per_epoch):
                # Use normalized observations for policy
                act, value_r, value_c, logp = agent.step(obs)
                next_obs, reward, cost, terminated, truncated, info = algo_instance._env.step(act)
                # Get original observations from info
                original_next_obs = info.get('original_obs', next_obs)
                
                # Log values
                algo_instance._env._log_value(reward=reward, cost=cost, info=info)
                
                if algo_instance._cfgs.algo_cfgs.use_cost:
                    logger.store({'Value/cost': value_c})
                logger.store({'Value/reward': value_r})
                
                # Store normalized observations as default (for RL algorithm)
                # and original observations separately
                buffer.store(
                    obs=obs,  # Store normalized observation (default for RL)
                    next_obs=next_obs,  # Store normalized next observation (default for RL)
                    obs_original=original_obs,  # Store original observation
                    next_obs_original=original_next_obs,  # Store original next observation
                    act=act,
                    reward=reward,
                    cost=cost,
                    value_r=value_r,
                    value_c=value_c,
                    logp=logp,
                    terminals=terminated,
                    timeout=truncated,
                )
                
                # Update for next iteration
                obs = next_obs
                original_obs = original_next_obs
                
                epoch_end = step >= steps_per_epoch - 1
                if epoch_end:
                    num_dones = int(terminated.contiguous().sum())
                    num_envs = terminated.shape[0] if terminated.dim() > 0 else 1
                    if num_envs - num_dones:
                        logger.log(
                            f'\nWarning: trajectory cut off when rollout by epoch\
                                in {num_envs - num_dones} of {num_envs} environments.',
                        )
                
                for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                    if epoch_end or done or time_out:
                        last_value_r = torch.zeros(1)
                        last_value_c = torch.zeros(1)
                        if not done:
                            if epoch_end:
                                _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                            if time_out:
                                _, last_value_r, last_value_c, _ = agent.step(
                                    info['final_observation'][idx],
                                )
                            last_value_r = last_value_r.unsqueeze(0)
                            last_value_c = last_value_c.unsqueeze(0)
                        
                        # Log metrics and reset for both done and timeout
                        if done or time_out:
                            algo_instance._env._log_metrics(logger, idx)
                            algo_instance._env._reset_log(idx)
                            
                            algo_instance._env._ep_ret[idx] = 0.0
                            algo_instance._env._ep_cost[idx] = 0.0
                            algo_instance._env._ep_len[idx] = 0.0
                        
                        # Finish path for both done and timeout
                        buffer.finish_path(last_value_r, last_value_c, idx)

        # Monkey patch the learn method to add periodic saving
        def modified_learn():
            start_time = time.time()
            algo_instance._logger.log('INFO: Start training with periodic buffer saving')
            
            full_trajectories = []

            for epoch in range(epochs):
                epoch_time = time.time()
                rollout_time = time.time()
                
                # Use custom rollout that saves original observations
                custom_rollout_with_original_obs(
                    steps_per_epoch=algo_instance._steps_per_epoch,
                    agent=algo_instance._actor_critic,
                    buffer=algo_instance._buf,
                    logger=algo_instance._logger,
                )
                algo_instance._logger.store({'Time/Rollout': time.time() - rollout_time})
                
                # Update
                update_time = time.time()
                algo_instance._update()
                algo_instance._logger.store({'Time/Update': time.time() - update_time})
                
                # Always accumulate replay buffer every epoch
                try:
                    print(f"Accumulating replay buffer at epoch {epoch}...")
                    full_trajectories = accumulate_replay_buffer(
                        full_trajectories,
                        algo_instance._buf, 
                        agent, 
                        epoch,
                        max_trajectories=replay_buffer_max_trajsize,
                        env=algo_instance._env,
                    )
                    print(f"Successfully accumulated {len(full_trajectories)} trajectories")
                except Exception as e:
                    print(f"Could not accumulate replay buffer at epoch {epoch}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with existing trajectories if accumulation fails
                
                # Save to disk only at save_freq intervals
                if epoch % save_freq == 0:
                    try:
                        save_replay_buffer(
                            full_trajectories,
                            agent, 
                            epoch,
                            max_trajectories=replay_buffer_max_trajsize,
                            cost_threshold=cost_threshold,
                            return_threshold=return_threshold,
                            env_name=env_id
                        )
                        print(f"Replay buffer saved to disk at epoch {epoch}")
                    except Exception as e:
                        print(f"Could not save replay buffer to disk at epoch {epoch}: {e}")
                
                # Model saving
                if algo_instance._cfgs.model_cfgs.exploration_noise_anneal:
                    algo_instance._actor_critic.annealing(epoch)
                
                if algo_instance._cfgs.model_cfgs.actor.lr is not None:
                    algo_instance._actor_critic.actor_scheduler.step()
                
                # Logging
                algo_instance._logger.store({
                    'TotalEnvSteps': (epoch + 1) * algo_instance._cfgs.algo_cfgs.steps_per_epoch,
                    'Time/FPS': algo_instance._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': (
                        0.0
                        if algo_instance._cfgs.model_cfgs.actor.lr is None
                        else algo_instance._actor_critic.actor_scheduler.get_last_lr()[0]
                    ),
                })
                
                algo_instance._logger.dump_tabular()
                
                # Save model
                if (epoch + 1) % algo_instance._cfgs.logger_cfgs.save_model_freq == 0 or (
                    epoch + 1
                ) == algo_instance._cfgs.train_cfgs.epochs:
                    algo_instance._logger.torch_save()
            
            # Final buffer save
            try:
                save_replay_buffer(
                    full_trajectories,
                    agent, 
                    "final",
                    cost_threshold=cost_threshold,
                    return_threshold=return_threshold,
                    env_name=env_id
                )
                print("Final replay buffer saved")
            except Exception as e:
                print(f"Could not save final buffer: {e}")
            
            # Return final metrics
            ep_ret = algo_instance._logger.get_stats('Metrics/EpRet')[0]
            ep_cost = algo_instance._logger.get_stats('Metrics/EpCost')[0]
            ep_len = algo_instance._logger.get_stats('Metrics/EpLen')[0]
            algo_instance._logger.close()
            algo_instance._env.close()
            
            return ep_ret, ep_cost, ep_len
        
        # Return the modified learn method
        return modified_learn
    
    # Replace the learn method
    algo_instance.learn = learn_with_periodic_saving(replay_buffer_save_freq)
    
    # Run training
    result = algo_instance.learn()
    
    print(f"Training completed with result: {result}")


if __name__ == '__main__':
    main()
