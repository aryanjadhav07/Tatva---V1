"""
train_ppo.py
------------
PPO training entry point for QuantumRL — Scaled to 2 Qubits.

Run with:
    python train_ppo.py
"""

import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from ppo_agent import PPOAgent, RolloutBuffer
from quantum_env import QuantumCircuitEnv
from utils import plot_training_curves, save_logs


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ppo(config: Config) -> None:
    """Main training loop for PPO agent in QuantumRL."""
    set_seeds(config.SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[PPO] Training Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[PPO] GPU Detected   : {gpu_name}")

    env = QuantumCircuitEnv(config)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print(f"[PPO] Training configuration: {config.NUM_QUBITS} Qubit(s)")
    print(f"[PPO] obs_size={obs_size}  action_size={action_size}")

    # Checkpoint save guard
    if os.path.exists(config.PPO_MODEL_PATH):
        print(f"[PPO][SAVE-GUARD] Warning: Existing model found at '{config.PPO_MODEL_PATH}'. "
              f"New checkpoints will overwrite this file.")

    agent = PPOAgent(obs_size, action_size, config, device)
    buffer = RolloutBuffer(config.PPO_ROLLOUT_STEPS, obs_size, device)

    episode_rewards = []
    episode_fidelities = []
    episode_steps_list = []
    action_counts = Counter()

    best_mean_fidelity = 0.0
    best_model_path = config.PPO_MODEL_PATH.replace('.pth', '_best.pth')

    print(f"[PPO] Starting training for {config.PPO_EPISODES} episodes ...\n")

    obs, _ = env.reset()
    obs_t = torch.FloatTensor(obs).to(device)

    episode_reward = 0.0
    episode_steps = 0
    current_episode = 0
    rollout_count = 0

    while current_episode < config.PPO_EPISODES:

        # Rollout collection phase
        buffer.reset()
        for step in range(config.PPO_ROLLOUT_STEPS):
            with torch.no_grad():
                action, log_prob, entropy, value = agent.select_action(obs_t)

            action_counts[action] += 1
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.add(obs_t.cpu(), action, log_prob.cpu(), reward, done, value.cpu())

            obs_t = torch.FloatTensor(next_obs).to(device)
            episode_reward += reward
            episode_steps += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_fidelities.append(info['fidelity'])
                episode_steps_list.append(episode_steps)

                current_episode += 1
                episode_reward = 0.0
                episode_steps = 0

                obs, _ = env.reset()
                obs_t = torch.FloatTensor(obs).to(device)

                if current_episode % 100 == 0:
                    mean_fid = float(np.mean(episode_fidelities[-100:]))
                    current_lr = agent.optimizer.param_groups[0]['lr']

                    print(
                        f"Episode {current_episode:5d} | "
                        f"Reward: {episode_rewards[-1]:7.3f} | "
                        f"Fidelity: {info['fidelity']:.4f} | "
                        f"Steps: {info['steps']:2d} | "
                        f"LR: {current_lr:.2e} | "
                        f"Mean Fid (100): {mean_fid:.4f}",
                        flush=True
                    )

                    if mean_fid > best_mean_fidelity:
                        best_mean_fidelity = mean_fid
                        agent.save(best_model_path)
                        agent.save(config.PPO_MODEL_PATH)
                        print(f"  *** New best model saved: {mean_fid:.4f} ***", flush=True)

                if current_episode >= config.PPO_EPISODES:
                    break

        # PPO update phase
        with torch.no_grad():
            _, _, _, last_value = agent.select_action(obs_t)

        buffer.compute_returns_and_advantages(
            last_value.cpu(), config.PPO_GAMMA, config.PPO_GAE_LAMBDA
        )
        losses = agent.update(buffer)
        rollout_count += 1

    if not os.path.exists(config.PPO_MODEL_PATH):
        agent.save(config.PPO_MODEL_PATH)

    final_mean_fidelity = float(np.mean(episode_fidelities[-100:]))

    os.makedirs(config.LOG_DIR, exist_ok=True)
    save_logs(
        {
            'rewards': episode_rewards,
            'fidelities': episode_fidelities,
            'steps': episode_steps_list,
        },
        config.PPO_LOG_PATH,
    )

    os.makedirs(config.PLOT_DIR, exist_ok=True)
    plot_training_curves(
        episode_rewards,
        episode_fidelities,
        episode_steps_list,
        config.PPO_PLOT_PATH,
    )

    print(f"\n[PPO] Training complete.")
    print(f"[PPO] Final 100-episode mean fidelity: {final_mean_fidelity:.4f}")
    print(f"[PPO] Best 100-episode mean fidelity:  {best_mean_fidelity:.4f}")

    if config.LOG_ACTION_HISTOGRAM:
        used_count = sum(1 for i in range(action_size) if action_counts.get(i, 0) > 0)
        least_used = sorted(
            ((i, action_counts.get(i, 0)) for i in range(action_size)),
            key=lambda x: (x[1], x[0]),
        )[:5]
        print(
            f"[PPO] Action usage: {used_count}/{action_size} actions "
            "used at least once"
        )
        print("[PPO] 5 least-used action indices:")
        for idx, count in least_used:
            print(f"  action {idx}: {count}")


if __name__ == '__main__':
    cfg = Config()
    train_ppo(cfg)
