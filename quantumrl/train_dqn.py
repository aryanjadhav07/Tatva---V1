"""
train_dqn.py
------------
DQN training entry point for QuantumRL — Scaled to 2 Qubits.

Run with:
    python train_dqn.py
"""

import os
import random
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dqn_agent import DQNAgent
from quantum_env import QuantumCircuitEnv
from utils import plot_training_curves, save_logs


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_dqn(config: Config) -> None:
    """Full DQN training loop for QuantumRL."""
    set_seeds(config.SEED)

    env = QuantumCircuitEnv(config)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print(f"[DQN] Training configuration: {config.NUM_QUBITS} Qubit(s)")
    print(f"[DQN] obs_size={obs_size}  action_size={action_size}")
    agent = DQNAgent(obs_size, action_size, config)
    print(f"[DQN] Training device: {agent.device}")

    # Checkpoint save guard
    if os.path.exists(config.DQN_MODEL_PATH):
        print(f"[DQN][SAVE-GUARD] Warning: Existing model found at '{config.DQN_MODEL_PATH}'. "
              f"New checkpoints will overwrite this file.")

    # Replay buffer warm-up (10000 random transitions)
    warmup_steps = getattr(config, 'DQN_WARMUP_STEPS', 10000)
    print(f"[DQN] Warming up PER replay buffer with {warmup_steps} random transitions...")
    warmup_obs, _ = env.reset()
    for _wu in range(warmup_steps):
        warmup_action = env.action_space.sample()
        warmup_next_obs, warmup_reward, warmup_term, warmup_trunc, _ = env.step(warmup_action)
        agent.buffer.push(
            warmup_obs, warmup_action, warmup_reward,
            warmup_next_obs, float(warmup_term or warmup_trunc)
        )
        warmup_obs = warmup_next_obs
        if warmup_term or warmup_trunc:
            warmup_obs, _ = env.reset()
    print(f"[DQN] Warm-up complete. Buffer size: {len(agent.buffer)}")

    episode_rewards = []
    episode_fidelities = []
    episode_steps = []
    action_counts = Counter()

    best_mean_fidelity = 0.0
    best_model_path = config.DQN_MODEL_PATH.replace('.pth', '_best.pth')

    print(f"[DQN] Starting training for {config.DQN_EPISODES} episodes ...\n")

    for episode in range(config.DQN_EPISODES):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        info = {'fidelity': 0.0, 'steps': 0}

        while not done:
            action = agent.select_action(obs)
            action_counts[action] += 1
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.buffer.push(obs, action, reward, next_obs, float(done))
            agent.update()

            obs = next_obs
            episode_reward += reward

        agent.decay_epsilon()

        if episode % config.DQN_TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        episode_rewards.append(episode_reward)
        episode_fidelities.append(info['fidelity'])
        episode_steps.append(info['steps'])

        if (episode + 1) % 100 == 0:
            mean_fid = float(np.mean(episode_fidelities[-100:]))
            print(
                f"Episode {episode + 1:5d} | "
                f"Reward: {episode_reward:7.3f} | "
                f"Fidelity: {info['fidelity']:.4f} | "
                f"Steps: {info['steps']:2d} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Mean Fid (100): {mean_fid:.4f}"
            )

            if mean_fid > best_mean_fidelity:
                best_mean_fidelity = mean_fid
                agent.save(best_model_path)
                agent.save(config.DQN_MODEL_PATH)
                print(f"  *** New best model saved: {mean_fid:.4f} ***")

    if not os.path.exists(config.DQN_MODEL_PATH):
        agent.save(config.DQN_MODEL_PATH)

    final_mean_fidelity = float(np.mean(episode_fidelities[-100:]))

    os.makedirs(config.LOG_DIR, exist_ok=True)
    save_logs(
        {
            'rewards': episode_rewards,
            'fidelities': episode_fidelities,
            'steps': episode_steps,
        },
        os.path.join(config.LOG_DIR, 'dqn_logs.json'),
    )

    os.makedirs(config.PLOT_DIR, exist_ok=True)
    plot_training_curves(
        episode_rewards,
        episode_fidelities,
        episode_steps,
        os.path.join(config.PLOT_DIR, 'dqn_training.png'),
    )

    print(f"\n[DQN] Training complete.")
    print(f"[DQN] Final Epsilon value: {agent.epsilon:.6f}")
    print(f"[DQN] Final 100-episode mean fidelity: {final_mean_fidelity:.4f}")
    print(f"[DQN] Best 100-episode mean fidelity:  {best_mean_fidelity:.4f}")

    if config.LOG_ACTION_HISTOGRAM:
        used_count = sum(1 for i in range(action_size) if action_counts.get(i, 0) > 0)
        least_used = sorted(
            ((i, action_counts.get(i, 0)) for i in range(action_size)),
            key=lambda x: (x[1], x[0]),
        )[:5]
        print(
            f"[DQN] Action usage: {used_count}/{action_size} actions "
            "used at least once"
        )
        print("[DQN] 5 least-used action indices:")
        for idx, count in least_used:
            print(f"  action {idx}: {count}")


if __name__ == '__main__':
    cfg = Config()
    train_dqn(cfg)
