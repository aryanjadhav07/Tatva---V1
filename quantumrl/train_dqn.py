"""
train_dqn.py
------------
DQN training entry point for QuantumRL.

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
from utils import (
    best_checkpoint_path,
    generate_random_statevector,
    plot_training_curves,
    save_logs,
)


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _greedy_eval_dqn(
    agent: DQNAgent,
    env: QuantumCircuitEnv,
    config: Config,
    episode: int,
) -> float:
    """
    Lightweight greedy evaluation of the DQN agent on freshly generated states.

    Runs BEST_CHECKPOINT_EVAL_STATES episodes with epsilon=0 (no exploration).
    Uses torch.no_grad() throughout and restores training mode afterwards.
    State seeds are derived from the current episode so the set is fresh at
    every call but still reproducible.

    Returns
    -------
    float
        Mean fidelity across the evaluation states.
    """
    n_states = getattr(config, 'BEST_CHECKPOINT_EVAL_STATES', 50)
    # Seed offset: use episode number so each eval gets a distinct, fresh set
    eval_seed = config.SEED + 20000 + episode

    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0          # fully greedy
    agent.q_net.eval()           # evaluation mode (affects LayerNorm, Dropout if any)

    fidelities = []
    with torch.no_grad():
        for i in range(n_states):
            target_sv = generate_random_statevector(config.NUM_QUBITS, seed=eval_seed + i)
            obs, _ = env.reset(target_sv=target_sv)
            done = False
            final_fidelity = 0.0
            while not done:
                action = agent.select_action(obs)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                final_fidelity = info['fidelity']
            fidelities.append(final_fidelity)

    agent.q_net.train()          # restore training mode
    agent.epsilon = saved_epsilon

    return float(np.mean(fidelities))


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
        print(
            f"[DQN][SAVE-GUARD] Warning: Existing model found at "
            f"'{config.DQN_MODEL_PATH}'. New checkpoints will overwrite this file."
        )

    # Derived best-checkpoint path (e.g. dqn_model_best.pth)
    best_path = best_checkpoint_path(config.DQN_MODEL_PATH)

    eval_interval = getattr(config, 'BEST_CHECKPOINT_EVAL_INTERVAL', 1000)
    print(
        f"[DQN] Best-checkpoint eval every {eval_interval} episodes "
        f"({getattr(config, 'BEST_CHECKPOINT_EVAL_STATES', 50)} states each)."
    )

    # Replay buffer warm-up
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

    # Best-checkpoint tracking
    best_eval_fidelity: float = float('-inf')
    best_eval_episode: int = -1

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

        # ── Per-100-episode progress print (training curve metric) ────────────
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

        # ── Periodic greedy evaluation for best-checkpoint tracking ──────────
        if (episode + 1) % eval_interval == 0:
            eval_fid = _greedy_eval_dqn(agent, env, config, episode)
            marker = ""
            if eval_fid > best_eval_fidelity:
                best_eval_fidelity = eval_fid
                best_eval_episode = episode + 1
                agent.save(best_path)
                marker = "  *** new best checkpoint saved ***"
            print(
                f"  [checkpoint-eval ep {episode + 1:5d}] "
                f"mean fidelity = {eval_fid:.4f}{marker}",
                flush=True,
            )

    # ── Save final model ──────────────────────────────────────────────────────
    agent.save(config.DQN_MODEL_PATH)

    # ── Final lightweight eval for honest end-of-training comparison ─────────
    final_eval_fid = _greedy_eval_dqn(agent, env, config, episode=config.DQN_EPISODES)

    # If the final model is also the best, update the record
    if final_eval_fid > best_eval_fidelity:
        best_eval_fidelity = final_eval_fid
        best_eval_episode = config.DQN_EPISODES
        agent.save(best_path)

    # ── Post-training logging & plots ─────────────────────────────────────────
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

    # ── End-of-training summary ───────────────────────────────────────────────
    print(f"\n[DQN] Training complete.")
    print(f"[DQN] Final Epsilon value: {agent.epsilon:.6f}")
    print(f"[DQN] {'─' * 52}")
    print(f"[DQN]  End-of-training checkpoint summary")
    print(f"[DQN] {'─' * 52}")
    print(f"[DQN]  Final model eval fidelity  : {final_eval_fid:.4f}")
    print(f"[DQN]  Best checkpoint fidelity   : {best_eval_fidelity:.4f}  "
          f"(episode {best_eval_episode})")
    delta = best_eval_fidelity - final_eval_fid
    if best_eval_episode == config.DQN_EPISODES or abs(delta) < 1e-6:
        print(f"[DQN]  The final model IS the best checkpoint.")
    else:
        print(
            f"[DQN]  An earlier checkpoint outperformed the final model "
            f"by {delta:+.4f}. Best checkpoint saved at: {best_path}"
        )
    print(f"[DQN] {'─' * 52}")

    if config.LOG_ACTION_HISTOGRAM:
        used_count = sum(1 for i in range(action_size) if action_counts.get(i, 0) > 0)
        least_used = sorted(
            ((i, action_counts.get(i, 0)) for i in range(action_size)),
            key=lambda x: (x[1], x[0]),
        )[:5]
        print(f"[DQN] Action usage: {used_count}/{action_size} actions used at least once")
        print("[DQN] 5 least-used action indices:")
        for idx, count in least_used:
            print(f"  action {idx}: {count}")


if __name__ == '__main__':
    cfg = Config()
    train_dqn(cfg)
