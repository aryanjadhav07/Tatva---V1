"""
evaluate.py
-----------
Evaluation script for QuantumRL — Scaled to 2 Qubits.

Loads pre-trained DQN and PPO models, evaluates both on the exact same
held-out test set of 500 Haar-random 2-qubit target statevectors (seed=1041),
and prints a side-by-side comparison table and grouped bar chart.

Run with:
    python evaluate.py
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent
from quantum_env import QuantumCircuitEnv
from utils import best_checkpoint_path, generate_target_states


def evaluate_dqn(agent: DQNAgent, env: QuantumCircuitEnv, test_states, config: Config):
    """Run greedy deterministic evaluation of a DQN agent on every test state."""
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0   # Greedy policy

    fidelities = []
    gate_counts = []
    successes = []

    for target_sv in test_states:
        obs, _ = env.reset(target_sv=target_sv)
        done = False
        final_fidelity = 0.0
        final_steps = 0

        while not done:
            action = agent.select_action(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            final_fidelity = info['fidelity']
            final_steps = info['steps']

        fidelities.append(final_fidelity)
        gate_counts.append(final_steps)
        successes.append(float(final_fidelity >= config.FIDELITY_THRESHOLD))

    agent.epsilon = original_epsilon   # Restore original epsilon

    return {
        'fidelities': fidelities,
        'gate_counts': gate_counts,
        'successes': successes,
    }


def evaluate_ppo(agent: PPOAgent, env: QuantumCircuitEnv, test_states, config: Config):
    """Run deterministic greedy evaluation of a PPO agent on every test state."""
    fidelities = []
    gate_counts = []
    successes = []

    for target_sv in test_states:
        obs, _ = env.reset(target_sv=target_sv)
        done = False
        final_fidelity = 0.0
        final_steps = 0

        while not done:
            action = agent.select_action_greedy(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            final_fidelity = info['fidelity']
            final_steps = info['steps']

        fidelities.append(final_fidelity)
        gate_counts.append(final_steps)
        successes.append(float(final_fidelity >= config.FIDELITY_THRESHOLD))

    return {
        'fidelities': fidelities,
        'gate_counts': gate_counts,
        'successes': successes,
    }


def plot_comparison(
    dqn_results: dict,
    ppo_results: dict,
    save_path: str,
    num_qubits: int = 1,
) -> None:
    """Save a four-metric grouped bar chart comparing DQN vs. PPO."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    metrics = ['Mean Fidelity', 'Median Fidelity', 'Success Rate (%)', 'Avg Gate Count']

    dqn_succ_gates = [g for g, s in zip(dqn_results['gate_counts'], dqn_results['successes']) if s]
    ppo_succ_gates = [g for g, s in zip(ppo_results['gate_counts'], ppo_results['successes']) if s]

    dqn_vals = [
        float(np.mean(dqn_results['fidelities'])),
        float(np.median(dqn_results['fidelities'])),
        float(np.mean(dqn_results['successes'])) * 100.0,
        float(np.mean(dqn_succ_gates)) if dqn_succ_gates else 0.0,
    ]
    ppo_vals = [
        float(np.mean(ppo_results['fidelities'])),
        float(np.median(ppo_results['fidelities'])),
        float(np.mean(ppo_results['successes'])) * 100.0,
        float(np.mean(ppo_succ_gates)) if ppo_succ_gates else 0.0,
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(f'DQN vs PPO — {num_qubits}-Qubit Evaluation Comparison ({len(dqn_results["fidelities"])} Test States)', color='white', fontsize=14, fontweight='bold')

    colors_dqn = '#e94560'
    colors_ppo = '#0f9b8e'

    for ax, metric, dv, pv in zip(axes, metrics, dqn_vals, ppo_vals):
        ax.set_facecolor('#16213e')
        bars = ax.bar(['DQN', 'PPO'], [dv, pv], color=[colors_dqn, colors_ppo],
                      width=0.5, edgecolor='white', linewidth=0.6)

        for bar, val in zip(bars, [dv, pv]):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01 * max(dv, pv, 1),
                f'{val:.3f}' if metric != 'Success Rate (%)' else f'{val:.1f}%',
                ha='center', va='bottom', color='white', fontsize=10, fontweight='bold',
            )

        ax.set_title(metric, color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#aaaaaa')
        ax.spines['bottom'].set_color('#444')
        ax.spines['top'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['right'].set_color('#444')
        ax.set_ylim(0, max(dv, pv, 0.1) * 1.25)
        ax.grid(True, axis='y', alpha=0.2, color='#444')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[evaluate] Comparison chart saved -> {save_path}")


def run_evaluation(config: Config, use_best: bool = False) -> None:
    """Load saved models and run full evaluation on held-out test states.

    Parameters
    ----------
    config : Config
        Active configuration object.
    use_best : bool
        When True, load the *_best* checkpoint instead of the final model.
        The best-checkpoint path is derived from the configured model path via
        ``best_checkpoint_path()``.  If the file does not exist the function
        exits with a clear error message.
    """
    test_seed = config.SEED + 999
    num_test_states = config.NUM_TEST_STATES
    print(f"[evaluate] Generating {num_test_states} {config.NUM_QUBITS}-qubit test states (seed={test_seed}) ...")
    test_states = generate_target_states(
        config.NUM_QUBITS, num_test_states, seed=test_seed
    )

    env = QuantumCircuitEnv(config)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Resolve model paths ───────────────────────────────────────────────────
    dqn_path = best_checkpoint_path(config.DQN_MODEL_PATH) if use_best else config.DQN_MODEL_PATH
    ppo_path = best_checkpoint_path(config.PPO_MODEL_PATH) if use_best else config.PPO_MODEL_PATH

    if use_best:
        print(f"[evaluate] --use-best: loading best checkpoints.")
        print(f"[evaluate]   DQN: {dqn_path}")
        print(f"[evaluate]   PPO: {ppo_path}")

    if not os.path.exists(dqn_path):
        label = "best-checkpoint" if use_best else "final model"
        raise FileNotFoundError(
            f"DQN {label} not found at {dqn_path}. "
            + ("Run train_dqn.py first — a best checkpoint is saved automatically "
               "once the first eval interval completes." if use_best
               else "Run train_dqn.py first.")
        )

    dqn_agent = DQNAgent(obs_size, action_size, config)
    dqn_agent.load(dqn_path)

    if not os.path.exists(ppo_path):
        label = "best-checkpoint" if use_best else "final model"
        raise FileNotFoundError(
            f"PPO {label} not found at {ppo_path}. "
            + ("Run train_ppo.py first — a best checkpoint is saved automatically "
               "once the first eval interval completes." if use_best
               else "Run train_ppo.py first.")
        )

    ppo_agent = PPOAgent(obs_size, action_size, config, device)
    ppo_agent.load(ppo_path)

    print(f"\n[evaluate] Evaluating DQN on {config.NUM_QUBITS}-qubit test set ...")
    dqn_results = evaluate_dqn(dqn_agent, env, test_states, config)

    dqn_mean_fid = float(np.mean(dqn_results['fidelities']))
    dqn_median_fid = float(np.median(dqn_results['fidelities']))
    dqn_success = float(np.mean(dqn_results['successes'])) * 100.0
    dqn_succ_gates = [g for g, s in zip(dqn_results['gate_counts'], dqn_results['successes']) if s]
    dqn_mean_gates = float(np.mean(dqn_succ_gates)) if dqn_succ_gates else float('nan')

    print(f"[evaluate] Evaluating PPO on {config.NUM_QUBITS}-qubit test set ...")
    ppo_results = evaluate_ppo(ppo_agent, env, test_states, config)

    ppo_mean_fid = float(np.mean(ppo_results['fidelities']))
    ppo_median_fid = float(np.median(ppo_results['fidelities']))
    ppo_success = float(np.mean(ppo_results['successes'])) * 100.0
    ppo_succ_gates = [g for g, s in zip(ppo_results['gate_counts'], ppo_results['successes']) if s]
    ppo_mean_gates = float(np.mean(ppo_succ_gates)) if ppo_succ_gates else float('nan')

    print("\n+==============================================+")
    print(f"|      {config.NUM_QUBITS}-QUBIT EVALUATION RESULTS ({num_test_states} states)    |")
    print("+==================+============+==============+")
    print("| Metric           |    DQN     |     PPO      |")
    print("+==================+============+==============+")
    print(f"| Mean Fidelity    |   {dqn_mean_fid:.4f}   |    {ppo_mean_fid:.4f}    |")
    print(f"| Median Fidelity  |   {dqn_median_fid:.4f}   |    {ppo_median_fid:.4f}    |")
    print(f"| Success Rate     |   {dqn_success:6.1f}%   |    {ppo_success:6.1f}%    |")
    print(f"| Avg Gate Count   |    {dqn_mean_gates:6.1f}    |     {ppo_mean_gates:6.1f}     |")
    print("+==================+============+==============+")

    save_plot_path = os.path.join(config.PLOT_DIR, 'dqn_vs_ppo_comparison.png')
    plot_comparison(dqn_results, ppo_results, save_plot_path, num_qubits=config.NUM_QUBITS)

    print("\n[evaluate] Evaluation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained DQN and PPO agents on a held-out test set.',
    )
    parser.add_argument(
        '--use-best',
        action='store_true',
        default=False,
        help=(
            'Load the best checkpoint (e.g. dqn_model_best.pth) instead of '
            'the final saved model.  The best checkpoint is saved automatically '
            'during training at every BEST_CHECKPOINT_EVAL_INTERVAL episodes.'
        ),
    )
    args = parser.parse_args()
    cfg = Config()
    run_evaluation(cfg, use_best=args.use_best)
