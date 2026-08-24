"""
utils.py
--------
Shared utility functions for QuantumRL:
  - Quantum fidelity computation
  - Random statevector generation
  - Observation encoding (state → NN input)
  - Training curve plotting
  - JSON log save/load
"""

import json
import os
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')          # Non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────
# Quantum utilities
# ─────────────────────────────────────────────────────────

def compute_fidelity(target_sv: np.ndarray, current_sv: np.ndarray) -> float:
    """
    Compute quantum state fidelity: F = |⟨target|current⟩|²

    Parameters
    ----------
    target_sv  : complex128 numpy array, normalized statevector
    current_sv : complex128 numpy array, normalized statevector

    Returns
    -------
    float in [0, 1]
    """
    return float(abs(np.dot(target_sv.conj(), current_sv)) ** 2)


def generate_random_statevector(n_qubits: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random normalized complex statevector of dimension 2^n_qubits.

    Parameters
    ----------
    n_qubits : int, number of qubits
    seed     : optional int for reproducibility

    Returns
    -------
    numpy array of shape (2^n_qubits,), dtype=complex128, unit norm
    """
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    real_part = rng.standard_normal(dim)
    imag_part = rng.standard_normal(dim)
    sv = (real_part + 1j * imag_part).astype(np.complex128)
    sv /= np.linalg.norm(sv)     # Normalize to unit norm
    return sv


def generate_target_states(
    n_qubits: int,
    num_states: int,
    seed: int = 42
) -> List[np.ndarray]:
    """
    Generate a reproducible list of random normalized statevectors.

    Each state uses a unique derived seed so results are deterministic
    yet diverse.

    Parameters
    ----------
    n_qubits   : int
    num_states : int, how many statevectors to generate
    seed       : int, base seed (different seeds → different sets)

    Returns
    -------
    List of num_states numpy complex128 arrays
    """
    states = []
    for i in range(num_states):
        sv = generate_random_statevector(n_qubits, seed=seed + i)
        states.append(sv)
    return states


def generate_curriculum_pool(
    pool_size: int,
    num_qubits: int,
    seed: int,
) -> List[np.ndarray]:
    """
    Generate a fixed pool of random target statevectors for curriculum training.

    Called once at the start of training.  The returned list is reused across
    episodes (sampled with replacement) so the agent sees repeated targets and
    can specialize before generalizing.

    Parameters
    ----------
    pool_size   : int, number of target states in the pool
    num_qubits  : int, number of qubits per statevector
    seed        : int, fixed seed for reproducible pool generation

    Returns
    -------
    List of pool_size numpy complex128 arrays
    """
    pool = []
    for i in range(pool_size):
        sv = generate_random_statevector(num_qubits, seed=seed + i)
        pool.append(sv)
    return pool


# ─────────────────────────────────────────────────────────
# Observation encoding
# ─────────────────────────────────────────────────────────

def encode_state(
    current_sv: np.ndarray,
    target_sv: np.ndarray,
    fidelity: float = 0.0,
    step: int = 0,
    max_steps: int = 20,
) -> np.ndarray:
    """
    Encode a (current, target) statevector pair + progress metrics as a flat float32 array.

    Concatenates: [Re(current), Im(current), Re(target), Im(target), fidelity, step/max_steps]
    Length = 4 * 2^n_qubits + 2 (18 floats for 2 qubits)

    Parameters
    ----------
    current_sv : complex128 array of length 2^n
    target_sv  : complex128 array of length 2^n
    fidelity   : float, current state fidelity
    step       : int, current episode step counter
    max_steps  : int, max episode step limit

    Returns
    -------
    numpy float32 array of length 4 * 2^n + 2
    """
    norm_step = float(step / max_steps) if max_steps > 0 else 0.0
    obs = np.concatenate([
        current_sv.real.astype(np.float32),
        current_sv.imag.astype(np.float32),
        target_sv.real.astype(np.float32),
        target_sv.imag.astype(np.float32),
        np.array([fidelity, norm_step], dtype=np.float32),
    ])
    return obs


# ─────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────

def _rolling_mean(data: List[float], window: int = 50) -> np.ndarray:
    """Compute rolling mean with given window size."""
    arr = np.array(data, dtype=np.float64)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def plot_training_curves(
    episode_rewards: List[float],
    episode_fidelities: List[float],
    episode_steps: List[int],
    save_path: str
) -> None:
    """
    Plot and save three-panel training diagnostics:
      1. Episode reward (with rolling mean overlay)
      2. Final fidelity per episode (with rolling mean overlay)
      3. Steps (gates) used per episode (with rolling mean overlay)

    Parameters
    ----------
    episode_rewards    : list of total rewards per episode
    episode_fidelities : list of final fidelity values per episode
    episode_steps      : list of gate counts per episode
    save_path          : string path (PNG) where figure is saved
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.patch.set_facecolor('#1a1a2e')

    datasets = [
        (episode_rewards,    'Episode Reward',   '#e94560', 'Reward'),
        (episode_fidelities, 'Fidelity',         '#0f3460', 'Fidelity'),
        (episode_steps,      'Gates per Episode','#533483', 'Gate Count'),
    ]

    for ax, (data, title, color, ylabel) in zip(axes, datasets):
        episodes = np.arange(1, len(data) + 1)
        ax.set_facecolor('#16213e')
        ax.plot(episodes, data, alpha=0.3, color=color, linewidth=0.8, label='Raw')

        if len(data) >= 50:
            rolled = _rolling_mean(data, window=50)
            offset = len(data) - len(rolled)
            ax.plot(
                np.arange(offset + 1, len(data) + 1),
                rolled, color=color, linewidth=2.0, label='Rolling mean (50)'
            )

        ax.set_title(title, color='white', fontsize=13, fontweight='bold')
        ax.set_xlabel('Episode', color='#aaaaaa', fontsize=10)
        ax.set_ylabel(ylabel, color='#aaaaaa', fontsize=10)
        ax.tick_params(colors='#aaaaaa')
        ax.spines['bottom'].set_color('#444')
        ax.spines['top'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['right'].set_color('#444')
        ax.legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
        ax.grid(True, alpha=0.2, color='#444')

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[utils] Training curve saved -> {save_path}")


# ─────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────

def save_logs(logs_dict: dict, filepath: str) -> None:
    """
    Persist a training-log dictionary to a JSON file.

    Parameters
    ----------
    logs_dict : dict with list values (rewards, fidelities, steps, …)
    filepath  : destination file path (.json)
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(logs_dict, f, indent=2)
    print(f"[utils] Logs saved -> {filepath}")


def load_logs(filepath: str) -> dict:
    """
    Load training logs from a JSON file.

    Parameters
    ----------
    filepath : path to .json log file

    Returns
    -------
    dict of lists
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────
# Checkpoint path utilities
# ─────────────────────────────────────────────────────────

def best_checkpoint_path(model_path: str) -> str:
    """
    Derive the *_best* checkpoint path from a configured model path.

    Inserts ``_best`` immediately before the file extension so the naming
    convention is consistent regardless of qubit count or directory layout.

    Examples
    --------
    >>> best_checkpoint_path('saved_models/1qubit/dqn_model.pth')
    'saved_models/1qubit/dqn_model_best.pth'
    >>> best_checkpoint_path('saved_models/2qubit/ppo_model.pth')
    'saved_models/2qubit/ppo_model_best.pth'
    >>> best_checkpoint_path('model')   # no extension
    'model_best'

    Parameters
    ----------
    model_path : str
        Path to the final (non-best) model checkpoint as stored in Config.

    Returns
    -------
    str
        Path with ``_best`` inserted before the file extension.
    """
    root, ext = os.path.splitext(model_path)
    return f"{root}_best{ext}"
