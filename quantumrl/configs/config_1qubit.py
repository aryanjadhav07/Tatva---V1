"""
config_1qubit.py
----------------
Central configuration dataclass for QuantumRL — Scoped to 1 Qubit (New Architecture).
All hyperparameters, paths, and environment settings live here.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Single source of truth for 1-Qubit QuantumRL (Dueling DQN + PER / High-Capacity PPO)."""

    # ──────────────────────────────────────────────
    # Environment (1-Qubit Configuration)
    # ──────────────────────────────────────────────
    NUM_QUBITS: int = 1           # Scoped to 1 Qubit
    MAX_STEPS: int = 15           # Maximum gates applied per episode
    FIDELITY_THRESHOLD: float = 0.99   # Target fidelity to declare success
    GATE_PENALTY: float = 0.005  # Per-step penalty

    # ──────────────────────────────────────────────
    # Gate set available to the agent
    # ──────────────────────────────────────────────
    GATES: List[str] = field(
        default_factory=lambda: ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'CNOT']
    )

    # 24-angle grid: 16 positive (π/8 … 2π) + 8 negative (-π/8 … -π)
    ROTATION_ANGLES: List[float] = field(
        default_factory=lambda: (
            [k * np.pi / 8 for k in range(1, 17)]   # π/8 … 2π  (16 angles)
            + [-k * np.pi / 8 for k in range(1, 9)]  # -π/8 … -π  (8 angles)
        )
    )

    # ──────────────────────────────────────────────
    # DQN Hyperparameters (1-Qubit Scoped + Dueling + PER)
    # ──────────────────────────────────────────────
    DQN_EPISODES: int = 25000
    DQN_BATCH_SIZE: int = 512
    DQN_BUFFER_SIZE: int = 200000
    DQN_LR: float = 0.0003
    DQN_GAMMA: float = 0.995
    DQN_EPSILON_START: float = 1.0
    DQN_EPSILON_END: float = 0.02
    DQN_EPSILON_DECAY: float = 0.99985
    DQN_TARGET_UPDATE_FREQ: int = 5
    DQN_HIDDEN_SIZE: int = 768
    DQN_WARMUP_STEPS: int = 10000

    # Prioritized Experience Replay (PER)
    PER_ALPHA: float = 0.6
    PER_BETA_START: float = 0.4
    PER_BETA_FRAMES: float = 50000

    # ──────────────────────────────────────────────
    # PPO Hyperparameters (1-Qubit Scoped)
    # ──────────────────────────────────────────────
    PPO_EPISODES: int = 25000
    PPO_ROLLOUT_STEPS: int = 8192
    PPO_EPOCHS: int = 15
    PPO_MINI_BATCH_SIZE: int = 512
    PPO_LR: float = 3e-4
    PPO_GAMMA: float = 0.995
    PPO_GAE_LAMBDA: float = 0.95
    PPO_CLIP_EPSILON: float = 0.2
    PPO_ENTROPY_COEF: float = 0.01
    PPO_VALUE_COEF: float = 1.0
    PPO_MAX_GRAD_NORM: float = 0.5
    PPO_HIDDEN_SIZE: int = 768
    PPO_MODEL_PATH: str = 'saved_models/1qubit/ppo_model.pth'
    PPO_LOG_PATH: str = 'logs/1qubit/ppo_logs.json'
    PPO_PLOT_PATH: str = 'plots/1qubit/ppo_training.png'

    # LR scheduler for PPO
    PPO_LR_DECAY: bool = True
    PPO_LR_MIN: float = 1e-5

    # ──────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────
    NUM_TEST_STATES: int = 500
    SEED: int = 42

    # ──────────────────────────────────────────────
    # Best-checkpoint tracking
    # ──────────────────────────────────────────────
    # How often (in episodes) to run lightweight greedy evaluation during training.
    # At each interval a small held-out set is evaluated and the model is saved if
    # it beats the best mean fidelity seen so far.
    BEST_CHECKPOINT_EVAL_INTERVAL: int = 1000
    # Number of freshly generated Haar-random states used for each lightweight
    # checkpoint evaluation.  Fresh states are drawn at every evaluation point so
    # no fixed held-out set is reused, keeping the signal unbiased.
    BEST_CHECKPOINT_EVAL_STATES: int = 50

    LOG_ACTION_HISTOGRAM: bool = True

    # Curriculum learning (disabled: fully random Haar states for generalization)
    CURRICULUM_ENABLED: bool = False
    CURRICULUM_POOL_SIZE: int = 30

    # File paths (qubit-scoped)
    DQN_MODEL_PATH: str = 'saved_models/1qubit/dqn_model.pth'
    LOG_DIR: str = 'logs/1qubit/'
    PLOT_DIR: str = 'plots/1qubit/'

    def __post_init__(self):
        """Ensure qubit-scoped paths if {n} formatting is present."""
        if '{n}' in self.DQN_MODEL_PATH:
            self.DQN_MODEL_PATH = self.DQN_MODEL_PATH.format(n=self.NUM_QUBITS)
        if '{n}' in self.PPO_MODEL_PATH:
            self.PPO_MODEL_PATH = self.PPO_MODEL_PATH.format(n=self.NUM_QUBITS)
        if '{n}' in self.PPO_LOG_PATH:
            self.PPO_LOG_PATH = self.PPO_LOG_PATH.format(n=self.NUM_QUBITS)
        if '{n}' in self.PPO_PLOT_PATH:
            self.PPO_PLOT_PATH = self.PPO_PLOT_PATH.format(n=self.NUM_QUBITS)
        if '{n}' in self.LOG_DIR:
            self.LOG_DIR = self.LOG_DIR.format(n=self.NUM_QUBITS)
        if '{n}' in self.PLOT_DIR:
            self.PLOT_DIR = self.PLOT_DIR.format(n=self.NUM_QUBITS)
