"""
synthesize_target.py
--------------------
Standalone inference script: Target State → Quantum Circuit

Given a user-specified target quantum state, loads the trained DQN and/or PPO
agent weights and runs greedy (non-exploratory) inference from |0⟩ to
synthesize a gate sequence that reaches that state.

Supports two saved-model layouts automatically (detected by key inspection):

  LEGACY (original 1-qubit models):
    obs=10, action=76, hidden=512/256
    DQN keys: net.0.*, net.3.*, net.6.*, net.9.*
    PPO keys: actor.0.*, ..., actor.9.* / critic.0.*, ..., critic.9.*

  NEW (Dueling DQN + PPO with hidden=768/384):
    obs=10, action=76, hidden=768/384
    DQN keys: feature_trunk.*, value_stream.*, advantage_stream.*
    PPO keys: actor.0.*, ..., actor.9.* / critic.0.*, ..., critic.9.*
              (same Sequential layout as legacy, different hidden dims)

This script defines compatible network wrappers for both layouts so .pth files
load cleanly without touching any training pipeline files.

Usage
-----
  # Interactive mode (no arguments):
      python synthesize_target.py

  # CLI mode (all arguments):
      python synthesize_target.py --target plus --agent both
      python synthesize_target.py --target custom --amplitudes "0.707,0.707"
      python synthesize_target.py --target random --agent ppo

Named presets (1-qubit):
  zero    : |0⟩  — ground state
  one     : |1⟩  — excited state
  plus    : |+⟩ = (|0⟩+|1⟩)/√2  — uniform superposition
  minus   : |−⟩ = (|0⟩−|1⟩)/√2  — phase state
  i_state : +i Y-eigenstate = (|0⟩+i|1⟩)/√2
  random  : Single Haar-random state (quick demo)

Custom amplitude input (--target custom):
  Enter comma-separated complex amplitudes, e.g.:  0.707, 0.707
  Two amplitudes required for the 1-qubit system.
  The vector is validated (correct dimension, unit norm within tolerance).
"""

import argparse
import math
import os
import sys
import textwrap
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ── Resolve quantumrl package path ────────────────────────────────────────────
# The script lives at the project root; quantumrl/ is one level below.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUANTUMRL_DIR = os.path.join(SCRIPT_DIR, 'quantumrl')
sys.path.insert(0, QUANTUMRL_DIR)

from config import Config
from quantum_env import QuantumCircuitEnv
from utils import generate_random_statevector

# ── Constants ─────────────────────────────────────────────────────────────────
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, 'outputs')
_INV_SQRT2 = 1.0 / math.sqrt(2.0)

# 1-qubit observation / action sizes (fixed regardless of hidden dim)
_OBS_SIZE = 10   # 4*(2^1)+2
_ACT_SIZE = 76   # 4 fixed + 3*24 rotation gates on 1 qubit

# Named preset statevectors (complex128, 1-qubit, unit-norm)
PRESETS: Dict[str, Optional[np.ndarray]] = {
    'zero':    np.array([1.0+0j, 0.0+0j],          dtype=np.complex128),
    'one':     np.array([0.0+0j, 1.0+0j],          dtype=np.complex128),
    'plus':    np.array([_INV_SQRT2, _INV_SQRT2],   dtype=np.complex128),
    'minus':   np.array([_INV_SQRT2, -_INV_SQRT2],  dtype=np.complex128),
    'i_state': np.array([_INV_SQRT2, _INV_SQRT2*1j], dtype=np.complex128),
    'random':  None,  # generated at runtime
}

NORM_TOLERANCE = 1e-5

# ── 1-qubit Config override ────────────────────────────────────────────────────
# The live Config() is 2-qubit; we need a 1-qubit env for inference.
# Python dataclasses don't allow overriding default values via subclass annotation
# without re-declaring the full dataclass, so we patch the instance directly.

def _make_one_qubit_config() -> Config:
    """Return a Config instance with NUM_QUBITS patched to 1."""
    cfg = Config()
    cfg.NUM_QUBITS = 1
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Network definitions — two layouts, auto-detected from saved weights
# ──────────────────────────────────────────────────────────────────────────────

def _make_mlp_branch(in_size: int, h1: int, h2: int, out_size: int) -> nn.Sequential:
    """
    Build a 3-hidden-layer MLP whose Sequential indices match saved .pth keys:
      index 0: Linear(in, h1)   index 1: LayerNorm(h1)   index 2: LeakyReLU (no params)
      index 3: Linear(h1, h1)   index 4: LayerNorm(h1)   index 5: LeakyReLU (no params)
      index 6: Linear(h1, h2)   index 7: LayerNorm(h2)   index 8: LeakyReLU (no params)
      index 9: Linear(h2, out)
    Works for both legacy (h1=512, h2=256) and new (h1=768, h2=384) hidden dims.
    """
    return nn.Sequential(
        nn.Linear(in_size, h1),   # 0
        nn.LayerNorm(h1),          # 1
        nn.LeakyReLU(0.01),       # 2
        nn.Linear(h1, h1),        # 3
        nn.LayerNorm(h1),          # 4
        nn.LeakyReLU(0.01),       # 5
        nn.Linear(h1, h2),        # 6
        nn.LayerNorm(h2),          # 7
        nn.LeakyReLU(0.01),       # 8
        nn.Linear(h2, out_size),   # 9
    )


class _FlatQNet(nn.Module):
    """Legacy DQN: single net.* Sequential (old architecture, keys: net.0 … net.9)."""
    def __init__(self, obs_size: int, action_size: int, h1: int, h2: int):
        super().__init__()
        self.net = _make_mlp_branch(obs_size, h1, h2, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DuelingQNet(nn.Module):
    """
    New Dueling DQN: feature_trunk + value_stream + advantage_stream.
    Keys: feature_trunk.0/1/3/4, value_stream.0/1/3, advantage_stream.0/1/3
    """
    def __init__(self, obs_size: int, action_size: int, h1: int, h2: int):
        super().__init__()
        self.feature_trunk = nn.Sequential(
            nn.Linear(obs_size, h1),  # 0
            nn.LayerNorm(h1),          # 1
            nn.LeakyReLU(0.01),       # 2
            nn.Linear(h1, h1),        # 3
            nn.LayerNorm(h1),          # 4
            nn.LeakyReLU(0.01),       # 5
        )
        self.value_stream = nn.Sequential(
            nn.Linear(h1, h2),   # 0
            nn.LayerNorm(h2),     # 1
            nn.LeakyReLU(0.01),  # 2
            nn.Linear(h2, 1),    # 3
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(h1, h2),          # 0
            nn.LayerNorm(h2),            # 1
            nn.LeakyReLU(0.01),         # 2
            nn.Linear(h2, action_size),  # 3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features   = self.feature_trunk(x)
        values     = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


class _ActorCritic(nn.Module):
    """
    Actor-Critic for PPO inference (both legacy and new share same key layout).
    Hidden dims are inferred from the saved state_dict at load time.
    """
    def __init__(self, obs_size: int, action_size: int, h1: int, h2: int):
        super().__init__()
        self.actor  = _make_mlp_branch(obs_size, h1, h2, action_size)
        self.critic = _make_mlp_branch(obs_size, h1, h2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(x), self.critic(x)


def _infer_hidden_dims(state_dict: dict) -> Tuple[int, int]:
    """
    Read h1 and h2 from a state_dict by inspecting the first Linear weight.
    Works for both flat-net (net.0.weight) and actor/critic (actor.0.weight) layouts.
    """
    for key in state_dict:
        if key.endswith('.0.weight'):
            h1 = state_dict[key].shape[0]
            # Find h2: the weight going from h1 to h2 is at index 6
            prefix = key[: key.rfind('.0.weight')]
            h2_key = f'{prefix}.6.weight'
            if h2_key in state_dict:
                h2 = state_dict[h2_key].shape[0]
                return h1, h2
            # Dueling: value_stream.0.weight or advantage_stream.0.weight
            break
    # Dueling fallback: value_stream.0.weight gives h2
    for key in state_dict:
        if 'value_stream.0.weight' in key or key == 'value_stream.0.weight':
            h2 = state_dict[key].shape[0]
            ft_key = 'feature_trunk.0.weight'
            h1 = state_dict[ft_key].shape[0] if ft_key in state_dict else 768
            return h1, h2
    return 512, 256  # safe fallback for oldest legacy models


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight inference wrappers
# ──────────────────────────────────────────────────────────────────────────────

class DQNInferenceAgent:
    """
    Greedy DQN inference. Automatically handles both:
      - Legacy flat net (keys: net.*)
      - New Dueling net (keys: feature_trunk.*, value_stream.*, advantage_stream.*)
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net: Optional[nn.Module] = None

    def load(self, path: str) -> None:
        sd = torch.load(path, map_location=self.device, weights_only=True)

        # Detect architecture from key names
        is_dueling = any(k.startswith('feature_trunk') for k in sd)
        obs_size   = sd['feature_trunk.0.weight'].shape[1] if is_dueling \
                     else sd['net.0.weight'].shape[1]
        act_size   = sd['advantage_stream.3.weight'].shape[0] if is_dueling \
                     else sd['net.9.weight'].shape[0]

        h1, h2 = _infer_hidden_dims(sd)

        if is_dueling:
            self.net = _DuelingQNet(obs_size, act_size, h1, h2).to(self.device)
            arch_label = f'Dueling (h1={h1}, h2={h2})'
        else:
            self.net = _FlatQNet(obs_size, act_size, h1, h2).to(self.device)
            arch_label = f'FlatNet (h1={h1}, h2={h2})'

        self.net.load_state_dict(sd)
        self.net.eval()
        print(f"[DQNAgent] Model loaded <- {path}  [{arch_label}]")

    def select_action(self, state: np.ndarray) -> int:
        """Greedy argmax Q — no exploration."""
        state_t = torch.tensor(state, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.net(state_t)
        return int(q_values.argmax(dim=1).item())


class PPOInferenceAgent:
    """
    Greedy PPO inference. Handles both legacy (h1=512) and new (h1=768) hidden dims
    automatically — PPO always uses the same actor/critic Sequential key layout.
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ac: Optional[nn.Module] = None

    def load(self, path: str) -> None:
        sd = torch.load(path, map_location=self.device, weights_only=True)

        obs_size = sd['actor.0.weight'].shape[1]
        act_size = sd['actor.9.weight'].shape[0]
        h1, h2   = _infer_hidden_dims(sd)

        self.ac = _ActorCritic(obs_size, act_size, h1, h2).to(self.device)
        self.ac.load_state_dict(sd)
        self.ac.eval()
        print(f"[PPOAgent] Model loaded <- {path}  [ActorCritic h1={h1}, h2={h2}]")

    def select_action_greedy(self, state: np.ndarray) -> int:
        """Deterministic argmax over actor logits — no sampling."""
        state_t = torch.tensor(state, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.ac(state_t)
        return int(logits.argmax(dim=1).item())


# ──────────────────────────────────────────────────────────────────────────────
# Target state handling
# ──────────────────────────────────────────────────────────────────────────────

def validate_statevector(sv: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Validate a candidate statevector:
      - Must have exactly 2^n_qubits amplitudes.
      - Must be unit-norm within NORM_TOLERANCE.

    Returns the vector as complex128 on success; raises ValueError with a
    descriptive message on failure.
    """
    expected_dim = 2 ** n_qubits
    if sv.ndim != 1 or len(sv) != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} amplitudes for a {n_qubits}-qubit system, "
            f"got {sv.size}. Please provide exactly {expected_dim} complex values."
        )
    norm = np.linalg.norm(sv)
    if abs(norm - 1.0) > NORM_TOLERANCE:
        raise ValueError(
            f"Statevector is not unit-normalized (norm = {norm:.6f}). "
            f"Please normalize your amplitudes so they satisfy |ψ|² = 1."
        )
    return sv.astype(np.complex128)


def parse_custom_amplitudes(raw: str, n_qubits: int) -> np.ndarray:
    """
    Parse a comma-separated string of complex amplitudes into a validated
    complex128 numpy array.

    Examples of valid input:
        "0.707, 0.707"
        "0.5+0.5j, 0.5-0.5j"
        "(0.5+0.5j),(0.5-0.5j)"
    """
    raw = raw.strip().strip('"').strip("'")
    parts = [p.strip().strip('()') for p in raw.split(',')]
    expected = 2 ** n_qubits
    if len(parts) != expected:
        raise ValueError(
            f"Expected {expected} comma-separated amplitudes, got {len(parts)}."
        )
    try:
        amplitudes = [complex(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"Could not parse amplitude(s) as complex numbers: {exc}\n"
            "Use Python complex notation, e.g. '0.707', '0.5+0.5j', '0-0.5j'."
        ) from exc

    sv = np.array(amplitudes, dtype=np.complex128)
    return validate_statevector(sv, n_qubits)


def get_target_state(
    target_name: str, n_qubits: int, cli_amplitudes: Optional[str] = None
) -> Tuple[np.ndarray, str]:
    """
    Resolve a target name (preset key, 'custom', or 'random') to a validated
    statevector.  Returns (statevector: np.ndarray complex128, display_label: str).
    """
    if target_name == 'random':
        sv = generate_random_statevector(n_qubits)
        return validate_statevector(sv, n_qubits), 'random (Haar)'

    if target_name in PRESETS and PRESETS[target_name] is not None:
        sv = PRESETS[target_name].copy()
        return validate_statevector(sv, n_qubits), target_name

    if target_name == 'custom':
        if cli_amplitudes is not None:
            # CLI path — amplitudes supplied via --amplitudes flag
            sv = parse_custom_amplitudes(cli_amplitudes, n_qubits)
        else:
            # Interactive path
            expected = 2 ** n_qubits
            print(
                f"\nEnter {expected} comma-separated complex amplitudes "
                f"for a {n_qubits}-qubit state."
            )
            print("  Example (1 qubit): 0.707, 0.707")
            raw = input("  Amplitudes: ").strip()
            sv = parse_custom_amplitudes(raw, n_qubits)
        return sv, 'custom'

    raise ValueError(
        f"Unknown target '{target_name}'. "
        f"Valid options: {list(PRESETS.keys()) + ['custom']}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Human-readable gate formatting
# ──────────────────────────────────────────────────────────────────────────────

def format_gate(step: int, gate_name: str, qubit_or_pair, angle: Optional[float]) -> str:
    """Return a human-readable description of one gate application."""
    if gate_name == 'CNOT':
        ctrl, tgt = qubit_or_pair
        return f"  Step {step:2d}: CNOT  ctrl=qubit {ctrl}  tgt=qubit {tgt}"
    if angle is not None:
        angle_deg = math.degrees(angle)
        return (
            f"  Step {step:2d}: {gate_name}({angle:.4f} rad = {angle_deg:.2f}°)"
            f"  on qubit {qubit_or_pair}"
        )
    return f"  Step {step:2d}: {gate_name}  on qubit {qubit_or_pair}"


# ──────────────────────────────────────────────────────────────────────────────
# Core inference runners
# ──────────────────────────────────────────────────────────────────────────────

def run_inference_dqn(
    agent: DQNInferenceAgent,
    env: QuantumCircuitEnv,
    target_sv: np.ndarray,
) -> Tuple[List[str], float, int]:
    """
    Run one greedy DQN episode against the given target state.
    Returns (gate_lines, fidelity, gate_count).
    """
    obs, _ = env.reset(target_sv=target_sv)
    gate_lines: List[str] = []
    done = False
    final_fidelity = 0.0
    final_steps = 0

    while not done:
        action = agent.select_action(obs)
        gate_name, qubit_or_pair, angle = env.action_list[action]
        gate_lines.append(format_gate(final_steps + 1, gate_name, qubit_or_pair, angle))
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        final_fidelity = info['fidelity']
        final_steps = info['steps']

    return gate_lines, final_fidelity, final_steps


def run_inference_ppo(
    agent: PPOInferenceAgent,
    env: QuantumCircuitEnv,
    target_sv: np.ndarray,
) -> Tuple[List[str], float, int]:
    """
    Run one greedy PPO episode against the given target state.
    Returns (gate_lines, fidelity, gate_count).
    """
    obs, _ = env.reset(target_sv=target_sv)
    gate_lines: List[str] = []
    done = False
    final_fidelity = 0.0
    final_steps = 0

    while not done:
        action = agent.select_action_greedy(obs)
        gate_name, qubit_or_pair, angle = env.action_list[action]
        gate_lines.append(format_gate(final_steps + 1, gate_name, qubit_or_pair, angle))
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        final_fidelity = info['fidelity']
        final_steps = info['steps']

    return gate_lines, final_fidelity, final_steps


# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_circuit_outputs(env: QuantumCircuitEnv, agent_name: str) -> Tuple[str, str]:
    """
    Save circuit diagram (PNG) and QASM code to the outputs/ directory.
    Returns (png_path, qasm_path).
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    png_path  = os.path.join(OUTPUTS_DIR, f'circuit_{agent_name}.png')
    qasm_path = os.path.join(OUTPUTS_DIR, f'circuit_{agent_name}.qasm')

    # ── PNG circuit diagram ───────────────────────────────────────────────────
    try:
        # Ensure Agg backend is active before each draw call.
        # Qiskit's mpl drawer can leave the backend in an inconsistent state
        # when multiple circuits are rendered in the same process.
        plt.switch_backend('Agg')
        fig = env.current_circuit.draw('mpl', fold=-1)
        if fig is not None and hasattr(fig, 'savefig'):
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            # Qiskit ≥1.2 returns None; use filename kwarg instead
            env.current_circuit.draw('mpl', filename=png_path, fold=-1)
    except Exception as exc:
        print(f"  [warn] Could not save PNG diagram: {exc}")
        png_path = f"(not saved — {exc})"

    # ── QASM export ───────────────────────────────────────────────────────────
    try:
        import qiskit.qasm2 as qasm2
        qasm_str = qasm2.dumps(env.current_circuit)
        with open(qasm_path, 'w', encoding='utf-8') as f:
            f.write(qasm_str)
    except Exception as exc:
        try:
            qasm_str = env.current_circuit.qasm()
            with open(qasm_path, 'w', encoding='utf-8') as f:
                f.write(qasm_str)
        except Exception as exc2:
            print(f"  [warn] Could not save QASM: {exc2}")
            qasm_path = f"(not saved — {exc2})"

    return png_path, qasm_path


def print_agent_result(
    agent_name: str,
    gate_lines: List[str],
    fidelity: float,
    gate_count: int,
    png_path: str,
    qasm_path: str,
    threshold: float,
    env: QuantumCircuitEnv,
) -> None:
    """Print the full result block for one agent."""
    success = fidelity >= threshold
    status  = "SUCCESS" if success else "PARTIAL (below threshold)"
    bar     = "=" * 60

    print(f"\n{bar}")
    print(f"  Agent  : {agent_name.upper()}")
    print(f"  Status : {status}")
    print(f"  Final fidelity : {fidelity:.6f}  (threshold = {threshold:.2f})")
    print(f"  Gate count     : {gate_count}")
    print(bar)

    print("\n  Gate Sequence:")
    if gate_lines:
        for line in gate_lines:
            print(line)
    else:
        print("  (no gates applied — already at target state)")

    print(f"\n  ASCII Circuit Diagram:")
    print()
    env.render()

    print(f"\n  Saved outputs:")
    print(f"    PNG  : {png_path}")
    print(f"    QASM : {qasm_path}")


def print_comparison_table(results: dict, threshold: float) -> None:
    """Print a side-by-side comparison table for all agents that were run."""
    if len(results) < 2:
        return

    agents = list(results.keys())
    print("\n" + "=" * 60)
    print("  SIDE-BY-SIDE COMPARISON")
    print("=" * 60)
    header_agents = "  ".join(f"{a.upper():>10}" for a in agents)
    print(f"  {'Metric':<22} {header_agents}")
    print("  " + "-" * 56)

    rows = [
        ("Final Fidelity",    "fidelity",   lambda v: f"{v:.6f}"),
        ("Gate Count",        "gate_count", lambda v: f"{v:>10}"),
        ("Success (>=thresh)","success",    lambda v: "       Yes" if v else "        No"),
    ]
    for label, key, fmt in rows:
        values = "  ".join(fmt(results[a][key]) for a in agents)
        print(f"  {label:<22} {values}")

    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Agent loading
# ──────────────────────────────────────────────────────────────────────────────

def load_dqn(obs_size: int, action_size: int, model_path: str) -> DQNInferenceAgent:
    """Load and return a DQN inference agent (auto-detects architecture)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"DQN model not found at: {model_path}\n"
            "Run train_dqn.py first to produce the saved weights."
        )
    agent = DQNInferenceAgent()
    agent.load(model_path)
    return agent


def load_ppo(obs_size: int, action_size: int, model_path: str) -> PPOInferenceAgent:
    """Load and return a PPO inference agent (auto-detects hidden dims)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"PPO model not found at: {model_path}\n"
            "Run train_ppo.py first to produce the saved weights."
        )
    agent = PPOInferenceAgent()
    agent.load(model_path)
    return agent


# ──────────────────────────────────────────────────────────────────────────────
# Interactive prompts
# ──────────────────────────────────────────────────────────────────────────────

def prompt_target() -> str:
    """Interactively ask the user for a target state name."""
    valid = list(PRESETS.keys()) + ['custom']
    descriptions = {
        'zero':    '|0⟩  — ground state',
        'one':     '|1⟩  — excited state',
        'plus':    '|+⟩ = (|0⟩+|1⟩)/√2  — uniform superposition',
        'minus':   '|−⟩ = (|0⟩−|1⟩)/√2  — phase state',
        'i_state': '+i Y-eigenstate = (|0⟩+i|1⟩)/√2',
        'random':  'Haar-random state (quick demo)',
        'custom':  'Enter your own complex amplitudes',
    }
    print("\nAvailable target states:")
    for name in valid:
        print(f"  {name:<10} — {descriptions.get(name, '')}")

    while True:
        choice = input("\nTarget state [default: plus]: ").strip().lower() or 'plus'
        if choice in valid:
            return choice
        print(f"  Invalid choice '{choice}'. Pick from: {', '.join(valid)}")


def prompt_agent() -> str:
    """Interactively ask the user which agent(s) to run."""
    while True:
        choice = (
            input("Agent to run — dqn / ppo / both [default: both]: ")
            .strip().lower() or 'both'
        )
        if choice in ('dqn', 'ppo', 'both'):
            return choice
        print("  Invalid choice. Enter 'dqn', 'ppo', or 'both'.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    valid_targets = list(PRESETS.keys()) + ['custom']
    parser = argparse.ArgumentParser(
        prog='synthesize_target.py',
        description=textwrap.dedent("""\
            Target-state-to-circuit synthesis using trained RL agents.
            Run with no arguments for interactive mode.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--target', '-t',
        choices=valid_targets,
        default=None,
        help=(
            f"Target quantum state. One of: {', '.join(valid_targets)}. "
            "Use 'custom' to enter raw complex amplitudes."
        ),
    )
    parser.add_argument(
        '--agent', '-a',
        choices=['dqn', 'ppo', 'both'],
        default=None,
        help="Which agent(s) to run: dqn, ppo, or both (default: both).",
    )
    parser.add_argument(
        '--amplitudes', '-amp',
        default=None,
        metavar='AMPLITUDES',
        help=(
            "Comma-separated complex amplitudes for --target custom. "
            "Example: '0.707,0.707'  (1-qubit system needs 2 values). "
            "Skips the interactive amplitude prompt when provided."
        ),
    )
    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Determine interactive vs CLI mode ─────────────────────────────────────
    interactive = (args.target is None) and (args.agent is None)

    if interactive:
        print("=" * 60)
        print("  QuantumRL — Target-State Circuit Synthesis")
        print("  (Interactive Mode — run with --help for CLI flags)")
        print("=" * 60)
        target_name  = prompt_target()
        agent_choice = prompt_agent()
    else:
        target_name  = args.target or 'plus'
        agent_choice = args.agent  or 'both'

    # ── Build 1-qubit environment (matches trained model dimensions) ──────────
    config_1q = _make_one_qubit_config()
    env = QuantumCircuitEnv(config_1q)
    n_qubits    = config_1q.NUM_QUBITS   # 1
    obs_size    = env.observation_space.shape[0]  # 10
    action_size = env.action_space.n              # 76

    # ── Resolve target statevector ────────────────────────────────────────────
    try:
        target_sv, target_label = get_target_state(
            target_name, n_qubits, cli_amplitudes=args.amplitudes
        )
    except ValueError as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Display target info ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Target state : {target_label}")
    print(f"  Amplitudes   : {np.round(target_sv, 4)}")
    print(f"  Norm         : {np.linalg.norm(target_sv):.8f}")
    print(f"  System       : {n_qubits} qubit  |  obs={obs_size}  actions={action_size}")
    print(f"  Agents       : {agent_choice}")
    print(f"{'=' * 60}")

    # ── Resolve model file paths ──────────────────────────────────────────────
    # config.*_MODEL_PATH is relative to quantumrl/; resolve from there.
    dqn_model_path = os.path.join(QUANTUMRL_DIR, config_1q.DQN_MODEL_PATH)
    ppo_model_path = os.path.join(QUANTUMRL_DIR, config_1q.PPO_MODEL_PATH)

    # ── Load requested agents ─────────────────────────────────────────────────
    print()
    dqn_agent: Optional[DQNInferenceAgent] = None
    ppo_agent: Optional[PPOInferenceAgent] = None

    if agent_choice in ('dqn', 'both'):
        print("[synthesize] Loading DQN ...")
        try:
            dqn_agent = load_dqn(obs_size, action_size, dqn_model_path)
        except FileNotFoundError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            sys.exit(1)

    if agent_choice in ('ppo', 'both'):
        print("[synthesize] Loading PPO ...")
        try:
            ppo_agent = load_ppo(obs_size, action_size, ppo_model_path)
        except FileNotFoundError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────────────────
    comparison_results: dict = {}

    if dqn_agent is not None:
        print("\n[synthesize] Running DQN inference (greedy, epsilon=0) ...")
        gate_lines, fidelity, gate_count = run_inference_dqn(dqn_agent, env, target_sv)
        png_path, qasm_path = save_circuit_outputs(env, 'dqn')
        print_agent_result(
            'dqn', gate_lines, fidelity, gate_count,
            png_path, qasm_path, config_1q.FIDELITY_THRESHOLD, env,
        )
        comparison_results['dqn'] = {
            'fidelity':   fidelity,
            'gate_count': gate_count,
            'success':    fidelity >= config_1q.FIDELITY_THRESHOLD,
        }

    if ppo_agent is not None:
        print("\n[synthesize] Running PPO inference (greedy, deterministic argmax) ...")
        gate_lines, fidelity, gate_count = run_inference_ppo(ppo_agent, env, target_sv)
        png_path, qasm_path = save_circuit_outputs(env, 'ppo')
        print_agent_result(
            'ppo', gate_lines, fidelity, gate_count,
            png_path, qasm_path, config_1q.FIDELITY_THRESHOLD, env,
        )
        comparison_results['ppo'] = {
            'fidelity':   fidelity,
            'gate_count': gate_count,
            'success':    fidelity >= config_1q.FIDELITY_THRESHOLD,
        }

    # ── Comparison table (only when both agents ran) ──────────────────────────
    print_comparison_table(comparison_results, config_1q.FIDELITY_THRESHOLD)

    print(f"\n[synthesize] Done. Output files saved to: {OUTPUTS_DIR}")


if __name__ == '__main__':
    main()
