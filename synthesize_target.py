"""
synthesize_target.py
--------------------
Standalone inference script: Target State → Quantum Circuit

Given a user-specified target quantum state, runs both trained agents (DQN and
PPO) and automatically selects the best result using a deterministic four-step
rule.  The non-selected agent's result is retained as secondary data.

Supports two saved-model layouts automatically (detected by key inspection):

  LEGACY (original 1-qubit models):
    obs=10, action=76, hidden=512/256
    DQN keys: net.0.*, net.3.*, net.6.*, net.9.*
    PPO keys: actor.0.*, ..., actor.9.* / critic.0.*, ..., critic.9.*

  NEW (Dueling DQN + PPO with hidden=768/384):
    obs=10, action=76, hidden=768/384
    DQN keys: feature_trunk.*, value_stream.*, advantage_stream.*
    PPO keys: actor.0.*, ..., actor.9.* / critic.0.*, ..., critic.9.*

Usage
-----
  # Interactive mode (no arguments):
      python synthesize_target.py

  # CLI mode:
      python synthesize_target.py --target plus
      python synthesize_target.py --target custom --amplitudes "0.707,0.707"
      python synthesize_target.py --target random --use-best

Named presets (1-qubit):
  zero    : |0⟩  — ground state
  one     : |1⟩  — excited state
  plus    : |+⟩ = (|0⟩+|1⟩)/√2  — uniform superposition
  minus   : |−⟩ = (|0⟩−|1⟩)/√2  — phase state
  i_state : +i Y-eigenstate = (|0⟩+i|1⟩)/√2
  random  : Single Haar-random state (quick demo)
"""

import argparse
import math
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ── Resolve quantumrl package path ────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
QUANTUMRL_DIR = os.path.join(SCRIPT_DIR, 'quantumrl')
sys.path.insert(0, QUANTUMRL_DIR)

from config import Config
from quantum_env import QuantumCircuitEnv
from utils import best_checkpoint_path, generate_random_statevector

# ── Constants ─────────────────────────────────────────────────────────────────
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, 'outputs')
_INV_SQRT2  = 1.0 / math.sqrt(2.0)

# Named preset statevectors (complex128, 1-qubit, unit-norm)
PRESETS: Dict[str, Optional[np.ndarray]] = {
    'zero':    np.array([1.0+0j, 0.0+0j],           dtype=np.complex128),
    'one':     np.array([0.0+0j, 1.0+0j],           dtype=np.complex128),
    'plus':    np.array([_INV_SQRT2, _INV_SQRT2],    dtype=np.complex128),
    'minus':   np.array([_INV_SQRT2, -_INV_SQRT2],   dtype=np.complex128),
    'i_state': np.array([_INV_SQRT2, _INV_SQRT2*1j], dtype=np.complex128),
    'random':  None,
}

NORM_TOLERANCE = 1e-5


# ── 1-qubit Config loader ──────────────────────────────────────────────────────
def _make_one_qubit_config() -> Config:
    """Return a standalone 1-qubit Config instance."""
    try:
        from configs.config_1qubit import Config as Config1Qubit
        return Config1Qubit()
    except ImportError:
        cfg = Config()
        cfg.NUM_QUBITS = 1
        return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Result data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """All data produced by one agent's inference run."""
    agent_name:  str
    fidelity:    float
    gate_count:  int
    gate_lines:  List[str]
    png_path:    str
    qasm_path:   str
    success:     bool   # fidelity >= FIDELITY_THRESHOLD


@dataclass
class SynthesisResult:
    """
    Structured output from synthesize().  Suitable for use by a calling API layer.

    Attributes
    ----------
    best_agent       : 'dqn' or 'ppo' — the agent selected as primary result.
    selection_reason : Human-readable string explaining why this agent was chosen.
    best_result      : Full AgentResult for the selected agent.
    other_result     : Full AgentResult for the non-selected agent.
    target_label     : Display name of the target state (e.g. 'plus', 'custom').
    target_sv        : The target statevector that was used.
    """
    best_agent:       str
    selection_reason: str
    best_result:      AgentResult
    other_result:     AgentResult
    target_label:     str
    target_sv:        np.ndarray


# ──────────────────────────────────────────────────────────────────────────────
# Selection rule
# ──────────────────────────────────────────────────────────────────────────────

def select_best_result(
    dqn_result: AgentResult,
    ppo_result: AgentResult,
    fidelity_threshold: float,
    tolerance: float = 1e-6,
) -> Tuple[AgentResult, AgentResult, str]:
    """
    Apply the four-step rule to determine which agent produced the better result.

    Step 1 — Success status differs:
        If exactly one agent reached fidelity_threshold, that agent wins.

    Step 2 — Both succeed or both fail:
        The agent with higher fidelity wins.

    Step 3 — Fidelity tied within tolerance:
        The agent with fewer gates wins.

    Step 4 — Still tied:
        DQN wins as a deterministic, reproducible default.

    Parameters
    ----------
    dqn_result          : AgentResult for the DQN agent.
    ppo_result          : AgentResult for the PPO agent.
    fidelity_threshold  : Value from config.FIDELITY_THRESHOLD (typically 0.99).
    tolerance           : Floating-point tolerance for fidelity equality check.

    Returns
    -------
    (best: AgentResult, other: AgentResult, reason: str)
    """
    dqn_ok = dqn_result.success
    ppo_ok = ppo_result.success

    # Step 1 — exactly one agent succeeded
    if dqn_ok and not ppo_ok:
        return dqn_result, ppo_result, "dqn succeeded, ppo did not"
    if ppo_ok and not dqn_ok:
        return ppo_result, dqn_result, "ppo succeeded, dqn did not"

    # Step 2 — both succeeded or both failed; compare fidelity
    fid_diff = dqn_result.fidelity - ppo_result.fidelity
    if abs(fid_diff) > tolerance:
        if fid_diff > 0:
            status = "both succeeded" if (dqn_ok and ppo_ok) else "both partial"
            return dqn_result, ppo_result, f"{status}, dqn had higher fidelity"
        else:
            status = "both succeeded" if (dqn_ok and ppo_ok) else "both partial"
            return ppo_result, dqn_result, f"{status}, ppo had higher fidelity"

    # Step 3 — fidelity tied within tolerance; fewer gates wins
    if dqn_result.gate_count != ppo_result.gate_count:
        if dqn_result.gate_count < ppo_result.gate_count:
            return dqn_result, ppo_result, "tied on fidelity, dqn had fewer gates"
        else:
            return ppo_result, dqn_result, "tied on fidelity, ppo had fewer gates"

    # Step 4 — fully tied; prefer DQN as deterministic default
    return dqn_result, ppo_result, "fully tied — defaulting to dqn"


# ──────────────────────────────────────────────────────────────────────────────
# Network definitions — two layouts, auto-detected from saved weights
# ──────────────────────────────────────────────────────────────────────────────

def _make_mlp_branch(in_size: int, h1: int, h2: int, out_size: int) -> nn.Sequential:
    """3-hidden-layer MLP matching saved .pth Sequential index layout."""
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
    def __init__(self, obs_size: int, action_size: int, h1: int, h2: int):
        super().__init__()
        self.net = _make_mlp_branch(obs_size, h1, h2, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DuelingQNet(nn.Module):
    def __init__(self, obs_size: int, action_size: int, h1: int, h2: int):
        super().__init__()
        self.feature_trunk = nn.Sequential(
            nn.Linear(obs_size, h1), nn.LayerNorm(h1), nn.LeakyReLU(0.01),
            nn.Linear(h1, h1),       nn.LayerNorm(h1), nn.LeakyReLU(0.01),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(h1, h2), nn.LayerNorm(h2), nn.LeakyReLU(0.01), nn.Linear(h2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(h1, h2), nn.LayerNorm(h2), nn.LeakyReLU(0.01),
            nn.Linear(h2, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature_trunk(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + (a - a.mean(dim=1, keepdim=True))


class _ActorCritic(nn.Module):
    def __init__(self, obs_size: int, action_size: int, h1: int, h2: int):
        super().__init__()
        self.actor  = _make_mlp_branch(obs_size, h1, h2, action_size)
        self.critic = _make_mlp_branch(obs_size, h1, h2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(x), self.critic(x)


def _infer_hidden_dims(state_dict: dict) -> Tuple[int, int]:
    """Infer h1, h2 from saved state_dict by inspecting weight shapes."""
    for key in state_dict:
        if key.endswith('.0.weight'):
            h1     = state_dict[key].shape[0]
            prefix = key[: key.rfind('.0.weight')]
            h2_key = f'{prefix}.6.weight'
            if h2_key in state_dict:
                return h1, state_dict[h2_key].shape[0]
            break
    for key in state_dict:
        if key == 'value_stream.0.weight':
            ft_key = 'feature_trunk.0.weight'
            h1 = state_dict[ft_key].shape[0] if ft_key in state_dict else 768
            return h1, state_dict[key].shape[0]
    return 512, 256


# ──────────────────────────────────────────────────────────────────────────────
# Inference wrappers
# ──────────────────────────────────────────────────────────────────────────────

class DQNInferenceAgent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net: Optional[nn.Module] = None

    def load(self, path: str) -> None:
        sd          = torch.load(path, map_location=self.device, weights_only=True)
        is_dueling  = any(k.startswith('feature_trunk') for k in sd)
        obs_size    = sd['feature_trunk.0.weight'].shape[1] if is_dueling else sd['net.0.weight'].shape[1]
        act_size    = sd['advantage_stream.3.weight'].shape[0] if is_dueling else sd['net.9.weight'].shape[0]
        h1, h2      = _infer_hidden_dims(sd)
        if is_dueling:
            self.net = _DuelingQNet(obs_size, act_size, h1, h2).to(self.device)
            arch     = f'Dueling h1={h1} h2={h2}'
        else:
            self.net = _FlatQNet(obs_size, act_size, h1, h2).to(self.device)
            arch     = f'FlatNet h1={h1} h2={h2}'
        self.net.load_state_dict(sd)
        self.net.eval()
        print(f"[DQNAgent] Model loaded <- {path}  [{arch}]")

    def select_action(self, state: np.ndarray) -> int:
        t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.net(t).argmax(dim=1).item())


class PPOInferenceAgent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ac: Optional[nn.Module] = None

    def load(self, path: str) -> None:
        sd       = torch.load(path, map_location=self.device, weights_only=True)
        obs_size = sd['actor.0.weight'].shape[1]
        act_size = sd['actor.9.weight'].shape[0]
        h1, h2   = _infer_hidden_dims(sd)
        self.ac  = _ActorCritic(obs_size, act_size, h1, h2).to(self.device)
        self.ac.load_state_dict(sd)
        self.ac.eval()
        print(f"[PPOAgent] Model loaded <- {path}  [ActorCritic h1={h1} h2={h2}]")

    def select_action_greedy(self, state: np.ndarray) -> int:
        t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.ac(t)
        return int(logits.argmax(dim=1).item())


# ──────────────────────────────────────────────────────────────────────────────
# Target state handling
# ──────────────────────────────────────────────────────────────────────────────

def validate_statevector(sv: np.ndarray, n_qubits: int) -> np.ndarray:
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
            "Please normalize your amplitudes so they satisfy |ψ|² = 1."
        )
    return sv.astype(np.complex128)


def parse_custom_amplitudes(raw: str, n_qubits: int) -> np.ndarray:
    raw   = raw.strip().strip('"').strip("'")
    parts = [p.strip().strip('()') for p in raw.split(',')]
    if len(parts) != 2 ** n_qubits:
        raise ValueError(
            f"Expected {2**n_qubits} comma-separated amplitudes, got {len(parts)}."
        )
    try:
        amplitudes = [complex(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"Could not parse amplitude(s) as complex numbers: {exc}\n"
            "Use Python complex notation, e.g. '0.707', '0.5+0.5j'."
        ) from exc
    return validate_statevector(np.array(amplitudes, dtype=np.complex128), n_qubits)


def get_target_state(
    target_name: str, n_qubits: int, cli_amplitudes: Optional[str] = None
) -> Tuple[np.ndarray, str]:
    if target_name == 'random':
        return validate_statevector(generate_random_statevector(n_qubits), n_qubits), 'random (Haar)'
    if target_name in PRESETS and PRESETS[target_name] is not None:
        return validate_statevector(PRESETS[target_name].copy(), n_qubits), target_name
    if target_name == 'custom':
        if cli_amplitudes is not None:
            return parse_custom_amplitudes(cli_amplitudes, n_qubits), 'custom'
        expected = 2 ** n_qubits
        print(f"\nEnter {expected} comma-separated complex amplitudes for a {n_qubits}-qubit state.")
        print("  Example (1 qubit): 0.707, 0.707")
        raw = input("  Amplitudes: ").strip()
        return parse_custom_amplitudes(raw, n_qubits), 'custom'
    raise ValueError(
        f"Unknown target '{target_name}'. "
        f"Valid options: {list(PRESETS.keys()) + ['custom']}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Gate formatting
# ──────────────────────────────────────────────────────────────────────────────

def format_gate(step: int, gate_name: str, qubit_or_pair, angle: Optional[float]) -> str:
    if gate_name == 'CNOT':
        ctrl, tgt = qubit_or_pair
        return f"  Step {step:2d}: CNOT  ctrl=qubit {ctrl}  tgt=qubit {tgt}"
    if angle is not None:
        return (
            f"  Step {step:2d}: {gate_name}({angle:.4f} rad = {math.degrees(angle):.2f}°)"
            f"  on qubit {qubit_or_pair}"
        )
    return f"  Step {step:2d}: {gate_name}  on qubit {qubit_or_pair}"


# ──────────────────────────────────────────────────────────────────────────────
# Inference runners
# ──────────────────────────────────────────────────────────────────────────────

def run_inference_dqn(
    agent: DQNInferenceAgent, env: QuantumCircuitEnv, target_sv: np.ndarray,
) -> Tuple[List[str], float, int]:
    obs, _ = env.reset(target_sv=target_sv)
    gate_lines: List[str] = []
    done = False
    final_fidelity, final_steps = 0.0, 0
    while not done:
        action = agent.select_action(obs)
        gate_lines.append(format_gate(final_steps + 1, *env.action_list[action]))
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        final_fidelity, final_steps = info['fidelity'], info['steps']
    return gate_lines, final_fidelity, final_steps


def run_inference_ppo(
    agent: PPOInferenceAgent, env: QuantumCircuitEnv, target_sv: np.ndarray,
) -> Tuple[List[str], float, int]:
    obs, _ = env.reset(target_sv=target_sv)
    gate_lines: List[str] = []
    done = False
    final_fidelity, final_steps = 0.0, 0
    while not done:
        action = agent.select_action_greedy(obs)
        gate_lines.append(format_gate(final_steps + 1, *env.action_list[action]))
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        final_fidelity, final_steps = info['fidelity'], info['steps']
    return gate_lines, final_fidelity, final_steps


# ──────────────────────────────────────────────────────────────────────────────
# File output helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_circuit_outputs(env: QuantumCircuitEnv, agent_name: str) -> Tuple[str, str]:
    """Save circuit PNG and QASM to outputs/. Returns (png_path, qasm_path)."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    png_path  = os.path.join(OUTPUTS_DIR, f'circuit_{agent_name}.png')
    qasm_path = os.path.join(OUTPUTS_DIR, f'circuit_{agent_name}.qasm')

    try:
        plt.switch_backend('Agg')
        fig = env.current_circuit.draw('mpl', fold=-1)
        if fig is not None and hasattr(fig, 'savefig'):
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            env.current_circuit.draw('mpl', filename=png_path, fold=-1)
    except Exception as exc:
        print(f"  [warn] Could not save PNG: {exc}")
        png_path = f"(not saved — {exc})"

    try:
        import qiskit.qasm2 as qasm2
        with open(qasm_path, 'w', encoding='utf-8') as f:
            f.write(qasm2.dumps(env.current_circuit))
    except Exception:
        try:
            with open(qasm_path, 'w', encoding='utf-8') as f:
                f.write(env.current_circuit.qasm())
        except Exception as exc2:
            print(f"  [warn] Could not save QASM: {exc2}")
            qasm_path = f"(not saved — {exc2})"

    return png_path, qasm_path


# ──────────────────────────────────────────────────────────────────────────────
# Agent loading
# ──────────────────────────────────────────────────────────────────────────────

def load_dqn(obs_size: int, action_size: int, model_path: str) -> DQNInferenceAgent:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"DQN model not found at: {model_path}\n"
            "Run train_dqn.py first to produce the saved weights."
        )
    agent = DQNInferenceAgent()
    agent.load(model_path)
    return agent


def load_ppo(obs_size: int, action_size: int, model_path: str) -> PPOInferenceAgent:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"PPO model not found at: {model_path}\n"
            "Run train_ppo.py first to produce the saved weights."
        )
    agent = PPOInferenceAgent()
    agent.load(model_path)
    return agent


# ──────────────────────────────────────────────────────────────────────────────
# Console printing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_agent_result(result: AgentResult, threshold: float, env: QuantumCircuitEnv) -> None:
    status = "SUCCESS" if result.success else "PARTIAL (below threshold)"
    bar    = "=" * 60
    print(f"\n{bar}")
    print(f"  Agent  : {result.agent_name.upper()}")
    print(f"  Status : {status}")
    print(f"  Final fidelity : {result.fidelity:.6f}  (threshold = {threshold:.2f})")
    print(f"  Gate count     : {result.gate_count}")
    print(bar)
    print("\n  Gate Sequence:")
    if result.gate_lines:
        for line in result.gate_lines:
            print(line)
    else:
        print("  (no gates applied — already at target state)")
    print("\n  ASCII Circuit Diagram:\n")
    try:
        env.render()
    except Exception:
        try:
            print(str(env.current_circuit.draw('text'))
                  .encode('ascii', errors='replace').decode('ascii'))
        except Exception:
            print("  (ASCII diagram omitted)")
    print(f"\n  Saved outputs:")
    print(f"    PNG  : {result.png_path}")
    print(f"    QASM : {result.qasm_path}")


def _print_best_result(synthesis: SynthesisResult, threshold: float) -> None:
    """Print the BEST RESULT summary section."""
    b   = synthesis.best_result
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  BEST RESULT")
    print(f"{bar}")
    print(f"  Selected agent : {synthesis.best_agent.upper()}")
    print(f"  Reason         : {synthesis.selection_reason}")
    print(f"  Fidelity       : {b.fidelity:.6f}  "
          f"({'SUCCESS' if b.success else 'PARTIAL'})")
    print(f"  Gate count     : {b.gate_count}")
    print(f"\n  Gate Sequence:")
    if b.gate_lines:
        for line in b.gate_lines:
            print(line)
    else:
        print("  (no gates applied)")
    print(f"\n  Output files:")
    print(f"    PNG  : {b.png_path}")
    print(f"    QASM : {b.qasm_path}")
    o = synthesis.other_result
    print(f"\n  Other agent ({o.agent_name.upper()}) : "
          f"fidelity={o.fidelity:.6f}  gates={o.gate_count}  "
          f"({'SUCCESS' if o.success else 'PARTIAL'})")
    print(bar)


# ──────────────────────────────────────────────────────────────────────────────
# Core synthesis function  (importable)
# ──────────────────────────────────────────────────────────────────────────────

def synthesize(
    target_sv: np.ndarray,
    target_label: str,
    dqn_model_path: str,
    ppo_model_path: str,
    fidelity_threshold: float,
    print_output: bool = True,
) -> SynthesisResult:
    """
    Run both DQN and PPO against ``target_sv`` and return a SynthesisResult.

    This is the importable core function intended for use by a web API layer
    or any other caller that wants structured data rather than console output.

    Parameters
    ----------
    target_sv           : Validated, normalised 1-qubit statevector (complex128).
    target_label        : Human-readable name for the target (e.g. 'plus').
    dqn_model_path      : Absolute path to the DQN .pth file.
    ppo_model_path      : Absolute path to the PPO .pth file.
    fidelity_threshold  : Value from Config.FIDELITY_THRESHOLD.
    print_output        : If True, prints all console sections (default True for CLI use).

    Returns
    -------
    SynthesisResult
        best_agent, selection_reason, best_result, other_result, target_label,
        target_sv — see SynthesisResult dataclass for full field documentation.
    """
    config_1q   = _make_one_qubit_config()
    env         = QuantumCircuitEnv(config_1q)
    obs_size    = env.observation_space.shape[0]
    action_size = env.action_space.n

    # ── Load both agents ──────────────────────────────────────────────────────
    dqn_agent = load_dqn(obs_size, action_size, dqn_model_path)
    ppo_agent = load_ppo(obs_size, action_size, ppo_model_path)

    # ── Run DQN inference ─────────────────────────────────────────────────────
    if print_output:
        print("\n[synthesize] Running DQN inference (greedy, epsilon=0) ...")
    dqn_gate_lines, dqn_fidelity, dqn_gate_count = run_inference_dqn(
        dqn_agent, env, target_sv
    )
    dqn_png, dqn_qasm = save_circuit_outputs(env, 'dqn')
    dqn_result = AgentResult(
        agent_name='dqn',
        fidelity=dqn_fidelity,
        gate_count=dqn_gate_count,
        gate_lines=dqn_gate_lines,
        png_path=dqn_png,
        qasm_path=dqn_qasm,
        success=(dqn_fidelity >= fidelity_threshold),
    )
    if print_output:
        _print_agent_result(dqn_result, fidelity_threshold, env)

    # ── Run PPO inference ─────────────────────────────────────────────────────
    if print_output:
        print("\n[synthesize] Running PPO inference (greedy, deterministic argmax) ...")
    ppo_gate_lines, ppo_fidelity, ppo_gate_count = run_inference_ppo(
        ppo_agent, env, target_sv
    )
    ppo_png, ppo_qasm = save_circuit_outputs(env, 'ppo')
    ppo_result = AgentResult(
        agent_name='ppo',
        fidelity=ppo_fidelity,
        gate_count=ppo_gate_count,
        gate_lines=ppo_gate_lines,
        png_path=ppo_png,
        qasm_path=ppo_qasm,
        success=(ppo_fidelity >= fidelity_threshold),
    )
    if print_output:
        _print_agent_result(ppo_result, fidelity_threshold, env)

    # ── Select best ───────────────────────────────────────────────────────────
    best_r, other_r, reason = select_best_result(
        dqn_result, ppo_result, fidelity_threshold
    )

    synthesis = SynthesisResult(
        best_agent=best_r.agent_name,
        selection_reason=reason,
        best_result=best_r,
        other_result=other_r,
        target_label=target_label,
        target_sv=target_sv,
    )

    if print_output:
        _print_best_result(synthesis, fidelity_threshold)

    return synthesis


# ──────────────────────────────────────────────────────────────────────────────
# Target prompt (interactive)
# ──────────────────────────────────────────────────────────────────────────────

def prompt_target() -> str:
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


# ──────────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    valid_targets = list(PRESETS.keys()) + ['custom']
    parser = argparse.ArgumentParser(
        prog='synthesize_target.py',
        description=textwrap.dedent("""\
            Target-state-to-circuit synthesis using trained RL agents.
            Both DQN and PPO run on every invocation; the best result is
            selected automatically and presented as the primary output.
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
        '--amplitudes', '-amp',
        default=None,
        metavar='AMPLITUDES',
        help=(
            "Comma-separated complex amplitudes for --target custom. "
            "Example: '0.707,0.707'  (1-qubit system needs 2 values)."
        ),
    )
    parser.add_argument(
        '--use-best',
        action='store_true',
        default=False,
        help=(
            'Load the best checkpoint (dqn_model_best.pth / ppo_model_best.pth) '
            'instead of the final saved model.'
        ),
    )
    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser  = build_parser()
    args    = parser.parse_args()

    # interactive mode when no target is given
    interactive = args.target is None

    if interactive:
        print("=" * 60)
        print("  QuantumRL — Target-State Circuit Synthesis")
        print("  (Interactive Mode — run with --help for CLI flags)")
        print("=" * 60)
        target_name = prompt_target()
    else:
        target_name = args.target

    # ── Config and environment info ───────────────────────────────────────────
    config_1q   = _make_one_qubit_config()
    n_qubits    = config_1q.NUM_QUBITS

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
    print(f"  Running      : DQN + PPO (both agents, best selected automatically)")
    print(f"{'=' * 60}")

    # ── Resolve model paths ───────────────────────────────────────────────────
    def _resolve(rel_path: str) -> str:
        for candidate in [
            os.path.join(QUANTUMRL_DIR, rel_path),
            os.path.join(SCRIPT_DIR, rel_path),
            rel_path,
        ]:
            if os.path.exists(candidate):
                return candidate
        return os.path.join(QUANTUMRL_DIR, rel_path)

    dqn_model_path = _resolve(config_1q.DQN_MODEL_PATH)
    ppo_model_path = _resolve(config_1q.PPO_MODEL_PATH)

    if args.use_best:
        dqn_model_path = _resolve(best_checkpoint_path(config_1q.DQN_MODEL_PATH))
        ppo_model_path = _resolve(best_checkpoint_path(config_1q.PPO_MODEL_PATH))
        print(f"[synthesize] --use-best: loading best checkpoints.")
        print(f"[synthesize]   DQN: {dqn_model_path}")
        print(f"[synthesize]   PPO: {ppo_model_path}")

    # ── Run synthesis (both agents, select best) ──────────────────────────────
    try:
        result = synthesize(
            target_sv=target_sv,
            target_label=target_label,
            dqn_model_path=dqn_model_path,
            ppo_model_path=ppo_model_path,
            fidelity_threshold=config_1q.FIDELITY_THRESHOLD,
            print_output=True,
        )
    except FileNotFoundError as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[synthesize] Done. Output files saved to: {OUTPUTS_DIR}")


if __name__ == '__main__':
    main()
