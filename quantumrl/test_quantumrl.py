"""
test_quantumrl.py
-----------------
Unit tests for QuantumRL using pytest.
Tests environment, fidelity calculations, DQN agent, PPO agent, and end-to-end pipeline execution.
"""

import os
import pytest
import numpy as np
import torch
from qiskit.quantum_info import Statevector

from config import Config
from quantum_env import QuantumCircuitEnv
from dqn_agent import DQNAgent, DuelingQNetwork as QNetwork, PrioritizedReplayBuffer as ReplayBuffer
from ppo_agent import PPOAgent, ActorCritic, RolloutBuffer
from utils import compute_fidelity, generate_random_statevector, generate_target_states, encode_state


class TestConfig:
    def test_default_config(self):
        cfg = Config()
        assert cfg.NUM_QUBITS in (1, 2)
        assert cfg.MAX_STEPS in (15, 20)
        assert cfg.FIDELITY_THRESHOLD == 0.99
        assert 'H' in cfg.GATES
        assert 'X' in cfg.GATES


class TestUtils:
    def test_fidelity_identical(self):
        sv = np.array([1.0, 0.0], dtype=np.complex128)
        fid = compute_fidelity(sv, sv)
        assert pytest.approx(fid, abs=1e-6) == 1.0

    def test_fidelity_orthogonal(self):
        sv1 = np.array([1.0, 0.0], dtype=np.complex128)
        sv2 = np.array([0.0, 1.0], dtype=np.complex128)
        fid = compute_fidelity(sv1, sv2)
        assert pytest.approx(fid, abs=1e-6) == 0.0

    def test_generate_random_statevector(self):
        sv = generate_random_statevector(n_qubits=2, seed=42)
        assert len(sv) == 4
        norm = np.linalg.norm(sv)
        assert pytest.approx(norm, abs=1e-6) == 1.0

    def test_encode_state(self):
        sv1 = np.array([1.0, 0.0], dtype=np.complex128)
        sv2 = np.array([0.0, 1.0], dtype=np.complex128)
        encoded = encode_state(sv1, sv2, fidelity=0.5, step=1, max_steps=15)
        assert encoded.shape == (4 * (2 ** 1) + 2,)
        assert encoded.dtype == np.float32


class TestQuantumEnv:
    def test_env_init_and_reset(self):
        cfg = Config()
        target_sv = generate_random_statevector(cfg.NUM_QUBITS, seed=10)
        env = QuantumCircuitEnv(cfg, target_sv=target_sv)
        
        obs, info = env.reset()
        assert obs.shape == (4 * (2 ** cfg.NUM_QUBITS) + 2,)
        assert env.steps == 0
        assert env.current_circuit is not None

    def test_env_step(self):
        cfg = Config()
        target_sv = generate_random_statevector(cfg.NUM_QUBITS, seed=10)
        env = QuantumCircuitEnv(cfg, target_sv=target_sv)
        env.reset()

        action = 0  # e.g., H gate on qubit 0
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (4 * (2 ** cfg.NUM_QUBITS) + 2,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert 'fidelity' in info
        assert 'steps' in info
        assert info['steps'] == 1


class TestDQNAgent:
    def test_qnetwork(self):
        net = QNetwork(obs_size=10, action_size=7, hidden_size=64)
        x = torch.randn(4, 10)
        q_vals = net(x)
        assert q_vals.shape == (4, 7)

    def test_replay_buffer(self):
        buf = ReplayBuffer(capacity=10)
        s = np.zeros(10, dtype=np.float32)
        buf.push(s, 0, 1.0, s, False)
        assert len(buf) == 1

        states, actions, rewards, next_states, dones, weights, indices = buf.sample(1)
        assert states.shape == (1, 10)
        assert actions.shape == (1,)
        assert rewards.shape == (1,)

    def test_agent_action_selection(self):
        cfg = Config()
        agent = DQNAgent(obs_size=10, action_size=7, config=cfg)
        state = np.zeros(10, dtype=np.float32)
        action = agent.select_action(state)
        assert 0 <= action < 7


class TestPPOAgent:
    def test_actor_critic(self):
        ac = ActorCritic(obs_size=10, action_size=7, hidden_size=64)
        x = torch.randn(4, 10)
        logits, val = ac(x)
        probs = torch.softmax(logits, dim=-1)
        assert probs.shape == (4, 7)
        assert val.shape == (4, 1)
        assert pytest.approx(probs.sum(dim=-1).detach().numpy(), abs=1e-5) == np.ones(4)

    def test_agent_update(self):
        cfg = Config(PPO_EPOCHS=2, PPO_MINI_BATCH_SIZE=2)
        device = torch.device('cpu')
        agent = PPOAgent(obs_size=10, action_size=7, config=cfg, device=device)
        buf = RolloutBuffer(rollout_steps=4, obs_size=10, device=device)
        
        obs_t = torch.zeros(10)
        buf.add(obs_t, 0, torch.tensor(-1.0), 1.0, False, torch.tensor(0.5))
        buf.add(obs_t, 1, torch.tensor(-1.0), 0.5, True, torch.tensor(0.2))
        buf.add(obs_t, 0, torch.tensor(-1.0), 0.8, False, torch.tensor(0.4))
        buf.add(obs_t, 1, torch.tensor(-1.0), 0.3, True, torch.tensor(0.1))
        buf.compute_returns_and_advantages(torch.tensor(0.0), gamma=0.99, gae_lambda=0.95)
        
        losses = agent.update(buf)
        assert 'policy_loss' in losses
        assert 'value_loss' in losses


class TestIntegration:
    def test_mini_training_dqn(self):
        cfg = Config(DQN_EPISODES=2, DQN_BATCH_SIZE=2, DQN_BUFFER_SIZE=100)
        env = QuantumCircuitEnv(cfg)
        target_sv = generate_random_statevector(cfg.NUM_QUBITS, seed=1)
        obs, _ = env.reset(target_sv=target_sv)
        
        agent = DQNAgent(obs_size=len(obs), action_size=env.action_space.n, config=cfg)
        
        for _ in range(5):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.buffer.push(obs, action, reward, next_obs, float(terminated or truncated))
            agent.update()
            obs = next_obs
            if terminated or truncated:
                break

    def test_mini_training_ppo(self):
        cfg = Config(PPO_EPISODES=2, PPO_ROLLOUT_STEPS=4, PPO_EPOCHS=1)
        device = torch.device('cpu')
        env = QuantumCircuitEnv(cfg)
        target_sv = generate_random_statevector(cfg.NUM_QUBITS, seed=1)
        obs, _ = env.reset(target_sv=target_sv)

        agent = PPOAgent(obs_size=len(obs), action_size=env.action_space.n, config=cfg, device=device)
        
        action, log_prob, entropy, value = agent.select_action(torch.FloatTensor(obs).to(device))
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(action, int)
