from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tqdm import trange  # type: ignore
except Exception:  # pragma: no cover
    trange = None  # type: ignore

from agent_memory import MemoryManager
from experiments.adapters import get_adapter
from experiments.models import PolicyValueNet
from experiments.rl_interface import MemoryRLEnvironment, MemoryRLConfig


def build_feature_vector(state, capacity: int) -> List[float]:
    cap = max(capacity, 1)
    features = [
        state.heap_used / cap,
        state.heap_free / cap,
        state.fragmentation,
        state.objects / max(1.0, cap / 64.0),
        state.last_reward,
        1.0 if state.last_hit else 0.0,
        state.last_gc_pause,
        math.tanh((state.metadata.get("importance", 0.0) or 0.0) * 2.0),
        math.tanh((state.metadata.get("ttl", -1.0) or -1.0) / 300.0),
        state.metadata.get("evidence_prob", 0.0),
        state.metadata.get("progress_signal", 0.0),
        state.metadata.get("action_cost", 0.0),
        min(state.step / 1000.0, 1.0),
    ]
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL policy for MemoryManager actions")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps-per-episode", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capacity", type=int, default=32768)
    parser.add_argument("--query-interval", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/rl_runs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--reward-hit", type=float, default=1.0)
    parser.add_argument("--reward-miss", type=float, default=-0.5)
    parser.add_argument("--reward-gc-scale", type=float, default=1e-6)
    parser.add_argument("--penalty-fragmentation", type=float, default=0.2)
    parser.add_argument("--penalty-pause", type=float, default=0.5)
    parser.add_argument("--progress-bar", action="store_true", help="Display tqdm progress for episodes")
    parser.add_argument("--stream", type=str, default="simulated", choices=["simulated", "hotpot"])
    parser.add_argument("--hotpot-split", type=str, default="train[:200]")
    parser.add_argument("--hotpot-max-samples", type=int, default=200)
    parser.add_argument("--hotpot-context-tokens", type=int, default=0)
    parser.add_argument("--hotpot-filler-path", type=str, default="data/text8")
    parser.add_argument("--hotpot-filler-chunk", type=int, default=4096)
    parser.add_argument("--hotpot-filler-overlap", type=int, default=128)
    parser.add_argument("--adapter", type=str, default="", help="Optional benchmark adapter name (e.g., ruler_hotpot)")
    parser.add_argument(
        "--policy-mode",
        type=str,
        default="combined",
        choices=["combined", "gc", "query", "write", "alloc"],
        help="Select which action head to train (combined, gc, query, write, alloc).",
    )
    parser.add_argument("--skip-query-penalty", type=float, default=0.05)
    parser.add_argument("--skip-write-penalty", type=float, default=0.02)
    parser.add_argument("--allocator-switch-penalty", type=float, default=0.01)
    parser.add_argument("--query-hit-bonus", type=float, default=0.3)
    parser.add_argument("--query-cost-weight", type=float, default=0.05)
    parser.add_argument("--write-evidence-bonus", type=float, default=0.4)
    parser.add_argument("--write-progress-bonus", type=float, default=0.2)
    parser.add_argument("--alloc-fragmentation-bonus", type=float, default=0.3)
    parser.add_argument("--gc-hit-bonus", type=float, default=0.3)
    parser.add_argument("--gc-hit-drop-penalty", type=float, default=0.25)
    parser.add_argument("--gc-pressure-threshold", type=float, default=0.82)
    parser.add_argument("--gc-pressure-penalty", type=float, default=0.05)
    parser.add_argument("--hit-history-window", type=int, default=16)
    parser.add_argument("--support-hit-reward", type=float, default=1.5)
    parser.add_argument("--support-miss-penalty", type=float, default=-0.2)
    parser.add_argument("--distractor-penalty", type=float, default=0.1)
    parser.add_argument("--filler-penalty", type=float, default=0.15)
    parser.add_argument("--support-bonus-k", type=int, default=5)
    parser.add_argument("--answer-hit-reward", type=float, default=1.0)
    parser.add_argument("--answer-miss-penalty", type=float, default=-0.2)
    parser.add_argument("--checkpoint", type=str, default="", help="Path to policy checkpoint for warm start/eval")
    parser.add_argument("--eval-only", action="store_true", help="Run greedy evaluation without updates")
    parser.add_argument("--eval-log", type=str, default="", help="Optional JSONL path for evaluation results")
    return parser.parse_args()


def select_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dir(base: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = Path(base) / f"rl_run_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    output_root = ensure_output_dir(args.output)

    config = MemoryRLConfig(
        capacity=args.capacity,
        query_interval=args.query_interval,
        reward_hit=args.reward_hit,
        reward_miss=args.reward_miss,
        gc_reward_scale=args.reward_gc_scale,
        penalty_fragmentation=args.penalty_fragmentation,
        penalty_pause=args.penalty_pause,
        seed=args.seed,
        top_k=args.top_k,
        skip_query_penalty=args.skip_query_penalty,
        skip_write_penalty=args.skip_write_penalty,
        allocator_switch_penalty=args.allocator_switch_penalty,
        query_hit_bonus=args.query_hit_bonus,
        query_cost_weight=args.query_cost_weight,
        write_evidence_bonus=args.write_evidence_bonus,
        write_progress_bonus=args.write_progress_bonus,
        alloc_fragmentation_bonus=args.alloc_fragmentation_bonus,
        gc_hit_bonus=args.gc_hit_bonus,
        gc_hit_drop_penalty=args.gc_hit_drop_penalty,
        gc_pressure_threshold=args.gc_pressure_threshold,
        gc_pressure_penalty=args.gc_pressure_penalty,
        hit_history_window=args.hit_history_window,
        support_hit_reward=args.support_hit_reward,
        support_miss_penalty=args.support_miss_penalty,
        distractor_penalty=args.distractor_penalty,
        filler_penalty=args.filler_penalty,
        support_bonus_k=args.support_bonus_k,
        answer_hit_reward=args.answer_hit_reward,
        answer_miss_penalty=args.answer_miss_penalty,
    )

    experience_iter = None
    adapter = None
    if args.adapter:
        adapter = get_adapter(
            args.adapter,
            config={
                "split": args.hotpot_split,
                "max_samples": args.hotpot_max_samples,
                "long_context_tokens": args.hotpot_context_tokens,
                "filler_path": args.hotpot_filler_path,
                "filler_chunk_size": args.hotpot_filler_chunk,
                "filler_overlap": args.hotpot_filler_overlap,
            },
        )
        if adapter is None:
            raise ValueError(f"Failed to instantiate adapter '{args.adapter}'")
        adapter.reset()
        experience_iter = adapter.experience_stream()
    elif args.stream == "hotpot":
        from experiments.hotpot_rl_stream import HotpotStreamConfig, hotpot_long_stream

        stream_cfg = HotpotStreamConfig(
            split=args.hotpot_split,
            max_samples=args.hotpot_max_samples,
            long_context_target=args.hotpot_context_tokens or None,
            filler_path=args.hotpot_filler_path,
            filler_chunk_size=args.hotpot_filler_chunk,
            filler_overlap=args.hotpot_filler_overlap,
        )
        experience_iter = iter(hotpot_long_stream(stream_cfg))

    manager = MemoryManager(capacity=args.capacity)
    env = MemoryRLEnvironment(
        manager,
        config,
        experience_stream=experience_iter,
        action_mode=args.policy_mode,
    )

    if args.policy_mode == "combined":
        action_space = ["no_op", "run_gc", "cycle_allocator", "tick"]
    elif args.policy_mode == "gc":
        action_space = ["skip", "run_gc"]
    elif args.policy_mode == "query":
        action_space = ["skip_query", "query"]
    elif args.policy_mode == "write":
        action_space = ["skip_write", "write"]
    elif args.policy_mode == "alloc":
        action_space = ["stay", "first_fit", "best_fit"]

    feature_dim = len(build_feature_vector(env.reset(), args.capacity))
    policy = PolicyValueNet(feature_dim, args.hidden_dim, len(action_space)).to(device)
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        policy.load_state_dict(state_dict)

    if args.eval_only:
        policy.eval()
        optimizer = None
    else:
        policy.train()
        optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)

    stats_log: List[Dict[str, float]] = []

    episode_iterator = range(args.episodes)
    progress = None
    if args.progress_bar and trange is not None:
        progress = trange(args.episodes, desc="episodes")
        episode_iterator = progress

    for episode in episode_iterator:
        state = env.reset()
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        rewards: List[float] = []
        hits = 0
        support_hits = 0
        answer_hits = 0

        heap_samples: List[float] = []
        frag_samples: List[float] = []

        for step in range(args.steps_per_episode):
            heap_samples.append(state.heap_used / args.capacity)
            frag_samples.append(state.fragmentation)
            features = torch.tensor(build_feature_vector(state, args.capacity), dtype=torch.float32, device=device)
            logits, value = policy(features)
            if args.eval_only:
                action_idx = torch.argmax(logits)
                distribution = torch.distributions.Categorical(logits=logits.detach())
                log_prob = distribution.log_prob(action_idx).detach()
                entropy = distribution.entropy().detach()
            else:
                distribution = torch.distributions.Categorical(logits=logits)
                action_idx = distribution.sample()
                log_prob = distribution.log_prob(action_idx)
                entropy = distribution.entropy()
            action = action_idx.item()
            next_state, reward = env.step(action)

            if not args.eval_only:
                log_probs.append(log_prob)
                values.append(value)
                entropies.append(entropy)
            rewards.append(reward)
            hits += 1 if next_state.last_hit else 0
            support_hits += 1 if (next_state.metadata.get("support_hit") or 0.0) else 0
            answer_hits += 1 if (next_state.metadata.get("answer_hit") or 0.0) else 0

            state = next_state

        episode_reward = sum(rewards)
        hit_rate = hits / max(1, args.steps_per_episode)
        support_hit_rate = support_hits / max(1, args.steps_per_episode)
        answer_hit_rate = answer_hits / max(1, args.steps_per_episode)

        if not args.eval_only:
            returns = []
            future_return = 0.0
            for reward in reversed(rewards):
                future_return = reward + args.gamma * future_return
                returns.append(future_return)
            returns.reverse()

            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
            values_tensor = torch.stack(values)
            log_probs_tensor = torch.stack(log_probs)
            entropies_tensor = torch.stack(entropies)

            advantages = returns_tensor - values_tensor.detach()
            policy_loss = -(log_probs_tensor * advantages).mean() - args.entropy_coef * entropies_tensor.mean()
            value_loss = args.value_loss_coef * (returns_tensor - values_tensor).pow(2).mean()
            loss = policy_loss + value_loss

            optimizer.zero_grad()  # type: ignore[union-attr]
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()  # type: ignore[union-attr]

            status_message = (
                f"Episode {episode:03d}: reward={episode_reward:.3f} "
                f"hit_rate={hit_rate:.3f} loss={loss.item():.4f}"
            )
            policy_loss_val = policy_loss.item()
            value_loss_val = value_loss.item()
        else:
            status_message = (
                f"Eval {episode:03d}: reward={episode_reward:.3f} "
                f"hit_rate={hit_rate:.3f}"
            )
            policy_loss_val = 0.0
            value_loss_val = 0.0

        stats = {
            "episode": episode,
            "reward": episode_reward,
            "hit_rate": hit_rate,
            "support_hit_rate": support_hit_rate,
            "answer_hit_rate": answer_hit_rate,
            "loss": policy_loss_val + value_loss_val,
            "policy_loss": policy_loss_val,
            "value_loss": value_loss_val,
            "heap_used_mean": sum(heap_samples) / max(1, len(heap_samples)),
            "fragmentation_mean": sum(frag_samples) / max(1, len(frag_samples)),
        }
        stats_log.append(stats)
        if progress is not None:
            progress.write(status_message)
        else:
            print(status_message)

    if not args.eval_only:
        torch.save(policy.state_dict(), output_root / "policy.pt")
        with (output_root / "config.json").open("w", encoding="utf-8") as handle:
            json.dump({"args": vars(args), "config": asdict(config)}, handle, indent=2)

    if args.eval_only and args.eval_log:
        log_target = Path(args.eval_log)
        log_target.parent.mkdir(parents=True, exist_ok=True)
    else:
        suffix = "evaluation_log.jsonl" if args.eval_only else "training_log.jsonl"
        log_target = output_root / suffix

    with log_target.open("w", encoding="utf-8") as handle:
        for row in stats_log:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
