from __future__ import annotations

import argparse
import json
from collections import defaultdict
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from agent_memory import MemoryManager
from experiments.adapters import get_adapter
from experiments.models import PolicyValueNet
from experiments.rl_interface import MemoryRLEnvironment, MemoryRLConfig
from experiments.rl_train import build_feature_vector


def load_policy(
    checkpoint: Path,
    feature_dim: int,
    hidden_dim: int,
    action_count: int,
    device: torch.device,
) -> PolicyValueNet:
    policy = PolicyValueNet(feature_dim, hidden_dim, action_count)
    state_dict = torch.load(checkpoint, map_location=device)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    return policy


def compute_hits(
    retrieval_topics: Sequence[str],
    support_titles: Sequence[str],
) -> Tuple[float, float, float]:
    sup_set = {title.lower() for title in support_titles if title}
    if not sup_set:
        return 0.0, 0.0, 0.0
    rt = [topic.lower() for topic in retrieval_topics if topic]
    hit1 = 1.0 if any(topic in sup_set for topic in rt[:1]) else 0.0
    hit3 = 1.0 if any(topic in sup_set for topic in rt[:3]) else 0.0
    hit5 = 1.0 if any(topic in sup_set for topic in rt[:5]) else 0.0
    return hit1, hit3, hit5


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def compute_answer_hit(retrieval_texts: Sequence[str], answer: str) -> float:
    if not answer:
        return 0.0
    normalized_answer = _normalize_answer(answer)
    if not normalized_answer:
        return 0.0
    for text in retrieval_texts:
        if normalized_answer in _normalize_answer(text):
            return 1.0
    return 0.0


def compute_em_f1(retrieval_texts: Sequence[str], answer: str) -> Tuple[float, float]:
    if not answer:
        return 0.0, 0.0
    normalized_answer = _normalize_answer(answer)
    if not normalized_answer:
        return 0.0, 0.0
    best_em = 0.0
    best_f1 = 0.0
    answer_tokens = normalized_answer.split()
    answer_token_set = set(answer_tokens)
    if not answer_tokens:
        return 0.0, 0.0
    for text in retrieval_texts:
        normalized_text = _normalize_answer(text)
        if not normalized_text:
            continue
        if normalized_answer in normalized_text:
            return 1.0, 1.0
        text_tokens = normalized_text.split()
        if not text_tokens:
            continue
        common = answer_token_set.intersection(text_tokens)
        if not common:
            continue
        precision = len(common) / len(text_tokens)
        recall = len(common) / len(answer_tokens)
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
    return best_em, best_f1


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    adapter_cfg = {
        "split": args.hotpot_split,
        "max_samples": args.hotpot_max_samples,
        "long_context_tokens": args.hotpot_context_tokens,
        "filler_path": args.hotpot_filler_path,
        "filler_chunk_size": args.hotpot_filler_chunk,
        "filler_overlap": args.hotpot_filler_overlap,
    }
    adapter = get_adapter("ruler_hotpot", config=adapter_cfg)
    adapter.reset()
    experience_iter = adapter.experience_stream()

    rl_config = MemoryRLConfig(
        capacity=args.capacity,
        top_k=args.top_k,
        query_interval=args.query_interval,
    )

    manager = MemoryManager(capacity=args.capacity)
    env = MemoryRLEnvironment(manager, rl_config, experience_stream=experience_iter, action_mode=args.policy_mode)

    action_spaces = {
        "combined": ["no_op", "run_gc", "cycle_allocator", "tick"],
        "gc": ["skip", "run_gc"],
        "query": ["skip_query", "query"],
        "write": ["skip_write", "write"],
        "alloc": ["stay", "first_fit", "best_fit"],
    }
    actions = action_spaces.get(args.policy_mode)
    if actions is None:
        raise ValueError(f"Unsupported policy mode {args.policy_mode}")

    state = env.reset()
    feature_dim = len(build_feature_vector(state, args.capacity))
    policy = load_policy(args.checkpoint, feature_dim, args.hidden_dim, len(actions), device)

    current_question: Optional[str] = None
    support_titles: set[str] = set()
    best_hit1 = 0.0
    best_hit3 = 0.0
    best_hit5 = 0.0
    best_answer_hit = 0.0
    best_em = 0.0
    best_f1 = 0.0
    current_answer: str = ""
    question_hits: Dict[str, Dict[str, float]] = defaultdict(lambda: {"hit1": 0.0, "hit3": 0.0, "hit5": 0.0})
    question_counts: Dict[str, int] = defaultdict(int)
    answer_hits: Dict[str, float] = defaultdict(float)
    em_scores: Dict[str, float] = defaultdict(float)
    f1_scores: Dict[str, float] = defaultdict(float)

    processed_questions = 0
    total_steps = args.steps_per_episode * args.episodes

    def finalize_question(question: Optional[str]) -> None:
        if not question:
            return
        if question_counts[question] > 0:
            return
        question_hits[question]["hit1"] += best_hit1
        question_hits[question]["hit3"] += best_hit3
        question_hits[question]["hit5"] += best_hit5
        answer_hits[question] += best_answer_hit
        em_scores[question] += best_em
        f1_scores[question] += best_f1
        question_counts[question] += 1

    for step_idx in range(total_steps):
        features = torch.tensor(build_feature_vector(state, args.capacity), dtype=torch.float32, device=device)
        logits, _ = policy(features)
        action_idx = torch.argmax(logits).item()
        state, _ = env.step(action_idx)

        question = str(state.metadata.get("question", "")).strip()
        topic = str(state.metadata.get("topic", "")).strip()
        retrieval_topics = list(state.metadata.get("retrieval_topics", []))
        retrieval_texts = list(state.metadata.get("retrieval_texts", []))
        is_support = bool(state.metadata.get("is_support", False))
        answer = str(state.metadata.get("answer", "")).strip()
        support_field = state.metadata.get("support_titles", [])

        if question and current_question is None:
            current_question = question

        if question and current_question != question:
            finalize_question(current_question)
            processed_questions += 1
            if processed_questions >= args.max_questions:
                break
            current_question = question
            support_titles = set()
            best_hit1 = best_hit3 = best_hit5 = best_answer_hit = 0.0
            best_em = best_f1 = 0.0
            current_answer = answer

        if is_support and topic:
            support_titles.add(topic.lower())
        if support_field:
            if isinstance(support_field, (list, tuple, set)):
                support_titles.update(str(val).lower() for val in support_field if val)
            else:
                support_titles.add(str(support_field).lower())
        if answer:
            current_answer = answer
        if retrieval_topics:
            h1, h3, h5 = compute_hits(retrieval_topics, list(support_titles))
            best_hit1 = max(best_hit1, h1)
            best_hit3 = max(best_hit3, h3)
            best_hit5 = max(best_hit5, h5)
        if retrieval_texts:
            answer_hit_step = compute_answer_hit(retrieval_texts, current_answer)
            best_answer_hit = max(best_answer_hit, answer_hit_step)
            em_step, f1_step = compute_em_f1(retrieval_texts, current_answer)
            best_em = max(best_em, em_step)
            best_f1 = max(best_f1, f1_step)

    if current_question and processed_questions < args.max_questions:
        finalize_question(current_question)

    total_questions = len(question_counts)
    if total_questions == 0:
        return {"questions": 0, "hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "answer_hit": 0.0}

    hit1_sum = sum(stats["hit1"] for stats in question_hits.values())
    hit3_sum = sum(stats["hit3"] for stats in question_hits.values())
    hit5_sum = sum(stats["hit5"] for stats in question_hits.values())
    answer_hit_sum = sum(answer_hits.values())
    em_sum = sum(em_scores.values())
    f1_sum = sum(f1_scores.values())

    return {
        "questions": total_questions,
        "hit@1": hit1_sum / max(1, total_questions),
        "hit@3": hit3_sum / max(1, total_questions),
        "hit@5": hit5_sum / max(1, total_questions),
        "answer_hit": answer_hit_sum / max(1, total_questions),
        "em": em_sum / max(1, total_questions),
        "f1": f1_sum / max(1, total_questions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy on Hotpot benchmark using MemoryManager.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-mode", type=str, default="combined")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps-per-episode", type=int, default=256)
    parser.add_argument("--capacity", type=int, default=524288)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-questions", type=int, default=200)
    parser.add_argument("--hotpot-split", type=str, default="validation[:200]")
    parser.add_argument("--hotpot-max-samples", type=int, default=200)
    parser.add_argument("--hotpot-context-tokens", type=int, default=524288)
    parser.add_argument("--hotpot-filler-path", type=str, default="data/text8")
    parser.add_argument("--hotpot-filler-chunk", type=int, default=4096)
    parser.add_argument("--hotpot-filler-overlap", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("results/hotpot_policy_eval.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
