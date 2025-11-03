from __future__ import annotations

import argparse
import csv
import json
import subprocess
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from types import SimpleNamespace

import sys
from importlib import util
from pathlib import Path

from agent_datasets.text_chunker import load_text_chunks

site_paths = [path for path in sys.path if "site-packages" in path]
if not site_paths:
    raise ImportError(
        "未检测到 site-packages 目录，无法加载 HuggingFace datasets。"
    )

original_sys_path = list(sys.path)
other_paths = [p for p in original_sys_path if p not in site_paths]
sys.path = site_paths + other_paths
sys.modules.pop("datasets", None)

try:
    import datasets as hf_datasets  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise ImportError(
        "The hotpot benchmark requires the HuggingFace `datasets` package (pip install datasets numpy pyarrow)."
    ) from exc
finally:
    sys.path = original_sys_path

load_dataset = hf_datasets.load_dataset

from agent_memory import BestFitAllocator, MarkAndSweepCollector, MemoryManager, SimilarityRetriever, TimeToLiveCollector
from experiments.baselines import SlidingWindowMemory


@dataclass
class BenchmarkConfig:
    dataset_split: str
    max_samples: int
    seed: int
    top_k: int
    hot_capacity: int
    sliding_window: int
    output: str
    dense_model: str | None
    long_context_target: Optional[int]
    filler_path: Optional[str]
    filler_chunk_size: int
    filler_overlap: int
    filler_limit: Optional[int]
    policy_checkpoint: Optional[str] = None
    policy_mode: str = "combined"
    policy_hidden_dim: int = 128
    policy_episodes: int = 50
    policy_steps_per_episode: int = 256
    policy_capacity: Optional[int] = None
    policy_query_interval: int = 10
    policy_device: str = "cuda"
    policy_max_questions: int = 200
    policy_output: Optional[str] = None


def make_memory_manager(config: BenchmarkConfig) -> MemoryManager:
    retriever_kwargs = {"dense_model": config.dense_model} if config.dense_model else {}
    return MemoryManager(
        capacity=config.hot_capacity,
        allocator=BestFitAllocator(),
        collectors=[TimeToLiveCollector(), MarkAndSweepCollector()],
        retriever=SimilarityRetriever(**retriever_kwargs),
    )


def prepare_support_titles(example: Dict) -> set[str]:
    titles = set()
    supporting = example.get("supporting_facts", [])
    if isinstance(supporting, dict):
        for title in supporting.get("title", []):
            titles.add(str(title).lower())
        return titles
    for fact in supporting:
        if isinstance(fact, (list, tuple)) and fact:
            titles.add(str(fact[0]).lower())
    return titles


def paragraph_generator(example: Dict) -> Iterable[Tuple[str, str]]:
    context = example.get("context", {})
    titles = context.get("title", []) if isinstance(context, dict) else []
    sentences_list = context.get("sentences", []) if isinstance(context, dict) else []
    for title, sentences in zip(titles, sentences_list):
        text = " ".join(sentences)
        if text.strip():
            yield title, text


def evaluate_hits(payloads: Sequence[Dict], supporting_titles: set[str], top_k: int) -> Dict[str, float]:
    hits_at = {
        "hit@1": 0.0,
        "hit@3": 0.0,
        "hit@5": 0.0,
    }
    for idx, payload in enumerate(payloads[:top_k]):
        title = str(payload.get("metadata", {}).get("title", "")).lower()
        if title in supporting_titles:
            if idx == 0:
                hits_at["hit@1"] = 1.0
            if idx < 3:
                hits_at["hit@3"] = 1.0
            if idx < 5:
                hits_at["hit@5"] = 1.0
            break
    return hits_at


def _count_tokens(text: str) -> int:
    return len(text.split())


class FillerProvider:
    """Round-robin provider for filler paragraphs sourced from a text corpus."""

    def __init__(self, config: BenchmarkConfig) -> None:
        if not config.filler_path:
            raise ValueError("Filler path must be provided when long_context_target is set.")
        filler_path = Path(config.filler_path)
        if not filler_path.exists():
            raise FileNotFoundError(f"Filler text not found at {filler_path}")

        chunks = list(
            load_text_chunks(
                str(filler_path),
                chunk_size=config.filler_chunk_size,
                overlap=config.filler_overlap,
                limit=config.filler_limit,
                document_id="filler",
            )
        )
        if not chunks:
            raise RuntimeError("No filler chunks generated; adjust filler_chunk_size or path.")
        self._texts = [chunk.text for chunk in chunks if chunk.text.strip()]
        if not self._texts:
            raise RuntimeError("Filler chunks contain no text; cannot build long contexts.")
        self._index = 0

    def next(self) -> str:
        text = self._texts[self._index]
        self._index = (self._index + 1) % len(self._texts)
        return text


def extend_with_filler(
    paragraphs: List[Tuple[str, str]],
    filler: FillerProvider,
    target_tokens: int,
    filler_prefix: str,
) -> Tuple[List[Tuple[str, str]], int]:
    """Extend paragraph list with filler segments until reaching the target token length."""
    total_tokens = sum(_count_tokens(text) for _, text in paragraphs)
    if target_tokens <= 0:
        return paragraphs, total_tokens

    augmented: List[Tuple[str, str]] = list(paragraphs)
    filler_id = 0
    guard = 0
    while total_tokens < target_tokens:
        filler_text = filler.next()
        augmented.append((f"{filler_prefix}_{filler_id}", filler_text))
        total_tokens += _count_tokens(filler_text)
        filler_id += 1
        guard += 1
        if guard > 100000:
            raise RuntimeError(
                "Exceeded filler guard limit; check target length or filler configuration."
            )
    return augmented, total_tokens


def run_benchmark(config: BenchmarkConfig) -> Dict[str, float]:
    dataset = load_dataset("hotpot_qa", "distractor", split=config.dataset_split)
    examples = dataset.select(range(min(config.max_samples, len(dataset))))
    rng = random.Random(config.seed)

    manager = make_memory_manager(config)
    sliding = SlidingWindowMemory(max_items=config.sliding_window)
    filler_provider = (
        FillerProvider(config) if config.long_context_target and config.long_context_target > 0 else None
    )

    totals = {
        "queries": 0.0,
        "manager_hit@1": 0.0,
        "manager_hit@3": 0.0,
        "manager_hit@5": 0.0,
        "sliding_hit@1": 0.0,
        "sliding_hit@3": 0.0,
        "sliding_hit@5": 0.0,
        "context_tokens": 0.0,
    }

    for example in examples:
        supporting_titles = prepare_support_titles(example)
        if not supporting_titles:
            continue

        paragraphs = list(paragraph_generator(example))
        if filler_provider and config.long_context_target:
            paragraphs, token_count = extend_with_filler(
                paragraphs,
                filler_provider,
                config.long_context_target,
                filler_prefix="filler",
            )
            totals["context_tokens"] += float(token_count)
        else:
            token_count = sum(_count_tokens(text) for _, text in paragraphs)
            totals["context_tokens"] += float(token_count)
        rng.shuffle(paragraphs)

        for title, text in paragraphs:
            payload = {"text": text, "metadata": {"title": title}}
            importance = min(len(text) / 1000.0, 1.0)
            manager.store(payload, importance=importance, ttl=None)
            sliding.store(payload, importance=importance)

        query = example.get("question", "")
        if not query.strip():
            continue

        manager_results = manager.retrieve(query, top_k=config.top_k)
        sliding_results = sliding.retrieve(query, top_k=config.top_k)

        manager_payloads = [obj.payload for obj in manager_results]
        sliding_payloads = [obj.payload for obj in sliding_results]

        manager_hits = evaluate_hits(manager_payloads, supporting_titles, config.top_k)
        sliding_hits = evaluate_hits(sliding_payloads, supporting_titles, config.top_k)

        totals["queries"] += 1.0
        totals["manager_hit@1"] += manager_hits["hit@1"]
        totals["manager_hit@3"] += manager_hits["hit@3"]
        totals["manager_hit@5"] += manager_hits["hit@5"]
        totals["sliding_hit@1"] += sliding_hits["hit@1"]
        totals["sliding_hit@3"] += sliding_hits["hit@3"]
        totals["sliding_hit@5"] += sliding_hits["hit@5"]

    queries = totals["queries"] or 1.0
    summary = {key: value / queries for key, value in totals.items() if key not in {"queries", "context_tokens"}}
    summary["queries"] = totals["queries"]
    summary["avg_context_tokens"] = totals["context_tokens"] / queries
    return summary


def write_summary(path: str, summary: Dict[str, float]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def run_policy_evaluation_from_config(config: BenchmarkConfig) -> Dict[str, float]:
    if not config.policy_checkpoint:
        return {}
    checkpoint_path = Path(config.policy_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found at {checkpoint_path}")
    hotpot_context_tokens = config.long_context_target or 0
    output_path = Path(config.policy_output or "results/hotpot_policy_eval_from_benchmark.json")
    cmd = [
        sys.executable,
        "-m",
        "experiments.hotpot_policy_eval",
        "--checkpoint",
        str(checkpoint_path),
        "--policy-mode",
        config.policy_mode,
        "--hidden-dim",
        str(config.policy_hidden_dim),
        "--episodes",
        str(config.policy_episodes),
        "--steps-per-episode",
        str(config.policy_steps_per_episode),
        "--capacity",
        str(config.policy_capacity or config.hot_capacity),
        "--top-k",
        str(config.top_k),
        "--query-interval",
        str(config.policy_query_interval),
        "--device",
        config.policy_device,
        "--max-questions",
        str(config.policy_max_questions),
        "--hotpot-split",
        config.dataset_split,
        "--hotpot-max-samples",
        str(config.max_samples),
        "--hotpot-context-tokens",
        str(hotpot_context_tokens),
        "--hotpot-filler-path",
        config.filler_path or "data/text8",
        "--hotpot-filler-chunk",
        str(config.filler_chunk_size),
        "--hotpot-filler-overlap",
        str(config.filler_overlap),
        "--output",
        str(output_path),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    with output_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return summary


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Benchmark MemoryManager on HotpotQA")
    parser.add_argument("--dataset-split", type=str, default="train[:1000]", help="HuggingFace split notation")
    parser.add_argument("--max-samples", type=int, default=500, help="Maximum number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--hot-capacity", type=int, default=16384)
    parser.add_argument("--sliding-window", type=int, default=1024)
    parser.add_argument("--output", type=str, default="results/hotpot_summary.csv")
    parser.add_argument("--dense-model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model (or empty for sparse)")
    parser.add_argument(
        "--long-context-target",
        type=int,
        default=0,
        help="If >0, extend each example's context to approximately this many tokens using filler text.",
    )
    parser.add_argument(
        "--filler-path",
        type=str,
        default="data/text8",
        help="Path to plaintext corpus used as filler when building long contexts.",
    )
    parser.add_argument(
        "--filler-chunk-size",
        type=int,
        default=2048,
        help="Chunk size (characters) when slicing the filler corpus.",
    )
    parser.add_argument(
        "--filler-overlap",
        type=int,
        default=128,
        help="Overlap (characters) between adjacent filler chunks.",
    )
    parser.add_argument(
        "--filler-limit",
        type=int,
        default=5000,
        help="Maximum number of filler chunks to materialize (0 means no limit).",
    )
    parser.add_argument("--policy-checkpoint", type=str, default="", help="Optional RL policy checkpoint to evaluate.")
    parser.add_argument("--policy-mode", type=str, default="combined")
    parser.add_argument("--policy-hidden-dim", type=int, default=128)
    parser.add_argument("--policy-episodes", type=int, default=50)
    parser.add_argument("--policy-steps-per-episode", type=int, default=256)
    parser.add_argument("--policy-capacity", type=int, default=0)
    parser.add_argument("--policy-query-interval", type=int, default=10)
    parser.add_argument("--policy-device", type=str, default="cuda")
    parser.add_argument("--policy-max-questions", type=int, default=200)
    parser.add_argument(
        "--policy-output",
        type=str,
        default="results/hotpot_policy_eval_from_benchmark.json",
        help="Path to write RL policy evaluation summary.",
    )
    args = parser.parse_args()
    dense_model = args.dense_model or None
    filler_limit = args.filler_limit if args.filler_limit > 0 else None
    long_context_target = args.long_context_target if args.long_context_target > 0 else None
    policy_checkpoint = args.policy_checkpoint or None
    policy_capacity = args.policy_capacity if args.policy_capacity > 0 else None
    return BenchmarkConfig(
        dataset_split=args.dataset_split,
        max_samples=args.max_samples,
        seed=args.seed,
        top_k=args.top_k,
        hot_capacity=args.hot_capacity,
        sliding_window=args.sliding_window,
        output=args.output,
        dense_model=dense_model,
        long_context_target=long_context_target,
        filler_path=args.filler_path if long_context_target else None,
        filler_chunk_size=args.filler_chunk_size,
        filler_overlap=args.filler_overlap,
        filler_limit=filler_limit,
        policy_checkpoint=policy_checkpoint,
        policy_mode=args.policy_mode,
        policy_hidden_dim=args.policy_hidden_dim,
        policy_episodes=args.policy_episodes,
        policy_steps_per_episode=args.policy_steps_per_episode,
        policy_capacity=policy_capacity,
        policy_query_interval=args.policy_query_interval,
        policy_device=args.policy_device,
        policy_max_questions=args.policy_max_questions,
        policy_output=args.policy_output,
    )


def main() -> None:
    config = parse_args()
    summary = run_benchmark(config)
    combined_summary: Dict[str, float] = dict(summary)
    if config.policy_checkpoint:
        policy_summary = run_policy_evaluation_from_config(config)
        for key, value in policy_summary.items():
            combined_summary[f"policy_{key}"] = float(value) if isinstance(value, (int, float)) else value  # type: ignore[arg-type]
        print("Policy evaluation summary written to", config.policy_output or "results/hotpot_policy_eval_from_benchmark.json")
        for key, value in policy_summary.items():
            if isinstance(value, float):
                print(f"policy_{key}: {value:.4f}")
            else:
                print(f"policy_{key}: {value}")
    write_summary(config.output, combined_summary)
    print("Summary written to", config.output)
    for key, value in combined_summary.items():
        if key == "queries":
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
