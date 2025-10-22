from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import sys
from importlib import util

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


def run_benchmark(config: BenchmarkConfig) -> Dict[str, float]:
    dataset = load_dataset("hotpot_qa", "distractor", split=config.dataset_split)
    examples = dataset.select(range(min(config.max_samples, len(dataset))))
    rng = random.Random(config.seed)

    manager = make_memory_manager(config)
    sliding = SlidingWindowMemory(max_items=config.sliding_window)

    totals = {
        "queries": 0.0,
        "manager_hit@1": 0.0,
        "manager_hit@3": 0.0,
        "manager_hit@5": 0.0,
        "sliding_hit@1": 0.0,
        "sliding_hit@3": 0.0,
        "sliding_hit@5": 0.0,
    }

    for example in examples:
        supporting_titles = prepare_support_titles(example)
        if not supporting_titles:
            continue

        paragraphs = list(paragraph_generator(example))
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
    summary = {key: value / queries for key, value in totals.items() if key != "queries"}
    summary["queries"] = totals["queries"]
    return summary


def write_summary(path: str, summary: Dict[str, float]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


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
    args = parser.parse_args()
    dense_model = args.dense_model or None
    return BenchmarkConfig(
        dataset_split=args.dataset_split,
        max_samples=args.max_samples,
        seed=args.seed,
        top_k=args.top_k,
        hot_capacity=args.hot_capacity,
        sliding_window=args.sliding_window,
        output=args.output,
        dense_model=dense_model,
    )


def main() -> None:
    config = parse_args()
    summary = run_benchmark(config)
    write_summary(config.output, summary)
    print("Summary written to", config.output)
    for key, value in summary.items():
        if key == "queries":
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
