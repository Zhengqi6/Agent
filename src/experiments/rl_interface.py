from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from agent_memory import BestFitAllocator, FirstFitAllocator, MemoryManager
from experiments.environment import Experience, SimulatedEnvironment


@dataclass
class MemoryRLConfig:
    """
    Configuration options for the reinforcement learning loop.

    reward_hit: positive reward when the retriever returns the ground-truth topic.
    reward_miss: penalty applied when retrieval fails.
    reward_gc: scaling factor for bytes freed by GC when the agent chooses to collect.
    penalty_fragmentation: scaling factor for the fragmentation ratio (higher means stronger penalty).
    """

    capacity: int = 32768
    query_interval: int = 10
    reward_hit: float = 1.0
    reward_miss: float = -0.5
    penalty_fragmentation: float = 0.2
    penalty_pause: float = 0.5
    seed: Optional[int] = None
    top_k: int = 1
    allocator_cycle: Optional[List[str]] = None
    root_queue_limit: int = 256
    size_hint_min: int = 512
    size_hint_max: int = 8192
    gc_reward_scale: float = 1e-6
    gc_decision_interval: int = 1
    skip_query_penalty: float = 0.05
    skip_write_penalty: float = 0.02
    allocator_switch_penalty: float = 0.01
    query_hit_bonus: float = 0.3
    query_cost_weight: float = 0.05
    write_evidence_bonus: float = 0.4
    write_progress_bonus: float = 0.2
    alloc_fragmentation_bonus: float = 0.3
    gc_hit_bonus: float = 0.3
    gc_hit_drop_penalty: float = 0.25
    gc_pressure_threshold: float = 0.82
    gc_pressure_penalty: float = 0.05
    hit_history_window: int = 16
    support_hit_reward: float = 1.5
    support_miss_penalty: float = -0.2
    distractor_penalty: float = 0.1
    filler_penalty: float = 0.15
    support_bonus_k: int = 5
    answer_hit_reward: float = 1.0
    answer_miss_penalty: float = -0.2


@dataclass
class MemoryRLState:
    """Observation handed to the RL policy."""

    step: int
    heap_used: int
    heap_free: int
    fragmentation: float
    objects: int
    last_reward: float
    last_hit: bool
    last_gc_pause: float
    topic: Optional[str] = None
    metadata: Dict[str, float] = field(default_factory=dict)


class MemoryRLEnvironment:
    """
    Thin wrapper that feeds synthetic experience streams into a MemoryManager and exposes
    a reward signal suitable for reinforcement learning algorithms.
    """

    def __init__(
        self,
        manager: MemoryManager,
        config: MemoryRLConfig,
        *,
        experience_stream: Optional[Iterator[Experience]] = None,
        action_mode: str = "combined",
    ) -> None:
        self.manager = manager
        self.config = config
        self.env = None if experience_stream is not None else SimulatedEnvironment(seed=config.seed)
        self._experience_stream = experience_stream
        self.action_mode = action_mode
        self.step_idx = 0
        self._last_reward = 0.0
        self._last_hit = False
        self._last_gc_pause = 0.0
        self._allocator_cycle = config.allocator_cycle or ["first_fit", "best_fit"]
        self._allocator_index = 0
        self._apply_allocator(self._allocator_cycle[0])
        self._root_queue: List[str] = []
        self._hit_history: List[float] = []
        self._current_question: Optional[str] = None
        self._question_support_rewarded = False
        self._question_answer_rewarded = False
        self._current_question_has_answer = False
        if self.action_mode == "combined":
            self._action_space = ["no_op", "run_gc", "cycle_allocator", "tick"]
        elif self.action_mode == "gc":
            self._action_space = ["skip", "run_gc"]
        elif self.action_mode == "query":
            self._action_space = ["skip_query", "query"]
        elif self.action_mode == "write":
            self._action_space = ["skip_write", "write"]
        elif self.action_mode == "alloc":
            self._action_space = ["stay", "first_fit", "best_fit"]
        else:
            raise ValueError(f"Unsupported action_mode '{self.action_mode}'")

    # -- Core interaction -----------------------------------------------------
    def reset(self) -> MemoryRLState:
        """Reset internal counters and return the initial observation."""
        self.step_idx = 0
        self._last_reward = 0.0
        self._last_hit = False
        self._last_gc_pause = 0.0
        self.manager = MemoryManager(self.config.capacity)
        self._allocator_index = 0
        self._apply_allocator(self._allocator_cycle[self._allocator_index])
        self._root_queue.clear()
        self._current_question = None
        self._question_support_rewarded = False
        self._question_answer_rewarded = False
        self._current_question_has_answer = False
        stats = self.manager.stats()
        return MemoryRLState(
            step=self.step_idx,
            heap_used=int(stats["heap_used"]),
            heap_free=int(stats["heap_free"]),
            fragmentation=float(stats["fragmentation"]),
            objects=int(stats["objects"]),
            last_reward=self._last_reward,
            last_hit=self._last_hit,
            last_gc_pause=self._last_gc_pause,
            topic=None,
        )

    def step(self, action: int | str) -> Tuple[MemoryRLState, float]:
        """Execute one environment step for the selected action mode."""
        self.step_idx += 1
        stats_before = self.manager.stats()
        prev_hit_avg = self._hit_average()
        experience = self._next_experience()
        importance = experience.importance
        ttl = experience.ttl
        action_str = self._normalize_action(action)
        pending_question_penalty = 0.0

        question = str(experience.metadata.get("question", "") or "").strip()
        answer_candidate = str(experience.metadata.get("answer", "") or "").strip()
        if question:
            if self._current_question is None:
                self._current_question = question
                self._question_support_rewarded = False
                self._question_answer_rewarded = False
                self._current_question_has_answer = bool(answer_candidate)
            elif question != self._current_question:
                if not self._question_support_rewarded:
                    pending_question_penalty += self.config.support_miss_penalty
                if self._current_question_has_answer and not self._question_answer_rewarded:
                    pending_question_penalty += self.config.answer_miss_penalty
                self._current_question = question
                self._question_support_rewarded = False
                self._question_answer_rewarded = False
                self._current_question_has_answer = bool(answer_candidate)

        store_enabled = not (self.action_mode == "write" and action_str == "skip_write")
        perform_query = not (self.action_mode == "query" and action_str == "skip_query")

        obj_id: Optional[str] = None
        aux_penalty = 0.0
        if store_enabled or self.action_mode not in {"write"}:
            payload = {
                "text": experience.text,
                "metadata": {
                    "topic": experience.metadata["topic"],
                    "tags": ",".join(experience.tags),
                },
            }
            text_len = len(experience.text)
            size_hint = max(
                self.config.size_hint_min,
                min(text_len // 32, self.config.size_hint_max),
            )
            obj = self.manager.store(payload, importance=importance, ttl=ttl, size=size_hint)
            obj_id = obj.id
            self.manager.add_root(obj.id)
            self._root_queue.append(obj.id)
            if len(self._root_queue) > self.config.root_queue_limit:
                removed = self._root_queue.pop(0)
                self.manager.remove_root(removed)

        freed_bytes = 0
        gc_pause = 0.0
        should_decide_gc = self.step_idx % max(1, self.config.gc_decision_interval) == 0

        if action_str == "run_gc" and should_decide_gc:
            before = self.manager.space.available()
            event = self.manager.run_gc(trigger="rl_policy")
            after = self.manager.space.available()
            freed_bytes = max(0, after - before)
            gc_pause = event.get("pause_duration", 0.0)
        elif action_str == "cycle_allocator":
            self._allocator_index = (self._allocator_index + 1) % len(self._allocator_cycle)
            self._apply_allocator(self._allocator_cycle[self._allocator_index])
            aux_penalty += self.config.allocator_switch_penalty
        elif action_str == "tick":
            self.manager.tick()
        elif action_str in {"first_fit", "best_fit"}:
            self._apply_allocator(action_str)
            aux_penalty += self.config.allocator_switch_penalty

        query_topic = experience.metadata.get("topic", "")

        support_bonus = 0.0
        distractor_penalty = 0.0
        filler_penalty = 0.0
        support_hit = False

        support_titles_raw = experience.metadata.get("support_titles", [])
        if isinstance(support_titles_raw, (list, tuple, set)):
            support_titles = [str(title) for title in support_titles_raw if title]
        elif support_titles_raw:
            support_titles = [str(support_titles_raw)]
        else:
            support_titles = []
        answer_text = answer_candidate
        if answer_text:
            self._current_question_has_answer = True

        results = []
        if perform_query:
            results = self._execute_query(experience)
        reward, hit, support_hit, distractor_penalty, filler_penalty = self._compute_reward(
            experience,
            freed_bytes,
            gc_pause,
            results=results,
        )
        retrieval_topics = []
        retrieval_texts = []
        if perform_query and results:
            retrieval_topics = [
                result.payload.get("metadata", {}).get("topic", "")
                for result in results[: self.config.top_k]
            ]
            retrieval_texts = [
                str(result.payload.get("text", ""))
                for result in results[: self.config.top_k]
            ]
        elif self.action_mode == "query":
            reward -= self.config.skip_query_penalty

        answer_hit = False
        if retrieval_texts and answer_text:
            answer_hit = self._answer_in_texts(answer_text, retrieval_texts)
            if answer_hit and not self._question_answer_rewarded:
                reward += self.config.answer_hit_reward
                self._question_answer_rewarded = True

        if support_hit and not self._question_support_rewarded:
            reward += self.config.support_hit_reward
            self._question_support_rewarded = True

        if self.action_mode == "write":
            if store_enabled:
                evidence = experience.metadata.get("evidence_prob", 0.0)
                progress = experience.metadata.get("progress_signal", 0.0)
                reward += self.config.write_evidence_bonus * evidence
                reward += self.config.write_progress_bonus * progress
            else:
                reward -= self.config.skip_write_penalty

        if self.action_mode == "query" and perform_query:
            evidence = experience.metadata.get("evidence_prob", 0.0)
            cost = experience.metadata.get("cost", 0.0)
            reward += self.config.query_hit_bonus * evidence
            reward -= self.config.query_cost_weight * cost

        reward -= aux_penalty
        reward -= distractor_penalty
        reward -= filler_penalty
        reward += pending_question_penalty

        stats = self.manager.stats()
        self._update_hit_history(hit)
        new_hit_avg = self._hit_average()

        if self.action_mode == "gc":
            if action_str == "run_gc":
                reward += self.config.gc_hit_bonus * max(new_hit_avg - prev_hit_avg, 0.0)
                reward -= self.config.gc_hit_drop_penalty * max(prev_hit_avg - new_hit_avg, 0.0)
            else:
                heap_used_ratio = stats_before.get("heap_used", 0.0) / max(1.0, float(self.config.capacity))
                if heap_used_ratio >= self.config.gc_pressure_threshold:
                    reward -= self.config.gc_pressure_penalty

        if self.action_mode == "alloc" and action_str in {"first_fit", "best_fit"}:
            reward += self.config.alloc_fragmentation_bonus * (
                stats_before.get("fragmentation", 0.0) - stats.get("fragmentation", 0.0)
            )

        self._last_reward = reward
        self._last_hit = hit
        self._last_gc_pause = gc_pause

        state = MemoryRLState(
            step=self.step_idx,
            heap_used=int(stats["heap_used"]),
            heap_free=int(stats["heap_free"]),
            fragmentation=float(stats["fragmentation"]),
            objects=int(stats["objects"]),
            last_reward=reward,
            last_hit=hit,
            last_gc_pause=gc_pause,
            topic=experience.metadata["topic"],
            metadata={
                "importance": importance,
                "ttl": ttl or -1.0,
                "evidence_prob": experience.metadata.get("evidence_prob", 0.0),
                "progress_signal": experience.metadata.get("progress_signal", 0.0),
                "action_cost": experience.metadata.get("cost", 0.0),
                "last_action": action_str,
                "last_object_id": obj_id or "",
                "question": experience.metadata.get("question", query_topic),
                "topic": experience.metadata.get("topic", ""),
                "tags": ",".join(experience.tags),
                "is_support": bool(experience.metadata.get("is_support", False)),
                "support_titles": support_titles,
                "answer": answer_text,
                "retrieval_topics": retrieval_topics,
                "retrieval_texts": retrieval_texts,
                "support_hit": 1.0 if support_hit else 0.0,
                "answer_hit": 1.0 if answer_hit else 0.0,
            },
        )
        return state, reward

    # -- Reward calculation ---------------------------------------------------
    def _compute_reward(
        self,
        experience: Experience,
        freed_bytes: int,
        gc_pause: float,
        *,
        results: Optional[List] = None,
    ) -> Tuple[float, bool, bool, float, float]:
        """Reward combines retrieval outcome, GC efficiency, and heap health."""
        query_topic = experience.metadata.get("topic", "")
        if results is None:
            if self.env is not None:
                query = self.env.make_query(query_topic)
            else:
                query = experience.metadata.get("question", query_topic)
            results = self.manager.retrieve(query, top_k=self.config.top_k)
        hit = False
        support_hit = False
        distractor_penalty = 0.0
        filler_penalty = 0.0

        if results:
            top_payload = results[0].payload
            metadata = top_payload.get("metadata", {})
            top_topic = metadata.get("topic")
            hit = top_topic == experience.metadata["topic"]

            truth_titles = {
                str(title).lower()
                for title in experience.metadata.get("support_titles", [])
                if title
            }
            if not truth_titles and experience.metadata.get("is_support", False):
                truth_titles.add(str(experience.metadata.get("topic", "")).lower())

            retrieved_support = set()
            for idx, result in enumerate(results[: max(1, self.config.support_bonus_k)]):
                meta = result.payload.get("metadata", {})
                topic = str(meta.get("topic", "")).lower()
                tags = str(meta.get("tags", "")).split(",")
                if topic and "support" in tags:
                    retrieved_support.add(topic)

            for idx, result in enumerate(results[: self.config.top_k]):
                meta = result.payload.get("metadata", {})
                tags = str(meta.get("tags", "")).split(",")
                if "filler" in tags:
                    filler_penalty += self.config.filler_penalty / max(1, idx + 1)
                elif "distractor" in tags:
                    distractor_penalty += self.config.distractor_penalty / max(1, idx + 1)

            if truth_titles:
                if truth_titles & retrieved_support:
                    support_hit = True

        stats = self.manager.stats()
        reward = self.config.reward_hit if hit else self.config.reward_miss
        reward += self.config.gc_reward_scale * float(freed_bytes)
        reward -= self.config.penalty_fragmentation * float(stats["fragmentation"])
        reward -= self.config.penalty_pause * gc_pause
        return reward, hit, support_hit, distractor_penalty, filler_penalty

    # -- Helpers --------------------------------------------------------------
    def _normalize_action(self, action: int | str) -> str:
        if isinstance(action, int):
            idx = max(0, min(action, len(self._action_space) - 1))
            mapped = self._action_space[idx]
        else:
            mapped = action

        if self.action_mode == "gc":
            return "run_gc" if mapped in {"run_gc", "gc", "1"} else "no_op"
        if self.action_mode == "query":
            return "query" if mapped in {"query", "run_query", "1"} else "skip_query"
        if self.action_mode == "write":
            return "write" if mapped in {"write", "1"} else "skip_write"
        if self.action_mode == "alloc":
            if mapped in {"first_fit", "best_fit"}:
                return mapped
            return "stay"
        return mapped

    def _next_experience(self) -> Experience:
        if self._experience_stream is not None:
            try:
                return next(self._experience_stream)
            except StopIteration as exc:  # pragma: no cover
                raise RuntimeError("Experience stream exhausted") from exc
        assert self.env is not None
        return self.env.next_experience()

    def _apply_allocator(self, name: str) -> None:
        if name == "best_fit":
            self.manager.allocator = BestFitAllocator()
        else:
            self.manager.allocator = FirstFitAllocator()

    def _execute_query(self, experience: Experience):
        query_topic = experience.metadata.get("topic", "")
        if self.env is not None:
            query = self.env.make_query(query_topic)
        else:
            query = experience.metadata.get("question", query_topic)
        return self.manager.retrieve(query, top_k=self.config.top_k)

    @staticmethod
    def _normalize_answer_text(text: str) -> str:
        return " ".join(str(text).strip().lower().split())

    def _answer_in_texts(self, answer: str, texts: Sequence[str]) -> bool:
        normalized_answer = self._normalize_answer_text(answer)
        if not normalized_answer:
            return False
        for candidate in texts:
            if normalized_answer in self._normalize_answer_text(candidate):
                return True
        return False

    def _hit_average(self) -> float:
        if not self._hit_history:
            return 0.0
        window = min(len(self._hit_history), self.config.hit_history_window)
        if window <= 0:
            return 0.0
        return sum(self._hit_history[-window:]) / window

    def _update_hit_history(self, hit: bool) -> None:
        self._hit_history.append(1.0 if hit else 0.0)
        max_len = max(self.config.hit_history_window * 2, 32)
        if len(self._hit_history) > max_len:
            del self._hit_history[: len(self._hit_history) - max_len]
