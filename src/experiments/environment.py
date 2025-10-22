from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Experience:
    text: str
    tags: List[str]
    importance: float
    ttl: Optional[float]
    metadata: Dict[str, str]


class SimulatedEnvironment:
    """
    Generate experience streams that emulate an agent interacting with tasks.

    Each iteration produces slightly noisy events so allocation and garbage
    collection policies can be stress-tested.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.random = random.Random(seed)
        self.topics = ["research", "planning", "reflection", "execution", "postmortem"]
        self.task_templates = {
            "research": [
                "Looked up {item} and summarised findings about {topic}.",
                "Compared {item} alternatives and noted pros/cons.",
            ],
            "planning": [
                "Outlined action plan for {topic} with {count} steps.",
                "Estimated resources for {topic} and updated schedule.",
            ],
            "reflection": [
                "Captured lessons learned from {topic}.",
                "Identified risks after {topic} session.",
            ],
            "execution": [
                "Implemented feature {item} related to {topic}.",
                "Ran experiment on {topic} and captured output snapshot.",
            ],
            "postmortem": [
                "Reviewed results for {topic} and flagged follow-ups.",
                "Archived artefacts linked to {topic}.",
            ],
        }
        self.catalogue = ["vector index", "memory policy", "allocation routine", "garbage collector", "scoring rubric"]
        self.topic_keywords = {
            "research": ["research", "analysis", "literature"],
            "planning": ["plan", "roadmap", "timeline"],
            "reflection": ["retrospective", "insight", "lessons"],
            "execution": ["implementation", "run", "experiment"],
            "postmortem": ["review", "summary", "follow-up"],
        }

    def next_experience(self) -> Experience:
        topic = self.random.choice(self.topics)
        template = self.random.choice(self.task_templates[topic])
        text = template.format(
            item=self.random.choice(self.catalogue),
            topic=self.random.choice(self.topics),
            count=self.random.randint(2, 6),
        )
        importance = round(self.random.random(), 2)
        ttl = self._sample_ttl(topic)
        tags = [topic, "experiment"]
        metadata = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "topic": topic,
        }
        return Experience(text=text, tags=tags, importance=importance, ttl=ttl, metadata=metadata)

    def make_query(self, topic: str) -> str:
        """Construct a query string that targets the provided topic."""
        keywords = self.topic_keywords.get(topic, [topic])
        focus_term = self.random.choice(keywords)
        return f"{topic} {focus_term} memory"

    def _sample_ttl(self, topic: str) -> Optional[float]:
        if topic in {"reflection", "postmortem"}:
            return None  # keep indefinitely
        base = 120.0 if topic == "planning" else 60.0
        jitter = self.random.uniform(-20.0, 20.0)
        return max(15.0, base + jitter)
