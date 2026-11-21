"""Prompt sets used by the benchmarks.

brief: the weekly media brief from the PR Radar project. A system prompt with
       the house rules plus a fact sheet (about 730 input tokens), answered in
       about 150 tokens. Long input, short output: a RAG-style request.
chat:  short synthetic chat questions (about 60 input tokens) that ask for an
       answer of a set length (capped by max_tokens). Short input, long
       output: a typical assistant turn.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "prompts"

CHAT_SYSTEM = "You are a helpful assistant. Answer in plain prose without headings or lists."

TOPICS = [
    "how a hash map handles collisions",
    "why the sky is blue",
    "what a mortgage offset account is",
    "how vaccines train the immune system",
    "the difference between TCP and UDP",
    "how compound interest works",
    "what causes jet lag and how to reduce it",
    "how a refrigerator moves heat",
    "why sourdough bread rises",
    "how public-key cryptography works",
    "what inflation does to savings",
    "how noise-cancelling headphones work",
    "how to prepare for a job interview",
    "why some batteries lose capacity over time",
    "how a bill becomes law in a parliament",
    "what a content delivery network does",
    "how plants turn light into sugar",
    "how to write a clear bug report",
    "what a database index is for",
    "how tides are caused by the moon",
]

TASKS = [
    "Explain {topic} to a new colleague in about {words} words.",
    "I keep getting confused about {topic}. Can you walk me through it in roughly {words} words?",
    "Write a short explanation of {topic} for a curious teenager, around {words} words.",
]


def chat_prompts(n: int = 160, words: int = 180, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        # The length target moves a little with each pass over the topics, so
        # no two prompts are identical and no prompt is served from cache.
        target = words + 5 * (i // len(TOPICS))
        task = rng.choice(TASKS).format(topic=topic, words=target)
        out.append(
            {
                "id": f"chat-{i:03d}",
                "messages": [
                    {"role": "system", "content": CHAT_SYSTEM},
                    {"role": "user", "content": task},
                ],
            }
        )
    return out


def brief_prompts(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or DATA_DIR / "brief.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_workload(name: str, **kwargs: Any) -> list[dict[str, Any]]:
    if name == "brief":
        return brief_prompts()
    if name == "chat":
        return chat_prompts(**kwargs)
    p = Path(name)
    if p.exists():
        return brief_prompts(p)
    raise ValueError(f"unknown workload {name!r} (use brief, chat or a .jsonl path)")
