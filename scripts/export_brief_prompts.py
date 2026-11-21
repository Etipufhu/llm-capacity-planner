"""Copy the brief prompts (system + fact sheet, no answer) from a PR Radar checkout.

    python scripts/export_brief_prompts.py ../pr-radar/data/brief_sft/train.jsonl

Writes data/prompts/brief.jsonl with all prompts in a fixed shuffled order. The
reference answer length is kept so runs can be compared with the target.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "prompts" / "brief.jsonl"


def main(src: str, n: int = 1000) -> None:
    lines = Path(src).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    random.Random(7).shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(rows[:n]):
            msgs = r["messages"]
            rec = {
                "id": f"brief-{i:03d}",
                "messages": [m for m in msgs if m["role"] != "assistant"],
                "ref_answer_chars": len(msgs[-1]["content"]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {min(n, len(rows))} prompts to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
