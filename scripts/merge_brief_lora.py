"""Merge the PR Radar brief LoRA into Qwen3-1.7B and save it for `ollama create`.

Runs on the CPU, in an environment with torch, transformers and peft (for
example the PR Radar venv):

    python scripts/merge_brief_lora.py ../pr-radar/adapters/brief-lora-qwen3-1.7b models/brief-qwen3-1.7b

Then import it into Ollama with the Modelfile in configs/ (see README).
"""

from __future__ import annotations

import sys


def main(adapter: str, out: str, base: str = "Qwen/Qwen3-1.7B") -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"merged model written to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
