"""Start and stop a private `ollama serve` with a given set of OLLAMA_* settings.

Server-side settings (parallel slots, context length, KV cache type, flash
attention) are environment variables read at start-up, so every benchmark
config gets its own server process on a port that does not clash with a
regular Ollama install.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_HOST = "127.0.0.1:11435"


def find_ollama(explicit: str | None = None) -> str:
    candidates = [
        explicit,
        os.environ.get("CAPPLAN_OLLAMA"),
        str(Path(".tools/ollama/ollama.exe")),
        str(Path(".tools/ollama/bin/ollama")),
        shutil.which("ollama"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(Path(c).resolve())
    raise FileNotFoundError("ollama executable not found; set CAPPLAN_OLLAMA or see README")


class OllamaServer:
    def __init__(
        self,
        exe: str | None = None,
        host: str = DEFAULT_HOST,
        env: dict[str, Any] | None = None,
        log_path: str | Path | None = None,
    ):
        self.exe = find_ollama(exe)
        self.host = host
        self.env = {k: str(v) for k, v in (env or {}).items()}
        self.log_path = Path(log_path) if log_path else None
        self.proc: subprocess.Popen | None = None
        self._log = None

    @property
    def api(self) -> str:
        return f"http://{self.host}"

    @property
    def openai_base(self) -> str:
        return f"{self.api}/v1"

    def _environ(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.env)
        env["OLLAMA_HOST"] = self.host
        env.setdefault("OLLAMA_KEEP_ALIVE", "-1")
        return env

    def start(self, timeout: float = 60.0) -> None:
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_path, "w", encoding="utf-8")  # noqa: SIM115
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(
            [self.exe, "serve"],
            env=self._environ(),
            stdout=self._log or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"ollama serve exited with code {self.proc.returncode}")
            try:
                httpx.get(f"{self.api}/api/version", timeout=2).raise_for_status()
                return
            except httpx.HTTPError:
                time.sleep(0.5)
        self.stop()
        raise TimeoutError("ollama serve did not come up")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            if sys.platform == "win32":
                # /T also ends the model runner child process, which holds the VRAM
                subprocess.run(
                    ["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                    capture_output=True,
                )
            else:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        if self._log:
            self._log.close()
            self._log = None

    def __enter__(self) -> OllamaServer:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def version(self) -> str | None:
        try:
            return httpx.get(f"{self.api}/api/version", timeout=5).json().get("version")
        except httpx.HTTPError:
            return None

    def has_model(self, model: str) -> bool:
        tags = httpx.get(f"{self.api}/api/tags", timeout=10).json().get("models", [])
        names = {m["name"] for m in tags} | {m["name"].removesuffix(":latest") for m in tags}
        return model in names

    def pull(self, model: str) -> None:
        if self.has_model(model):
            return
        subprocess.run([self.exe, "pull", model], env=self._environ(), check=True)

    def loaded(self) -> list[dict[str, Any]]:
        """Models in memory with their size and VRAM share (from /api/ps)."""
        try:
            return httpx.get(f"{self.api}/api/ps", timeout=5).json().get("models", [])
        except httpx.HTTPError:
            return []

    def unload_all(self) -> None:
        for m in self.loaded():
            httpx.post(
                f"{self.api}/api/generate",
                json={"model": m["name"], "keep_alive": 0},
                timeout=30,
            )
