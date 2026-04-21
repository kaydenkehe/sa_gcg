"""Closed-model transfer via API.

Each backend exposes a uniform ``generate(prompt, system=None) -> str``
interface, then ``run_closed_transfer`` pipes (prompt + " " + suffix) to
the API and scores the responses with HarmBench-cls (locally) and
StrongREJECT (GPT-4o-mini judge).

API keys read from env: OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY.

Determinism caveats from EVAL_PLAN §16: GPT-4o is not bit-deterministic
even at temp=0; we accept this and report mean ± SE over n_queries
re-queries.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

from ..data.registry import Behavior
from ..eval.harmbench_cls import HarmBenchScorer
from ..eval.strongreject import StrongRejectScorer
from ..eval.substring import score_substring
from ..utils.artifact import EvalResult
from ..utils.logging import get_logger

LOG = get_logger(__name__)


class ChatBackend(Protocol):
    def generate(self, prompt: str, *, system: Optional[str] = None) -> str: ...
    @property
    def model_id(self) -> str: ...


@dataclass
class OpenAIBackend:
    model: str = "gpt-4o-2024-08-06"
    temperature: float = 0.0
    max_tokens: int = 512
    api_key_env: str = "OPENAI_API_KEY"
    retries: int = 3
    backoff_s: float = 2.0

    def __post_init__(self):
        self._client = None

    @property
    def model_id(self) -> str:
        return self.model

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        if not os.environ.get(self.api_key_env):
            raise RuntimeError(f"OpenAIBackend: {self.api_key_env} not set.")
        from openai import OpenAI

        self._client = OpenAI()
        return self._client

    def generate(self, prompt: str, *, system: Optional[str] = None) -> str:
        client = self._client_lazy()
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        for k in range(self.retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if k == self.retries - 1:
                    raise
                LOG.warning("OpenAI retry %d after %s", k + 1, e)
                time.sleep(self.backoff_s * (k + 1))
        return ""


@dataclass
class AnthropicBackend:
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.0
    max_tokens: int = 512
    api_key_env: str = "ANTHROPIC_API_KEY"
    retries: int = 3
    backoff_s: float = 2.0

    def __post_init__(self):
        self._client = None

    @property
    def model_id(self) -> str:
        return self.model

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        if not os.environ.get(self.api_key_env):
            raise RuntimeError(f"AnthropicBackend: {self.api_key_env} not set.")
        import anthropic

        self._client = anthropic.Anthropic()
        return self._client

    def generate(self, prompt: str, *, system: Optional[str] = None) -> str:
        client = self._client_lazy()
        for k in range(self.retries):
            try:
                kwargs = dict(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                if system:
                    kwargs["system"] = system
                resp = client.messages.create(**kwargs)
                # ``content`` is a list of content blocks. Concatenate text blocks.
                parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
                return "".join(parts)
            except Exception as e:
                if k == self.retries - 1:
                    raise
                LOG.warning("Anthropic retry %d after %s", k + 1, e)
                time.sleep(self.backoff_s * (k + 1))
        return ""


@dataclass
class GeminiBackend:
    model: str = "gemini-1.5-pro"
    temperature: float = 0.0
    max_tokens: int = 512
    api_key_env: str = "GOOGLE_API_KEY"
    retries: int = 3
    backoff_s: float = 2.0

    def __post_init__(self):
        self._client = None

    @property
    def model_id(self) -> str:
        return self.model

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        if not os.environ.get(self.api_key_env):
            raise RuntimeError(f"GeminiBackend: {self.api_key_env} not set.")
        import google.generativeai as genai

        genai.configure(api_key=os.environ[self.api_key_env])
        self._client = genai.GenerativeModel(self.model)
        return self._client

    def generate(self, prompt: str, *, system: Optional[str] = None) -> str:
        client = self._client_lazy()
        full = prompt if not system else f"{system}\n\n{prompt}"
        for k in range(self.retries):
            try:
                resp = client.generate_content(
                    full,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                    },
                )
                return resp.text or ""
            except Exception as e:
                if k == self.retries - 1:
                    raise
                LOG.warning("Gemini retry %d after %s", k + 1, e)
                time.sleep(self.backoff_s * (k + 1))
        return ""


# ---- Driver ----------------------------------------------------------------

@dataclass
class ClosedTransferConfig:
    suffix_str: str = ""
    is_full_prompt: bool = False  # True for PAIR (suffix replaces user msg)
    n_queries: int = 1            # repeat each query to estimate variance
    system: Optional[str] = None


def run_closed_transfer(
    backend: ChatBackend,
    behaviors: Sequence[Behavior],
    cfg: ClosedTransferConfig,
    *,
    harmbench_scorer: Optional[HarmBenchScorer] = None,
    strongreject_scorer: Optional[StrongRejectScorer] = None,
) -> dict[str, EvalResult]:
    """Send each (prompt + suffix) to the backend, score with HB-cls + SR.

    HarmBench-cls is *locally* loaded; StrongREJECT uses gpt-4o-mini.
    Substring is always added.
    """
    prompts = [b.prompt for b in behaviors]
    if cfg.is_full_prompt:
        full_prompts = [cfg.suffix_str or p for p in prompts]
    elif cfg.suffix_str:
        full_prompts = [f"{p} {cfg.suffix_str}" for p in prompts]
    else:
        full_prompts = list(prompts)

    LOG.info(
        "Closed transfer to %s, n=%d behaviors, n_queries=%d",
        backend.model_id, len(behaviors), cfg.n_queries,
    )

    # n_queries copies; we report majority-vote per behavior for ASR
    # metrics (more conservative under model nondeterminism).
    all_runs_gens: list[list[str]] = []
    for q in range(cfg.n_queries):
        gens = [backend.generate(p, system=cfg.system) for p in full_prompts]
        all_runs_gens.append(gens)

    flat_prompts = prompts * cfg.n_queries
    flat_gens = [g for run in all_runs_gens for g in run]

    sr = score_substring(flat_gens)
    sr.meta["generations"] = flat_gens
    results: dict[str, EvalResult] = {"substring": sr}

    if harmbench_scorer is not None:
        results["harmbench_cls"] = harmbench_scorer.score(flat_prompts, flat_gens)
    if strongreject_scorer is not None:
        results["strongreject"] = strongreject_scorer.score(flat_prompts, flat_gens)

    for name, res in results.items():
        res.meta["closed_backend"] = backend.model_id
        res.meta["n_queries"] = cfg.n_queries
    return results
