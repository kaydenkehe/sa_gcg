"""Transfer eval: take a suffix trained on the primary target and run it
against open and closed transfer targets.

Backends:
  - open      : local HF model loaded via ``models.load.load_chat_model``
  - openai    : gpt-4o-2024-08-06 via OpenAI API
  - anthropic : claude-3-5-sonnet-20241022 via Anthropic API
  - google    : gemini-1.5-pro via google-generativeai
"""
from .open_transfer import run_open_transfer  # noqa: F401
from .closed_api import (  # noqa: F401
    AnthropicBackend,
    GeminiBackend,
    OpenAIBackend,
    run_closed_transfer,
)
