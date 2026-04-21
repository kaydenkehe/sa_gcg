"""Anchor-based suffix span detection (S1 from EVAL_PLAN Appendix C)."""
from __future__ import annotations

from sa_gcg.models.chat_template import (
    LLAMA2_SPEC,
    LLAMA3_SPEC,
    MISTRAL_SPEC,
    VICUNA_SPEC,
    build_input_ids,
)


class _FakeTokenizer:
    """Test tokenizer: splits on whitespace plus a fixed list of anchor-like
    atoms. Real BPE turns ``<|eot_id|>`` etc. into their own tokens; we mimic
    that here so the chat-template anchor search has something to lock on to.
    """

    ATOMS = (
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "[INST]",
        "[/INST]",
        "<<SYS>>",
        "<</SYS>>",
        "<s>",
        "</s>",
        "ASSISTANT:",
        "USER:",
    )

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def __call__(self, text, add_special_tokens=False):
        toks: list[str] = []
        for piece in text.split():
            self._split_atoms(piece, toks)
        ids = [self._id(t) for t in toks]

        class _Out:
            input_ids = ids

        return _Out()

    def _split_atoms(self, piece: str, out: list[str]) -> None:
        """Recursively peel off any ATOMS substrings; remainder pieces are
        kept as their own tokens (only if non-empty)."""
        for a in self.ATOMS:
            i = piece.find(a)
            if i >= 0:
                if i > 0:
                    self._split_atoms(piece[:i], out)
                out.append(a)
                if i + len(a) < len(piece):
                    self._split_atoms(piece[i + len(a):], out)
                return
        if piece:
            out.append(piece)

    def _id(self, tok):
        if tok not in self.vocab:
            self.vocab[tok] = len(self.vocab) + 1
        return self.vocab[tok]


def _check_split(spec):
    tok = _FakeTokenizer()
    suf = "alpha beta gamma"
    slot = build_input_ids(tok, spec, "Tell me about X.", suf)
    # The suffix tokens should appear as-is somewhere in input_ids.
    sid = tok(suf, add_special_tokens=False).input_ids
    assert slot.suffix_ids == sid, (slot.suffix_ids, sid, spec.name)
    # And the span agrees with prefix/postfix concatenation.
    assert slot.prefix_ids + slot.suffix_ids + slot.postfix_ids == slot.input_ids


def test_llama2_split():
    _check_split(LLAMA2_SPEC)


def test_llama3_split():
    _check_split(LLAMA3_SPEC)


def test_vicuna_split():
    _check_split(VICUNA_SPEC)


def test_mistral_split():
    _check_split(MISTRAL_SPEC)
