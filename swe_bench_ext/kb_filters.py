from __future__ import annotations

import re


_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_FUNC_CALL_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)")
_FILE_PATH_RE = re.compile(
    r"(?:\.{0,2}/)?(?:[a-zA-Z0-9_\-]+/)+[a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+"
)
_CLASS_METHOD_RE = re.compile(r"\b[A-Z][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*")
_INLINE_CODE_RE = re.compile(r"`[^`]+`")


def filter_kb_for_novice(kb: str) -> str:
    """
    Reduce implementation-specific leakage for novice responders.

    Goal: keep user-facing behavior descriptions, remove code-ish details.
    """
    kb = kb or ""
    # Strip whole code blocks first to avoid partial replacements inside code.
    kb = _CODE_BLOCK_RE.sub("[code example omitted]", kb)
    kb = _FILE_PATH_RE.sub("[file]", kb)
    kb = _CLASS_METHOD_RE.sub("[component]", kb)
    kb = _FUNC_CALL_RE.sub("[function]", kb)
    kb = _INLINE_CODE_RE.sub("[identifier]", kb)
    return kb


def enhance_kb_for_expert(kb: str) -> str:
    """Pass-through hook for expert-only augmentation (no-op for now)."""
    return kb or ""


def filter_knowledge_base_for_responder(kb: str, responder_type: str) -> str:
    rt = (responder_type or "").strip().lower()
    if rt == "novice":
        return filter_kb_for_novice(kb)
    if rt == "expert":
        return enhance_kb_for_expert(kb)
    return kb or ""

