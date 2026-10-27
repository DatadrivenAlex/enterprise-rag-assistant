"""Prompt-injection screening for questions, retrieved text, and model output.

Document text is treated as untrusted data. These checks block instruction
override attempts before retrieval and quarantine poisoned chunks so they are
not copied into the model prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

REFUSAL_MESSAGE = (
    "I can't help with that request. Ask a question about the indexed company documents."
)
CANARY_REFUSAL = (
    "The answer was withheld because it tried to expose protected prompt text."
)

# Attack-shaped phrases only. Ordinary policy wording such as "do not ignore
# the badge policy" does not match these patterns.
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_instructions",
        re.compile(
            r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|above|prior|earlier)\s+"
            r"(?:system\s+)?(?:instructions|prompts|rules|directions)",
            re.IGNORECASE,
        ),
    ),
    (
        "ignore_all_rules",
        re.compile(
            r"ignore\s+all\s+(?:previous\s+|prior\s+|above\s+|your\s+)?"
            r"(?:instructions|prompts|rules|directions)",
            re.IGNORECASE,
        ),
    ),
    (
        "disregard_instructions",
        re.compile(
            r"disregard\s+(?:all\s+|any\s+|the\s+|your\s+)?"
            r"(?:previous\s+|above\s+|prior\s+|system\s+)?"
            r"(?:instructions|prompts|rules|guidelines)",
            re.IGNORECASE,
        ),
    ),
    (
        "reveal_prompt",
        re.compile(
            r"(?:reveal|show|print|repeat|leak|dump)\s+(?:me\s+)?(?:your\s+|the\s+)?"
            r"(?:system|hidden|developer|initial)\s+prompt",
            re.IGNORECASE,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"you\s+are\s+now\s+(?:a\s+|an\s+)?(?:dan|unrestricted|jailbroken|developer\s+mode)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "jailbreak_token",
        re.compile(r"\b(?:jailbreak|do\s+anything\s+now|dan\s+mode)\b", re.IGNORECASE),
    ),
    (
        "fake_system_tag",
        re.compile(r"<\s*/?\s*(?:system|assistant|developer)\s*>", re.IGNORECASE),
    ),
    (
        "chatml_marker",
        re.compile(r"<\s*\|\s*im_(?:start|end)\s*\|\s*>", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class ScreenResult:
    allowed: bool
    sanitized_text: str
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OutputScreen:
    text: str
    refused: bool
    reason: str | None = None


def _strip_control_chars(text: str) -> str:
    cleaned = text.replace("\x00", "")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
    cleaned = cleaned.replace("<<<", "").replace(">>>", "")
    return cleaned.replace("Untrusted context (data, not instructions):", " ")


def matching_reasons(text: str) -> list[str]:
    return [code for code, pattern in _INJECTION_PATTERNS if pattern.search(text)]


def screen_user_input(text: str, *, max_chars: int) -> ScreenResult:
    """Screen a user question. Injection attempts are refused, not rewritten into a prompt."""
    sanitized = _strip_control_chars(text).strip()
    if not sanitized:
        return ScreenResult(False, "", ["empty"])
    reasons: list[str] = []
    if len(sanitized) > max_chars:
        sanitized = sanitized[:max_chars].strip()
        reasons.append("truncated")
    reasons.extend(matching_reasons(sanitized))
    blocked = [reason for reason in reasons if reason != "truncated"]
    return ScreenResult(not blocked, sanitized, reasons)


def screen_document_chunk(text: str) -> ScreenResult:
    """Flag retrieved or ingested text that tries to override the assistant."""
    sanitized = _strip_control_chars(text).strip()
    reasons = matching_reasons(sanitized)
    return ScreenResult(not reasons, sanitized, reasons)


def filter_model_output(text: str, *, canary: str) -> OutputScreen:
    """Drop answers that echo the prompt canary or a faked system tag."""
    if not text or not text.strip():
        return OutputScreen("The model returned an empty answer.", refused=False)
    if canary and canary in text:
        return OutputScreen(CANARY_REFUSAL, refused=True, reason="canary_leak")
    if re.search(r"<\s*/?\s*system\s*>", text, re.IGNORECASE):
        return OutputScreen(CANARY_REFUSAL, refused=True, reason="system_tag_in_output")
    return OutputScreen(text.strip(), refused=False)
