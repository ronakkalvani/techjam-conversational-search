"""Tokenisation and normalisation helpers.

Everything here is pure and deterministic. No module-level mutable state.
"""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")
WS_RE = re.compile(r"\s+")

# Deliberately small stopword list. Constraint strings are short and quoting
# product metadata, so aggressive stopping loses signal.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be been but by for from had has have i if in into is it
    its me my of on or our please so some that the their them then there these
    they this to was were will with would you your looking want need
    """.split()
)


def flatten(value: object) -> str:
    """Flatten heterogeneous Amazon metadata into a single string.

    Mirrors the shape the catalog uses: strings, lists and ``details`` dicts.
    """
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if item in (None, "", []):
                continue
            parts.append(f"{key} {flatten(item)}")
        return " ".join(parts)
    if isinstance(value, (list, tuple)):
        return " ".join(flatten(item) for item in value if item not in (None, ""))
    return str(value)


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return WS_RE.sub(" ", str(text or "").lower()).strip()


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, stopwords removed, length > 1."""
    return [
        token
        for token in TOKEN_RE.findall(str(text or "").lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def content_tokens(text: str) -> list[str]:
    """Tokens preserved in order with duplicates removed."""
    return list(dict.fromkeys(tokenize(text)))


def singular(token: str) -> str:
    """Very small, conservative plural stripper.

    Only handles the regular English cases that matter for apparel vocabulary
    ("shirts" -> "shirt"). It never rewrites short tokens, so "gas" or "1s"
    are left alone.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and token[-3] in "sxzh":
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def token_variants(token: str) -> tuple[str, ...]:
    """Token plus its singular form, deduplicated."""
    stem = singular(token)
    return (token,) if stem == token else (token, stem)
