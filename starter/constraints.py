"""Constraint records and customer-message parsing.

The parser recognises the shapes the deterministic customer policy produces,
but every rule fails soft: anything it cannot classify is handed back as
``residual`` text and still contributes as lexical evidence. That keeps the
agent useful if the private evaluator paraphrases its replies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .facets import classify_constraint
from .text import normalize

# --------------------------------------------------------------------------
# Patterns.
#
# Design principle: parsing is *turn-aware* and defaults to treating text as
# information. Turn 1 is always the customer's opening; every later turn is a
# reply to a question we asked, so anything not recognised as a refusal, an
# override or a nudge is read as a disclosure. Regexes therefore only have to
# recognise the negative cases well - they never have to enumerate every way a
# customer might phrase a requirement.
# --------------------------------------------------------------------------

# Verbs that introduce the kind of item wanted, in an opening line.
_CATEGORY_RE = re.compile(
    r"(?:looking (?:for|at)|shopping for|hoping to find|trying to find|"
    r"searching for|interested in|show me|browsing|after|seeking|find me|"
    r"i want|i need|i'?m after|want|need)\s+(.+?)"
    r"(?=\s*(?:,\s*(?:but|though)\b|\.\s|\.$|;|\s+[-\u2013\u2014]\s|$))",
    re.IGNORECASE,
)

# Cue words that mark a clause as framing rather than content.
_CUE_RE = re.compile(
    r"\b(?:matter|matters|care|important|importance|key|must[- ]?have|need|"
    r"needs|require|requires|requirement|priorit|thing|things|looking for|"
    r"want|wants|has|have|it'?s)\b",
    re.IGNORECASE,
)

# Lead-in clauses stripped from the front of a disclosure when there is no colon.
_LEADIN_RE = re.compile(
    r"^\s*(?:well|so|ok(?:ay)?|sure|hmm+|honestly|right|yeah|yes|actually)?[,\s]*"
    r"(?:(?:the\s+)?(?:key|main|important|biggest)\s+(?:thing|things|point|factor)s?"
    r"|must[- ]?haves?|priorit(?:y|ies)|requirements?|what\s+i\s+(?:really\s+)?"
    r"(?:care about|want|need|look for))"
    r"\s*(?:for me\s*)?(?:are|is|would be)?\s*[:,-]?\s*",
    re.IGNORECASE,
)

_OVERRIDE_RE = re.compile(
    r"(?:ignore|forget|disregard|scratch|never mind)\b.{0,60}?"
    r"(?:what i (?:need|want|really need|really want) is|"
    r"instead(?:,)? i (?:need|want)|what i'?m after is)"
    r"\s*:?\s*(.+)$",
    re.IGNORECASE,
)
_OVERRIDE_HINT_RE = re.compile(
    r"\b(?:ignore|forget|disregard|scratch|never mind)\b\s+"
    r"(?:my|the|that|what)\b.{0,40}?\b(?:earlier|previous|last|first|said)\b"
    r"|\bchange of plan\b|\bactually,?\s+(?:forget|ignore|disregard)\b",
    re.IGNORECASE,
)

# Refusals. "additional" distinguishes an exhausted attribute from a boundary
# "no preference at all", but both simply block the attribute from being asked
# again, so a misread between them is harmless.
_NO_ADDITIONAL_RE = re.compile(
    r"(?:(?:do\s*n[o']?t|don'?t|do not|no)\s+have\s+an?\s+additional\s+preference"
    r"|nothing (?:further|more|else)|no (?:strong )?(?:feelings|opinions|views)|"
    r"no additional)"
    r"(?:[^a-z]{0,12}(?:on|for|about|regarding)\s+([a-z_]+))?",
    re.IGNORECASE,
)
_NO_PREFERENCE_RE = re.compile(
    r"(?:(?:do\s*n[o']?t|don'?t|do not)\s+have\s+an?\s+preference|no preference|"
    r"don'?t mind|not fussed|no strong preference)"
    r"(?:[^a-z]{0,12}(?:on|for|about|regarding)\s+([a-z_]+))?",
    re.IGNORECASE,
)
_JUDGMENT_RE = re.compile(
    r"(?:use your (?:judgment|judgement|discretion)|your call|you (?:can )?(?:decide|choose)|"
    r"whatever you think|up to you)",
    re.IGNORECASE,
)
_ASK_ONE_RE = re.compile(
    r"(?:not quite right|aren'?t right|not what i meant|not quite what|"
    r"ask me about|ask me something)",
    re.IGNORECASE,
)
_EXPLORING_RE = re.compile(
    r"still exploring|just (?:looking|browsing|seeing)|not (?:sure|decided) yet|"
    r"haven'?t decided|nothing (?:firm|specific) in mind|seeing what'?s out there|"
    r"what is out there|no specifics",
    re.IGNORECASE,
)

_EDGE = " -;,.:\t\n\"'\u2013\u2014"


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip(_EDGE)


def split_disclosure(payload: str) -> list[str]:
    """Split a disclosure payload into individual requirements."""
    parts = [_clean(part) for part in re.split(r";\s*", payload)]
    return [part for part in parts if len(part) > 1]


def extract_payload(text: str) -> str:
    """Strip framing from a reply, keeping the requirement itself.

    Prefers the text after the first colon when the clause before it is short
    framing ("What I really care about: X"). Requirements frequently contain
    their own colons ("Material:alloy"), which is why the split is on the first
    colon and only when the head looks like framing.
    """
    head, separator, tail = text.partition(":")
    if separator and tail.strip() and len(head) < 100 and _CUE_RE.search(head):
        return tail.strip()
    stripped = _LEADIN_RE.match(text)
    if stripped:
        remainder = text[stripped.end():].strip()
        if remainder:
            return remainder
    return text.strip()


@dataclass
class Constraint:
    """One requirement the customer has expressed."""

    text: str
    attribute: str
    source_turn: int
    confidence: float = 1.0
    polarity: int = 1          # +1 wanted, -1 unwanted (reserved)
    hard: bool = False         # eligible for structured filtering
    active: bool = True        # cleared by an override rather than deleted
    superseded_by: str | None = None

    @property
    def key(self) -> str:
        return normalize(self.text)

    def weight(self, soft_weight: float) -> float:
        """Effective ranking weight, discounted when superseded."""
        if self.active:
            return self.confidence
        return self.confidence * soft_weight


@dataclass
class ParsedMessage:
    """Structured reading of one customer turn."""

    kind: str = "statement"
    category: str | None = None
    new_constraints: list[str] = field(default_factory=list)
    override_value: str | None = None
    exhausted_attribute: str | None = None
    unavailable_attribute: str | None = None
    residual: str = ""
    prompted_for_question: bool = False


def parse_message(message: str, turn: int) -> ParsedMessage:
    """Read one customer turn. Never raises.

    ``turn`` matters: turn 1 is the opening statement, and every later turn is
    a reply to a question we just asked. That lets the reply path default to
    "this is a requirement" instead of needing a pattern for every phrasing.
    """
    result = ParsedMessage()
    try:
        text = str(message or "").strip()
    except Exception:  # pragma: no cover - defensive
        return result
    if not text:
        return result

    # 1. Override rewrites earlier belief, so it is checked first at any turn.
    override = _OVERRIDE_RE.search(text)
    if override:
        value = _clean(override.group(1))
        if value:
            result.kind = "override"
            result.override_value = value
            result.new_constraints = split_disclosure(value)
            return result
    if _OVERRIDE_HINT_RE.search(text):
        # Override language we cannot fully parse: still invalidate, and take
        # whatever follows the last cue verb as the replacement.
        tail = re.split(r"\b(?:is|need|want|after)\b\s*:?\s*", text, maxsplit=1)
        value = _clean(tail[-1]) if len(tail) > 1 else ""
        result.kind = "override"
        result.override_value = value or None
        result.new_constraints = split_disclosure(value) if value else []
        return result

    # 2. Opening turn: name the category, then treat any remainder as a
    #    stated requirement.
    if turn <= 1:
        result.kind = "opening"
        category = _CATEGORY_RE.search(text)
        if category:
            result.category = _clean(category.group(1))
            remainder = text[category.end():].strip(_EDGE + " ")
        else:
            # No recognisable lead-in verb. Use the first sentence as the
            # category query rather than discarding the strongest signal.
            first, _, rest = text.partition(".")
            result.category = _clean(first)
            remainder = rest.strip()
        if remainder and not _EXPLORING_RE.search(remainder):
            result.new_constraints = split_disclosure(extract_payload(remainder))
        result.residual = remainder
        return result

    # 3. Reply turns. Refusals and nudges first - everything else is content.
    no_additional = _NO_ADDITIONAL_RE.search(text)
    if no_additional:
        result.kind = "exhausted"
        result.exhausted_attribute = (no_additional.group(1) or "").lower() or None
        return result
    no_preference = _NO_PREFERENCE_RE.search(text)
    if no_preference:
        result.kind = "no_preference"
        result.unavailable_attribute = (no_preference.group(1) or "").lower() or None
        return result
    if _JUDGMENT_RE.search(text):
        result.kind = "no_preference"
        return result
    if _ASK_ONE_RE.search(text):
        result.kind = "prompt"
        result.prompted_for_question = True
        return result

    # 4. Default: a substantive reply to our question is a disclosure.
    payload = extract_payload(text)
    constraints = split_disclosure(payload)
    if constraints:
        result.kind = "disclosure"
        result.new_constraints = constraints
        result.residual = text
        return result

    result.residual = text
    return result


class ConstraintSet:
    """Active and superseded constraints for one session."""

    def __init__(self) -> None:
        self.items: list[Constraint] = []
        self._seen: set[str] = set()
        self.override_count: int = 0

    def __len__(self) -> int:
        return sum(1 for item in self.items if item.active)

    @property
    def active(self) -> list[Constraint]:
        return [item for item in self.items if item.active]

    @property
    def all_texts(self) -> list[str]:
        return [item.text for item in self.items]

    def add(self, text: str, turn: int, hard: bool = False, confidence: float = 1.0) -> Constraint | None:
        """Add a requirement, ignoring exact repeats."""
        cleaned = _clean(text)
        if len(cleaned) < 2:
            return None
        key = normalize(cleaned)
        if key in self._seen:
            return None
        self._seen.add(key)
        constraint = Constraint(
            text=cleaned,
            attribute=classify_constraint(cleaned),
            source_turn=turn,
            confidence=confidence,
            hard=hard,
        )
        self.items.append(constraint)
        return constraint

    def apply_override(self, replacement: str | None, turn: int) -> list[Constraint]:
        """Retire the superseded requirement and register the replacement.

        Constraints are marked inactive rather than deleted: products excluded
        only by the old preference become eligible again, but the retired text
        still contributes weak evidence, because the customer originally
        described the same product.
        """
        self.override_count += 1
        retired: list[Constraint] = []
        replacement_attribute = classify_constraint(replacement) if replacement else None

        candidates = [item for item in self.items if item.active]
        if candidates:
            # Prefer an active constraint of the same attribute; otherwise the
            # most recently stated one.
            same_attribute = [c for c in candidates if c.attribute == replacement_attribute]
            target = (same_attribute or candidates)[-1]
            target.active = False
            target.hard = False
            target.superseded_by = replacement
            retired.append(target)

        if replacement:
            for part in split_disclosure(replacement):
                self.add(part, turn, hard=True, confidence=1.0)
        return retired

    def texts_for_ranking(self, soft_weight: float) -> list[tuple[str, float, str]]:
        """(text, weight, attribute) triples for the ranker."""
        return [
            (item.text, item.weight(soft_weight), item.attribute)
            for item in self.items
            if item.weight(soft_weight) > 0.0
        ]
