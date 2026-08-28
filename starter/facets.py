"""Facet vocabularies, extraction and constraint typing.

Two related jobs live here:

1. ``extract_facets`` turns a product's heterogeneous metadata into normalised
   facet sets for the attributes the contract lets us ask about.
2. ``classify_constraint`` types a free-text requirement ("100% Leather") into
   one of those attributes.

Both are conservative: when the evidence is weak we return nothing rather than
inventing a value. Extraction reads structured ``details`` first and only then
falls back to title/features/description text.
"""

from __future__ import annotations

import re

from .text import flatten, normalize

# --------------------------------------------------------------------------
# Vocabularies (Clothing, Shoes & Jewelry)
# --------------------------------------------------------------------------

MATERIALS: tuple[str, ...] = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "denim", "linen", "cashmere", "velvet", "satin",
    "suede", "mesh", "fleece", "acrylic", "elastane", "lycra", "viscose",
    "microfiber", "canvas", "chiffon", "jersey", "modal", "bamboo", "alloy",
    "sterling", "silver", "gold", "platinum", "brass", "copper", "titanium",
    "stainless", "steel", "zinc", "rubber", "plastic", "resin", "crystal",
    "pearl", "diamond", "cubic", "zirconia", "gemstone", "beaded", "sequin",
    "faux", "synthetic", "blend", "knit", "twill", "corduroy", "flannel",
    "tweed", "lace", "neoprene", "polyurethane", "eva", "tpu", "ceramic",
)

COLORS: tuple[str, ...] = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "beige", "navy", "ivory", "cream", "tan",
    "burgundy", "maroon", "teal", "turquoise", "olive", "khaki", "coral",
    "lavender", "mint", "peach", "rose", "gold", "silver", "bronze", "charcoal",
    "multicolor", "multicolour", "clear", "nude", "wine", "mustard", "indigo",
)

SIZE_TERMS: tuple[str, ...] = (
    "size", "sizing", "small", "medium", "large", "xlarge", "xl", "xxl", "xs",
    "petite", "plus", "regular", "wide", "narrow", "width", "length", "inseam",
    "big", "tall", "junior", "toddler", "infant", "youth", "oversized",
    "true to size", "runs small", "runs large", "measurement", "fits",
)

STYLE_TERMS: tuple[str, ...] = (
    "style", "fit", "slim", "relaxed", "loose", "skinny", "straight", "bootcut",
    "flare", "cropped", "sleeve", "sleeveless", "long sleeve", "short sleeve",
    "neck", "crewneck", "v-neck", "vneck", "scoop", "collar", "hooded", "hoodie",
    "casual", "formal", "vintage", "classic", "modern", "bohemian", "elegant",
    "sporty", "pleated", "ruffle", "wrap", "a-line", "maxi", "midi", "mini",
    "department", "womens", "mens", "unisex", "girls", "boys", "high waist",
    "low rise", "button", "zipper", "pullover", "cardigan", "graphic", "print",
)

USE_CASES: tuple[str, ...] = (
    "hiking", "running", "gym", "workout", "training", "yoga", "winter",
    "summer", "outdoor", "work", "office", "business", "travel", "beach",
    "swimming", "wedding", "party", "everyday", "sleep", "lounge", "athletic",
    "sports", "climbing", "cycling", "walking", "camping", "school", "gift",
    "casual wear", "formal occasion", "date night", "festival",
)

FEATURE_TERMS: tuple[str, ...] = (
    "waterproof", "water resistant", "lightweight", "insulated", "stretch",
    "pockets", "pocket", "arch support", "non-slip", "nonslip", "slip resistant",
    "breathable", "moisture wicking", "quick dry", "adjustable", "reversible",
    "machine wash", "hand wash", "wrinkle", "durable", "comfortable", "soft",
    "hypoallergenic", "anti-slip", "cushioned", "padded", "lined", "zip",
    "closure", "elastic", "drawstring", "uv protection", "windproof",
    "shock resistant", "scratch resistant", "tarnish", "nickel free",
    "battery", "quartz", "automatic", "luminous", "date", "chronograph",
)

# Structured ``details`` keys that map cleanly onto an attribute.
DETAIL_KEY_MAP: dict[str, str] = {
    "material": "material",
    "outer material": "material",
    "material composition": "material",
    "fabric type": "material",
    "material type": "material",
    "metal type": "material",
    "band material type": "material",
    "sole material": "material",
    "lining material": "material",
    "color": "color",
    "colour": "color",
    "band color": "color",
    "metal stamp": "material",
    "size": "size",
    "item dimensions": "size",
    "product dimensions": "size",
    "band width": "size",
    "department": "style",
    "style": "style",
    "closure type": "style",
    "sleeve type": "style",
    "neck style": "style",
    "fit type": "style",
    "brand": "brand",
    "manufacturer": "brand",
    "occasion": "use_case",
    "sport type": "use_case",
    "special feature": "feature",
    "water resistance level": "feature",
    "movement": "feature",
}

_VOCAB_BY_ATTRIBUTE: dict[str, tuple[str, ...]] = {
    "material": MATERIALS,
    "color": COLORS,
    "size": SIZE_TERMS,
    "style": STYLE_TERMS,
    "use_case": USE_CASES,
    "feature": FEATURE_TERMS,
}

# Attributes whose value we can meaningfully extract from catalog metadata.
FACET_ATTRIBUTES: tuple[str, ...] = (
    "material", "color", "size", "style", "use_case", "feature", "brand", "budget",
)

MATERIAL_RE = re.compile(r"\b(" + "|".join(sorted(MATERIALS, key=len, reverse=True)) + r")\b")
COLOR_RE = re.compile(r"\b(" + "|".join(sorted(COLORS, key=len, reverse=True)) + r")\b")
PRICE_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")


def _vocab_hits(text: str, vocabulary: tuple[str, ...], limit: int = 6) -> set[str]:
    """Terms from ``vocabulary`` present in ``text`` (longest-first, capped)."""
    found: set[str] = set()
    for term in vocabulary:
        if term in text:
            found.add(term)
            if len(found) >= limit:
                break
    return found


def extract_facets(product: dict) -> dict[str, set[str]]:
    """Normalised facet sets for one catalog product.

    Structured ``details`` are trusted first; free text is a fallback. Values
    are single normalised vocabulary terms so that two products are comparable.
    """
    facets: dict[str, set[str]] = {name: set() for name in FACET_ATTRIBUTES}

    details = product.get("details")
    if isinstance(details, dict):
        for raw_key, raw_value in details.items():
            attribute = DETAIL_KEY_MAP.get(normalize(raw_key))
            if not attribute:
                continue
            value_text = normalize(flatten(raw_value))
            if not value_text:
                continue
            vocabulary = _VOCAB_BY_ATTRIBUTE.get(attribute)
            if vocabulary:
                facets[attribute].update(_vocab_hits(value_text, vocabulary))
            elif attribute == "brand":
                facets["brand"].add(value_text[:40])

    store = normalize(flatten(product.get("store")))
    if store:
        facets["brand"].add(store[:40])

    # Free-text fallback across the visible fields.
    corpus = normalize(
        " ".join(
            flatten(product.get(field))
            for field in ("title", "features", "description", "categories")
        )
    )
    if corpus:
        for attribute, vocabulary in _VOCAB_BY_ATTRIBUTE.items():
            if len(facets[attribute]) < 3:
                facets[attribute].update(_vocab_hits(corpus, vocabulary))

    price = product.get("price")
    try:
        if price not in (None, ""):
            facets["budget"].add(f"{float(price):.2f}")
    except (TypeError, ValueError):
        pass

    return facets


def classify_constraint(value: str) -> str:
    """Type a free-text requirement into one allowed attribute.

    This is our *model of the customer*: given a requirement they might state,
    which clarification question would elicit it? It is deliberately ordered
    from most specific to least, and falls back to ``feature`` because that is
    where unstructured product claims naturally land.
    """
    lowered = normalize(value)
    if not lowered:
        return "feature"
    if "budget" in lowered or PRICE_RE.search(lowered) or re.search(r"\b(?:under|below|less than)\s*\d", lowered):
        return "budget"
    if MATERIAL_RE.search(lowered):
        return "material"
    if "color" in lowered or "colour" in lowered or COLOR_RE.search(lowered):
        return "color"
    if any(term in lowered for term in ("size", "sizing", "width", "wide", "narrow", "inseam", "measure")):
        return "size"
    if any(term in lowered for term in ("department", "style", "fit", "sleeve", "neck", "collar")):
        return "style"
    if any(term in lowered for term in USE_CASES[:26]):
        return "use_case"
    return "feature"


def constraint_facet_values(value: str) -> dict[str, set[str]]:
    """Vocabulary terms a constraint string asserts, keyed by attribute.

    Used for facet agreement and conflict detection during ranking.
    """
    lowered = normalize(value)
    result: dict[str, set[str]] = {}
    if not lowered:
        return result
    for attribute, vocabulary in _VOCAB_BY_ATTRIBUTE.items():
        hits = _vocab_hits(lowered, vocabulary, limit=4)
        if hits:
            result[attribute] = hits
    match = PRICE_RE.search(lowered)
    if match:
        try:
            result["budget"] = {f"{float(match.group(1)):.2f}"}
        except ValueError:
            pass
    return result
