"""NSC identifier canonicalization.

Rules (Section 14 of specification):
- Canonical format is 'NSC-<integer>' (e.g. NSC-740).
- Accepts integers, integer strings, 'NSC-740', 'NSC 740', 'nsc740', ' NSC-740 '.
- Rejects unparseable values, NaN, None, empty strings cleanly by raising ValueError.
- Do NOT zero-pad (e.g. 740 -> NSC-740, not NSC-000740).
"""

from __future__ import annotations

import re
from typing import Any

_NSC_REGEX = re.compile(r"^(?:NSC[\s\-_]*)?(\d+)$", re.IGNORECASE)


def canonicalize_nsc_id(value: Any) -> str:
    """Canonicalize any valid representation of an NSC ID to 'NSC-<number>'.

    Examples
    --------
    >>> canonicalize_nsc_id(740)
    'NSC-740'
    >>> canonicalize_nsc_id("740")
    'NSC-740'
    >>> canonicalize_nsc_id("NSC-740")
    'NSC-740'
    >>> canonicalize_nsc_id("NSC 740")
    'NSC-740'
    >>> canonicalize_nsc_id("nsc740")
    'NSC-740'
    >>> canonicalize_nsc_id(" NSC-740 ")
    'NSC-740'
    """
    if value is None:
        raise ValueError("nsc_id is missing or None")
    if isinstance(value, float) and value != value:  # NaN check
        raise ValueError("nsc_id is NaN")
    
    text = str(value).strip()
    if not text:
        raise ValueError("nsc_id is empty")

    match = _NSC_REGEX.match(text)
    if match:
        number_str = str(int(match.group(1)))  # stripping leading zeros if any, e.g. 0740 -> 740
        return f"NSC-{number_str}"

    raise ValueError(f"unparseable nsc_id: {value!r}")
