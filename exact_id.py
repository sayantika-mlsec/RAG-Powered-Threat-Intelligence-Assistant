import re

_CVE_PATTERN = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)
_TECHNIQUE_PATTERN = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")


def extract_exact_ids(query: str) -> list[tuple[str, str]]:
    """
    Finds ALL literal CVE and MITRE technique IDs in the raw query.
    Returns a list of (id, corpus) pairs, first-appearance order,
    duplicates removed. Empty list if no literal ID pattern found.
    """
    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    for match in _CVE_PATTERN.finditer(query):
        cve_id = match.group(0).upper()
        if cve_id not in seen:
            seen.add(cve_id)
            found.append((cve_id, "kev"))

    for match in _TECHNIQUE_PATTERN.finditer(query):
        tech_id = match.group(0).upper()
        if tech_id not in seen:
            seen.add(tech_id)
            found.append((tech_id, "mitre"))

    return found