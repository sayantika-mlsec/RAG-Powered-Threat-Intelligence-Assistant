import json
import os
import logging

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Output directories ────────────────────────────────────────────────────────
MITRE_DIR = "./threat_reports/mitre_techniques"
CISA_DIR  = "./threat_reports/cisa_vulnerabilities"

os.makedirs(MITRE_DIR, exist_ok=True)
os.makedirs(CISA_DIR,  exist_ok=True)


# ── MITRE ATT&CK Flattener ────────────────────────────────────────────────────

def flatten_mitre(json_path: str):
    """
    Reads enterprise-attack.json and writes one .txt file per attack-pattern.

    Field names match ingest.py's _extract_field() calls exactly:
      TECHNIQUE_ID, TACTIC, DATE_ADDED

    TACTIC is extracted from kill_chain_phases[0].phase_name.
    DATE_ADDED is extracted from the 'created' STIX property.
    """
    logger.info(f"Loading MITRE ATT&CK from: {json_path}")

    if not os.path.exists(json_path):
        logger.error(f"File not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    objects = data.get("objects", [])
    logger.info(f"Total objects in bundle: {len(objects)}")

    count_written  = 0
    count_skipped  = 0
    count_no_id    = 0

    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue

        # ── Extract core fields ───────────────────────────────────────────────
        name = obj.get("name", "Unknown Technique")
        desc = obj.get("description", "No description provided.").strip()

        # Date extraction: Slice ISO 8601 (YYYY-MM-DDTHH:MM:SS.SSSZ) to YYYY-MM-DD
        created_raw = obj.get("created", "")
        date_added = created_raw[:10] if created_raw else "unknown"

        # Technique ID from external_references
        ext_id = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                ext_id = ref.get("external_id")
                break

        if not ext_id:
            count_no_id += 1
            logger.warning(f"No external_id found for technique '{name}' — skipping.")
            continue

        # Skip sub-techniques (T1566.001) — they duplicate parent content
        # Comment this block out if you want sub-techniques included
        if "." in ext_id:
            count_skipped += 1
            continue

        # Tactic from kill_chain_phases
        phases = obj.get("kill_chain_phases", [])
        if phases:
            # Join all tactics — some techniques span multiple phases
            tactic = " | ".join(
                p.get("phase_name", "unknown").replace("-", " ").title()
                for p in phases
            )
        else:
            tactic = "unknown"
            logger.warning(f"[{ext_id}] No kill_chain_phases found — tactic set to 'unknown'.")

        # Detection field (bonus context for the LLM)
        detection = obj.get("x_mitre_detection", "").strip()

        # Platforms (Windows, Linux, macOS, etc.)
        platforms = ", ".join(obj.get("x_mitre_platforms", []))

        # Revoked techniques are outdated — skip them
        if obj.get("revoked", False):
            count_skipped += 1
            logger.debug(f"[{ext_id}] Revoked — skipping.")
            continue

        # ── Write file ────────────────────────────────────────────────────────
        file_path = os.path.join(MITRE_DIR, f"{ext_id}.txt")
        with open(file_path, "w", encoding="utf-8") as out:
            out.write(f"TECHNIQUE_ID: {ext_id}\n")
            out.write(f"TACTIC: {tactic}\n")
            out.write(f"DATE_ADDED: {date_added}\n\n")
            out.write(f"Technique Name: {name}\n")
            if platforms:
                out.write(f"Platforms: {platforms}\n")
            out.write(f"\n{desc}\n")
            if detection:
                out.write(f"\nDetection:\n{detection}\n")

        count_written += 1

    logger.info(
        f"MITRE flatten complete — "
        f"written={count_written}, "
        f"skipped_subtechniques={count_skipped}, "
        f"skipped_no_id={count_no_id}"
    )

# ── CISA KEV Flattener ────────────────────────────────────────────────────────

def flatten_cisa_kev(json_path: str):
    """
    Reads known_exploited_vulnerabilities.json and writes one .txt per CVE.

    Field names match ingest.py's _extract_field() calls exactly:
      VULNERABILITY_ID, TACTIC, DATE_ADDED

    TACTIC is hardcoded to 'Exploitation' — CISA KEV entries are all
    actively exploited vulnerabilities, so this is semantically correct
    and gives the LLM useful tactic context.
    """
    logger.info(f"Loading CISA KEV from: {json_path}")

    if not os.path.exists(json_path):
        logger.error(f"File not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vulnerabilities = data.get("vulnerabilities", [])
    logger.info(f"Total vulnerabilities in feed: {len(vulnerabilities)}")

    count_written = 0
    count_skipped = 0

    for vuln in vulnerabilities:
        cve_id     = vuln.get("cveID", "").strip()
        vendor     = vuln.get("vendorProject", "Unknown Vendor").strip()
        product    = vuln.get("product", "Unknown Product").strip()
        desc       = vuln.get("shortDescription", "No description provided.").strip()
        date_added = vuln.get("dateAdded", "unknown").strip()
        due_date   = vuln.get("dueDate", "unknown").strip()
        action     = vuln.get("requiredAction", "").strip()
        vuln_name  = vuln.get("vulnerabilityName", "").strip()

        if not cve_id:
            count_skipped += 1
            logger.warning("Entry missing cveID — skipping.")
            continue

        # ── Write file ────────────────────────────────────────────────────────
        file_path = os.path.join(CISA_DIR, f"{cve_id}.txt")
        with open(file_path, "w", encoding="utf-8") as out:
            out.write(f"VULNERABILITY_ID: {cve_id}\n")
            out.write(f"TACTIC: Exploitation\n")
            out.write(f"DATE_ADDED: {date_added}\n\n")
            out.write(f"Vulnerability Name: {vuln_name}\n")
            out.write(f"Vendor: {vendor}\n")
            out.write(f"Product: {product}\n")
            out.write(f"Patch Due Date: {due_date}\n\n")
            out.write(f"{desc}\n")
            if action:
                out.write(f"\nRequired Action:\n{action}\n")

        count_written += 1

    logger.info(
        f"CISA KEV flatten complete — "
        f"written={count_written}, "
        f"skipped={count_skipped}"
    )


# ── Execution ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    flatten_mitre("enterprise-attack.json")
    flatten_cisa_kev("known_exploited_vulnerabilities.json")

    # Final summary
    mitre_count = len(list(os.scandir(MITRE_DIR)))
    cisa_count  = len(list(os.scandir(CISA_DIR)))

    logger.info("─" * 60)
    logger.info(f"MITRE techniques : {mitre_count} files → {MITRE_DIR}")
    logger.info(f"CISA KEV entries : {cisa_count} files → {CISA_DIR}")
    logger.info(f"Total .txt files : {mitre_count + cisa_count}")
    logger.info("─" * 60)
    logger.info("Next step: run ingest.py to rebuild ./brain/")