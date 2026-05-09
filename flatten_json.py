import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create the folders if they don't exist
os.makedirs("./threat_reports/mitre_techniques", exist_ok=True)
os.makedirs("./threat_reports/cisa_vulnerabilities", exist_ok=True)

def flatten_mitre(json_path):
    logger.info(f"Flattening MITRE ATT&CK: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    count = 0
    for obj in data.get("objects", []):
        if obj.get("type") == "attack-pattern":
            name = obj.get("name")
            desc = obj.get("description", "No description provided.")
            
            ext_id = "Unknown"
            for ref in obj.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    ext_id = ref.get("external_id")
            
            file_path = f"./threat_reports/mitre_techniques/{ext_id}.txt"
            with open(file_path, "w", encoding="utf-8") as out:
                out.write(f"Technique Name: {name}\n")
                out.write(f"Technique ID: {ext_id}\n\n")
                out.write(desc)
            count += 1

    logger.info(f"Created {count} MITRE text files.")

def flatten_cisa_kev(json_path):
    logger.info(f"Flattening CISA KEV: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    count = 0
    for vuln in data.get("vulnerabilities", []):
        cve_id = vuln.get("cveID")
        vendor = vuln.get("vendorProject")
        product = vuln.get("product")
        desc = vuln.get("shortDescription")
        
        file_path = f"./threat_reports/cisa_vulnerabilities/{cve_id}.txt"
        with open(file_path, "w", encoding="utf-8") as out:
            out.write(f"Vulnerability ID: {cve_id}\n")
            out.write(f"Target: {vendor} {product}\n\n")
            out.write(desc)
        count += 1

    logger.info(f"Created {count} CISA KEV text files.")

# Execute the extraction
flatten_mitre("enterprise-attack.json")
flatten_cisa_kev("known_exploited_vulnerabilities.json")