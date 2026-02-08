"""
Tag Inference - Infer vulnerability tags from descriptions and metadata

Provides:
- TagInference: Derives namespaced tags from all available data sources

Layer: Transform/Inferences
Dependencies: dataset, registry, flatten transforms

CRITICAL: TagInference adds inferred_tags column to VulnDataset.
This is separate from VulnRecord.tags (user-provided tags).
Tags are namespaced: "category:tag" (e.g., "threat:rce", "status:actively-exploited").
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.transforms.base import BaseInference
from vulnflow.utils import json

logger = logging.getLogger(__name__)

# Pre-compiled CVSS vector component regex
_CVSS_METRIC_RE = re.compile(r"(?:^|/)([A-Z]{1,2}):([A-Z])")

# ============================================================================
# Keyword/mapping tables (module-level for easy extension)
# ============================================================================

# Threat keywords: description substring -> tag suffix
THREAT_KEYWORDS: dict[str, str] = {
    "remote code execution": "rce",
    "arbitrary code execution": "rce",
    "sql injection": "sqli",
    "cross-site scripting": "xss",
    "cross site scripting": "xss",
    "buffer overflow": "buffer-overflow",
    "heap overflow": "buffer-overflow",
    "stack overflow": "buffer-overflow",
    "stack-based buffer overflow": "buffer-overflow",
    "heap-based buffer overflow": "buffer-overflow",
    "privilege escalation": "privesc",
    "denial of service": "dos",
    "path traversal": "path-traversal",
    "directory traversal": "path-traversal",
    "command injection": "command-injection",
    "xxe": "xxe",
    "xml external entity": "xxe",
    "csrf": "csrf",
    "cross-site request forgery": "csrf",
    "deserialization": "deserialization",
    "authentication bypass": "auth-bypass",
    "authorization bypass": "auth-bypass",
    "information disclosure": "info-disclosure",
    "information leak": "info-disclosure",
    "server-side request forgery": "ssrf",
    "ssrf": "ssrf",
    "open redirect": "open-redirect",
    "race condition": "race-condition",
    "use-after-free": "use-after-free",
    "use after free": "use-after-free",
    "integer overflow": "integer-overflow",
    "integer underflow": "integer-overflow",
    "null pointer dereference": "null-deref",
    "null dereference": "null-deref",
    "memory leak": "memory-leak",
    "insecure default": "insecure-default",
    "default credentials": "hardcoded-creds",
    "hardcoded credentials": "hardcoded-creds",
    "hard-coded credentials": "hardcoded-creds",
    "hardcoded password": "hardcoded-creds",
    "hard-coded password": "hardcoded-creds",
}

# CWE ID -> vector tag suffix
CWE_TO_VECTOR: dict[str, str] = {
    # Web
    "CWE-79": "web",
    "CWE-89": "web",
    "CWE-94": "web",
    "CWE-352": "web",
    "CWE-601": "web",
    "CWE-918": "web",
    # Filesystem
    "CWE-22": "filesystem",
    "CWE-59": "filesystem",
    "CWE-73": "filesystem",
    # OS
    "CWE-78": "os",
    "CWE-77": "os",
    # Memory
    "CWE-119": "memory",
    "CWE-120": "memory",
    "CWE-125": "memory",
    "CWE-787": "memory",
    "CWE-416": "memory",
    "CWE-190": "memory",
    "CWE-476": "memory",
    "CWE-122": "memory",
    "CWE-121": "memory",
    # Deserialization
    "CWE-502": "deserialization",
    # Authentication
    "CWE-287": "authentication",
    "CWE-306": "authentication",
    "CWE-798": "authentication",
    "CWE-862": "authentication",
    "CWE-863": "authentication",
    # Crypto
    "CWE-326": "crypto",
    "CWE-327": "crypto",
    "CWE-295": "crypto",
    "CWE-330": "crypto",
    # Configuration
    "CWE-16": "configuration",
    "CWE-732": "configuration",
}

# (vendor, product) -> product tag suffix
PRODUCT_CATEGORIES: dict[tuple[str, str], str] = {
    # Web servers
    ("apache", "http_server"): "web-server",
    ("apache", "tomcat"): "web-server",
    ("nginx", "nginx"): "web-server",
    ("microsoft", "iis"): "web-server",
    ("eclipse", "jetty"): "web-server",
    # Databases
    ("oracle", "database"): "database",
    ("mysql", "mysql"): "database",
    ("postgresql", "postgresql"): "database",
    ("mongodb", "mongodb"): "database",
    ("redis", "redis"): "database",
    ("microsoft", "sql_server"): "database",
    ("elastic", "elasticsearch"): "database",
    # Operating systems
    ("microsoft", "windows"): "os",
    ("linux", "linux_kernel"): "os",
    ("apple", "macos"): "os",
    ("apple", "mac_os_x"): "os",
    ("canonical", "ubuntu_linux"): "os",
    ("redhat", "enterprise_linux"): "os",
    # Libraries
    ("apache", "log4j"): "library",
    ("openssl", "openssl"): "library",
    ("jquery", "jquery"): "library",
    ("apache", "struts"): "library",
    ("vmware", "spring_framework"): "library",
    # Browsers
    ("google", "chrome"): "browser",
    ("mozilla", "firefox"): "browser",
    ("microsoft", "edge"): "browser",
    ("apple", "safari"): "browser",
    # CMS
    ("wordpress", "wordpress"): "cms",
    ("drupal", "drupal"): "cms",
    ("joomla\\!", "joomla\\!"): "cms",
    # VPN / Firewall
    ("fortinet", "fortigate"): "firewall",
    ("fortinet", "fortios"): "firewall",
    ("paloaltonetworks", "pan-os"): "firewall",
    ("cisco", "adaptive_security_appliance"): "firewall",
    ("citrix", "netscaler"): "vpn",
    ("ivanti", "connect_secure"): "vpn",
    # Container
    ("docker", "docker"): "container",
    ("kubernetes", "kubernetes"): "container",
}

# CVSS vector metric -> (value, tag suffix)
CVSS_ATTACK_VECTOR: dict[str, str] = {
    "N": "network",
    "A": "adjacent",
    "L": "local",
    "P": "physical",
}

CVSS_PRIVILEGES: dict[str, str] = {
    "N": "no-auth",
    "L": "low-priv",
    "H": "high-priv",
}

CVSS_USER_INTERACTION: dict[str, str] = {
    "N": "no-interaction",
    "R": "requires-interaction",
}

CVSS_COMPLEXITY: dict[str, str] = {
    "L": "low-complexity",
    "H": "high-complexity",
}

# NVD reference tag -> ref tag suffix
REFERENCE_TAG_MAP: dict[str, str] = {
    "Patch": "has-patch",
    "Vendor Advisory": "has-advisory",
    "Exploit": "has-exploit",
    "Mitigation": "has-mitigation",
    "US Government Resource": "govt-tracked",
}

# Impact keywords: description substring -> tag suffix
IMPACT_KEYWORDS: dict[str, str] = {
    "execute arbitrary code": "code-exec",
    "execute code": "code-exec",
    "code execution": "code-exec",
    "sensitive data": "data-leak",
    "sensitive information": "data-leak",
    "data exposure": "data-leak",
    "data breach": "data-leak",
    "gain elevated privileges": "privilege-gain",
    "gain administrative": "privilege-gain",
    "escalate privileges": "privilege-gain",
    "denial of service": "dos",
    "crash the application": "dos",
    "crash the service": "dos",
}


def _parse_json_safe(raw: str | None) -> list[dict[str, Any]]:
    """Parse JSON string, return empty list on failure."""
    if not raw or raw == "[]":
        return []
    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def _parse_cvss_vector(vector_string: str | None) -> dict[str, str]:
    """Parse CVSS vector string into metric -> value dict.

    E.g. "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    -> {"AV": "N", "AC": "L", "PR": "N", ...}
    """
    if not vector_string or not vector_string.startswith("CVSS:"):
        return {}

    metrics: dict[str, str] = {}
    for match in _CVSS_METRIC_RE.finditer(vector_string):
        metrics[match.group(1)] = match.group(2)
    return metrics


@register_transform(default=False)
class TagInference(BaseInference):
    """Infer namespaced tags from vulnerability data.

    Uses rule-based methods to derive tags from all available data sources.
    Each tag is namespaced as "category:tag" for easy filtering.

    Tag categories:
    - threat: Vulnerability type (threat:rce, threat:sqli, etc.)
    - vector: Attack vector from CWE (vector:web, vector:memory, etc.)
    - product: Product category from CPE (product:web-server, product:database, etc.)
    - attack: Attack surface from CVSS (attack:network, attack:no-auth, etc.)
    - status: Exploitation status (status:actively-exploited, status:high-epss, etc.)
    - ref: Reference indicators (ref:has-patch, ref:has-advisory, etc.)
    - impact: Impact type from CVSS+description (impact:confidentiality, etc.)

    Output: inferred_tags column (JSON-encoded sorted list of tag strings)
    """

    name: str = "TagInference"
    depends_on: list[str] = [
        "KEVFlattenTransform",
        "EPSSFlattenTransform",
        "CWEFlattenTransform",
        "CVSSFlattenTransform",
        "CPEFlattenTransform",
    ]

    SCHEMA_VERSION: str = "2.0.0"
    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition(
            "inferred_tags", "list[str]", "Inferred classification tags", "TagInference"
        ),
    ]

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        """Initialize TagInference.

        Args:
            confidence_threshold: Minimum confidence (0.0-1.0) to add tag.
                - 1.0: Only direct keyword/structured data matches
                - 0.7: Balanced (default) - includes reference tags (0.9)
                - 0.5: Aggressive - includes all inference rules
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold

    def transform(self, dataset: VulnDataset) -> None:
        """Infer tags from all available data sources.

        Args:
            dataset: VulnDataset to transform (adds inferred_tags column)
        """
        n = len(dataset)

        # Read all source columns once (safe for missing columns)
        descriptions = dataset.get_column_or_none("description")
        cwes_json = dataset.get_column_or_none("cwes")
        cpes_json = dataset.get_column_or_none("cpes")
        cvss_vectors = dataset.get_column_or_none("cvss_vector_string")
        is_kev = dataset.get_column_or_none("is_kev")
        epss_prob = dataset.get_column_or_none("epss_probability")
        kev_ransom = dataset.get_column_or_none("kev_ransomware")
        exploits_json = dataset.get_column_or_none("exploits")
        refs_json = dataset.get_column_or_none("references")

        # Build output
        all_tags: list[str] = []

        for i in range(n):
            tags: list[str] = []

            # --- Threat tags from description ---
            desc = descriptions[i] if descriptions is not None else None
            threat_tags, conf = self._infer_threat_tags(desc)
            if conf >= self.confidence_threshold:
                tags.extend(threat_tags)

            # --- Vector tags from CWE ---
            cwes_raw = cwes_json[i] if cwes_json is not None else None
            cwes_data = _parse_json_safe(cwes_raw)
            vector_tags, conf = self._infer_vector_tags(cwes_data)
            if conf >= self.confidence_threshold:
                tags.extend(vector_tags)

            # --- Product tags from CPE ---
            cpes_raw = cpes_json[i] if cpes_json is not None else None
            cpes_data = _parse_json_safe(cpes_raw)
            product_tags, conf = self._infer_product_tags(cpes_data)
            if conf >= self.confidence_threshold:
                tags.extend(product_tags)

            # --- Attack surface tags from CVSS vector ---
            cvss_vec = cvss_vectors[i] if cvss_vectors is not None else None
            attack_tags, conf = self._infer_attack_surface_tags(cvss_vec)
            if conf >= self.confidence_threshold:
                tags.extend(attack_tags)

            # --- Exploitation status tags from KEV/EPSS/exploits ---
            kev_val = is_kev[i] if is_kev is not None else None
            epss_val = epss_prob[i] if epss_prob is not None else None
            ransom_val = kev_ransom[i] if kev_ransom is not None else None
            exploits_raw = exploits_json[i] if exploits_json is not None else None
            exploit_tags, conf = self._infer_exploitation_tags(
                kev_val, epss_val, ransom_val, exploits_raw
            )
            if conf >= self.confidence_threshold:
                tags.extend(exploit_tags)

            # --- Reference tags ---
            refs_raw = refs_json[i] if refs_json is not None else None
            refs_data = _parse_json_safe(refs_raw)
            ref_tags, conf = self._infer_reference_tags(refs_data)
            if conf >= self.confidence_threshold:
                tags.extend(ref_tags)

            # --- Impact tags from CVSS + description ---
            impact_tags, conf = self._infer_impact_tags(desc, cvss_vec)
            if conf >= self.confidence_threshold:
                tags.extend(impact_tags)

            # Group by category as dict[str, list[str]]
            grouped: dict[str, list[str]] = {}
            for tag in sorted(set(tags)):
                category, _, value = tag.partition(":")
                grouped.setdefault(category, []).append(value)

            all_tags.append(json.dumps(grouped))

        dataset.set_column("inferred_tags", np.array(all_tags, dtype=object))

    # ========================================================================
    # Rule methods
    # ========================================================================

    def _infer_threat_tags(
        self, description: str | None
    ) -> tuple[list[str], float]:
        """Infer threat-type tags from description keywords.

        Confidence: 1.0 (direct keyword match)
        """
        if not description:
            return [], 0.0

        desc_lower = description.lower()
        tags: list[str] = []

        for keyword, tag_suffix in THREAT_KEYWORDS.items():
            if keyword in desc_lower:
                tags.append(f"threat:{tag_suffix}")

        if tags:
            return list(set(tags)), 1.0

        return [], 0.0

    def _infer_vector_tags(
        self, cwes: list[dict[str, Any]]
    ) -> tuple[list[str], float]:
        """Infer attack vector tags from CWE IDs.

        Confidence: 1.0 (direct CWE mapping)
        """
        if not cwes:
            return [], 0.0

        tags: list[str] = []
        for cwe in cwes:
            cwe_id = cwe.get("id") if isinstance(cwe, dict) else None
            if cwe_id and cwe_id in CWE_TO_VECTOR:
                tags.append(f"vector:{CWE_TO_VECTOR[cwe_id]}")

        if tags:
            return list(set(tags)), 1.0

        return [], 0.0

    def _infer_product_tags(
        self, cpes: list[dict[str, Any]]
    ) -> tuple[list[str], float]:
        """Infer product category tags from CPE vendor/product.

        Confidence: 1.0 (known vendor/product pair)
        """
        if not cpes:
            return [], 0.0

        tags: list[str] = []
        for cpe in cpes:
            vendor = cpe.get("vendor") if isinstance(cpe, dict) else None
            product = cpe.get("product") if isinstance(cpe, dict) else None

            if vendor and product:
                key = (vendor.lower(), product.lower())
                if key in PRODUCT_CATEGORIES:
                    tags.append(f"product:{PRODUCT_CATEGORIES[key]}")

        if tags:
            return list(set(tags)), 1.0

        return [], 0.0

    def _infer_attack_surface_tags(
        self, cvss_vector: str | None
    ) -> tuple[list[str], float]:
        """Infer attack surface tags from CVSS vector string.

        Parses AV, PR, UI, AC metrics from CVSS 3.x vectors.

        Confidence: 1.0 (direct from structured CVSS data)
        """
        metrics = _parse_cvss_vector(cvss_vector)
        if not metrics:
            return [], 0.0

        tags: list[str] = []

        # Attack Vector (AV)
        av = metrics.get("AV")
        if av and av in CVSS_ATTACK_VECTOR:
            tags.append(f"attack:{CVSS_ATTACK_VECTOR[av]}")

        # Privileges Required (PR)
        pr = metrics.get("PR")
        if pr and pr in CVSS_PRIVILEGES:
            tags.append(f"attack:{CVSS_PRIVILEGES[pr]}")

        # User Interaction (UI)
        ui = metrics.get("UI")
        if ui and ui in CVSS_USER_INTERACTION:
            tags.append(f"attack:{CVSS_USER_INTERACTION[ui]}")

        # Attack Complexity (AC)
        ac = metrics.get("AC")
        if ac and ac in CVSS_COMPLEXITY:
            tags.append(f"attack:{CVSS_COMPLEXITY[ac]}")

        if tags:
            return list(set(tags)), 1.0

        return [], 0.0

    def _infer_exploitation_tags(
        self,
        is_kev_val: Any,
        epss_probability: Any,
        kev_ransomware: Any,
        exploits_raw: str | None,
    ) -> tuple[list[str], float]:
        """Infer exploitation status tags from KEV, EPSS, and exploit data.

        Confidence levels:
        - KEV status: 1.0 (authoritative)
        - EPSS high (>=0.5): 0.9
        - EPSS moderate (>=0.1): 0.8
        - Has exploits: 0.9
        """
        tags: list[str] = []
        max_conf = 0.0

        # KEV: actively exploited (1.0 confidence)
        if is_kev_val is True:
            tags.append("status:actively-exploited")
            max_conf = max(max_conf, 1.0)

        # KEV: ransomware use (1.0 confidence)
        if kev_ransomware is True:
            tags.append("status:ransomware")
            max_conf = max(max_conf, 1.0)

        # EPSS probability thresholds
        if epss_probability is not None:
            try:
                prob = float(epss_probability)
                if prob >= 0.5:
                    tags.append("status:high-epss")
                    max_conf = max(max_conf, 0.9)
                elif prob >= 0.1:
                    tags.append("status:moderate-epss")
                    max_conf = max(max_conf, 0.8)
            except (TypeError, ValueError):
                pass

        # Exploits present
        exploits_data = _parse_json_safe(exploits_raw)
        if exploits_data:
            tags.append("status:has-public-exploit")
            max_conf = max(max_conf, 0.9)

        if tags:
            return list(set(tags)), max_conf

        return [], 0.0

    def _infer_reference_tags(
        self, refs: list[dict[str, Any]]
    ) -> tuple[list[str], float]:
        """Infer reference indicator tags from NVD reference tags.

        Confidence: 0.9 (structured reference tag data)
        """
        if not refs:
            return [], 0.0

        tags: list[str] = []
        for ref in refs:
            ref_tags = ref.get("tags", []) if isinstance(ref, dict) else []
            for nvd_tag, suffix in REFERENCE_TAG_MAP.items():
                if nvd_tag in ref_tags:
                    tags.append(f"ref:{suffix}")

        if tags:
            return list(set(tags)), 0.9

        return [], 0.0

    def _infer_impact_tags(
        self, description: str | None, cvss_vector: str | None
    ) -> tuple[list[str], float]:
        """Infer impact tags from CVSS C/I/A metrics and description keywords.

        Confidence:
        - CVSS-derived: 1.0 (structured data)
        - Description-derived: 0.8 (keyword heuristic)

        Returns highest confidence among matched tags.
        """
        tags: list[str] = []
        max_conf = 0.0

        # CVSS impact metrics (C, I, A)
        metrics = _parse_cvss_vector(cvss_vector)
        if metrics:
            c_val = metrics.get("C")
            if c_val and c_val in ("H", "L"):
                tags.append("impact:confidentiality")
                max_conf = max(max_conf, 1.0)

            i_val = metrics.get("I")
            if i_val and i_val in ("H", "L"):
                tags.append("impact:integrity")
                max_conf = max(max_conf, 1.0)

            a_val = metrics.get("A")
            if a_val and a_val in ("H", "L"):
                tags.append("impact:availability")
                max_conf = max(max_conf, 1.0)

        # Description-based impact keywords
        if description:
            desc_lower = description.lower()
            for keyword, suffix in IMPACT_KEYWORDS.items():
                if keyword in desc_lower:
                    tags.append(f"impact:{suffix}")
                    max_conf = max(max_conf, 0.8)

        if tags:
            return list(set(tags)), max_conf

        return [], 0.0
