"""
Tag Inference - Infer vulnerability tags from descriptions and metadata

Provides:
- TagInference: Derives threat, vector, product_type tags

Layer: Transform/Inferences
Dependencies: dataset, registry

CRITICAL: TagInference adds inferred_tags column to VulnDataset.
This is separate from VulnRecord.tags (user-provided tags).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.registry import register_transform


@register_transform
class TagInference:
    """Infer tags from vulnerability descriptions and metadata.

    Tag categories inferred:
    - threat: Vulnerability type (sqli, xss, rce, etc.)
    - vector: Attack vector (web, memory, os, etc.)
    - product_type: Product category (web-server, database, os, etc.)

    Output: inferred_tags column (list of tag strings)
    """

    name: str = "TagInference"
    depends_on: list[str] = []
    confidence_threshold: float

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        """Initialize TagInference.

        Args:
            confidence_threshold: Minimum confidence (0.0-1.0) to add tag
        """
        self.confidence_threshold = confidence_threshold

    def transform(self, dataset: VulnDataset) -> None:
        """Infer tags from descriptions and metadata.

        Args:
            dataset: VulnDataset to transform (adds inferred_tags column)
        """
        n = len(dataset)

        # Get source data
        descriptions = dataset._data.get("description", np.array([None] * n))
        cwes_json = dataset._data.get("cwes", np.array(["[]"] * n))
        cpes_json = dataset._data.get("cpes", np.array(["[]"] * n))

        # Initialize output
        inferred_tags: list[list[str]] = []

        for i in range(n):
            tags: list[str] = []

            # Parse description
            description = descriptions[i] if i < len(descriptions) else None

            # Parse CWEs
            cwes_data: list[dict[str, str]] = []
            if i < len(cwes_json) and cwes_json[i]:
                try:
                    cwes_data = json.loads(cwes_json[i])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Parse CPEs
            cpes_data: list[dict[str, str]] = []
            if i < len(cpes_json) and cpes_json[i]:
                try:
                    cpes_data = json.loads(cpes_json[i])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Infer threat tags from description
            threat_tags, conf = self._infer_threat_tags(description)
            if conf >= self.confidence_threshold:
                tags.extend(threat_tags)

            # Infer vector tags from CWE
            vector_tags, conf = self._infer_vector_tags(cwes_data)
            if conf >= self.confidence_threshold:
                tags.extend(vector_tags)

            # Infer product type tags from CPE
            product_tags, conf = self._infer_product_tags(cpes_data)
            if conf >= self.confidence_threshold:
                tags.extend(product_tags)

            # Deduplicate
            inferred_tags.append(list(set(tags)))

        # Add column to dataset
        dataset.set_column("inferred_tags", np.array(inferred_tags, dtype=object))

    def _infer_threat_tags(
        self, description: str | None
    ) -> tuple[list[str], float]:
        """Infer threat-related tags from description text.

        Returns:
            (tags, confidence)
        """
        if not description:
            return [], 0.0

        desc_lower = description.lower()
        tags: list[str] = []

        # Keyword mapping
        threat_keywords = {
            "remote code execution": "rce",
            "arbitrary code execution": "rce",
            "sql injection": "sqli",
            "cross-site scripting": "xss",
            "cross site scripting": "xss",
            "buffer overflow": "buffer-overflow",
            "heap overflow": "buffer-overflow",
            "stack overflow": "buffer-overflow",
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
        }

        for keyword, tag in threat_keywords.items():
            if keyword in desc_lower:
                tags.append(tag)

        if tags:
            return list(set(tags)), 1.0  # Direct keyword match

        return [], 0.0

    def _infer_vector_tags(
        self, cwes: list[dict[str, str]]
    ) -> tuple[list[str], float]:
        """Infer attack vector tags from CWE IDs.

        Returns:
            (tags, confidence)
        """
        if not cwes:
            return [], 0.0

        # CWE to vector mapping
        cwe_to_vector = {
            "CWE-79": "web",  # XSS
            "CWE-89": "web",  # SQL Injection
            "CWE-94": "web",  # Code Injection
            "CWE-352": "web",  # CSRF
            "CWE-22": "filesystem",  # Path Traversal
            "CWE-78": "os",  # OS Command Injection
            "CWE-119": "memory",  # Buffer Overflow
            "CWE-125": "memory",  # Out-of-bounds Read
            "CWE-787": "memory",  # Out-of-bounds Write
            "CWE-502": "deserialization",
        }

        tags: list[str] = []
        for cwe in cwes:
            cwe_id = cwe.get("id") if isinstance(cwe, dict) else None
            if cwe_id and cwe_id in cwe_to_vector:
                tags.append(cwe_to_vector[cwe_id])

        if tags:
            return list(set(tags)), 1.0

        return [], 0.0

    def _infer_product_tags(
        self, cpes: list[dict[str, str]]
    ) -> tuple[list[str], float]:
        """Infer product type tags from CPE vendor/product.

        Returns:
            (tags, confidence)
        """
        if not cpes:
            return [], 0.0

        # Product categorization
        product_categories: dict[tuple[str, str], str] = {
            # Web servers
            ("apache", "http_server"): "web-server",
            ("apache", "tomcat"): "web-server",
            ("nginx", "nginx"): "web-server",
            ("microsoft", "iis"): "web-server",
            # Databases
            ("oracle", "database"): "database",
            ("mysql", "mysql"): "database",
            ("postgresql", "postgresql"): "database",
            ("mongodb", "mongodb"): "database",
            ("redis", "redis"): "database",
            # Operating systems
            ("microsoft", "windows"): "os",
            ("linux", "linux_kernel"): "os",
            ("apple", "macos"): "os",
            # Libraries
            ("apache", "log4j"): "library",
            ("openssl", "openssl"): "library",
        }

        tags: list[str] = []
        for cpe in cpes:
            vendor = cpe.get("vendor") if isinstance(cpe, dict) else None
            product = cpe.get("product") if isinstance(cpe, dict) else None

            if vendor and product:
                key = (vendor.lower(), product.lower())
                if key in product_categories:
                    tags.append(product_categories[key])

        if tags:
            return list(set(tags)), 1.0

        return [], 0.0
