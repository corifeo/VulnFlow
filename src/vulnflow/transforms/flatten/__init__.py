"""Flatten Transforms - Select best item from arrays.

Provides:
- BaseFlatten: Abstract base for flatten transforms
- CVSSFlattenTransform: Select authoritative CVSS score
- EPSSFlattenTransform: Select most recent EPSS score
- CPEPrimaryTransform: Select primary CPE
- CWEPrimaryTransform: Select primary CWE
"""

from vulnflow.transforms.flatten.base import BaseFlatten
from vulnflow.transforms.flatten.cpe import CPEPrimaryTransform
from vulnflow.transforms.flatten.cvss import CVSSFlattenTransform
from vulnflow.transforms.flatten.cwe import CWEPrimaryTransform
from vulnflow.transforms.flatten.epss import EPSSFlattenTransform

__all__ = [
    "BaseFlatten",
    "CPEPrimaryTransform",
    "CVSSFlattenTransform",
    "CWEPrimaryTransform",
    "EPSSFlattenTransform",
]
