"""Flatten Transforms - Select best item from arrays.

Provides:
- BaseFlatten: Abstract base for flatten transforms
- CVSSFlattenTransform: Select authoritative CVSS score
- EPSSFlattenTransform: Select most recent EPSS score
- KEVFlattenTransform: Extract KEV fields
- CPEFlattenTransform: Select primary CPE
- CWEFlattenTransform: Select primary CWE
- CAPECFlattenTransform: Select primary CAPEC
- STRIDEFlattenTransform: Extract STRIDE threat booleans
"""

from vulnflow.transforms.flatten.base import BaseFlatten
from vulnflow.transforms.flatten.capec import CAPECFlattenTransform
from vulnflow.transforms.flatten.cpe import CPEFlattenTransform
from vulnflow.transforms.flatten.cvss import CVSSFlattenTransform
from vulnflow.transforms.flatten.cwe import CWEFlattenTransform
from vulnflow.transforms.flatten.epss import EPSSFlattenTransform
from vulnflow.transforms.flatten.kev import KEVFlattenTransform
from vulnflow.transforms.flatten.stride import STRIDEFlattenTransform

__all__ = [
    "BaseFlatten",
    "CAPECFlattenTransform",
    "CPEFlattenTransform",
    "CVSSFlattenTransform",
    "CWEFlattenTransform",
    "EPSSFlattenTransform",
    "KEVFlattenTransform",
    "STRIDEFlattenTransform",
]
