"""Flatten Transforms - Select best item from arrays.

Provides:
- BaseFlatten: Abstract base for flatten transforms
- CVSSFlattenTransform: Select authoritative CVSS score
- EPSSFlattenTransform: Select most recent EPSS score
- CPEPrimaryTransform: Select primary CPE
- CWEPrimaryTransform: Select primary CWE
"""

from vulnflow.transforms.flatten.base import BaseFlatten

__all__ = ["BaseFlatten"]
