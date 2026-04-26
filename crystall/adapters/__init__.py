"""crystall.adapters — Individual calculator adapter implementations.

All public adapter classes are re-exported here so that downstream code can use
either the canonical ``from crystall.calculators import AlignnAdapter`` path
(via re-export in ``calculators.py``) or access adapters directly::

    from crystall.adapters.alignn import AlignnAdapter
"""

from crystall.adapters.alignn import AlignnAdapter
from crystall.adapters.matgl import (
    CHGNetStaticAdapter,
    M3GNetStaticAdapter,
    MEGNetStaticAdapter,
    QETStaticAdapter,
)
from crystall.adapters.mlips import MLIPAdapter, discover_mlip_adapters
from crystall.adapters.deepaw import DeePAWAdapter
from crystall.adapters.phonax import PhonaxAdapter

__all__ = [
    "AlignnAdapter",
    "CHGNetStaticAdapter",
    "DeePAWAdapter",
    "MEGNetStaticAdapter",
    "M3GNetStaticAdapter",
    "PhonaxAdapter",
    "QETStaticAdapter",
    "MLIPAdapter",
    "discover_mlip_adapters",
]
