"""Known MPDD field paths and MongoDB operators, plus typo/validation helpers
for user-supplied `other_targets` keys.
"""
import difflib
import importlib.resources
import warnings
from typing import Any, Dict

from ruamel.yaml import YAML
from pymatgen.core import Element


def _load_known_fields() -> frozenset:
    """Load common MPDD field paths from `common_mpdd_fields.yaml` plus dynamically
    derived `material.compositionDictionary.<element>` leaves (one per pymatgen
    Element). Used as the candidate list for typo suggestions on user-supplied
    `other_targets` keys.
    """
    fields: set = set()
    try:
        with importlib.resources.files("crystall.utils").joinpath("common_mpdd_fields.yaml").open("r") as f:
            data = YAML(typ='safe', pure=True).load(f)
        fields.update(data.get('fields', []) or [])
    except (FileNotFoundError, ModuleNotFoundError) as e:
        warnings.warn(f"Could not load common_mpdd_fields.yaml; typo suggestions disabled: {e}")
    fields.update(f'material.compositionDictionary.{e.symbol}' for e in Element)
    return frozenset(fields)


_KNOWN_FIELDS = _load_known_fields()

_KNOWN_OPERATORS = frozenset({
    '$eq', '$ne', '$gt', '$gte', '$lt', '$lte',
    '$in', '$nin', '$exists', '$regex', '$options', '$type',
    '$and', '$or', '$not', '$nor',
    '$all', '$elemMatch', '$size', '$mod',
})


def _suggest_field_typos(other_targets: Dict[str, Any]) -> None:
    """Warn when a key in `other_targets`, or a MongoDB operator inside one of
    its values, looks like a typo of a known field or operator.

    Compares each unknown key to the set of common MPDD fields (and operators,
    for `$`-prefixed keys inside operator-form values) via `difflib`. Only fires
    when there is a close match — silent otherwise.
    """
    for key, value in other_targets.items():
        # Field-name typo check
        if key not in _KNOWN_FIELDS:
            suggestion = difflib.get_close_matches(key, _KNOWN_FIELDS, n=1, cutoff=0.75)
            if suggestion:
                warnings.warn(
                    f"other_targets key {key!r} is not a common MPDD field; "
                    f"did you mean {suggestion[0]!r}?"
                )
        # Operator-name typo check (only when value is operator-form)
        if isinstance(value, dict) and any(k.startswith('$') for k in value):
            for op_key in value:
                if op_key in _KNOWN_OPERATORS:
                    continue
                suggestion = difflib.get_close_matches(op_key, _KNOWN_OPERATORS, n=1, cutoff=0.75)
                if suggestion:
                    warnings.warn(
                        f"other_targets[{key!r}] uses {op_key!r}, which is not a "
                        f"known MongoDB operator; did you mean {suggestion[0]!r}?"
                    )


def _warn_unrecognized_fields(other_targets: Dict[str, Any]) -> None:
    """Warn when `other_targets` contains fields not on the known MPDD field list.
    Useful for cases where a query returns no prototypes, to help the user debug `other_targets`
    """
    unrecognized = [k for k in other_targets if k not in _KNOWN_FIELDS]
    if unrecognized:
        warnings.warn(
            f"No prototypes found, and other_targets contains uncommon MPDD "
            f"fields: {unrecognized}. Please double-check the field names."
        )
