"""Species interpretation supplied to the canonical observation engine."""

from dataclasses import dataclass
from typing import Callable, Mapping


@dataclass(frozen=True)
class ObservationPolicy:
    accepts: Callable
    evidence: Callable
    classify: Callable
    source_priority: Mapping[str, int]
    detail_to_bucket: Mapping[str, str]
    common_name: str
    scientific_name: str
    identity_prefix: str


@dataclass(frozen=True)
class CountPolicy:
    expected_columns: Mapping[str, str]
    other_column: str
    unknown_label: str
    other_bucket: str
    buckets: tuple[str, ...]
