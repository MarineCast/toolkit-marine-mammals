"""Bind reusable source adapters to the killer-whale source contract."""

from marine_mammal_toolkit.tools.observations.process.adapters import (
    adapt_snapshot as _adapt_snapshot,
)
from .source_policy import twm_evidence, TWM_IDENTITY_FIELDS
from .interpretation import OBSERVATION_POLICY


def adapt_snapshot(source, snapshot):
    return _adapt_snapshot(
        source,
        snapshot,
        scientific_name=OBSERVATION_POLICY.scientific_name,
        twm_evidence=twm_evidence,
        twm_identity_fields=TWM_IDENTITY_FIELDS,
    )
