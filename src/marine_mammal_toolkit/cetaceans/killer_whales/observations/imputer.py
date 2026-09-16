"""Killer-whale composition of reusable selective imputation algorithms."""

from marine_mammal_toolkit.tools.observations.impute.components import (
    ImputationComponents,
)
from marine_mammal_toolkit.tools.observations.impute.model import (
    SelectiveDateContextImputer as BinaryImputer,
)
from marine_mammal_toolkit.tools.observations.impute.policy import AcceptancePolicy
from .features import DateContextFeatureBuilder, prepare_model_frame
from .imputation_policy import other_support_veto


class KillerWhaleImputer(BinaryImputer):
    def __init__(self, config=None):
        super().__init__(
            config,
            components=ImputationComponents(
                feature_builder=DateContextFeatureBuilder,
                prepare_frame=prepare_model_frame,
                acceptance_policy=AcceptancePolicy,
                support_veto=other_support_veto,
            ),
        )
