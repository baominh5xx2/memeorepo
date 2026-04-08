from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


IGNORE_LABEL = 255
BACKGROUND_LABEL = 0
TUMOR_LABEL = 1
STROMA_LABEL = 2


@dataclass(frozen=True)
class DomainLabelSpec:
    domain_name: str
    index_to_target: Dict[int, int]
    rgb_to_target: Dict[Tuple[int, int, int], int]


# BCSS palette documented in dataset readme.
BCSS_SPEC = DomainLabelSpec(
    domain_name="BCSS",
    index_to_target={
        0: TUMOR_LABEL,
        1: STROMA_LABEL,
        2: IGNORE_LABEL,
        3: IGNORE_LABEL,
        4: BACKGROUND_LABEL,
    },
    rgb_to_target={
        (255, 0, 0): TUMOR_LABEL,
        (0, 255, 0): STROMA_LABEL,
        (0, 0, 255): IGNORE_LABEL,
        (153, 0, 255): IGNORE_LABEL,
        (255, 255, 255): BACKGROUND_LABEL,
    },
)


# LUAD palette documented in dataset readme.
LUAD_SPEC = DomainLabelSpec(
    domain_name="Hist",
    index_to_target={
        0: TUMOR_LABEL,
        1: IGNORE_LABEL,
        2: IGNORE_LABEL,
        3: STROMA_LABEL,
        4: BACKGROUND_LABEL,
    },
    rgb_to_target={
        (205, 51, 51): TUMOR_LABEL,
        (0, 255, 0): IGNORE_LABEL,
        (65, 105, 225): IGNORE_LABEL,
        (255, 165, 0): STROMA_LABEL,
        (255, 255, 255): BACKGROUND_LABEL,
    },
)
