"""
BMHD: Beltrami MHD & Non-Symmetric 3D Stellarator Engine.
"""

__version__ = "0.4.0"

from bmhd.mhd import (
    IssanNonSymmetricEquilibriumEngine,
    CerfonFreidbergEquilibrium,
)
from bmhd.engine import (
    BMHDModel,
    TheoremAuditResult,
    SteppedPressureBeltramiTriad,
    SkewSymmetricFluxTriad,
    BeltramiChordShell,
    TemporalPlasmaContinuitySplitter,
    build_kelvin_mandel_so6_rotor,
    rotate_stiffness_so6,
)

__all__ = [
    "__version__",
    "IssanNonSymmetricEquilibriumEngine",
    "CerfonFreidbergEquilibrium",
    "BMHDModel",
    "TheoremAuditResult",
    "SteppedPressureBeltramiTriad",
    "SkewSymmetricFluxTriad",
    "BeltramiChordShell",
    "TemporalPlasmaContinuitySplitter",
    "build_kelvin_mandel_so6_rotor",
    "rotate_stiffness_so6",
]
