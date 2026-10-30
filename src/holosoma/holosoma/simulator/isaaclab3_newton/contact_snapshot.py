"""Read the actual MJWarp constraint decision, not a geometric approximation.

Arrays have solver contact indexing, NOT Newton raw-contact indexing. Distances
and positions belong to the last solver contact evaluation, not recomputed from
the post-integration pose. Force is deliberately a separate recording channel.
"""
from somaforge_core.newton_contacts import (  # noqa: F401
    FIELDS, constraint_decision, snapshot_solver_contacts,
)

