
import kwant
import numpy as np
from math import pi
import logging


# Configure logging (replace prints)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# LATTICE: create once and pass references around
zig_zag = kwant.lattice.general(
    [(1, 0), (np.sin(pi / 6), np.cos(pi / 6))],
    [(0, 0), (1 / 2, 1 / (np.sqrt(3) * 2))],
)
a, b = zig_zag.sublattices


def make_lead(t_a, t_b, t_c, W):
    """
    Create a translationally-invariant lead builder, a semi-infinite lattice oriented in the -x dir.
    Args:
        t_a, t_b, t_c: lists/arrays of hoppings (length W or W-1 as appropriate)
        W: width (int)
    Returns:
        kwant.Builder (unfinalized)
    """
    lead = kwant.Builder(kwant.TranslationalSymmetry(zig_zag.vec((-1, 0))))
    # On-site zeros
    lead[(a(0, j) for j in range(W))] = 0.0
    lead[(b(0, j) for j in range(W))] = 0.0

    for y in range(W):
        lead[a(0, y), b(0, y)] = float(t_a[y])
        lead[b(0, y), a(1, y)] = float(t_b[y])
    for y in range(W - 1):
        lead[b(0, y), a(0, y + 1)] = float(t_c[y])

    return lead
