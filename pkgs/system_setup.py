
import kwant
import numpy as np
from math import pi
import logging


# Configure logging (replace prints)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .lead_setup import make_lead

# LATTICE: create once and pass references around
zig_zag = kwant.lattice.general(
    [(1, 0), (np.sin(pi / 6), np.cos(pi / 6))],
    [(0, 0), (1 / 2, 1 / (np.sqrt(3) * 2))],
)
a, b = zig_zag.sublattices


def make_system(
    L,
    W,
    t_a,
    t_b,
    t_c,
    V_disorder=0.0,
    rng=None,
):
    """
    Build a kwant.Builder system with random (uniform) disorder on hoppings and/or on-site.
    This version vectorizes the random sampling for speed.

    Args:
        L: length (int)
        W: width (int)
        t_a, t_b, t_c: arrays/lists of base hoppings (lengths W, W, W-1)
        V_disorder: scalar disorder amplitude; uniform(-V_disorder, V_disorder)
        rng: numpy.random.Generator instance (if None, a new one is created)
    Returns:
        kwant.Builder with attached leads (unfinalized)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Convert to numpy arrays for easy broadcasting
    t_a = np.asarray(t_a, dtype=float)
    t_b = np.asarray(t_b, dtype=float)
    t_c = np.asarray(t_c, dtype=float)

    syst = kwant.Builder()

    # ⚡ Performance improvement:
    # Vectorize random draws for onsite and hoppings so we avoid Python-level
    # random.uniform calls in tight loops.
    # Create arrays of random perturbations that match shape of sites/hoppings.

    # On-site disorder array (shape L x W) - used if needed (here onsite is set to zero in your code)
    if V_disorder > 0.0:
        onsite_perturb = rng.uniform(-V_disorder, V_disorder, size=(L, W))
    else:
        onsite_perturb = np.zeros((L, W), dtype=float)

    # Hopping perturbations shaped to match how hoppings are assigned
    # For t_a (intra-cell): shape (L, W)
    t_a_perturb = rng.uniform(-V_disorder, V_disorder, size=(L, W)) if V_disorder > 0 else np.zeros((L, W))
    # For t_b (to next cell): shape (L-1, W)
    t_b_perturb = rng.uniform(-V_disorder, V_disorder, size=(max(L - 1, 0), W)) if V_disorder > 0 else np.zeros((max(L - 1, 0), W))
    # For t_c (between y and y+1): shape (L, W-1)
    t_c_perturb = rng.uniform(-V_disorder, V_disorder, size=(L, max(W - 1, 0))) if V_disorder > 0 else np.zeros((L, max(W - 1, 0)))

    # Add onsite terms (explicit loops required to add sites to builder)
    for x in range(L):
        for y_idx in range(W):
            onsite_value = float(0.0 + onsite_perturb[x, y_idx])
            syst[a(x, y_idx)] = onsite_value
            syst[b(x, y_idx)] = onsite_value

    # Add hoppings using pre-computed perturbations
    for x in range(L):
        for y_idx in range(W):
            # t_a is intra-cell a(x,y) <-> b(x,y)
            value = float(t_a[y_idx] + t_a_perturb[x, y_idx])
            syst[a(x, y_idx), b(x, y_idx)] = value

    for x in range(L - 1):
        for y_idx in range(W):
            value = float(t_b[y_idx] + t_b_perturb[x, y_idx])
            syst[b(x, y_idx), a(x + 1, y_idx)] = value

    for x in range(L):
        for y_idx in range(W - 1):
            value = float(t_c[y_idx] + t_c_perturb[x, y_idx])
            syst[b(x, y_idx), a(x, y_idx + 1)] = value

    # Attach identical leads created from clean hoppings
    lead_builder = make_lead(t_a=t_a, t_b=t_b, t_c=t_c, W=W)
    syst.attach_lead(lead_builder)
    syst.attach_lead(lead_builder.reversed())

    return syst

