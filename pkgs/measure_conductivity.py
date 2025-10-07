import kwant
import numpy as np
from math import pi
import logging

from .system_setup import make_system

# Configure logging (replace prints)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def measure_conductivity_once(params):
    """
    Worker function to build system, finalize, compute scattering matrix and return
    processed transmission quantity.

    Args:
        params: dict with keys:
            'V' : disorder amplitude
            'L' : system length
            'W' : width
            't_a','t_b','t_c' : base hopping arrays
            'E' : energy
            'rng_seed' (optional) : int for reproducible realization
    Returns:
        float x (the script's transformed log/T measure), or np.nan for invalids.
    """
    V = params['V']
    L = params['L']
    W = params['W']
    t_a = params['t_a']
    t_b = params['t_b']
    t_c = params['t_c']
    E = params['E']
    seed = params.get('rng_seed', None)

    rng = np.random.default_rng(seed)

    syst = make_system(L=L, W=W, t_a=t_a, t_b=t_b, t_c=t_c, V_disorder=V, rng=rng)
    try:
        syst_f = syst.finalized()
        smatrix = kwant.smatrix(syst_f, E)
        T = smatrix.transmission(1, 0)
    except Exception as exc:
        # On failure, log and return nan to be filtered by caller
        logger.debug("kwant smatrix failure for L=%s seed=%s : %s", L, seed, exc)
        return np.nan

    # Process T into x similar to your original logic but cleaner:
    if T == 0:
        return np.nan  # signal invalid / zero transmission
    if T >= 1.0:
        # In original you had special sentinel values; better to return a numeric but flagged
        return np.nan

    # Original code computed logg = np.log(T) and returned it
    return float(np.log(T))
