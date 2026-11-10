"""Saturated steam/water properties as polynomials in pressure (fitted to IAPWS-IF97).
All functions take p in Pa and return SI units. Derivatives are with respect to p [Pa]."""
import numpy as np
from steam_coeffs import COEFFS

_MPA = 1e6

def _val(name, p):
    return np.polyval(COEFFS[name], p / _MPA)

def _der(name, p):
    return np.polyval(np.polyder(COEFFS[name]), p / _MPA) / _MPA

def props(p):
    """Return dict with rho_s, rho_w, h_s, h_w, t_s and d/dp of each ('d' prefix)."""
    out = {}
    for k in ("rho_s", "rho_w", "h_s", "h_w", "t_s"):
        out[k] = _val(k, p)
        out["d" + k] = _der(k, p)
    out["h_c"] = out["h_s"] - out["h_w"]
    return out
