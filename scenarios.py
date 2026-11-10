"""Input signal builders: steps (Sec. 5) and PRBS-like excitation (Sec. 6)."""
import numpy as np

OPERATING_POINTS = {
    # p [Pa], qs [kg/s], hf [J/kg]
    "Medium load": dict(p=8.5e6, qs=50.0, hf=1.0e6),
    "High load":   dict(p=10.5e6, qs=100.0, hf=1.0e6),
}


def step_inputs(u0, channel, delta, t_step=20.0):
    """channel in {'Q','qf','qs'}; delta in W or kg/s."""
    idx = {"Q": 0, "qf": 1, "qs": 2}[channel]
    def u(t):
        v = list(u0)
        if t >= t_step:
            v[idx] += delta
        return tuple(v)
    return u


def prbs_inputs(u0, channel, amplitude, t_end, hold=(60, 360), seed=1):
    """Random-binary sequence with random hold times (stand-in for the plant PRBS)."""
    rng = np.random.default_rng(seed)
    idx = {"Q": 0, "qf": 1, "qs": 2}[channel]
    edges, levels, t, lvl = [0.0], [0.0], 0.0, 1
    while t < t_end:
        t += rng.uniform(*hold); lvl = -lvl
        edges.append(t); levels.append(0.5 * amplitude * lvl)
    edges = np.array(edges); levels = np.array(levels)
    def u(t):
        v = list(u0)
        v[idx] += levels[np.searchsorted(edges, t, side="right") - 1]
        return tuple(v)
    return u
