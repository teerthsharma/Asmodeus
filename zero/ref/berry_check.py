"""Gauge-invariant Berry phases and lattice Chern numbers (Fukui-Hatsugai-Suzuki, cond-mat/0503172).

Convention: A = i<psi|d psi>, gamma = i * loop-integral <psi|d psi>, discretised as
gamma = -arg prod_j <psi_j|psi_{j+1}> over a closed loop. Surfaces are oriented by the outward normal.
"""
import numpy as np


def spin_state(n, s):
    """Spin coherent state pointing along unit vector n, for s = 1/2 or 1 (z-basis, m = +s first)."""
    t = np.arccos(np.clip(n[2], -1.0, 1.0))
    p = np.arctan2(n[1], n[0])
    c, d = np.cos(t / 2), np.exp(1j * p) * np.sin(t / 2)
    if s == 0.5:
        return np.array([c, d])
    if s == 1.0:
        return np.array([c * c, np.sqrt(2) * c * d, d * d])
    raise ValueError(s)


def berry_phase(states):
    """-arg of the closed-loop product of overlaps <psi_j|psi_{j+1}>; gauge-invariant."""
    prod = 1.0 + 0j
    for a, b in zip(states, states[1:] + states[:1]):
        prod *= np.vdot(a, b)
    return -np.angle(prod)


def solid_angle(a, b, c):
    """Signed solid angle of the geodesic triangle (a, b, c) on S^2 (Van Oosterom-Strackee)."""
    return 2 * np.arctan2(np.dot(a, np.cross(b, c)), 1 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a))


def icosphere(level):
    """Octahedron subdivided `level` times and projected to S^2; faces outward-oriented."""
    V = [np.array(v, float) for v in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]]
    F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    for _ in range(level):
        mid, G = {}, []

        def m(i, j):
            k = (min(i, j), max(i, j))
            if k not in mid:
                v = V[i] + V[j]
                V.append(v / np.linalg.norm(v))
                mid[k] = len(V) - 1
            return mid[k]

        for a, b, c in F:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            G += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        F = G
    return np.array(V), F


def chern_number(V, F, state_fn):
    """(1/2pi) * sum of triangle Berry phases over a closed outward-oriented triangulation."""
    psi = [state_fn(v) for v in V]
    return sum(berry_phase([psi[i], psi[j], psi[k]]) for i, j, k in F) / (2 * np.pi)
