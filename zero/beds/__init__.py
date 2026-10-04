"""Beds for T2. make_bed(name, n_traj, T, obs, seed) -> common.py Bed dict.

state = the manifold's embedded coordinates (S2 in R3, T2 in R4, Klein in R4, SO(3) in R9, R2 in R2).
It is EVAL ONLY. obs='lift' = fixed random 2-layer MLP of state to D=256 (+ noise); the lift is keyed by
bed NAME, not by seed, so train/eval/tuning splits share one observation function.

All dynamics are first order (overdamped): x_{t+1} = Exp_x(dt * (drift(x) + a_t)). Second-order physics
would make one frame non-Markov (velocity hidden) and hand the arms with a history backbone a free edge.
Klein is non-orientable and not parallelisable, so it has no global action frame: actions are zero and
the flow is driven by isotropic noise.
"""
import zlib

import numpy as np

D_OBS, H_LIFT, NOISE = 256, 64, 0.02
KR, Kr = 1.0, 0.5                          # Klein embedding radii (R > r)


# ---------- geometry ----------
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def klein_embed(u, v):
    """Embedding K -> R4, invariant under (u,v) ~ (u+2pi, -v) ~ (u, v+2pi)."""
    rho = KR + Kr * np.cos(v)
    return np.stack([rho * np.cos(u), rho * np.sin(u),
                     Kr * np.sin(v) * np.cos(u / 2), Kr * np.sin(v) * np.sin(u / 2)], -1)


def klein_uv(x):
    u = np.arctan2(x[..., 1], x[..., 0])
    rho = np.hypot(x[..., 0], x[..., 1])
    rs = x[..., 2] * np.cos(u / 2) + x[..., 3] * np.sin(u / 2)
    return u, np.arctan2(rs / Kr, (rho - KR) / Kr)


def quat_to_mat(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    return R


def rotvec(R):
    """Log map SO(3) -> R3 (rotation vector). R: [..., 9] row-major."""
    M = R.reshape(R.shape[:-1] + (3, 3))
    c = np.clip((np.trace(M, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    ang = np.arccos(c)
    ax = np.stack([M[..., 2, 1] - M[..., 1, 2], M[..., 0, 2] - M[..., 2, 0], M[..., 1, 0] - M[..., 0, 1]], -1)
    s = np.linalg.norm(ax, axis=-1, keepdims=True)
    # near pi the antisymmetric part vanishes: take the axis from the symmetric part instead
    sym = (M + np.swapaxes(M, -1, -2)) / 2 - c[..., None, None] * np.eye(3)
    i = np.argmax(np.diagonal(sym, axis1=-2, axis2=-1), -1)
    ax_pi = np.take_along_axis(sym, i[..., None, None].repeat(3, -1), -2)[..., 0, :]
    ax_pi = ax_pi * np.sign(np.take_along_axis(ax, i[..., None], -1) + 1e-300)
    axis = np.where(s > 1e-6, ax / np.maximum(s, 1e-300),
                    ax_pi / np.maximum(np.linalg.norm(ax_pi, axis=-1, keepdims=True), 1e-300))
    return axis * ang[..., None]


def geo_s2(a, b):
    return np.arccos(np.clip((a * b).sum(-1), -1, 1))


def geo_t2(a, b):
    ta, tb = np.arctan2(a[..., 1::2], a[..., 0::2]), np.arctan2(b[..., 1::2], b[..., 0::2])
    return np.linalg.norm(wrap(tb - ta), axis=-1)


def geo_klein(a, b):
    """Flat quotient metric: min over the deck group (u,v) -> (u + 2pi k, (-1)^k v + 2pi l)."""
    u1, v1 = klein_uv(a)
    u2, v2 = klein_uv(b)
    best = None
    for k in (-1, 0, 1):
        d = np.hypot(u2 - u1 + 2 * np.pi * k, wrap((-1) ** k * v2 - v1))
        best = d if best is None else np.minimum(best, d)
    return best


def geo_so3(a, b):
    return np.arccos(np.clip(((a * b).sum(-1) - 1) / 2, -1, 1))


def geo_r2(a, b):
    return np.linalg.norm(a - b, axis=-1)


# ---------- torn reference charts (EVAL/calibration only) ----------
def torn_chart(name, s):
    if name == "s2":
        return np.stack([np.arctan2(s[..., 1], s[..., 0]), s[..., 2]], -1)
    if name == "t2":
        return np.arctan2(s[..., 1::2], s[..., 0::2])
    if name == "klein":
        return np.stack(klein_uv(s), -1)
    if name == "so3":
        return rotvec(s)
    if name == "r2":
        x, y = s[..., 0] - 0.13, s[..., 1] - 0.1
        return np.stack([np.arctan2(y, x), np.hypot(x, y)], -1)
    if name == "berry0":
        return np.stack([np.arctan2(s[..., 1], s[..., 0]), s[..., 2]], -1)
    if name == "berry":                              # Hopf/toroidal coords (arg psi1, arg psi2, |psi1|): two seams
        p1, p2 = s[..., 0] + 1j * s[..., 1], s[..., 2] + 1j * s[..., 3]
        return np.stack([np.angle(p1), np.angle(p2), abs(p1)], -1)
    raise KeyError(name)


def sample_uniform(name, n, rng):
    if name == "s2":
        x = rng.normal(size=(n, 3)); return x / np.linalg.norm(x, axis=1, keepdims=True)
    if name == "t2":
        th = rng.uniform(-np.pi, np.pi, (n, 2)); return np.stack([np.cos(th[:, 0]), np.sin(th[:, 0]), np.cos(th[:, 1]), np.sin(th[:, 1])], 1)
    if name == "klein":
        uv = rng.uniform(-np.pi, np.pi, (n, 2)); return klein_embed(uv[:, 0], uv[:, 1])
    if name == "so3":
        return quat_to_mat(rng.normal(size=(n, 4)))
    if name == "r2":
        return rng.uniform(-1, 1, (n, 2))
    raise KeyError(name)


# ---------- dynamics ----------
def sim_s2(N, T, rng, render):
    dt, sig, grav = 0.1, 1.5, 0.3
    p = sample_uniform("s2", N, rng)
    act = rng.normal(0, sig, (N, T, 3))
    out = np.empty((N, T, 3))
    for t in range(T):
        out[:, t] = p
        f = act[:, t] - grav * np.array([0, 0, 1.0])
        w = dt * (f - (f * p).sum(1, keepdims=True) * p)
        a = np.linalg.norm(w, axis=1, keepdims=True)
        p = np.cos(a) * p + np.sinc(a / np.pi) * w
        p /= np.linalg.norm(p, axis=1, keepdims=True)
    return out, act, None


def hopf(psi):
    """S^3 -> S^2. psi: [..., 4] = (Re p1, Im p1, Re p2, Im p2)."""
    p1, p2 = psi[..., 0] + 1j * psi[..., 1], psi[..., 2] + 1j * psi[..., 3]
    c = np.conj(p1) * p2
    return np.stack([2 * c.real, 2 * c.imag, abs(p1) ** 2 - abs(p2) ** 2], -1)


BERRY_SIG, BERRY_OMEGA = 0.8, 1.8


def _berry(N, T, rng, sig=None, omega=None):
    """s2's n-update rule with psi horizontally transported along n_t. No gravity (it leans the walk south); instead
    an azimuthal drift omega * (e_z x n) (a rotation about z: preserves the uniform measure, so no lean) makes the
    base paths circulate and enclose solid angle, so the Berry phase is measurable at small steps."""
    dt = 0.1
    sig = BERRY_SIG if sig is None else sig
    omega = BERRY_OMEGA if omega is None else omega
    ez = np.array([0, 0, 1.0])
    psi = rng.normal(size=(N, 4)); psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    p = hopf(psi)
    act = rng.normal(0, sig, (N, T, 3))
    out, nout = np.empty((N, T, 4)), np.empty((N, T, 3))
    for t in range(T):
        out[:, t], nout[:, t] = psi, p
        f = act[:, t] + omega * np.cross(ez, p)
        w = dt * (f - (f * p).sum(1, keepdims=True) * p)
        a = np.linalg.norm(w, axis=1, keepdims=True)
        q = np.cos(a) * p + np.sinc(a / np.pi) * w
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        # minimal SU(2) rotation n -> q: R = cos(b/2) I - i (k.sigma) sin(b/2), sin(b/2) k = (n x q) / (2 cos(b/2))
        c = np.sqrt((1 + (p * q).sum(1)) / 2)
        v = np.cross(p, q) / (2 * c[:, None])
        p1, p2 = psi[:, 0] + 1j * psi[:, 1], psi[:, 2] + 1j * psi[:, 3]
        vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
        n1 = c * p1 - 1j * (vz * p1 + (vx - 1j * vy) * p2)
        n2 = c * p2 - 1j * ((vx + 1j * vy) * p1 - vz * p2)
        psi = np.stack([n1.real, n1.imag, n2.real, n2.imag], 1)
        psi /= np.linalg.norm(psi, axis=1, keepdims=True)
        p = q
    return out, nout, act


def sim_berry(N, T, rng, render):
    psi, _, act = _berry(N, T, rng)
    return psi, act, None


def sim_berry0(N, T, rng, render):
    _, n, act = _berry(N, T, rng)
    return n, act, None


def sim_klein(N, T, rng, render):
    dt, sig = 0.1, 0.5
    u, v = rng.uniform(-np.pi, np.pi, (2, N))
    out = np.empty((N, T, 4))
    for t in range(T):
        out[:, t] = klein_embed(u, v)
        du = 0.5 + 0.3 * np.cos(v)
        dv = 0.5 * np.sin(v) * np.cos(u)            # f_v(u+2pi, -v) = -f_v(u, v): a field on K
        u = u + dt * du + np.sqrt(dt) * sig * rng.normal(size=N)
        v = v + dt * dv + np.sqrt(dt) * sig * rng.normal(size=N)
    return out, np.zeros((N, T, 2)), None


def sim_r2(N, T, rng, render):
    dt, sig = 0.1, 1.0
    x = rng.uniform(-1, 1, (N, 2))
    act = rng.normal(0, sig, (N, T, 2))
    out = np.empty((N, T, 2))
    for t in range(T):
        out[:, t] = x
        y = (x + dt * act[:, t] + 1) % 4
        x = np.where(y > 2, 4 - y, y) - 1               # reflecting walls
    return out, act, None


T2_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<visual><global offwidth="64" offheight="64"/></visual>
<worldbody><light pos="0 -3 1" dir="0 1 0"/><camera name="cam" pos="0 -2.6 0" xyaxes="1 0 0 0 0 1"/>
<body><joint type="hinge" axis="0 1 0" damping="3"/><geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" mass="0.1"/>
<body pos="0 0 -0.5"><joint type="hinge" axis="0 1 0" damping="3"/><geom type="capsule" fromto="0 0 0 0 0 -0.35" size="0.025" mass="0.07"/>
</body></body></worldbody></mujoco>"""

SO3_XML = """<mujoco><option timestep="0.002" gravity="0 0 0"/>
<visual><global offwidth="64" offheight="64"/></visual>
<worldbody><light pos="0.4 -3 1" dir="0 1 0"/><camera name="cam" pos="0 -1.4 0" xyaxes="1 0 0 0 0 1"/>
<body><joint type="ball" damping="0.6"/>
<geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.035" mass="0.05"/>
<geom type="capsule" fromto="0 0 0 0 0.2 0" size="0.025" mass="0.05"/>
<geom type="box" pos="0 0 0.08" size="0.03 0.03 0.08" mass="0.05"/>
</body></worldbody></mujoco>"""


def _sim_mujoco(xml, N, T, rng, render, init, sig, to_state, skip=25):
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    rend = mujoco.Renderer(m, 64, 64) if render else None
    act = rng.normal(0, sig, (N, T, m.nv))
    states, frames = [], []
    for i in range(N):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = init(rng)
        traj, fr = [], []
        for t in range(T):
            mujoco.mj_forward(m, d)
            traj.append(to_state(d.qpos.copy()))
            if rend is not None:
                rend.update_scene(d, camera="cam"); fr.append(rend.render().mean(-1).ravel() / 255.0)
            d.qfrc_applied[:] = act[i, t]
            for _ in range(skip):
                mujoco.mj_step(m, d)
        states.append(traj); frames.append(fr)
    if rend is not None:
        rend.close()
    return np.array(states), act, (np.array(frames) if render else None)


def sim_t2(N, T, rng, render):
    return _sim_mujoco(T2_XML, N, T, rng, render, lambda r: r.uniform(-np.pi, np.pi, 2), 10.0,
                       lambda q: np.array([np.cos(q[0]), np.sin(q[0]), np.cos(q[1]), np.sin(q[1])]))


def sim_so3(N, T, rng, render):
    def init(r):
        q = r.normal(size=4); return q / np.linalg.norm(q)
    return _sim_mujoco(SO3_XML, N, T, rng, render, init, 2.4, quat_to_mat)


def step_t2(s0, s1):
    return wrap(np.arctan2(s1[..., 1::2], s1[..., 0::2]) - np.arctan2(s0[..., 1::2], s0[..., 0::2]))


def step_so3(s0, s1):
    M0, M1 = s0.reshape(s0.shape[:-1] + (3, 3)), s1.reshape(s1.shape[:-1] + (3, 3))
    rel = np.swapaxes(M0, -1, -2) @ M1                 # body-frame increment (MuJoCo ball qvel is local)
    return rotvec(rel.reshape(rel.shape[:-2] + (9,)))


BEDS = {
    "s2": dict(n=2, atlas=2, betti_z2=(1, 0, 1), geo=geo_s2, sim=sim_s2, tangent_step=lambda a, b: b - a),
    "t2": dict(n=2, atlas=3, betti_z2=(1, 2, 1), geo=geo_t2, sim=sim_t2, tangent_step=step_t2),
    "klein": dict(n=2, atlas=3, betti_z2=(1, 2, 1), geo=geo_klein, sim=sim_klein, tangent_step=None),
    "so3": dict(n=3, atlas=4, betti_z2=(1, 1, 1, 1), geo=geo_so3, sim=sim_so3, tangent_step=step_so3),
    "r2": dict(n=2, atlas=1, betti_z2=(1, 0, 0), geo=geo_r2, sim=sim_r2, tangent_step=lambda a, b: b - a),
    # TO1: total space S^3 (Hopf bundle over the s2 dynamics) vs the gamma = 0 control on the base S^2.
    # Both share one rng stream ("rng" key) so berry / berry0 with one seed have identical n_t and actions.
    "berry": dict(n=3, atlas=2, betti_z2=(1, 0, 0, 1), geo=geo_s2, sim=sim_berry, tangent_step=None, rng="berry"),
    "berry0": dict(n=2, atlas=2, betti_z2=(1, 0, 1), geo=geo_s2, sim=sim_berry0, tangent_step=None, rng="berry"),
}


# ---------- observation ----------
def _lift_weights(name, S):
    rng = np.random.default_rng(zlib.crc32(("lift-" + name).encode()))
    W1 = rng.normal(0, 2.0 / np.sqrt(S), (S, H_LIFT)); b1 = rng.normal(0, 0.5, H_LIFT)
    W2 = rng.normal(0, 1.0 / np.sqrt(H_LIFT), (H_LIFT, D_OBS))
    return W1, b1, W2


def lift(name, s):
    W1, b1, W2 = _lift_weights(name, s.shape[-1])
    return np.tanh(s @ W1 + b1) @ W2


def make_bed(name, n_traj, T, obs="lift", seed=0):
    spec = BEDS[name]
    rng = np.random.default_rng([seed, zlib.crc32(spec.get("rng", name).encode())])
    state, act, frames = spec["sim"](n_traj, T, rng, obs == "render")
    if obs == "lift":
        o = lift(name, state) + NOISE * rng.normal(size=state.shape[:2] + (D_OBS,))
    elif obs == "render":
        if frames is None:
            raise NotImplementedError(f"render not available for {name}; use obs='lift'")
        o = frames + NOISE * rng.normal(size=frames.shape)
    else:
        raise ValueError(obs)
    return dict(obs=o.astype(np.float32), act=act.astype(np.float32), state=state.astype(np.float64),
                geo=spec["geo"], meta=dict(name=name, n=spec["n"], atlas=spec["atlas"],
                                           betti_z2=spec["betti_z2"], seed=seed, obs=obs))


def closed_loops(name, n_loops, L, seed=0):
    """TO7 held-out closed loops with known free-homotopy class. s = j/L (endpoint excluded, so last -> first closes).
    t2 classes a/b/ab/0, all sign +1; klein a (orientation-reversing, sign -1) / b / 0."""
    rng = np.random.default_rng([seed, zlib.crc32(("loops-" + name).encode())])
    classes = ["a", "b", "ab", "0"] if name == "t2" else ["a", "b", "0"]
    s = np.arange(L) / L
    bump = np.sin(np.pi * s)[:, None]                  # vanishes at both ends
    cls, sign, st = [], [], []
    for i in range(n_loops):
        c = classes[i % len(classes)]
        x0 = rng.uniform(-np.pi, np.pi, 2)
        d = rng.normal(size=2); d *= 0.2 / np.linalg.norm(d)
        if c == "0":
            r = 0.6 * np.stack([np.cos(2 * np.pi * s), np.sin(2 * np.pi * s)], -1)
            x = x0 + r + bump * d     # small circle, still contractible
        else:
            if name == "t2":
                w = np.array([c in ("a", "ab"), c in ("b", "ab")], float)
                x = x0 + 2 * np.pi * s[:, None] * w + bump * d
            else:                                      # klein cover path (u0,v0) -> (u0+2pi, -v0) or (u0, v0+2pi)
                end = np.array([2 * np.pi, -2 * x0[1]]) if c == "a" else np.array([0, 2 * np.pi])
                x = x0 + s[:, None] * end + bump * d
        cls.append(c); sign.append(-1 if (name == "klein" and c == "a") else 1); st.append(x)
    th = np.stack(st)                                  # [n_loops, L, 2] cover / angle coords
    if name == "t2":
        state = np.stack([np.cos(th[..., 0]), np.sin(th[..., 0]), np.cos(th[..., 1]), np.sin(th[..., 1])], -1)
    elif name == "klein":
        state = klein_embed(th[..., 0], th[..., 1])
    else:
        raise KeyError(name)
    obs = lift(name, state) + NOISE * rng.normal(size=state.shape[:2] + (D_OBS,))
    return dict(state=state.astype(np.float64), obs=obs.astype(np.float32), cls=cls, sign=np.array(sign))
