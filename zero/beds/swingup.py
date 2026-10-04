"""T5 bed: torque-limited double-pendulum swing-up (MuJoCo). State space T^2 x R^2 (angles, velocities).

Same geometry and camera as beds.T2_XML, but light damping (DAMP = 0.02, not 3) and two torque-limited motors:
gear TAU = 0.5 N m, while holding link 1 horizontal against gravity needs ~0.71 N m, so the pendulum must
pump energy; it cannot lift itself quasi-statically. (DAMP, TAU) picked so the oracle ceiling holds: at
(0.005, 0.25) and (0.01, 0.5) true-dynamics CEM reaches the top but drops it; at (0.02, 0.25) it never gets up. q = 0 hangs down, q1 = pi, q2 = 0 is upright.
Control period 0.05 s (5 implicitfast steps of 0.01 s), episode 200 control steps (10 s).

state = (cos q1, sin q1, cos q2, sin q2, qd1, qd2) float64, EVAL ONLY (L-GIVEN). obs = beds.lift('swingup',
lift_state(state)) + noise; lift_state scales velocities by 1/VSCALE. Velocity is in the lift, so one lifted
frame is Markov (unlike obs='render', where velocity is hidden and a history backbone gets an edge).

Success (preregisterable): tip height above the pivot >= H_SUCC = 0.75 (max 0.85) for >= HOLD = 20
consecutive control steps (1 s) at any point within the episode horizon.

Dataset: uniform random angles on T^2, velocities N(0, 2 rad/s), actions = clipped OU noise ("noisy random
controller"). Starts cover the goal region without an expert; evaluation starts hang near rest.

evaluate(plan_fn, ...) steps all episodes in lockstep and calls plan_fn(obs [n, D]) -> act [n, 2] in [-1, 1].
Only a plan_fn with attribute .oracle = True (calibration ceiling) is also passed state=(qpos, qvel).
"""
import os
import zlib

import mujoco
import numpy as np
from mujoco import rollout as mjr

from beds import NOISE, geo_t2, lift

L1, L2, TAU, VSCALE, DAMP = 0.5, 0.35, 0.5, 20.0, 0.02
SKIP, EP_LEN, H_SUCC, HOLD = 5, 200, 0.75, 20
NTHREAD = min(16, os.cpu_count() or 1)

XML = f"""<mujoco><option timestep="0.01" integrator="implicitfast" gravity="0 0 -9.81"/>
<visual><global offwidth="64" offheight="64"/></visual>
<worldbody><light pos="0 -3 1" dir="0 1 0"/><camera name="cam" pos="0 -2.6 0" xyaxes="1 0 0 0 0 1"/>
<body><joint name="j1" type="hinge" axis="0 1 0" damping="{DAMP}"/><geom type="capsule" fromto="0 0 0 0 0 -{L1}" size="0.04" mass="0.1"/>
<body pos="0 0 -{L1}"><joint name="j2" type="hinge" axis="0 1 0" damping="{DAMP}"/><geom type="capsule" fromto="0 0 0 0 0 -{L2}" size="0.025" mass="0.07"/>
</body></body></worldbody>
<actuator><motor joint="j1" gear="{TAU}" ctrllimited="true" ctrlrange="-1 1"/>
<motor joint="j2" gear="{TAU}" ctrllimited="true" ctrlrange="-1 1"/></actuator></mujoco>"""

_M = mujoco.MjModel.from_xml_string(XML)
_DATAS = [mujoco.MjData(_M) for _ in range(NTHREAD)]
_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS
_NS = mujoco.mj_stateSize(_M, _SPEC)


# ---------- state helpers ----------
def embed(qpos, qvel):
    return np.concatenate([np.cos(qpos[..., :1]), np.sin(qpos[..., :1]), np.cos(qpos[..., 1:]),
                           np.sin(qpos[..., 1:]), qvel], -1)


def lift_state(s):
    return np.concatenate([s[..., :4], s[..., 4:] / VSCALE], -1)


def tip_height(s):
    """Tip z above the pivot: -L1 cos q1 - L2 cos(q1 + q2)."""
    c1, s1, c2, s2 = np.moveaxis(s[..., :4], -1, 0)
    return -L1 * c1 - L2 * (c1 * c2 - s1 * s2)


def geo(a, b):
    return np.sqrt(geo_t2(a[..., :4], b[..., :4]) ** 2 + (((a[..., 4:] - b[..., 4:]) / VSCALE) ** 2).sum(-1))


def longest_hold(ok):
    """ok bool [n, T] -> longest run of consecutive True per row."""
    best, run = np.zeros(len(ok), int), np.zeros(len(ok), int)
    for t in range(ok.shape[1]):
        run = np.where(ok[:, t], run + 1, 0)
        best = np.maximum(best, run)
    return best


GOAL_STATE = embed(np.array([np.pi, 0.0]), np.zeros(2))


# ---------- simulation (mujoco.rollout: the oracle uses exactly the bed's dynamics) ----------
def _init_state(qpos, qvel):
    d, out = _DATAS[0], np.empty((len(qpos), _NS))
    for i, (p, v) in enumerate(zip(qpos, qvel)):
        mujoco.mj_resetData(_M, d)
        d.qpos[:], d.qvel[:] = p, v
        mujoco.mj_getState(_M, d, out[i], _SPEC)
    return out


def simulate(qpos, qvel, act, chunk=1024):
    """qpos, qvel [n, 2], act [n, H, 2] -> (qpos, qvel) [n, H, 2] after each control step.
    Rollouts run in chunks of `chunk` (deterministic physics, so the output does not depend on chunk size;
    this only bounds peak RAM)."""
    out = np.empty(act.shape[:2] + (_NS,))
    for i in range(0, len(act), chunk):
        ctrl = np.repeat(np.clip(act[i:i + chunk], -1, 1), SKIP, axis=1)
        st, _ = mjr.rollout(_M, _DATAS, _init_state(qpos[i:i + chunk], qvel[i:i + chunk]), ctrl,
                            persistent_pool=True)
        out[i:i + chunk] = st[:, SKIP - 1::SKIP]
    return out[..., 1:3], out[..., 3:5]


def _render(qpos, rend, d):
    out = []
    for p in qpos:
        d.qpos[:] = p
        mujoco.mj_forward(_M, d)
        rend.update_scene(d, camera="cam")
        out.append(rend.render().mean(-1).ravel() / 255.0)
    return np.array(out)


class _Obs:
    def __init__(self, obs, rng):
        self.obs, self.rng = obs, rng
        if obs == "render":
            self.rend, self.d = mujoco.Renderer(_M, 64, 64), mujoco.MjData(_M)
        elif obs != "lift":
            raise ValueError(obs)

    def __call__(self, qpos, qvel):
        if self.obs == "lift":
            o = lift("swingup", lift_state(embed(qpos, qvel)))
        else:
            o = _render(qpos.reshape(-1, 2), self.rend, self.d).reshape(qpos.shape[:-1] + (-1,))
        return (o + NOISE * self.rng.normal(size=o.shape)).astype(np.float32)


# ---------- offline dataset (common.py Bed contract) ----------
def make_dataset(n_traj, T, obs="lift", seed=0, ou=0.85, sigma=0.7):
    """Noisy controller: a_t = clip(beta * sign(qd_t) + OU_t), beta ~ U(0, 1) per trajectory. The sign(qd)
    term pumps energy so damped trajectories keep visiting the upright region (data coverage only; no
    model ever sees beta or the controller)."""
    rng = np.random.default_rng([seed, zlib.crc32(b"swingup")])
    qpos, qvel = np.empty((n_traj, T, 2)), np.empty((n_traj, T, 2))
    qpos[:, 0], qvel[:, 0] = rng.uniform(-np.pi, np.pi, (n_traj, 2)), rng.normal(0, 2.0, (n_traj, 2))
    beta = rng.uniform(0, 1, (n_traj, 1))
    act = np.empty((n_traj, T, 2))
    noise = rng.normal(0, sigma, (n_traj, 2))
    for t in range(T):
        act[:, t] = np.clip(beta * np.sign(qvel[:, t]) + noise, -1, 1)
        noise = ou * noise + np.sqrt(1 - ou ** 2) * rng.normal(0, sigma, (n_traj, 2))
        if t + 1 < T:
            qp, qv = simulate(qpos[:, t], qvel[:, t], act[:, t, None])
            qpos[:, t + 1], qvel[:, t + 1] = qp[:, 0], qv[:, 0]
    o = _Obs(obs, rng)(qpos, qvel)
    return dict(obs=o, act=act.astype(np.float32), state=embed(qpos, qvel), geo=geo,
                meta=dict(name="swingup", n=4, atlas=3, betti_z2=(1, 2, 1), seed=seed, obs=obs,
                          dt=SKIP * _M.opt.timestep, tau=TAU, goal_state=GOAL_STATE))


def goal_obs(obs="lift"):
    """Noise-free goal observation (upright, at rest) for latent-goal planners."""
    if obs == "lift":
        return lift("swingup", lift_state(GOAL_STATE)).astype(np.float32)
    return _render(np.array([[np.pi, 0.0]]), mujoco.Renderer(_M, 64, 64), mujoco.MjData(_M))[0].astype(np.float32)


# ---------- evaluation ----------
def evaluate(plan_fn, n_episodes, seed, horizon=EP_LEN, obs="lift"):
    rng = np.random.default_rng([seed, zlib.crc32(b"swingup-eval")])
    qpos, qvel = rng.normal(0, 0.05, (n_episodes, 2)), rng.normal(0, 0.05, (n_episodes, 2))
    see = _Obs(obs, rng)
    oracle = getattr(plan_fn, "oracle", False)
    tip = np.empty((n_episodes, horizon))
    for t in range(horizon):
        o = see(qpos, qvel)
        a = plan_fn(o, state=(qpos.copy(), qvel.copy())) if oracle else plan_fn(o)
        qp, qv = simulate(qpos, qvel, np.asarray(a, float).reshape(n_episodes, 1, 2))
        qpos, qvel = qp[:, 0], qv[:, 0]
        tip[:, t] = tip_height(embed(qpos, qvel))
    hold = longest_hold(tip >= H_SUCC)
    ok = hold >= HOLD
    return dict(success_rate=float(ok.mean()), success=ok, hold=hold, tip=tip)


def random_policy(seed=0):
    rng = np.random.default_rng(seed)
    return lambda o: rng.uniform(-1, 1, (len(o), 2))


def oracle_mpc(horizon=30, n_samples=256, n_elites=16, iters=5, cost="height", warm=True, seed=0, wv=0.0):
    """CALIBRATION ONLY: CEM over true dynamics from the true state (mujoco state copy).
    cost='height': running sum of (L1 + L2 - tip)^2 + wv |qd / VSCALE|^2 (ceiling).
    cost='goal_l1': terminal ||lift(s_H) - lift(goal)||_1 in noise-free observation space = the
    arms.planner.cem_plan energy with a PERFECT model and identity encoder (tests the energy, not the model).
    warm=False + defaults (800, 32, 10) reproduces cem_plan's sampling exactly."""
    rng = np.random.default_rng(seed)
    gl = lift("swingup", lift_state(GOAL_STATE))
    mem = {}

    def plan(o, state):
        qpos, qvel = state
        n = len(qpos)
        mu = mem.get("mu")
        if mu is None or not warm or len(mu) != n:
            mu = np.zeros((n, horizon, 2))
        else:
            mu = np.concatenate([mu[:, 1:], np.zeros((n, 1, 2))], 1)
        std = np.ones_like(mu)
        for _ in range(iters):
            a = np.clip(mu[:, None] + std[:, None] * rng.normal(size=(n, n_samples, horizon, 2)), -1, 1)
            qp, qv = simulate(np.repeat(qpos, n_samples, 0), np.repeat(qvel, n_samples, 0),
                              a.reshape(n * n_samples, horizon, 2))
            s = embed(qp, qv)
            if cost == "height":
                c = ((L1 + L2 - tip_height(s)) ** 2 + wv * ((s[..., 4:] / VSCALE) ** 2).sum(-1)).sum(-1)
            elif cost == "goal_l1":
                c = np.abs(lift("swingup", lift_state(s[:, -1])) - gl).sum(-1)
            else:
                raise ValueError(cost)
            c = c.reshape(n, n_samples)
            el = np.argsort(c, 1)[:, :n_elites]
            elite = np.take_along_axis(a, el[:, :, None, None], 1)
            mu, std = elite.mean(1), elite.std(1)
        mem["mu"] = mu
        return mu[:, 0]
    plan.oracle = True
    return plan
