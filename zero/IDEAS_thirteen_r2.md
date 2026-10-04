# IDEAS r2 (Thirteen): where Asmodeous should shine

**Sources.** 5 sonnet nurses, 2026-10-04, using arXiv abstracts or ar5iv pages, dataset pages, READMEs and
Wikipedia.
- Facts are summary-level.
- **UNV** marks anything a nurse could not verify; it must be checked against the primary source before it
  enters PREREG.
- A sixth batch (topology and equivariance benchmarks) did not launch because the concurrent-agent cap was
  hit. Item 12 covers it from beds we already own.

**Parts named below.**
- LK: lazy charts, i.e. top-m POVM charts that compute only when read.
- PH: phase state, gauge links and holonomy.
- CTC: the observer loop that reads predicted futures before committing.
- EV: event-driven time, i.e. per-chart send-on-delta.

**Rules for every bar.**
- Equal active FLOPs and params against the named owned baseline, under L-BR.
- 3 eval seeds plus tune seed 3.
- Kill = the stated "≤" case.

**Ranking.** Items 1–4 are cheap, small and decisive. 5–8 need a download or a port. 9–12 are stretch or
honesty items.

---

### 1. Event Camera Dataset `shapes_rotation`: SO(3) from events (EV + LK + PH)
- **Data:** https://rpg.ifi.uzh.ch/davis_data.html ; paper https://arxiv.org/abs/1610.08336.
  - License CC BY-NC-SA 3.0, non-commercial; research use is fine.
  - `shapes_rotation.zip` is 0.16 GB (HEAD request).
  - Contains DAVIS events, IMU, and mocap pose with orientation, under pure rotation at increasing speed.
- **Why Asmodeous shines here:**
  - The input is *already* events, so EV is native: a chart over a quiet image region never fires.
  - The target is a path on SO(3). Zhou et al. (https://arxiv.org/abs/1812.07035) prove that every
    representation in ≤ 4 real dimensions is discontinuous. That is the C1 atlas argument on real data.
  - Holonomy of the rotation path (PH) is measurable against mocap.
- **Bar:**
  - Angular-velocity error and h = 10 orientation error (deg) ≤ 0.8× a 6D-representation single-chart
    baseline (Zhou's continuous 6D).
  - Chart events per second ≤ 0.5× the input event rate, at equal FLOPs.
- **Kill:** ≥ 0.95×, or no compute saving.
- **Note:** no published small-model number on this sequence was found (UNV). Our 6D baseline is the
  comparator.

### 2. Stochastic Moving MNIST: multimodal futures at collisions (LK + CTC)
- **Data:** SVG paper https://arxiv.org/abs/1802.07687.
  - Generated on the fly from MNIST, 64×64, 5 context frames, predict 10. A digit leaves a wall in a *random*
    direction.
  - License follows MNIST (UNV).
- **Why Asmodeous shines here:**
  - The bounce is a branching transition, which is exactly where MoP-JEPA's collapse argument applies.
  - The CTC loop reads candidate futures per chart.
  - Lazy charts cover distinct bounce modes.
- **Bar:** best-of-K SSIM at the post-collision frames ≥ SVG-LP's (re-run at equal FLOPs; the paper's
  numbers are UNV). Mode coverage of bounce directions ≥ 0.8 with a router-gated readout (T4b protocol:
  codebook and shuffle controls).
- **Kill:** ≤ a dense single predictor plus a codebook control.

### 3. TEM-style and CSCG-style synthetic graph worlds: loops, aliasing, shortcuts (PH + LK + gluing)
- **Sources:**
  - CSCG: https://github.com/vicariousinc/naturecomm_cscg (MIT, archived; bioRxiv 10.1101/864421). A 6×8 room
    with 4 unique observations recovers perfectly; error correction fixes 50/55.
  - TEM: https://github.com/djcrw/generalising-structural-knowledge (license UNV). Square, hex, line and
    family-tree worlds plus a 4-lap circular track.
  - Both are generated synthetically, far under 1 GB, and run on CPU.
- **Why Asmodeous shines here:**
  - Aliased observations, where one view belongs to many places, are a chart-selection problem.
  - Loop closure is holonomy: a learned connection must return trivial holonomy around a contractible loop
    and a nontrivial one around a non-contractible one, as on a lap track or a torus world.
  - We also add a Klein-bottle grid world, which CSCG and TEM never test. Its sign holonomy is −1, the
    O7/O8 π law.
- **Bar:**
  - Transition-graph recovery ≥ CSCG at 6×8 and 9×11. CSCG degrades at 9×11, so that is the
    discriminating size.
  - Holonomy sign correct on ≥ 95% of loops for torus vs Klein worlds.
- **Kill:** below CSCG at 9×11, or identical holonomy patterns on torus and Klein.

### 4. HNN/LNN pendulum, double pendulum, two-body: phases on S¹/T², conserved quantities (PH + Cayley)
- **Sources:**
  - HNN: https://github.com/greydanus/hamiltonian-nn (Apache-2.0); https://arxiv.org/abs/1906.01563. Tiny
    data.
  - LNN: https://github.com/MilesCranmer/lagrangian_nns (license UNV); https://arxiv.org/abs/2003.04630.
- **Published numbers:**
  - HNN: two-body energy MSE 6.3e4 → 39 (baseline → HNN); ideal pendulum 42e-3 → 25e-3.
  - LNN: double-pendulum energy error 8% → 0.4% of max potential energy.
- **Why Asmodeous shines here:**
  - Unitary (Cayley) chart steps conserve norm by construction.
  - Angles are phases, so their wrapping is native and not an L2 discontinuity.
  - The double pendulum is chaotic, which makes it the TO4 bed: λ₁ > 0 in the latent while the integer
    readout stays protected.
- **Bar:** double-pendulum energy error ≤ LNN's 0.4% and two-body energy MSE ≤ HNN's 39, at ≤ the papers'
  model sizes (3×200 and 4×500). TO4: λ₁ > 0 with flip rate < 1%.
- **Kill:** energy error > baseline MLP / 2.

### 5. DMC swing-ups from pixels (pendulum, cartpole, acrobot): T² planning (CTC + PH)
- **Source:** https://github.com/google-deepmind/dm_control (Apache-2.0).
- **Published numbers:**
  - PlaNet cartpole swingup 821 return from 64×64 pixels after 1,000 episodes, 10–20 h on a V100
    (https://arxiv.org/pdf/1811.04551).
  - TD-MPC2 1M multitask model costs 3.7 GPU-days on a 3090 (https://arxiv.org/pdf/2310.16828).
- **Why Asmodeous shines here:** angles live on S¹ or T², and swing-up is energy pumping through a branching
  decision (which way to swing). It is the public twin of our T5 MuJoCo swing-up.
- **Bar:** cartpole-swingup return ≥ 821 at ≤ 1,000 episodes on the 4060, in ≤ 24 h wall time.
- **Kill:** < 700.

### 6. DINO-WM tasks (Wall, Reach, PointMaze, PushT): latent planning with CEM (CTC + LK)
- **Source:** https://github.com/gaoyuezhou/dino_wm (MIT); https://arxiv.org/abs/2411.04983.
- **Published numbers:**
  - DINO-WM success: Maze 0.98, Wall 0.96, Reach 0.92, PushT 0.90.
  - On PushT, DreamerV3 scores 0.30 and TD-MPC2 0.00.
  - About 19M params. CEM planning takes 53 s on an A6000.
- **Why Asmodeous shines here:**
  - The CEM cost is the bottleneck. A CTC loop that reads futures only on lazy charts should cut planning
    time.
  - Wall is a door-branching task. MoP-JEPA ported its head into DINO-WM, so a direct comparison exists.
- **Bar:** success within 0.03 of DINO-WM on Wall and PushT at ≤ 0.25× planning MACs.
- **Kill:** > 0.1 below DINO-WM at equal planning compute.

### 7. OGBench beyond pointmaze (antmaze-stitch, teleport, puzzle-3x3): stitching and branching (LK + CTC)
- **Source:** https://github.com/seohongpark/ogbench (MIT); https://arxiv.org/abs/2410.20092. We already
  have it installed in a venv.
- **Published numbers:** antmaze-medium-stitch HIQL 94%; puzzle-3x3-play GCIQL 95%. Pointmaze-teleport's
  ceiling is 0.49 with an oracle executor (our measurement).
- **Why Asmodeous shines here:**
  - Stitch data creates successor branching, so T4b's graph protocol transfers directly.
  - The ant's joints are a T^8 torus.
  - The puzzle has discrete combinatorial futures.
- **Bar:** antmaze-medium-stitch execution ≥ 0.5 with a world model plus graph search. The published GCRL
  numbers are external context, as in MoP-JEPA v1.
- **Kill:** ≤ dense JEPA plus the same planner.

### 8. TUM-VI / EuRoC attitude from IMU: SO(3) with drift (PH + EV)
- **Sources:**
  - TUM-VI: https://arxiv.org/abs/1804.06120. IMU 200 Hz, mocap 120 Hz, 1024² camera at 20 Hz. Size and
    license UNV.
  - Baseline: Brossard et al., gyro-denoising open-loop attitude (https://arxiv.org/abs/2002.10718). Its
    number is UNV from the abstract.
- **Why Asmodeous shines here:**
  - Attitude is a path on SO(3), and drift is a holonomy error.
  - The quaternion double cover q ≈ −q is a two-sheet chart problem
    (https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation).
- **Bar:** open-loop attitude error ≤ 0.9× Brossard at equal params. This needs its paper's numbers pinned
  first.
- **Kill:** ≥ 1.0×.

### 9. DVS128 Gesture: event-driven compute vs accuracy (EV)
- **Source:** https://research.ibm.com/interactive/dvsgesture/. 11 classes, 29 subjects. License and size
  UNV.
- **Published numbers:** TrueNorth 96.5% (IBM page). SLAYER 93.64% (https://arxiv.org/abs/1810.08646,
  search snippet).
- **Comparator for compute:** asynchronous sparse convolutions reduce compute "up to 20×"
  (https://arxiv.org/abs/2003.09148). That is the only verified sparse-compute number.
- **Why Asmodeous shines here:** per-chart send-on-delta should beat frame-dense compute by a large factor.
- **Bar:** accuracy ≥ 93.6% at ≤ 0.1× the dense-frame MACs.
- **Kill:** accuracy < 90% at that budget.
- **Caveat:** this is classification, not world modelling. It is an EV systems demo, not a C-claim.

### 10. Atari-100k MsPacman: the tunnel is a teleport/periodic edge (PH + LK)
- **Source:** ALE (GPL-2.0). IRIS mean human-normalised score 1.046 (https://arxiv.org/abs/2209.00588).
- **Why Asmodeous shines here:** the wrap-around tunnel makes the maze a cylinder; one chart tears there.
  Sticky actions add stochasticity.
- **Bar:** MsPacman score ≥ IRIS's per-game score at equal world-model params. The per-game number is UNV,
  so pin it first.
- **Kill:** below.
- **Cost:** heavy. Stretch only.

### 11. Crafter: long-horizon partial observability (CTC + replay)
- **Source:** https://github.com/danijar/crafter (MIT). Scores: DreamerV3 14.5±1.6, Curious Replay 19.4±1.6,
  human 50.5.
- **Why Asmodeous shines here:** Curious Replay beats DreamerV3 with *replay prioritisation*, which is the
  same lever as our rare-mode replay (IDEAS_thirteen #5).
- **Bar:** Crafter score ≥ 14.5 at 1M steps on the 4060.
- **Kill:** < 10.
- **Cost:** stretch.

### 12. Owned topology beds as the public face (PH readouts)
- **Beds:** Berry vs Berry0, S², T², Klein, SO(3) (`beds/`, already built).
- **Why:** no public benchmark tests non-orientability or Chern numbers. Topological autoencoders and chart
  AE (1912.10094) only report reconstruction and persistence diagrams.
- **Plan:** a small public "topology-of-world-models" suite, consisting of our beds plus a Klein grid world
  (item 3), released as one package of synthetic generators (MIT, under 1 GB).
- **Bar:** the TO1/TO7 bars as preregistered. Topological-AE (https://arxiv.org/abs/1906.00722) runs as an
  owned baseline. That entry is UNV; batch 6 never launched.

---

## Picks

**Run first, all cheap and decisive.**
- #1 `shapes_rotation`: SO(3) and events, real data, 0.16 GB.
- #3 synthetic graph worlds with a Klein variant: topology, CPU only.
- #4 double pendulum: phases, chaos and TO4.
- #2 SM-MNIST: branching futures.

**Then** #5 and #6, the public planning benches that compare directly against PlaNet, DINO-WM and
MoP-JEPA's DINO-WM port.

**UNV items to pin before PREREG:**
- the SVG and Brossard numbers;
- TUM-VI size and license;
- the DVS Gesture license;
- the IRIS per-game MsPacman score.
