# Taub: plan after the 3-hour session

## Wander (no agenda)

**Path.** Fetched 2026-10-04 on en.wikipedia.org. 20 articles opened, plus 2 that returned 404 (Harmonic
analyser, Huygens's clocks). The longest chain is 7 hops.

- Engineering → Antikythera mechanism → Epicyclic gearing → Harmonic drive
- Engineering → Al-Jazari
- Antikythera → Analog computer → Torque amplifier → Capstan equation
- Analog computer → Tide-predicting machine → Gibbs phenomenon → Mach bands → Lateral inhibition
- Torque amplifier → Servomechanism → Centrifugal governor → Escapement → Shortt–Synchronome clock
  → Hit-and-miss engine → Phase-locked loop → Delta-sigma modulation
- Engineering → Jacquard machine

**Five things that surprised me.**

1. **Capstan equation: T_load = T_hold · e^{μφ}.**
   - Five turns at μ = 0.6 multiply the force by about 1.5 × 10⁸, independent of drum radius.
   - That exponential is what made the differential analyser possible. Its ball-and-disk integrators
     were too weak to drive the next integrator until a capstan torque amplifier sat between them.
   - (Capstan equation, Torque amplifier)
2. **The Shortt clock.**
   - Its master pendulum does no work at all. It sits in a vacuum (Q 25,000 → 110,000) and is touched
     once every 30 s.
   - A slave pendulum runs the clock. A hit-and-miss synchroniser pushes the slave only when it lags.
   - Result: about 200 µs/day, one second in 12 years. (Shortt–Synchronome clock)
3. **Hit-and-miss engines decide whether to fire every cycle.**
   - At or below set speed: fire. Above it: hold the exhaust valve open and coast on a ~1,000 lb
     flywheel.
   - Event-driven compute in cast iron, 1890s–1940s. (Hit-and-miss engine)
4. **The Gibbs overshoot is a constant.**
   - A truncated Fourier series overshoots a jump by about 8.95% of the jump, however many terms are
     added: ∫₀^π sin t / t dt.
   - The fix is not more terms but a different summation (Fejér, σ-approximation).
   - The famous story that Michelson's analyser "showed" it is false. (Gibbs phenomenon)
5. **Delta-sigma modulation.**
   - A 1-bit quantiser inside a feedback integrator becomes a high-resolution converter. The loop is a
     high-pass filter on the quantisation noise and a low-pass on the signal.
   - Each doubling of the oversampling ratio buys order + ½ bits. (Delta-sigma modulation)

Also noted:
- Maxwell's 1868 "On Governors" (Centrifugal governor) was born from governors that "hunt": they
  oscillate around set speed.
- Lateral inhibition runs Mach bands, touch localisation and Notch–Delta cell-fate choice: one cell
  wins and inhibits its neighbours (Lateral inhibition).
- The escapement turns continuous energy into discrete ticks. That is the original
  continuous-to-token converter (Escapement).

**Instinct connections to Asmodeous (hunches, not findings).**

- **Hunch 1, the Shortt clock is the observer.** The QF-LM think loop is a master/slave pair.
  - ψ "thinks" freely in an isolated, high-Q unitary evolution: the master in its vacuum.
  - Measurement, the token, is the rare hit that couples it to the world: the slave.
  - Shortt's lesson is to touch the master as rarely as possible. A loop that measures every step
    lowers Q, which is the Zeno effect.
  - Testable now: hold the instrument's back-action to the null outcome for k steps and see whether
    long-range coherence improves. Kutner's zeno_probe is the obvious place.
- **Hunch 2, hit-and-miss beats throttling.**
  - My T-EV numbers say per-row skip loses on the GPU. The engine says skip whole cycles and let the
    flywheel carry state.
  - For the LM, the flywheel is the CUDA graph. Skip the whole think segment for the whole batch when the
    batch-max surprise is below ε, so one cheap device predicate gates one graph branch.
  - Hunch: the batch-level hit/miss decision keeps most of the FLOP saving and loses none of the graph.
- **Hunch 3, delta-sigma as the event trigger.**
  - The current trigger is ‖Δ‖ ≥ ε against the last fired frame: a 1-bit quantiser with no integrator.
  - Delta-sigma says to integrate the residual error and fire when the integral crosses ε. The error
    then becomes shaped noise, not bias, and drift between triggers is bounded.
  - That is a one-line change to EventDriven.triggers and worth an A/B on the T-EV beds.
- **Hunch 4, Gibbs at chart seams.**
  - Any smooth glued latent across a seam should ring at a fixed fraction of the seam jump, however
    many charts are added.
  - If the T2 discontinuity rate plateaus with K instead of falling, that is the expected cause.
  - The owned fix is a different summation (Fejér-like smoothing of the partition of unity), not more
    charts.
- **Hunch 5, capstan amplification as depth.**
  - Each friction stage multiplies the signal exponentially, and stages chain.
  - The butterfly think unitaries are log N stages of 2×2 rotations. A small angle change compounds
    across stages the same way, which is good for expressivity and bad for stability.
  - Watch gradient norms per butterfly stage.

## Small plan (after this session, ≤ 10 steps)

1. Land `qflm_graph.py` (nurse in flight). Report whether it meets the PLAN item-4 bars: ≥ 3× eager tok/s
   and ≥ 10 tok/s serving.
2. Batch-level hit/miss think gate (Hunch 2): one device predicate, two captured graphs, selected per
   step. Bench against the per-row masked version.
3. Delta-sigma trigger (Hunch 3): integrate the residual and fire on crossing. A/B the trigger rate and
   ΔQ against the current trigger on s2/so3/swing-up, with test-first parity at ε = 0.
4. Re-bench T-EV on an idle GPU (3 runs). The current table comes from a contended single run.
5. `run_t3.py --dry-run` on both beds once the tree is committed. Hand the wall estimate to Inspector,
   including the open soft K = 256 cost (~95 GPU-h).
6. Shortt probe (Hunch 1) with Kutner: null-hold k ∈ {0, 2, 4} against long-context ppl on the item-5
   small cell.
7. Watch per-stage gradient norms on QF-LM butterflies during item 5 (Hunch 5). Report only.
8. Item 5 harness: tokens/s probe for QF-LM, GPT and GRU at matched params on the 4060 (the M0 harness
   was stopped mid-build and needs restarting on PLAN's interface).
9. Kaggle: verify Wilson's checkpoint + resume across two chained 12-h sessions on a dummy run before
   any real pretraining.
10. Keep the asmodeous strict-xfail in `test_event_driven.py` until Kutner fixes the float32 zeros at
    `asmodeous.py:237,243`, then remove the marker.
