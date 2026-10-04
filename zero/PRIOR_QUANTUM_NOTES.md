# Prior quantum work: what ran on IBM hardware, and what Asmodeus ० can take from it

Sources read on 2026-10-04 through the GitHub API (read-only, by Taub plus two sonnet nurses):
- `teerthsharma/topobridge-q` at HEAD;
- `teerthsharma/laamba-silence` `docs/V5-QUANTUM-BRIDGE.md`;
- `teerthsharma/EPSILON-PHASE` (private).

Tags: [V] means re-verified by Taub, either by recomputing from the raw logs or by running the repo code.
[N] means nurse-read and not re-verified.

The real hardware evidence is two IBM jobs, one each on ibm_marrakesh and ibm_fez, from one repo
(topobridge-q). Both use terminal measurement only. No repo contains mid-circuit measurement, Zeno-style
repeated measurement, a teleportation protocol, or a contextuality test.

## 1. What exists

**topobridge-q**
- `topobridge_q.py` is one 83 KB file.
- `_build_wormhole_circuit` (line 761) builds a 6-qubit circuit from `docs/qpu_problem_6q.json`.
  - Its two complexes are 3-edge cycle graphs, and the bridge matrix J is synthetic.
  - The circuit does the following:
    - prepares H+CX Bell pairs, left i ↔ right 3+i;
    - applies a CX chain along each cycle basis;
    - writes the message with X gates on the left register (message = 111);
    - runs 2 Trotter steps of RZZ + XX + YY over all 9 J pairs, with Rz/CRz phases;
    - ends with `measure_all()`. [V]
- Each QPU job holds 3 circuits [V]:
  - bridge ON (`coupling_scale=1.0`);
  - bridge OFF (`coupling_scale=0.0`);
  - a QAOA p=2 circuit.
- "Recovery fidelity" is P(the right 3 bits equal the message), via `_recovery_fidelity`, around line 2030.
  [V]
- Despite the names ("wormhole teleportation", `wormhole_teleportation_experiment`, line 836), the circuit
  is state preparation plus a unitary plus final readout. It has no Bell-basis measurement and no
  classically conditioned X or Z. [V for the absence of conditional ops in the ON/OFF builder; N for the
  grep over the whole repo]

**laamba-silence `docs/V5-QUANTUM-BRIDGE.md`**
- A design document only: an ASCII pipeline, a Qiskit stub containing `# ... build circuit ...`, and an
  illustrative JSON output (fidelity 0.97) not taken from any run. [N]
- `ibm_brisbane` appears only in that document, and no job id exists for it. [N]
- `docs/SCOPE.md:44` lists V5 as unchecked, and `CHANGELOG.md:79` calls it "Planned". [N]

**EPSILON-PHASE**
- A 2-D wind-tunnel simulator with a stochastic-resonance and subtractive-dither numeric layer. [N]
- It has no quantum code, no teleportation and no hardware runs (one commit, b203735). [N]

## 2. Raw hardware numbers

Files:
- `topobridge-q/docs/logs/ibm_marrakesh_job_d81tcavtjchs73bnbtg0.json`
- `topobridge-q/docs/logs/ibm_fez_job_d81tdk6gbeec73al1je0.json`

Both jobs were Heron r2 (156 qubits), status DONE, with 4096 shots per circuit, 3 circuits, and
`num_randomizations` 32. [V]

| Quantity | ibm_marrakesh | ibm_fez | Ideal (statevector) |
|---|---|---|---|
| created → finished (UTC, 2026-05-13) | 01:38:51 → 01:39:06 | 01:41:36 → 01:42:04 | — |
| billed `quantum_seconds` | 5 | 5 | — |
| P(R = 111), bridge ON | 734/4096 = 0.1792 | 691/4096 = 0.1687 | 0.2252 |
| P(R = 111), bridge OFF | 490/4096 = 0.1196 | 517/4096 = 0.1262 | 0.1250 (exact: with J = 0 the right marginal is uniform) |
| ON − OFF | +0.060 (z ≈ 7.6) | +0.043 (z ≈ 5.4) | +0.100 |
| ideal excess over 1/8 retained | 54% | 44% | 100% |
| P(R = 111), QAOA circuit | 532/4096 = 0.1299 | 506/4096 = 0.1235 | — |
| distinct 6-bit outcomes, per circuit | 64/64 | 64/64 | — |

Provenance of the table:
- Counts, timestamps and quantum seconds were recomputed by Taub from `full_counts` in the logs. [V]
- The ideal values come from `topobridge_q._build_wormhole_circuit` run as a `Statevector` on the repo's
  problem JSON. No repo file stores them. [V]
- The z values are two-proportion, shot-noise only, and ignore drift. [N]

Controls run on the ideal simulator only, with bridge ON [V]:
- message 000 gives P(R = 111) = 0.1542;
- no message injected gives 0.1542.

So about 0.071 of the 0.100 ideal excess is message-dependent. The remaining 0.029 is the bridge reshaping
the output distribution with no message at all. The QPU never ran this control.

Mitigation and cost [N]:
- Mitigation was `dynamical_decoupling.enable=True` and `twirling.enable_gates=True` (topobridge_q.py
  ~2011–2013), with transpile `optimization_level=3`.
- No readout or measurement mitigation was used: no TREX, ZNE or PEC.
- `qubit_properties` is `{}` in both logs, so no calibration snapshot exists. [V]
- Total spend was 10 QPU-seconds: 2 jobs × 12,288 shots, with no repeats and no confidence intervals.

Claims the logs do not support [N, confirm before citing]:
- README "OTOC < 0.01 on both backends": OTOC exists only from statevectors (line ~903), with simulator
  mean 0.023.
- README says T1/T2 values are logged: they are empty.
- "+50% / +34% advantage": measured against an OFF baseline that sits at chance; the absolute gains are
  0.060 and 0.043.
- The QAOA "certified" energy equals brute force by construction, because all 64 bitstrings were
  observed.
- CZ counts and depths disagree across the certificate and transpiled files: 155 vs 159 CZ for ON, and
  depth 268–297.
- laamba-silence README "Verified on IBM hardware" for topological codes: TECHNICAL.md says error
  correction is not implemented.

## 3. Lessons for Asmodeus ० (these are lessons, not facts)

- **L-Q1 Observer = choice of measurement, and the prior work never exercised it.**
  - Every prior measurement is terminal and fixed. No circuit chooses what to measure based on an earlier
    outcome.
  - So "observed → exists" has no hardware precedent here.
  - Project rule (Seal, 2026-10-04): Asmodeus ० runs on a classical GPU only; no IBM/QPU jobs, ever.
    Measurement-as-observer stays an abstraction (§2), and T6 contextuality is tested classically.
- **L-Q2 Always run the control that separates signal from reshaping.**
  - topobridge-q's ON − OFF gap mixes message transfer (0.071 ideal) with bridge reshaping (0.029 ideal).
  - The atlas analogue: T3/T7 must compare lazy against soft, and also against "router present, charts
    frozen". That separates routing gain from capacity gain.
  - Same rule as the topobridge control: one control per confound, or the effect is not attributed.
- **L-Q3 Chance-level baselines inflate relative numbers.**
  - "+50%" was measured over a baseline at chance.
  - Report absolute deltas and the chance level. For example, `npersist` = 1.0 is copy-last; quote
    deltas in those units.
- **L-Q4 Single runs are not results.**
  - One job per backend, with no repeats and no confidence intervals, is the same failure PREREG's
    3-run / 3-seed rule exists to prevent.
  - The T7 sweep showed about 30% run-to-run drift at B = 1024 (densew_1024: 970 → 696 µs). Hardware
    noise is not special.
- **L-Q5 Noise-robust readout means comparing to the ideal, not to the other arm.**
  - The useful hardware figure is "44–54% of ideal excess retained" (vs statevector).
  - For the atlas: report every arm's metric next to its own ideal or noise-free ceiling, such as T2's
    oracle chart. Report it at fixed noise.
- **L-Q6 Teleportation ↔ chart switching is an analogy with no prior implementation.**
  - Teleportation moves a state between registers using shared entanglement (the overlap) plus a
    classical message (the measurement outcome) and a conditional correction (a gluing map).
  - That has the shape of a chart transition φ_k⁻¹∘φ_j on an overlap.
  - Treat it as a possible framing for §4.5 gluing, not as evidence. The prior repos contain no
    teleportation to build on.
- **L-Q7 No QPU.** The QPU budget figures above are history. This project runs nothing on quantum
  hardware (Seal's rule). Any quantum check is a CPU statevector simulation, like §4.

## 4. The missing control: no-message runs on the ideal simulator, all 8 messages [V]

Method:
- `_build_wormhole_circuit(problem, msg, depth=2, coupling_scale, inject_message)` on the repo problem,
  exact `Statevector`.
- R is the right register; the metric is P(R = reversed(msg)), as in `_recovery_fidelity`.
- With `inject_message=False` the output does not depend on the message argument (max difference 0.0).

| msg | target R | ON | OFF | no-message ON | message gain (ON − no-msg) | reshaping (no-msg − 1/8) |
|---|---|---|---|---|---|---|
| 000 | 000 | 0.1113 | 0.1250 | 0.1113 | +0.0000 | −0.0137 |
| 001 | 100 | 0.0557 | 0.1250 | 0.0524 | +0.0033 | −0.0726 |
| 010 | 010 | 0.2054 | 0.1250 | 0.1589 | +0.0465 | +0.0339 |
| 011 | 110 | 0.1428 | 0.1250 | 0.1450 | −0.0022 | +0.0200 |
| 100 | 001 | 0.0448 | 0.1250 | 0.0393 | +0.0055 | −0.0857 |
| 101 | 101 | 0.0640 | 0.1250 | 0.2000 | −0.1360 | +0.0750 |
| 110 | 011 | 0.1349 | 0.1250 | 0.1388 | −0.0039 | +0.0138 |
| 111 | 111 | 0.2252 | 0.1250 | 0.1542 | +0.0710 | +0.0292 |
| mean | | 0.1230 | 0.1250 | 0.1250 | −0.0020 | 0.0000 |

**Fact.**
- Averaged over all 8 messages, ideal bridge-ON recovery is 0.123, which is chance.
- The message gain is positive for only 2 of 8 messages: 111 (+0.071) and 010 (+0.047).
- It is strongly negative for 101 (−0.136).
- The hardware message, 111, is the most favourable message for this instance.

**Corrected hardware gain over the no-message control (estimate, not measured).**
- The QPU never ran the no-message circuit.
- The estimate assumes hardware keeps the same fraction r of every ideal deviation from uniform that it
  kept for ON-111 (a global-depolarising assumption).
- r = (hw ON − 1/8) / (0.2252 − 1/8): 0.541 on marrakesh, 0.436 on fez.
- The predicted hardware no-message P(R = 111) is 0.125 + r × 0.0292: 0.1408 on marrakesh, 0.1377 on fez.

| Backend | ON (measured) | no-message (estimated) | gain over no-message | quoted gain over OFF |
|---|---|---|---|---|
| ibm_marrakesh | 0.1792 | 0.1408 | **+0.038** | +0.060 |
| ibm_fez | 0.1687 | 0.1377 | **+0.031** | +0.043 |

**Lesson L-Q8.**
- The prior hardware result is one favourable message out of 8. The ideal circuit transfers no message
  on average.
- Do not cite it as message transfer. The full 8-message + no-message control was run only on the
  classical CPU statevector simulator (§4). No QPU reruns: this project uses a classical GPU only.
- For Asmodeus ०: evaluate over the full input distribution, never a hand-picked input.
