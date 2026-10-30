# V0.6.1 validation — 2026-10-29

The 68-test regression suite passed after updating two UI tests for the conditional model selector (66 passed initially; all six tests in the affected UI module passed on rerun). Two additional model-discovery and dropdown tests passed. API responses were mocked; no real account credentials were used. Windows startup was not executed in this environment.

The default continue-versus-augment contrast reproduces difference -0.593787 and raw p=0.009817, with ** on the chart. A rendered chart was visually checked. Exported pairwise p-values, assistant guidance and annotations are all unadjusted.

Coverage includes:

- Eight observed paths, six embedded strategies and deterministic responder maintenance.
- Exact stress reduction calculation and responder probabilities 1 versus nonresponder 1/3.
- Hand-calculated six-strategy IPW values with responder sharing.
- Categorical Q2 recovery of non-monotone A/B/A+B effects; missing declared arms rejected.
- Null scenario removes assignment, switching and augmentation effects.
- Combination coefficients and dosage change augmented trajectories only.
- Holdout isolation and all 28 paired comparisons with unadjusted two-sided p-values.
- Legacy binary engine regression tests, probability checks and timing validation.
- Coping custom-label CSV roundtrip, raw-stress/utility conversion, and rejection of inconsistent reduction.
- Default UI, dosage changes, augmentation on/off, upload, custom variables, stale results and exports.
- AI revision tasks, selected atomic edits, before/after history and stale-proposal guards.
- Current coping sources and design/figure counts. Tree and bar chart rendered and visually checked.

Real OpenAI/Ollama calls, actual Windows launcher execution, empirical effectiveness,
clinical measurement validity and research contribution have not been verified.
Mocked AI protocol tests do not establish model quality or account access.
The bundled upload example is synthetic and must not be presented as an employee study.
