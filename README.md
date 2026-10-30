# SMART Discovery Lab — V0.6

AI-assisted design and analysis of adaptive employee coping support.

## Run on Windows

Extract into a new folder, close the earlier app, and double-click
`START_WINDOWS.bat`. Python 3.12 is required. The launcher installs requirements
and serves the app on this PC at `http://localhost:8501`.

## Default example

A is problem-focused coping support (identify controllable work stressors and
practice action planning). B is emotion-focused support (reappraisal, emotion
regulation and relaxation). These are illustrative program definitions, not
validated treatment manuals. No work-arrangement change is required.

Randomize A/B first. At the interim assessment, reduction = baseline stress minus
interim stress. Reduction >= the prespecified threshold means response and
maintenance. Other employees are randomized equally to continue, switch or
augment (retain the original component and add the other). Everyone has the
same final measurement time. Eight observed paths support six complete embedded
strategies. The results also show two learned policies: eight displayed policies
are not eight randomized groups. Disabling augmentation returns six paths and
four embedded strategies.

Combination dosage is explicit: equal total time (split between components), or
added time (full delivery of each component). Changing this control changes the
simulation and saved protocol. A+B superiority does not by itself establish
mechanism synergy, particularly with extra contact time. Skills learned earlier
may persist after switching scheduled support.

## Workflow

1. Inspect/edit the variables and proposed hypotheses in Theory & evidence.
2. Inspect the protocol, measurement map and tree in SMART design.
3. Set dosage, coefficients, threshold and sample size; generate a synthetic run,
   or upload a completed compatible randomized trial.
4. Compare policies and paired contrasts. Bar charts show raw final stress (lower
   is better); policy tables use utility = negative stress (higher is better).
5. In Research assistant, choose a live provider and use Develop dynamic
   hypotheses, Check and improve design, or Plan next study from results.
   Review the suggested changes and select individual fields before applying.
6. After edits, rerun or reimport. Old results remain marked stale. Export the
   run bundle, including source records and before/after revision history.

AI field edits do not autonomously change simulation equations, arm availability,
dosage, or clinical measurement validity. Sidebar controls implement the listed
simulation choices. Offline mode is a deterministic checklist, not AI dialogue.
Live OpenAI calls require your local API key; Ollama requires a running local
model. No credentials or employee rows are placed in assistant context. The
assistant sees study settings, source summaries, aggregate results and recent
conversation. Do not put secrets in questions; questions are included in exports.

## Evidence and limitations

The default catalog now contains Folkman et al. (1986), Folkman & Lazarus (1985),
Demerouti et al. (2001), and Sonnentag & Fritz (2007), with original publication
links, evidence-status notes and boundaries on what each source establishes.
The original learning/scaffolding catalog has been removed from the default
application. Sources motivate constructs; no source supplies the simulation
coefficients or validates this SMART. Earlier coping theory already treats
coping as dynamic: the proposed contribution is facilitating rigorous tests of
sequential support decisions, not inventing the idea of dynamic coping.

Synthetic scores are arbitrary 0–10 work-stress units, not a clinical scale.
Only final stress is modeled; burnout and recovery require distinct measures.
Names remain editable but do not calibrate a different numerical mechanism.
Uploaded probabilities, response timing, adherence, interference and missingness
require research review. No clinical advice or real employee treatment is provided.
See METHODS.md, REFERENCES.md and VALIDATION.md for details.

## v0.6.1 review update
- OpenAI model dropdown, optional account model discovery, and a custom-ID fallback. Astra is the initial suggestion; access must be verified with your own key.
- Pairwise tables, figure stars, assistant context and exports use unadjusted two-sided p-values. Confidence intervals are pointwise. No family-wise error control is claimed.
- Slimmer policy bars, restrained colors, improved spacing and explicit p-value annotation.
