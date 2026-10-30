"""SMART Discovery Lab v0.6.1 — local, evidence-traceable SMART prototype."""
from __future__ import annotations

import io
import hashlib
import json
import os
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ai_design import AIDesignError, generate_draft
from core import SimulationConfig, simulate, run_analysis, evaluate_difference, pairwise_comparisons
from visuals import smart_tree, tree_figure, comparison_figure, figure_bytes
from data_io import import_trial
from copilot import OPENAI_MODEL_CHOICES, list_openai_models, ask, make_context, apply_edits, REVISION_TASKS, revision_question, prepare_revision
from theory import SOURCE_CATALOG, compile_design, default_theory_inputs
from research import FIELDS, defaults, validate, candidate, local_sources, literature_query, compile_research, display_frame, search_crossref, restore_engine_columns

st.set_page_config(page_title="SMART Discovery Lab · v0.6.1", page_icon="🧭", layout="wide")
st.markdown("""<style>
.block-container {padding-top:2rem;max-width:1480px;}
h1 {letter-spacing:-.035em;}
[data-testid='stMetric'] {background:#fff;border:1px solid #DFE5EF;border-radius:12px;padding:15px;}
.eyebrow {color:#177E89;font-size:.78rem;letter-spacing:.14em;font-weight:700;}
</style>""", unsafe_allow_html=True)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    raise TypeError(f"Cannot export {type(value).__name__}")


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=json_default).encode("utf-8")


def plain_table(value):
    """Keep nested design records readable and Arrow-compatible."""
    if isinstance(value, dict):
        value = [{"item": key, "detail": val} for key, val in value.items()]
    rows = []
    for row in value:
        if not isinstance(row, dict):
            row = {"detail": row}
        rows.append({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    return pd.DataFrame(rows)


if "research" not in st.session_state:
    st.session_state.research = defaults()
if "pending_local" in st.session_state:
    for key, value in st.session_state.pop("pending_local").items():
        st.session_state["edit_" + key] = value
if "inputs" not in st.session_state:
    st.session_state.inputs = default_theory_inputs()
if "approved_copilot" in st.session_state:
    proposed = st.session_state.pop("approved_copilot")
    st.session_state.research, st.session_state.inputs = proposed["research"], proposed["inputs"]
    st.session_state.setdefault("revision_history", []).append(proposed["record"])
    st.session_state.revision_notice = "Selected edits applied. Previous results are retained as an earlier run; generate new data or reimport data for the revised design. Hypothesis edits alone do not recalibrate simulation mechanisms."
    for key, value in st.session_state.research.items():
        st.session_state["var_" + key] = value
    for key, value in st.session_state.inputs.items():
        st.session_state["edit_" + key] = value
    st.session_state["response_threshold"] = float(st.session_state.inputs["response_threshold"])
    for key in ["online_sources", "online_query", "literature_terms", "local_candidate", "assistant_proposal"]:
        st.session_state.pop(key, None)
    st.session_state.setdefault("design_history", []).append({"research":dict(st.session_state.research), "inputs":dict(st.session_state.inputs), "source":"Researcher-approved assistant edits"})
if "pending_ai" in st.session_state:
    candidate = st.session_state.pop("pending_ai")
    for key in ("title", "research_question", "proposition_learning", "proposition_scaffolding", "proposition_motivation", "competing_prediction", "boundary_conditions", "response_threshold", "motivation_threshold"):
        st.session_state.inputs[key] = candidate["draft"][key]
        st.session_state["edit_" + key] = candidate["draft"][key]
    st.session_state.inputs["falsification"] = candidate["draft"]["falsification_plan"]
    st.session_state["edit_falsification"] = candidate["draft"]["falsification_plan"]
    st.session_state["response_threshold"] = float(candidate["draft"]["response_threshold"])
    st.session_state["motivation_threshold"] = float(candidate["draft"]["motivation_threshold"])
    st.session_state.ai_provenance = candidate

st.markdown('<div class="eyebrow">THEORY · DESIGN · LEARNING</div>', unsafe_allow_html=True)
st.title("SMART Discovery Lab")
st.write("AI-Assisted Design and Analysis of Adaptive Organizational Interventions")
st.caption("v0.6.1 · Research prototype · Synthetic or uploaded data · Individual employee as the unit of analysis")

with st.sidebar:
    st.header("Run an experiment")
    augmentation = st.checkbox("Include augmentation (A+B) for nonresponders", value=True, key="augmentation")
    dosage = st.selectbox("A+B contact-time convention", ["equal_time", "added_time"],
        format_func=lambda x: "Equal total time: split A / B" if x=="equal_time" else "Added time: full A plus full B", key="dosage")
    augmentation_effect = st.slider("Combination-specific effect (simulation assumption)", -1.5, 1.5, 0.2, 0.1, key="augmentation_effect")
    switch_effect = st.slider("Switching effect (simulation assumption)", -1.0, 1.0, 0.0, 0.1, key="switch_effect")
    n = st.slider("Synthetic employees", 200, 3000, 500, 100, key="n")
    seed = st.number_input("Simulation seed", min_value=0, max_value=2**31-1, value=42, step=1, key="seed")
    scenario = st.selectbox("Mechanism scenario", ["theory_aligned", "null", "reversed"], format_func=lambda x: {"theory_aligned":"Theory-aligned example", "null":"Null intervention effects", "reversed":"Reversed intervention effects"}[x], key="scenario")
    effect = st.slider("B versus A main-effect parameter (both stages)", -1.5, 2.0, 0.6, 0.1, key="effect")
    moderation = st.slider("Controllability × support matching coefficient", -1.0, 1.0, 0.3, 0.1, key="moderation")
    noise_sd = st.slider("Outcome noise SD", 0.2, 2.0, 1.0, 0.1, key="noise")
    with st.expander("Decision rules & randomization", expanded=False):
        response_threshold = st.slider("Minimum interim stress reduction", -2.0, 2.0, float(st.session_state.inputs["response_threshold"]), 0.1, key="response_threshold", help="Baseline stress minus interim stress; higher reduction is better. Prespecify before generating data.")
        motivation_threshold = 0.0  # Compatibility parameter; not a branching rule.
        p1 = st.slider("P(action 1), stage 1", 0.1, 0.9, 0.5, 0.05, key="p1")
        p2_low = st.slider("P(action 1), stage 2: below response threshold", 0.1, 0.9, 0.5, 0.05, key="p2_low", disabled=augmentation)
        p2_high = 0.5  # Legacy field is ignored; responder maintenance probability is 1.
        st.caption("Responders maintain. Nonresponders: A, B or A+B with probability 1/3 each when augmentation is enabled. Otherwise binary probabilities apply. Switching replaces scheduled support; augmentation retains and adds.")
    test_fraction = st.slider("Holdout fraction", 0.25, 0.5, 0.4, 0.05, key="test_fraction")
    run_clicked = st.button("Generate data & compare policies", type="primary", width="stretch", key="run_pipeline")
    st.caption("Generate creates a synthetic run and replaces the currently displayed uploaded-data analysis. Coefficients and thresholds are demonstration settings. They are not estimates extracted from cited papers.")

theory_tab, design_tab, data_tab, policy_tab, notes_tab, assistant_tab = st.tabs(["1 · Theory & evidence", "2 · SMART design", "3 · Data", "4 · Policy analysis", "5 · Methods & exports", "6 · Research assistant"])

with theory_tab:
    st.subheader("From established evidence to a new hypothesis")
    st.write("The local builder uses explicit, editable mappings. Free text records your reasoning; the structured controls define the executable two-stage experiment.")
    st.info("Workflow: 1. Save variables → 2. Build a local candidate (optional) and edit hypotheses → 3. Save hypotheses and rebuild design → 4. Generate data & compare policies.")
    with st.form("variables_form"):
        spec = dict(st.session_state.research)
        for key, (label, _) in FIELDS.items():
            spec[key] = st.text_input(label, value=spec[key], key="var_"+key)
        spec["action0"] = st.text_input("Intervention A (code 0): randomized independent variable", value=spec["action0"], key="var_action0")
        spec["action1"] = st.text_input("Intervention B (code 1): randomized independent variable", value=spec["action1"], key="var_action1")
        spec["direction"] = st.selectbox("Preferred outcome direction", ["Higher is better", "Lower is better"], index=0 if spec["direction"]=="Higher is better" else 1, key="var_direction")
        spec["mechanism"] = st.text_area("Proposed mechanism (research assumption)", value=spec["mechanism"], key="var_mechanism")
        spec["keywords"] = st.text_input("Literature concepts / search keywords", value=spec["keywords"], key="var_keywords", help="Update these when changing topics. Local matching is lexical, not an AI literature review.")
        if st.form_submit_button("Save variables and refresh references"):
            try:
                st.session_state.research = validate(spec)
                st.session_state.pop("local_candidate", None)
                st.session_state.pop("online_sources", None)
                st.session_state.pop("online_query", None)
                st.session_state.pop("literature_terms", None)
                st.success("Variables saved; design and local reference candidates updated. Rebuild hypotheses and rerun analysis for these constructs.")
            except ValueError as exc:
                st.error(str(exc))
    st.caption("The numeric prototype supports two baseline variables, two interim variables, two stage-1 actions, up to three stage-2 actions and one continuous outcome. Names and proposed mechanisms are editable; additional variables or new outcome families require extending the model.")
    if st.button("Build local candidate — no API key", key="generate_local"):
        st.session_state.local_candidate = candidate(st.session_state.research)
    if "local_candidate" in st.session_state:
        st.json(st.session_state.local_candidate, expanded=False)
        st.caption("Rule-based draft, not AI generation or an evidence-backed causal explanation. Review and edit its predictions.")
        if st.button("Apply local candidate to hypothesis editor", key="apply_local"):
            st.session_state.pending_local = st.session_state.local_candidate
            st.rerun()
    st.info("Use Research assistant for conversational help with your saved variables, design and results.")
    inputs = dict(st.session_state.inputs)
    with st.form("theory_form"):
        for key, label in [("title", "Study title"), ("research_question", "Research question")]:
            inputs[key] = st.text_input(label, value=inputs[key], key="edit_" + key)
        for key, label in [("proposition_learning", "Main effect hypothesis"), ("proposition_scaffolding", "Response-dependent intervention hypothesis"), ("proposition_motivation", "Moderator hypothesis")]:
            inputs[key] = st.text_area(label, value=inputs[key], height=90, key="edit_" + key)
        with st.expander("Theory development: alternatives and boundaries", expanded=True):
            for key, label in [("competing_prediction", "Competing prediction"), ("falsification", "What pattern would challenge the hypothesis?"), ("boundary_conditions", "Proposed boundary conditions")]:
                inputs[key] = st.text_area(label, value=inputs.get(key, ""), height=75, key="edit_" + key)
        if st.form_submit_button("Save hypotheses and rebuild design"):
            st.session_state.inputs = inputs
            st.success("Hypotheses saved. Use Generate data & compare policies to update the analysis snapshot.")
    st.subheader("Trace the evidence")
    st.caption("Local candidates are ranked by overlap with your saved constructs and search keywords. This small seed catalog is not comprehensive; relevance is not evidence of support.")
    matched_sources = local_sources(st.session_state.research)
    if not matched_sources:
        st.info("No local references match these concepts. Use the online search below or revise the keywords.")
    for source in matched_sources:
        heading = f"{source.get('authors', '')} ({source.get('year', '')}) · {source.get('journal', '')}"
        with st.expander(heading):
            st.markdown(f"**{source['title']}**")
            st.caption("Matched terms: " + ", ".join(source["matched_terms"]))
            url = source.get("accessible_source", source.get("url", "https://doi.org/" + source.get("doi", "")))
            st.markdown(f"[Source]({url}) · [DOI: {source.get('doi', '')}](https://doi.org/{source.get('doi', '')})")
            for key, label in [("supports", "What it supports"), ("does_not_establish", "What it does not establish"), ("design_use", "Proposed use in this study"), ("evidence_status", "Verification scope"), ("source_group", "Source group")]:
                if key in source:
                    st.write(f"**{label}:**", source[key])

    with st.expander("Search published references online — no API key", expanded=False):
        st.caption("Queries Crossref bibliographic metadata over the internet. Only your entered search terms are sent. Candidates require reading and researcher verification; no findings are generated from titles. Filtering examines the first 100 search matches, so it is not exhaustive.")
        terms = st.text_input("Search terms", value=literature_query(st.session_state.research), key="literature_terms")
        journals = st.text_area("Journal filter: exact names, one per line (blank = all journals)", value="Organization Science\nAcademy of Management Journal\nAcademy of Management Review\nAdministrative Science Quarterly\nStrategic Management Journal\nManagement Science\nMIS Quarterly\nInformation Systems Research\nJournal of Management Information Systems\nJournal of Management\nJournal of Applied Psychology\nPersonnel Psychology\nOrganizational Behavior and Human Decision Processes", key="journal_filter")
        st.caption("Editable selection of business and psychology journals; not a complete or currently certified UTD24/FT50 list. Add journals relevant to your question.")
        if st.button("Search references", key="search_references"):
            try:
                with st.spinner("Retrieving bibliographic candidates…"):
                    st.session_state.online_sources = search_crossref(terms, journals)
                    st.session_state.online_query = {"terms":terms, "journals":journals, "research":dict(st.session_state.research)}
            except Exception as exc:
                st.error("Reference search unavailable. Local design and analysis still work. " + str(exc))
        if "online_sources" in st.session_state:
            st.caption("Results for: " + st.session_state.online_query["terms"])
            if not st.session_state.online_sources:
                st.info("No matches among the first 100 candidates after filtering. Narrow the query or broaden the journal selection.")
            st.dataframe(pd.DataFrame(st.session_state.online_sources), hide_index=True, width="stretch")
            st.download_button("Download reference candidates", json_bytes({"query":st.session_state.online_query,"candidates":st.session_state.online_sources}), "reference_candidates.json", "application/json")

inputs = dict(st.session_state.inputs)
inputs.update(response_threshold=float(response_threshold), motivation_threshold=float(motivation_threshold))
st.session_state.inputs = inputs
probabilities = {"p1_guided": p1, "p2_guided_low": p2_low, "p2_guided_high": p2_high}
try:
    design = compile_research(compile_design(inputs, probabilities), st.session_state.research, augmentation, dosage)
except ValueError as exc:
    st.error(f"Design needs revision: {exc}")
    st.stop()

config = SimulationConfig(n=int(n), seed=int(seed), scaffolding_effect=float(effect), motivation_moderation=float(moderation), response_threshold=float(response_threshold), p1_guided=float(p1), p2_guided_low=float(p2_low), p2_guided_high=float(p2_high), scenario=scenario, noise_sd=float(noise_sd), case="coping", augmentation=augmentation, augmentation_effect=augmentation_effect, switch_effect=switch_effect, augmentation_dose=dosage)
fingerprint = json.dumps({"config": asdict(config), "inputs": inputs, "research": st.session_state.research, "holdout_fraction": test_fraction}, sort_keys=True)
if "snapshot" not in st.session_state or run_clicked:
    try:
        with st.spinner("Generating the SMART sample and fitting policies on the training set…"):
            df = simulate(config)
            analysis = run_analysis(df, test_fraction=float(test_fraction), split_seed=2026, motivation_threshold=float(motivation_threshold))
            st.session_state.snapshot = {"data": df, "analysis": analysis, "design": design, "config": asdict(config), "fingerprint": fingerprint, "created_at_utc": datetime.now(timezone.utc).isoformat(), "origin":"synthetic", "ai_provenance": st.session_state.get("ai_provenance", {}), "research":dict(st.session_state.research)}
    except ValueError as exc:
        st.error(f"The experiment could not be analyzed: {exc}")
        st.stop()
snapshot = st.session_state.snapshot
analysis = snapshot["analysis"]
df = snapshot["data"]
stale = snapshot["fingerprint"] != fingerprint
if stale:
    st.warning("Settings have changed. Data and policy results below still describe the last completed run. Press Generate data & compare policies to update them.")

with design_tab:
    st.subheader("A two-stage SMART with explicit decision histories")
    st.session_state.setdefault("design_history", [])
    if st.button("Save current design version", key="save_design_version"):
        st.session_state.design_history.append({"research":dict(st.session_state.research),"inputs":dict(inputs),"source":"Researcher saved version", "created_at_utc":datetime.now(timezone.utc).isoformat()})
        st.success("Design version saved for this session; include it in a downloaded run bundle to retain it.")
    if st.session_state.design_history:
        with st.expander("Compare current design with a saved version (no new data needed)"):
            vi = st.selectbox("Saved design version", list(range(len(st.session_state.design_history))), format_func=lambda x: "Version " + str(x+1), key="design_version")
            old = st.session_state.design_history[vi]
            differences=[]
            for group, current in [("research",st.session_state.research),("inputs",inputs)]:
                for k,value in current.items():
                    if old[group].get(k)!=value:
                        differences.append({"field":group+"."+k,"saved":str(old[group].get(k)),"current":str(value)})
            if differences:
                st.dataframe(pd.DataFrame(differences),hide_index=True,width="stretch")
            else:
                st.write("No differences from the selected version.")
    st.caption("This preview reflects the current saved hypotheses and controls. Download the completed run in Methods & exports to preserve the design attached to its results.")
    spec = st.session_state.research
    tree = tree_figure(spec, augmentation)
    st.pyplot(tree)
    st.download_button("Download study tree (SVG)",figure_bytes(tree),"smart_study_tree.svg","image/svg+xml")
    st.info(f"{8 if augmentation else 6} observed paths, {6 if augmentation else 4} embedded strategies. Response is measured, not randomized; paths are not independent randomized groups. Everyone has the same final follow-up time.")
    st.json(design["protocol"], expanded=True)
    st.caption("Response criterion: " + spec["interim_skill"] + " >= " + str(response_threshold) + ". Only this interim measure controls eligibility; the moderator does not.")
    st.dataframe(pd.DataFrame([
        {"strategy":f"{letter} first / {option}", "stage_1":spec[f"action{first}"],
         "responder":f"Maintain {letter}", "nonresponder":option}
        for first,letter in [(0,"A"),(1,"B")]
        for option in (["Continue", "Switch", "Augment A+B"] if augmentation else ["Continue", "Switch"])
    ]), hide_index=True, width="stretch")
    st.dataframe(plain_table(design["stages"]), hide_index=True, width="stretch")
    st.markdown("#### Measurement map")
    st.dataframe(plain_table(design["measurement_map"]), hide_index=True, width="stretch")
    st.markdown("#### Structural checks")
    st.dataframe(plain_table(design["checks"]), hide_index=True, width="stretch")
    st.caption("Passing structural checks does not establish construct validity, no interference, adequate power, or ethical acceptability of a real study.")
    with st.expander("Theory development and estimands"):
        st.json({k: design[k] for k in ("theories", "hypothesis_tests", "estimands", "assumptions") if k in design}, expanded=False)
    st.download_button("Download current design preview", json_bytes(design), "smart_design_preview.json", "application/json", key="download_design")

with data_tab:
    st.subheader("Upload a completed two-stage SMART")
    st.caption("Use the current custom-label template, or technical engine headers from a run bundle. Custom headers are mapped using the saved measurement names. Stage 1 must be randomized; only nonresponders are rerandomized at stage 2. Responders must maintain their initial action and have p2_observed = 1. Observational data and all-participant stage-2 randomization are not supported.")
    template = simulate(SimulationConfig(n=40, case="coping", augmentation=augmentation)).iloc[:0]
    st.download_button("Download CSV column template", display_frame(template,st.session_state.research).to_csv(index=False).encode(), "smart_upload_template.csv", "text/csv")
    with st.expander("Required format and timing"):
        st.write("One row per employee. Stage 1: 0=A, 1=B. Stage 2: 0=A, 1=B, 2=A+B when augmentation is enabled. Interim response is baseline minus interim stress; if interim_stress is supplied, this equality is checked. p1_observed and p2_observed are the probabilities of the action ACTUALLY RECEIVED, conditional on pre-decision history; they are not always P(action 1). Use raw outcome values, not sign-reversed utility. high_progress is optional and derived from the current threshold.")
        st.write("Optional timestamps: baseline_time, stage1_time, interim_time, stage2_time, outcome_time. Include all five; ISO date/time values must be strictly increasing. Without timestamps, measurement timing requires researcher attestation.")
        st.write("All required numeric fields must be complete and finite. No imputation is performed. Missingness, clustering, interference and treatment adherence require a study-specific analysis plan. Randomization cannot be verified from a CSV alone.")
    uploaded = st.file_uploader("Employee trial CSV", type=["csv"], key="trial_upload")
    randomized = st.checkbox("I confirm stage 1 was randomized, only nonresponders were rerandomized, responders maintained their first action, and supplied probabilities match that protocol.", key="confirm_randomized")
    timing = st.checkbox("I confirm baseline preceded stage 1, interim measurements preceded stage 2, and outcome followed stage 2.", key="confirm_timing")
    if st.button("Validate upload & analyze", key="analyze_upload"):
        try:
            if uploaded is None or not randomized:
                raise ValueError("Select a CSV and confirm its randomization protocol.")
            raw = restore_engine_columns(pd.read_csv(uploaded), st.session_state.research)
            actual = import_trial(raw, float(response_threshold), st.session_state.research["direction"], timing, augmentation=augmentation, coping=True)
            fitted = run_analysis(actual, test_fraction=float(test_fraction), split_seed=2026, motivation_threshold=float(motivation_threshold))
            st.session_state.snapshot = {"data":actual,"analysis":fitted,"design":design,"config":asdict(config),
                "fingerprint":fingerprint,"created_at_utc":datetime.now(timezone.utc).isoformat(),"ai_provenance":{},
                "research":dict(st.session_state.research),"origin":"uploaded", "timing_check":"timestamps" if "baseline_time" in raw else "researcher attestation"}
            st.rerun()
        except (ValueError, TypeError, UnicodeError) as exc:
            st.error("Upload not applied: " + str(exc))
    st.subheader("Completed run: " + snapshot.get("origin", "synthetic"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees", len(df))
    c2.metric("Training rows", len(analysis["train"]))
    c3.metric("Holdout rows", len(analysis["test"]))
    c4.metric("Data source", snapshot.get("origin", "synthetic"))
    shown = df.copy()
    if snapshot["research"]["direction"] == "Lower is better":
        shown["independent_competence"] *= -1
    st.dataframe(display_frame(shown.head(50), snapshot["research"]), hide_index=True, width="stretch")
    st.caption("Action codes refer to the saved intervention labels. Interim response thresholds the supplied stress reduction from baseline; lower raw stress is better. Synthetic values use an arbitrary 0–10 scale, not a validated instrument. Uploaded probabilities come from the CSV; sidebar probabilities and simulation coefficients apply only to synthetic generation. Displayed outcomes use the chosen direction; analysis uses higher-is-better utility.")
    support = df.groupby(["stage1_guided", "high_progress", "stage2_guided"], observed=True).size().rename("employees").reset_index()
    st.markdown("#### Observed path coverage")
    st.dataframe(display_frame(support, snapshot["research"]), hide_index=True, width="stretch")
    st.download_button("Download employee data (raw outcome scale)", display_frame(shown, snapshot["research"]).to_csv(index=False).encode("utf-8"), "smart_employees_v0_6.csv", "text/csv", key="download_data")

with policy_tab:
    st.subheader("Evaluate policies on held-out employees")
    st.write("Q-learning is fitted on the training sample. The policy values below use only the separate holdout sample and the known probabilities of both observed assignments.")
    st.caption("Analysis corresponds to: " + snapshot["research"]["independent_competence"] + ". " + snapshot["research"]["direction"] + ". Utility is the outcome for higher-is-better; its negative for lower-is-better.")
    results = analysis["results"].copy()
    all_contrasts = pairwise_comparisons(analysis)
    st.caption("Six embedded strategies when augmentation is enabled (four otherwise), plus two learned candidate policies. Learned candidates are constrained to maintain treatment for responders.")
    st.caption("Action 0: " + snapshot["research"]["action0"] + "; action 1: " + snapshot["research"]["action1"] + "; action 2: A+B when enabled. All responders maintain. Restricted Q-learning uses the moderator for treatment interactions; the full model includes other eligible history features. These are prediction specifications, not automatic tests of your hypotheses.")
    st.dataframe(results, hide_index=True, width="stretch", column_config={"estimate": st.column_config.NumberColumn("Estimated value", format="%.3f"), "se": st.column_config.NumberColumn("SE", format="%.3f"), "ci_lower": st.column_config.NumberColumn("95% lower", format="%.3f"), "ci_upper": st.column_config.NumberColumn("95% upper", format="%.3f"), "ess": st.column_config.NumberColumn("Effective sample size", format="%.1f")})
    chart_df = results.dropna(subset=["estimate"]).copy()
    if len(chart_df):
        base = alt.Chart(chart_df).encode(y=alt.Y("policy:N", sort=results["policy"].tolist(), title=None))
        points = base.mark_point(filled=True, size=85, color="#177E89").encode(x=alt.X("estimate:Q", title="Holdout mean utility (higher is better)", scale=alt.Scale(zero=False)), tooltip=["policy:N", alt.Tooltip("estimate:Q", format=".3f"), "matched_n:Q", alt.Tooltip("ess:Q", format=".1f")])
        intervals = base.mark_rule(color="#7E95A8").encode(x="ci_lower:Q", x2="ci_upper:Q")
        st.altair_chart((intervals + points).properties(height=max(280, len(chart_df)*31)), width="stretch")
    st.caption("Self-normalized IPW (Hájek). Approximate 95% intervals are conditional on the trained policy; they do not prove global optimality. Low matching counts reduce precision. Repeatedly tuning against this holdout removes its independent-validation role.")
    for warning in analysis["diagnostics"].get("warnings", []):
        st.warning(str(warning))
    with st.expander("Inspect learned recommendations", expanded=False):
        model_name = st.selectbox("Fitted policy", list(analysis["models"]), key="inspect_model")
        model = analysis["models"][model_name]
        sample = analysis["test"].head(20).copy()
        sample["recommended_stage1_guided"] = model.recommend_stage1(sample)
        sample["recommended_stage2_guided"] = model.recommend_stage2(sample)
        st.dataframe(display_frame(sample, snapshot["research"]), hide_index=True, width="stretch")
        st.caption("Stage-1 recommendations use baseline variables only. Stage-2 recommendations use the observed stage-1 history; these rows are not a rollout of a different counterfactual stage-1 history.")
        st.json(model.describe(), expanded=False)
    st.markdown("#### Policy effect: an explicit contrast (unadjusted)")
    names = list(analysis["policies"])
    ca, cb = st.columns(2)
    pa = ca.selectbox("Policy A", names, index=len(names)-1, key="contrast_a")
    pb = cb.selectbox("Reference policy B", names, index=0, key="contrast_b")
    contrast = {"policy_a":pa,"policy_b":pb, **evaluate_difference(analysis["test"], analysis["policies"][pa], analysis["policies"][pb])}
    if pa == pb:
        contrast.update(p_raw=1.0,significance="ns")
    else:
        pair = all_contrasts.loc[((all_contrasts.policy_a==pa)&(all_contrasts.policy_b==pb))|((all_contrasts.policy_a==pb)&(all_contrasts.policy_b==pa))].iloc[0]
        contrast.update(p_raw=float(pair.p_raw),significance=pair.significance)
    st.dataframe(pd.DataFrame([contrast]), hide_index=True, width="stretch")
    comparison = comparison_figure(analysis["results"],pa,pb,contrast,snapshot["research"])
    st.pyplot(comparison)
    comparison_svg = figure_bytes(comparison)
    st.download_button("Download comparison figure (SVG)",comparison_svg,"policy_comparison.svg","image/svg+xml")
    st.caption("Bracket significance uses unadjusted two-sided p-values for each paired contrast. All 95% CIs are pointwise (unadjusted). No family-wise error control is claimed. Chart values are on the raw outcome scale; the table contrast uses higher-is-better utility.")
    with st.expander("All pairwise tests — unadjusted"):
        st.dataframe(all_contrasts,hide_index=True,width="stretch")
    st.caption("Effect = value(A) − value(B), on the utility scale. The paired SE accounts for shared holdout employees. Positive favors A. Intervals are pointwise; selecting contrasts after seeing results is exploratory.")
    st.session_state.current_contrast = json.loads(pd.DataFrame([contrast]).to_json(orient="records"))[0]
    st.download_button("Download holdout policy comparison", results.to_csv(index=False).encode("utf-8"), "holdout_policy_values_v0_5.csv", "text/csv", key="download_results")

with notes_tab:
    st.subheader("A reproducible, reviewable run")
    st.write("Export the saved design, settings and seeds, split membership, data, fitted models, policy comparisons and figures. API credentials are excluded. Assistant questions and answers are included in the audit file.")
    export_data = df.copy()
    train_ids = set(analysis["train"]["employee_id"].tolist())
    export_data["analysis_split"] = np.where(export_data["employee_id"].isin(train_ids), "training", "holdout")
    report = {"version":"0.6.1", "pairwise_inference":"Unadjusted two-sided normal p-values; pointwise 95% CIs; no family-wise error control", "created_at_utc":snapshot["created_at_utc"], "data_origin":snapshot.get("origin","synthetic"), "simulation":snapshot["config"], "design":snapshot["design"], "diagnostics":analysis["diagnostics"], "models":{k:v.describe() for k,v in analysis["models"].items()}, "ai_draft_provenance":snapshot["ai_provenance"], "interpretation":"Synthetic runs do not validate theory. Uploaded-run inference requires verified design assumptions. No guarantee of policy improvement.", "selected_contrast":st.session_state.get("current_contrast"), "timing_check":snapshot.get("timing_check","generated sequence"), "simulation_settings_apply":snapshot.get("origin","synthetic")=="synthetic"}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("run_manifest.json", json_bytes(report))
        archive.writestr("assistant_audit.json", json_bytes(st.session_state.get("assistant_audit", [])))
        archive.writestr("revision_history.json", json_bytes(st.session_state.get("revision_history", [])))
        archive.writestr("design_history.json", json_bytes(st.session_state.get("design_history", [])))
        archive.writestr("design.json", json_bytes(snapshot["design"]))
        archive.writestr("analysis_utility_with_split.csv", export_data.to_csv(index=False))
        archive.writestr("holdout_policy_values.csv", results.to_csv(index=False))
        archive.writestr("source_catalog.json", json_bytes(snapshot["design"]["source_catalog"]))
        archive.writestr("policy_comparison.svg",comparison_svg)
        archive.writestr("all_pairwise_tests.csv",all_contrasts.to_csv(index=False))
        archive.writestr("smart_tree.dot",smart_tree(snapshot["research"], snapshot["config"]["augmentation"]))
        archive.writestr("smart_tree.svg",figure_bytes(tree_figure(snapshot["research"], snapshot["config"]["augmentation"])))
        archive.writestr("research_variables.json", json_bytes(snapshot["research"]))
        outcome_data = export_data.copy()
        if snapshot["research"]["direction"] == "Lower is better":
            outcome_data["independent_competence"] *= -1
        archive.writestr("raw_outcome_with_split.csv", outcome_data.to_csv(index=False))
        archive.writestr("data_with_custom_labels.csv", display_frame(outcome_data,snapshot["research"]).to_csv(index=False))
        archive.writestr("SCALE_NOTES.txt", "analysis_utility_with_split.csv contains higher-is-better utility in independent_competence. raw_outcome_with_split.csv contains the raw outcome in the selected direction. Use research_variables.json for construct mappings.")
        if st.session_state.get("online_query", {}).get("research") == snapshot["research"]:
            archive.writestr("online_reference_candidates.json", json_bytes({"query":st.session_state.online_query,"candidates":st.session_state.online_sources}))
    st.download_button("Download complete run bundle", buffer.getvalue(), "smart_discovery_lab_run.zip", "application/zip", key="download_run")
    with st.expander("Assumptions and limitations", expanded=True):
        for line in snapshot["design"].get("assumptions", []):
            st.write(line)
        for line in snapshot["design"].get("limitations", []):
            st.write(line)
        st.write("Real-data import is implemented for the supported schema. Validation on an actual employee trial and live model-account testing have not been completed.")
    methods = Path(__file__).with_name("METHODS.md")
    if methods.exists():
        with st.expander("Statistical methods and references"):
            st.markdown(methods.read_text(encoding="utf-8"))

with assistant_tab:
    st.subheader("SMART research assistant")
    st.write("Discuss your design, analysis choices and potential contributions. Proposed design edits are shown for review before they are applied.")
    provider = st.selectbox("Assistant mode", ["Offline checklist", "Local Ollama", "OpenAI API"], key="copilot_provider")
    ai_key = st.text_input("API key (OpenAI mode only)", type="password", key="copilot_key")
    if provider == "OpenAI API":
        key_tag = hashlib.sha256(ai_key.strip().encode()).hexdigest()
        if st.session_state.get("model_list_key_tag") != key_tag:
            st.session_state.pop("openai_model_ids", None)
            st.session_state.model_list_key_tag = key_tag
        if st.button("Load models from my API account", key="load_api_models", disabled=not ai_key.strip()):
            try:
                with st.spinner("Loading model names…"):
                    st.session_state.openai_model_ids = list_openai_models(ai_key)
            except ValueError as exc:
                st.error(str(exc))
        discovered = st.session_state.get("openai_model_ids")
        options = (sorted(discovered, key=lambda x: (x not in OPENAI_MODEL_CHOICES,
            OPENAI_MODEL_CHOICES.index(x) if x in OPENAI_MODEL_CHOICES else 99, x))
            if discovered is not None else list(OPENAI_MODEL_CHOICES))
        options += ["Other / custom model ID"]
        if st.session_state.get("openai_model_choice") not in options:
            st.session_state.openai_model_choice = options[0]
        selected_model = st.selectbox("OpenAI model", options, key="openai_model_choice")
        ai_model = (st.text_input("Custom API model ID", key="copilot_model")
            if selected_model == "Other / custom model ID" else selected_model)
        st.caption("Account model list loaded; listed models may not all support text and structured outputs."
            if discovered is not None else "Suggested model IDs; account access is not yet checked. Select Load models to query your own account.")
    elif provider == "Local Ollama":
        ai_model = st.text_input("Installed Ollama model name", key="copilot_model")
    else:
        ai_model = ""
    st.caption("Offline checklist is deterministic, not AI conversation. Local Ollama requires an installed running model on this PC. OpenAI mode sends the saved design, supplied source summaries, aggregate results and recent conversation to OpenAI. Employee rows and IDs are excluded. Keys remain in this app session and are excluded from exports. Do not include confidential details in your question.")
    if st.session_state.get("revision_notice"):
        st.success(st.session_state.pop("revision_notice"))
    st.markdown("**AI-assisted revision: choose a task → review the proposal → select edits → apply → rerun.**")
    revision_task = st.selectbox("Revision task", list(REVISION_TASKS), key="revision_task")
    revision_goal = st.text_area("Research goal / requested adjustment", key="revision_goal",
        placeholder="Example: Develop competing coping hypotheses about continuing versus switching problem-focused and emotion-focused support. Discuss how augmentation and its dosage could change the hypothesis.")
    st.caption("Only supported text fields and the response threshold can be applied. Augmentation and its dosage are controlled in the sidebar; the coping response is a supplied stress-reduction score. AI edits do not change those controls or simulation coefficients. Additional stages and new mechanisms require engine changes. Hypotheses generated after inspecting results are exploratory.")
    question = st.text_area("Ask about the current study", value="Review my SMART design. Explain the response criterion, second-stage rerandomization, and what the policy analysis can establish.", key="copilot_question")
    context = make_context(st.session_state.research, inputs, snapshot, stale, st.session_state.get("current_contrast"), st.session_state.get("online_sources") if st.session_state.get("online_query",{}).get("research")==st.session_state.research else [])
    context["current_protocol"] = design["protocol"]
    context["current_simulation_config"] = asdict(config)
    context["scope"] = f"Two stages; {8 if augmentation else 6} paths; {6 if augmentation else 4} embedded strategies; stage 2 includes A+B only when enabled; responders maintain."
    with st.expander("Inspect exactly what the assistant receives"):
        st.json(context, expanded=False)
    send_col, review_col = st.columns(2)
    send = send_col.button("Send question", key="ask_copilot", type="primary", disabled=provider=="Offline checklist")
    review = review_col.button("Review current study", key="review_copilot")
    st.caption("Use Send question for a live model conversation; Ctrl+Enter only commits the text field. Review current study runs a preset review. Offline mode supports the review checklist only. Live requests include the most recent four exchanges.")
    propose = st.button("Propose study adjustments", key="propose_revision", disabled=provider=="Offline checklist")
    if provider == "Offline checklist":
        st.caption("Select OpenAI API or Local Ollama to generate revision proposals. The offline checklist does not generate AI edits.")
    if send or review or propose:
        if review:
            question = "Review the saved study design, response rule, intervention timing and current analysis. Identify concrete limitations."
        try:
            if propose:
                question = revision_question(revision_task, revision_goal, context)
            with st.spinner("Preparing the study review or revision proposal…"):
                result = ask(context, question, st.session_state.get("conversation", []), provider, ai_model, ai_key)
            result["question"] = question
            result["task"] = revision_task if propose else "Conversation / review"
            result["design_fingerprint"] = fingerprint
            result["run_created_at"] = snapshot["created_at_utc"]
            result["context"] = context
            st.session_state.assistant_proposal = result
            st.session_state.setdefault("assistant_audit", []).append(result)
            st.session_state.setdefault("conversation", []).extend([{"role":"user","content":question},{"role":"assistant","content":result["reply"]["answer"]}])
        except ValueError as exc:
            st.error(str(exc))
    if "assistant_proposal" in st.session_state:
        result = st.session_state.assistant_proposal
        st.caption(result["provider"] + " · " + str(result.get("model") or "No model"))
        if result["design_fingerprint"] != fingerprint or result.get("run_created_at") != snapshot["created_at_utc"]:
            st.warning("This review describes an earlier design or analysis run. Ask again for a current interpretation.")
        st.markdown(result["reply"]["answer"])
        for limitation in result["reply"]["limitations"]:
            st.caption(limitation)
        if result["reply"]["source_ids"]:
            st.write("Source identifiers:", result["reply"]["source_ids"])
        edits = result["reply"]["edits"]
        if edits:
            comparison=[]
            for e in edits:
                group,key=e["field"].split(".")
                old=(st.session_state.research if group=="research" else inputs).get(key)
                comparison.append({"field":e["field"],"current":str(old),"proposed":e["value"],"reason":e["reason"]})
            st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch")
            selection_key = "revision_fields_" + result["created_at"]
            selected = st.multiselect("Fields to apply (review related fields together)",
                [e["field"] for e in edits], default=[e["field"] for e in edits], key=selection_key)
            changed = result["design_fingerprint"] != fingerprint or result.get("run_created_at") != snapshot["created_at_utc"]
            if changed:
                st.warning("Design or analysis changed after this proposal. Ask again before applying edits.")
            if st.button("Apply selected reviewed edits", disabled=changed or not selected, key="apply_copilot"):
                try:
                    updated_r, updated_i, record = prepare_revision(st.session_state.research, inputs,
                        result, selected, fingerprint, snapshot["created_at_utc"])
                    st.session_state.approved_copilot = {"research":updated_r, "inputs":updated_i, "record":record}
                    st.rerun()
                except ValueError as exc:
                    st.error("No changes applied: " + str(exc))
        st.download_button("Download assistant review", json_bytes(result), "smart_assistant_review.json", "application/json")
    if st.session_state.get("revision_history"):
        with st.expander("Applied revision history"):
            st.json(st.session_state.revision_history, expanded=False)
            st.download_button("Download revision history", json_bytes(st.session_state.revision_history),
                "study_revision_history.json", "application/json", key="download_revision_history")
    if st.button("Clear conversation", key="clear_conversation"):
        st.session_state.conversation = []
        st.session_state.pop("assistant_proposal",None)
        st.rerun()
    with st.expander("Conversation in this session", expanded=True):
        for message in st.session_state.get("conversation", []):
            st.markdown("**"+message["role"].title()+":** " + message["content"])

st.divider()
st.caption("SMART Discovery Lab · v0.6.1 · Evidence can motivate a hypothesis; randomized data can challenge it. Synthetic data demonstrate the workflow.")
