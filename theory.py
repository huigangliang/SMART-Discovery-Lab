"""Evidence-linked, deterministic design compiler for SMART Discovery Lab.

This module does not parse arbitrary prose into a validated causal model. It
compiles researcher-reviewed fields into one supported two-stage SMART and
keeps citations, proposed extensions, structural checks, and assumptions apart.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


SOURCE_POLICY = {
    "priority": "UTD24 management and organizational research, supplemented by established psychology journals and foundational originals",
    "journal_list_url": "https://jsom.utdallas.edu/the-utd-top-100-business-school-research-rankings/list-of-journals",
    "rule": "Prioritize direct support, research quality, and the original source. Journal membership alone does not validate a claim or its operationalization.",
    "scope": "Curated coping and occupational psychology primary studies; these are not presented as UTD24 articles. Relevance takes priority over journal-list membership.",
}

SOURCE_CATALOG = [{'accessible_source': 'https://pubmed.ncbi.nlm.nih.gov/3712234/',
  'authors': 'Susan Folkman, Richard S. Lazarus, Christine Dunkel-Schetter, Anita DeLongis, and Rand J. '
             'Gruen',
  'design_use': 'Motivate measured controllability as a candidate moderator; the intervention-matching '
                'hypothesis is a proposed extension.',
  'does_not_establish': 'Observational associations do not identify the best randomized support sequence or '
                        'prove a controllability-by-treatment causal interaction.',
  'doi': '10.1037/0022-3514.50.5.992',
  'evidence_status': 'Primary-study abstract and bibliographic record checked; full text not reviewed',
  'id': 'folkman_1986',
  'journal': 'Journal of Personality and Social Psychology',
  'source_group': 'Coping and occupational psychology; not represented as UTD24',
  'supports': 'Appraisal, coping options and forms of coping were associated within individuals across '
              'stressful encounters.',
  'title': 'Dynamics of a stressful encounter: Cognitive appraisal, coping, and encounter outcomes',
  'url': 'https://doi.org/10.1037/0022-3514.50.5.992',
  'volume_pages': '50(5), 992–1003',
  'year': 1986},
 {'accessible_source': 'https://pubmed.ncbi.nlm.nih.gov/2980281/',
  'authors': 'Susan Folkman and Richard S. Lazarus',
  'design_use': 'Motivate sequence-sensitive hypotheses. The claim is not that earlier coping theory was '
                'static; this tool supports experimental tests of dynamic support rules.',
  'does_not_establish': 'An examination study is not a workplace SMART and does not establish that combining '
                        'assigned supports is superior.',
  'doi': '10.1037/0022-3514.48.1.150',
  'evidence_status': 'Primary-study abstract and bibliographic record checked; full text not reviewed',
  'id': 'folkman_lazarus_1985',
  'journal': 'Journal of Personality and Social Psychology',
  'source_group': 'Coping and occupational psychology; not represented as UTD24',
  'supports': 'Coping and emotions changed across exam stages; participants used multiple forms of problem- '
              'and emotion-focused coping.',
  'title': 'If it changes it must be a process: Study of emotion and coping during three stages of a college '
           'examination',
  'url': 'https://doi.org/10.1037/0022-3514.48.1.150',
  'volume_pages': '48(1), 150–170',
  'year': 1985},
 {'accessible_source': 'https://pubmed.ncbi.nlm.nih.gov/11419809/',
  'authors': 'Evangelia Demerouti, Arnold B. Bakker, Friedhelm Nachreiner, and Wilmar B. Schaufeli',
  'design_use': 'Situate stress in work conditions; demands, resources and burnout would require separate '
                'measures for a full JD-R test.',
  'does_not_establish': 'This study does not validate the demo stress score or establish effects of the two '
                        'proposed coping programs.',
  'doi': '10.1037/0021-9010.86.3.499',
  'evidence_status': 'Primary-study abstract and bibliographic record checked; full text not reviewed',
  'id': 'demerouti_2001',
  'journal': 'Journal of Applied Psychology',
  'source_group': 'Coping and occupational psychology; not represented as UTD24',
  'supports': 'Work demands and resources related differently to exhaustion and disengagement across '
              'occupational groups.',
  'title': 'The job demands-resources model of burnout',
  'url': 'https://doi.org/10.1037/0021-9010.86.3.499',
  'volume_pages': '86(3), 499–512',
  'year': 2001},
 {'accessible_source': 'https://pubmed.ncbi.nlm.nih.gov/17638488/',
  'authors': 'Sabine Sonnentag and Charlotte Fritz',
  'design_use': 'Guide future measurement of recovery as a distinct outcome or mechanism; this release '
                'simulates stress, not the Recovery Experience Questionnaire.',
  'does_not_establish': 'Recovery experiences are not interchangeable with coping strategies, stress scores '
                        'or the demo controllability moderator.',
  'doi': '10.1037/1076-8998.12.3.204',
  'evidence_status': 'Primary-study abstract and bibliographic record checked; full text not reviewed',
  'id': 'sonnentag_fritz_2007',
  'journal': 'Journal of Occupational Health Psychology',
  'source_group': 'Coping and occupational psychology; not represented as UTD24',
  'supports': 'Four recovery experiences were distinguished: detachment, relaxation, mastery and control.',
  'title': 'The Recovery Experience Questionnaire: Development and validation of a measure for assessing '
           'recuperation and unwinding from work',
  'url': 'https://doi.org/10.1037/1076-8998.12.3.204',
  'volume_pages': '12(3), 204–221',
  'year': 2007}]

THEORY_OPTIONS = {
    "learning": "Work stress, appraisal and coping",
    "scaffolding": "Dynamic coping and support sequences",
    "motivation": "Controllability and recovery boundaries",
}

BASELINE_VARIABLES = ["baseline_skill", "motivation"]
STAGE2_VARIABLES = ["baseline_skill", "motivation", "stage1_guided", "interim_skill", "interim_motivation", "high_progress"]
PROBABILITY_KEYS = ("p1_guided", "p2_guided_low", "p2_guided_high")


def default_theory_inputs() -> dict[str, Any]:
    """Return independent editable inputs for the supported design family."""
    return {
        "title": 'Adaptive Coping Support for Work Stress',
        "research_question": 'When should employees with insufficient early stress reduction continue, switch or combine problem-focused and emotion-focused coping support?',
        "proposition_learning": 'H1: Problem-focused and emotion-focused support may differ in final work stress; neither is assumed universally superior.',
        "proposition_scaffolding": 'H2: Among nonresponders, continuing, switching and combining supports may have different final stress outcomes. Specify primary contrasts and dosage before data collection.',
        "proposition_motivation": 'H3: Perceived stressor controllability may modify relative support effects. This is a proposed interaction, not an established causal result.',
        "intervention0": 'Problem-focused coping support',
        "intervention1": 'Emotion-focused coping support',
        "outcome": 'Final work stress',
        "timepoint": 'End of week 2, after interim stress and before stage-2 assignment',
        "response_threshold": 0.5,
        "motivation_threshold": 0.0,
        "enabled_theories": ["learning", "scaffolding", "motivation"],
        "baseline_variables": list(BASELINE_VARIABLES),
        "stage2_variables": list(STAGE2_VARIABLES),
        "response_variable": "interim_skill",
        "outcome_variable": "independent_competence",
        "competing_prediction": 'Continuation may allow skills to develop; switching may improve fit; combination may help or dilute practice. Extra contact time may explain apparent augmentation benefits.',
        "falsification": 'Preregister policy contrasts and meaningful stress differences. Test on real randomized data; exploratory revisions require independent validation. Simulated patterns cannot validate coping theory.',
        "boundary_conditions": 'Individual work stress and personal support; no change to work arrangements. Spillovers, adherence and existing coping skills need assessment. Stress, burnout and recovery are distinct constructs.',
    }


def _finite_number(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be a finite number.")
    return number


def _normalize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(inputs, dict):
        raise ValueError("Theory inputs must be an object.")
    defaults = default_theory_inputs()
    unknown = sorted(set(inputs) - set(defaults))
    if unknown:
        raise ValueError("Unsupported theory input fields: " + ", ".join(unknown))
    normalized = deepcopy(defaults)
    normalized.update(deepcopy(inputs))
    for key, default in defaults.items():
        if isinstance(default, str):
            value = normalized[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be nonempty text.")
            if len(value) > 12000:
                raise ValueError(f"{key} exceeds the 12,000-character limit.")
            normalized[key] = value.strip()
    for key in ("response_threshold", "motivation_threshold"):
        normalized[key] = _finite_number(normalized[key], key)
    enabled = normalized["enabled_theories"]
    if not isinstance(enabled, list) or not enabled or any(not isinstance(x, str) or x not in THEORY_OPTIONS for x in enabled):
        raise ValueError("enabled_theories must select one or more supported theory IDs.")
    if len(set(enabled)) != len(enabled):
        raise ValueError("enabled_theories cannot contain duplicates.")
    for key, expected in (("baseline_variables", BASELINE_VARIABLES), ("stage2_variables", STAGE2_VARIABLES)):
        supplied = normalized[key]
        if not isinstance(supplied, list) or len(supplied) != len(expected) or any(not isinstance(x, str) for x in supplied) or set(supplied) != set(expected):
            raise ValueError(f"{key} must use the v0.5 supported columns: {', '.join(expected)}. Future measurements cannot enter an earlier decision.")
    if normalized["response_variable"] != "interim_skill":
        raise ValueError("v0.5 supports interim_skill as the pre-stage-2 response measure.")
    if normalized["outcome_variable"] != "independent_competence":
        raise ValueError("v0.5 supports independent_competence as its terminal outcome.")
    if normalized["intervention0"] == normalized["intervention1"]:
        raise ValueError("The two intervention labels must differ.")
    return normalized


def compile_design(inputs: dict[str, Any], probabilities: dict[str, float] | None = None) -> dict[str, Any]:
    """Compile a reviewed two-stage design; reject unsupported executable fields.

    Labels and prose document the researcher's protocol. Only the thresholds
    and probabilities below alter the supported simulator; prose does not
    silently change a data-generating equation or the meaning of action codes.
    """
    user = _normalize_inputs(inputs)
    probabilities = {} if probabilities is None else probabilities
    if not isinstance(probabilities, dict):
        raise ValueError("Randomization probabilities must be an object.")
    unknown = set(probabilities) - set(PROBABILITY_KEYS)
    if unknown:
        raise ValueError("Unsupported randomization fields: " + ", ".join(sorted(unknown)))
    probs = {key: _finite_number(probabilities.get(key, 0.5), key) for key in PROBABILITY_KEYS}
    for key, value in probs.items():
        if not 0.1 <= value <= 0.9:
            raise ValueError(f"{key} must lie between 0.10 and 0.90 for this prototype's positivity guardrail.")

    theories = [
        {"id":key, "label":THEORY_OPTIONS[key], "researcher_hypothesis":user[field],
         "source_ids":source_ids,
         "evidence_level":"Primary-study context; the workplace dynamic intervention hypothesis is untested.",
         "design_consequence":"Prespecify stress measurements, support content, dosage, response criteria and complete strategy contrasts."}
        for key,field,source_ids in [
            ("learning","proposition_learning",["folkman_1986","demerouti_2001"]),
            ("scaffolding","proposition_scaffolding",["folkman_lazarus_1985"]),
            ("motivation","proposition_motivation",["folkman_1986","sonnentag_fritz_2007"])]
    ]
    theories = [row for row in theories if row["id"] in user["enabled_theories"]]
    actions = [{"code": 0, "label": user["intervention0"], "implemented_role": "problem-focused support"},
               {"code": 1, "label": user["intervention1"], "implemented_role": "emotion-focused support"}]
    design = {
        "schema_version": "0.5",
        "title": user["title"],
        "research_question": user["research_question"],
        "inputs": user,
        "source_policy": deepcopy(SOURCE_POLICY),
        "provenance": {
            "mode": "Deterministic, evidence-linked design compiler",
            "ai_generated": False,
            "hypothesis_status": "Researcher-proposed extensions; not findings of the cited papers",
            "text_processing": "Preserves reviewed text. Does not infer arbitrary designs or new measurement validity from prose.",
            "code_scope": "Individual-level, two-stage, two-action SMART; only nonresponders are rerandomized at stage 2.",
        },
        "theories": theories,
        "stages": [
            {"stage": 1, "time": "Before first support period", "history": list(BASELINE_VARIABLES),
             "action_column": "stage1_guided", "actions": deepcopy(actions), "randomized": True,
             "probability_guided": probs["p1_guided"], "eligibility": "All employees"},
            {"stage": 2, "time": user["timepoint"], "history": list(STAGE2_VARIABLES),
             "action_column": "stage2_guided", "actions": deepcopy(actions), "randomized": True,
             "rerandomize_all": False, "eligibility": "Nonresponders only; responders maintain their initial action",
             "response_column": "high_progress", "response_measure": "interim_skill",
             "response_rule": f"interim_skill >= {user['response_threshold']:g}",
             "response_threshold": user["response_threshold"],
             "probability_guided_low": probs["p2_guided_low"],
             "probability_guided_high": probs["p2_guided_high"]},
        ],
        "outcome": {"column": "independent_competence", "label": user["outcome"],
                    "timing": "After stage 2", "direction": "higher is better", "level": "individual employee"},
        "measurement_map": [
            {"variable": "baseline_skill", "when": "Before stage 1", "construct": "Baseline work stress", "use": "Baseline adjustment and stage-1 tailoring"},
            {"variable": "motivation", "when": "Before stage 1", "construct": "Baseline perceived controllability", "use": "Baseline moderator; not autonomy, need satisfaction, or a randomized treatment"},
            {"variable": "interim_skill", "when": "Before stage 2", "construct": "Interim stress reduction", "use": "Stage-2 history only"},
            {"variable": "interim_motivation", "when": "Before stage 2", "construct": "Interim perceived controllability", "use": "Stage-2 effect modifier; may be affected by stage-1 assignment"},
            {"variable": "high_progress", "when": "Before stage 2", "construct": "Thresholded interim stress reduction", "use": "Assignment stratum; not a computed change score"},
            {"variable": "independent_competence", "when": "After stage 2", "construct": "Final work stress", "use": "Terminal outcome, never a tailoring variable"},
            {"variable": "p1_observed / p2_observed", "when": "At each randomization", "construct": "Probability of the action actually assigned", "use": "Product gives the probability used in two-stage policy weighting"},
        ],
        "estimands": [
            {"name": "Policy value", "definition": "Mean terminal outcome utility if eligible employees followed a specified two-stage regime", "analysis": "Self-normalized IPW (Hajek) using known sequential randomization probabilities"},
            {"name": "Value of a learned policy", "definition": "Value of the policy trained on the training split, evaluated on a separate holdout", "analysis": "Two-stage Q-learning, then holdout IPW; not a guarantee of a globally optimal policy"},
            {"name": "Theory-relevant comparison", "definition": "Preregistered contrasts between fixed and state-dependent strategies and proposed effect modification", "analysis": "The prototype demonstrates estimands; it does not identify a psychological mediation mechanism or validate theory using synthetic data"},
        ],
        "hypothesis_tests": {
            "competing_prediction": user["competing_prediction"],
            "falsification": user["falsification"],
            "boundary_conditions": user["boundary_conditions"],
            "theory_development": "Use competing predictions, mechanism measures, null/reversed results, and boundary conditions to refine theory. Optimization alone is not a theory test.",
        },
        "executable_config": {"response_threshold": user["response_threshold"],
                              "motivation_threshold": user["motivation_threshold"], **probs},
        "source_catalog": deepcopy(SOURCE_CATALOG),
        "assumptions": [
            "Treatment implementations are sufficiently well specified for consistency.",
            "Stage 1 and nonresponder stage 2 are randomized with recorded positive probabilities; responder maintenance has probability 1.",
            "No unmodeled interference between employees; organizational spillovers require an appropriate cluster/network design.",
            "Outcome measurement, adherence, and missingness require an explicit protocol; v0.5 simulates complete outcomes.",
            "Researcher review is required for construct validity, feasibility, timing, and ethics; code checks do not establish these conditions.",
        ],
        "limitations": [
            "Edited prose does not alter simulation equations or validate measures.",
            "Interim response is the supplied stress-reduction score; higher improvement is better.",
            "Controllability is a measured candidate moderator, not a randomized factor or mediation test.",
            "Simulation coefficients and the 0–10 stress scale are illustrative, not clinical estimates.",
            "Coping, recovery and burnout are distinct constructs; only stress is the simulated final outcome.",
        ],
    }
    design["checks"] = validate_design(design)
    return design


def validate_design(design: dict[str, Any]) -> list[dict[str, str]]:
    """Return structural checks, with errors for unsupported or invalid designs.

    Passing these checks establishes schema consistency, not causal identification
    for a real study, construct validity, or empirical support for a theory.
    """
    checks: list[dict[str, str]] = []
    def add(severity: str, check: str, detail: str) -> None:
        checks.append({"severity": severity, "check": check, "detail": detail})
    if not isinstance(design, dict):
        return [{"severity": "error", "check": "Design object", "detail": "Design must be an object."}]
    stages = design.get("stages")
    if not isinstance(stages, list) or len(stages) != 2 or any(not isinstance(s, dict) for s in stages):
        add("error", "Supported stage count", "Exactly two structured decision stages are required.")
        return checks
    s1, s2 = stages
    h1, h2 = s1.get("history", []), s2.get("history", [])
    valid_h1 = isinstance(h1, list) and all(isinstance(x, str) for x in h1) and len(h1) == len(BASELINE_VARIABLES) and set(h1) == set(BASELINE_VARIABLES)
    add("pass" if valid_h1 else "error", "Stage-1 information timing",
        "Only baseline_skill and motivation may determine the first decision; later state and outcome variables are unavailable then.")
    valid_h2 = isinstance(h2, list) and all(isinstance(x, str) for x in h2) and len(h2) == len(STAGE2_VARIABLES) and set(h2) == set(STAGE2_VARIABLES)
    add("pass" if valid_h2 else "error", "Stage-2 information timing",
        "Stage 2 uses supported history measured before its assignment; final outcomes cannot be tailoring variables.")
    for index, stage in enumerate(stages, 1):
        actions = stage.get("actions", [])
        valid_actions = (isinstance(actions, list) and len(actions) == 2 and
                         all(isinstance(a, dict) and type(a.get("code")) is int for a in actions) and
                         {a.get("code") for a in actions} == {0, 1})
        add("pass" if valid_actions else "error", f"Stage-{index} action support",
            "Both supported actions, A (0) and B (1), must remain available.")
        add("pass" if stage.get("randomized") is True else "error", f"Stage-{index} randomization",
            "Assignment must be randomized; a theory-based deterministic recommendation is an analysis target, not the assignment mechanism.")
    add("pass" if s2.get("rerandomize_all") is False else "error", "Second-stage eligibility",
        "Only nonresponders are rerandomized; responders deterministically maintain their initial action.")
    pp = [s1.get("probability_guided"), s2.get("probability_guided_low"), s2.get("probability_guided_high")]
    good_probs = all(not isinstance(p, bool) and isinstance(p, (int, float)) and math.isfinite(p) and 0.1 <= p <= 0.9 for p in pp)
    add("pass" if good_probs else "error", "Positive assignment probabilities",
        "Each modeled guided-action probability must be between 0.10 and 0.90; direct-action probability is its complement.")
    threshold = s2.get("response_threshold")
    good_threshold = not isinstance(threshold, bool) and isinstance(threshold, (int, float)) and math.isfinite(threshold)
    add("pass" if good_threshold else "error", "Preregistered response threshold", "The interim-skill threshold must be finite and specified before examining outcomes.")
    outcome = design.get("outcome", {})
    add("pass" if isinstance(outcome, dict) and outcome.get("column") == "independent_competence" and outcome.get("timing") == "After stage 2" else "error",
        "Terminal outcome", "Final stress is measured after both assignments.")
    add("warning", "Construct and implementation review", "Structural checks do not validate measurement scales, intervention fidelity, interference assumptions, or real-study causal identification.")
    add("warning", "Empirical status", "A synthetic result evaluates the implementation under stated assumptions. It cannot confirm or refute an organizational theory in the real world.")
    return checks
