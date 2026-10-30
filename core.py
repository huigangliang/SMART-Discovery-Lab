"""Statistical core for the SMART Discovery Lab.

Synthetic, individual-level, terminal-outcome, two-stage SMART prototype.
Only nonresponders are rerandomized at stage 2; responders maintain stage 1. Policy values are evaluated
on an untouched test split with the known observed assignment probabilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import math
import itertools
import numpy as np
import pandas as pd

OUTCOME = "independent_competence"
BASELINE = ("baseline_skill", "motivation")
HISTORY = BASELINE + ("stage1_guided", "interim_skill", "interim_motivation", "high_progress")
Q2_HISTORY = tuple(x for x in HISTORY if x != "high_progress")
REQUIRED = HISTORY + ("stage2_guided", "p1_observed", "p2_observed", OUTCOME)
ActionRule = Callable[[pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class SimulationConfig:
    n: int = 500
    seed: int = 42
    scaffolding_effect: float = 0.6
    motivation_moderation: float = 0.3
    response_threshold: float = 0.0
    p1_guided: float = 0.5
    p2_guided_low: float = 0.5
    p2_guided_high: float = 0.5
    scenario: str = "theory_aligned"
    noise_sd: float = 1.0
    augmentation: bool = False
    case: str = "legacy"
    augmentation_effect: float = 0.2
    switch_effect: float = 0.0
    augmentation_dose: str = "equal_time"

    def validate(self) -> None:
        if self.case not in ("legacy", "coping") or self.augmentation_dose not in ("equal_time", "added_time"):
            raise ValueError("Unsupported simulation case or dosage convention.")
        if not isinstance(self.augmentation, bool):
            raise ValueError("Augmentation must be boolean.")
        if not np.isfinite([self.augmentation_effect, self.switch_effect]).all():
            raise ValueError("Sequence and combination coefficients must be finite.")
        if not isinstance(self.n, (int, np.integer)) or self.n < 40:
            raise ValueError("Simulation needs an integer sample size of at least 40.")
        if not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("Random seed must be a nonnegative integer.")
        for name in ("p1_guided", "p2_guided_low", "p2_guided_high"):
            if not np.isfinite(getattr(self, name)) or not 0.1 <= getattr(self, name) <= 0.9:
                raise ValueError(f"{name} must be between 0.10 and 0.90.")
        for name in ("scaffolding_effect", "motivation_moderation", "response_threshold", "noise_sd"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if self.noise_sd < 0:
            raise ValueError("Noise standard deviation must be nonnegative.")
        if self.scenario not in ("theory_aligned", "null", "reversed"):
            raise ValueError("Scenario must be theory_aligned, null, or reversed.")


def simulate(config: SimulationConfig | None = None) -> pd.DataFrame:
    """Generate ordered randomizations with a fixed, pre-specified response cutpoint.

    Higher scores indicate more independent competence. Scores are arbitrary
    continuous synthetic units, not a validated scale. A2=1 means guided AI
    support; A2=0 means direct AI assistance, regardless of stage-1 assignment.
    The null scenario removes ALL treatment effects, including effects on the
    intermediate state. Aligned/reversed describe assumptions, not findings.
    """
    c = config or SimulationConfig()
    c.validate()
    if c.case == "coping":
        return simulate_coping(c)
    if c.augmentation:
        raise ValueError("Augmentation requires the coping simulation case.")
    rng = np.random.default_rng(c.seed)
    n = c.n
    direction = {"theory_aligned": 1.0, "null": 0.0, "reversed": -1.0}[c.scenario]
    baseline = rng.normal(size=n)
    motivation = rng.normal(size=n)
    a1 = rng.binomial(1, c.p1_guided, size=n)
    interim_skill = 0.5 * baseline + 0.35 * motivation + direction * 0.3 * a1 + rng.normal(0, c.noise_sd, n)
    interim_motivation = 0.75 * motivation + direction * 0.2 * a1 + rng.normal(0, 0.5 * c.noise_sd, n)
    response = (interim_skill >= c.response_threshold).astype(int)
    draw = rng.binomial(1, c.p2_guided_low, size=n)
    a2 = np.where(response == 1, a1, draw)
    p2 = np.where(response == 1, a1.astype(float), c.p2_guided_low)
    outcome = (
        1.0 + 0.5 * baseline + 0.3 * motivation + 0.4 * interim_skill
        + direction * (c.scaffolding_effect * a1 +
                       (0.45 + c.motivation_moderation * interim_motivation - 0.65 * response) * a2)
        + rng.normal(0, c.noise_sd, n)
    )
    df = pd.DataFrame({
        "employee_id": np.arange(1, n + 1),
        "baseline_skill": baseline, "motivation": motivation,
        "stage1_guided": a1, "interim_skill": interim_skill,
        "interim_motivation": interim_motivation, "high_progress": response,
        "stage2_guided": a2,
        "p1_guided": np.full(n, c.p1_guided), "p2_guided": p2,
        "p1_observed": np.where(a1 == 1, c.p1_guided, 1 - c.p1_guided),
        "p2_observed": np.where(a2 == 1, p2, 1 - p2),
        OUTCOME: outcome,
    })
    df.attrs["simulation_config"] = asdict(c)
    df.attrs["synthetic"] = True
    return df


def validate_data(df: pd.DataFrame, *, for_fit: bool = False) -> None:
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError("Required columns are missing: " + ", ".join(sorted(missing)))
    if len(df) == 0:
        raise ValueError("Dataset is empty.")
    values = df[list(REQUIRED)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Analysis columns must be numeric and contain no missing or infinite values.")
    for name in ("stage1_guided", "high_progress"):
        if not df[name].isin([0, 1]).all():
            raise ValueError(f"{name} must contain only 0 and 1.")
    if not df.stage2_guided.isin(stage2_actions(df)).all():
        raise ValueError("Stage-2 actions do not match the declared design.")
    if not ((df.p1_observed > 0) & (df.p1_observed < 1)).all():
        raise ValueError("Stage-1 observed probabilities must be strictly between 0 and 1.")
    responders = df.high_progress == 1
    if not (df.loc[responders, "stage2_guided"] == df.loc[responders, "stage1_guided"]).all():
        raise ValueError("Responders must maintain their stage-1 intervention.")
    if not np.allclose(df.loc[responders, "p2_observed"], 1):
        raise ValueError("Responders have deterministic maintenance: p2_observed must equal 1.")
    nrp = df.loc[~responders, "p2_observed"]
    if not ((nrp > 0) & (nrp < 1)).all():
        raise ValueError("Nonresponders must be randomized with observed probabilities strictly between 0 and 1.")
    if for_fit:
        if len(df) < 20:
            raise ValueError("At least 20 training observations are required.")
        if set(df.stage1_guided.unique()) != {0, 1}:
            raise ValueError("Training data need both stage-1 actions.")


def _matrix(df: pd.DataFrame, main: tuple[str, ...], interactions: tuple[str, ...], action: np.ndarray) -> np.ndarray:
    a = np.asarray(action, dtype=float)
    return np.column_stack([
        np.ones(len(df)), df[list(main)].to_numpy(dtype=float), a,
        df[list(interactions)].to_numpy(dtype=float) * a[:, None],
    ])


def _feature_names(main: tuple[str, ...], interactions: tuple[str, ...], action_name: str) -> list[str]:
    return ["intercept", *main, action_name, *(f"{action_name} × {x}" for x in interactions)]


@dataclass
class QLearningModel:
    """Two-stage linear Q-learning with a terminal outcome and zero interim reward."""
    beta1: np.ndarray
    beta2: np.ndarray
    interactions1: tuple[str, ...]
    interactions2: tuple[str, ...]
    theory_guided: bool
    train_n: int
    rank1: int
    rank2: int
    actions2: tuple[int, ...] = (0, 1)

    def q1(self, df: pd.DataFrame, action: int) -> np.ndarray:
        return _matrix(df, BASELINE, self.interactions1, np.full(len(df), action)) @ self.beta1

    def q2(self, df: pd.DataFrame, action: int) -> np.ndarray:
        return _matrix2(df, self.interactions2, np.full(len(df), action), self.actions2) @ self.beta2

    def recommend_stage1(self, df: pd.DataFrame) -> np.ndarray:
        # Exact and near ties choose action 0 deterministically.
        return (self.q1(df, 1) - self.q1(df, 0) > 1e-10).astype(int)

    def recommend_stage2(self, df: pd.DataFrame) -> np.ndarray:
        choice = np.argmax(np.column_stack([self.q2(df, a) for a in self.actions2]), axis=1)
        return np.where(df.high_progress == 1, df.stage1_guided, choice).astype(int)

    def describe(self) -> dict:
        return {
            "method": "Two-stage linear Q-learning (ordinary least squares)",
            "theory_guided": self.theory_guided,
            "train_n": self.train_n,
            "stage1_features": _feature_names(BASELINE, self.interactions1, "stage1_guided"),
            "stage2_features": _feature_names2(self.interactions2, self.actions2),
            "stage2_actions": list(self.actions2),
            "stage1_coefficients": self.beta1.tolist(),
            "stage2_coefficients": self.beta2.tolist(),
            "stage1_rank": self.rank1, "stage2_rank": self.rank2,
            "stage1_columns": len(self.beta1), "stage2_columns": len(self.beta2),
            "objective": "Maximize expected terminal outcome utility with responder maintenance; no interim reward or support cost.",
            "interpretation": "Estimated policy within this linear model class; not a proven globally optimal policy.",
            "theory_restriction": "Prespecified tailoring-feature subset; coefficient signs remain freely estimated." if self.theory_guided else "All available pre-decision covariates can interact with treatment.",
        }


def fit_q_learning(train: pd.DataFrame, theory_guided: bool = False) -> QLearningModel:
    """Fit Q2 among nonresponders, maximize over eligible actions, and fit baseline-only Q1.

    No information from the evaluation sample is used. Theory guidance changes
    the treatment-interaction feature set, without forcing theory-consistent
    coefficient signs. Both variants retain nonconstant pre-A2 main effects in the nonresponder Q2 model.
    """
    validate_data(train, for_fit=True)
    inter1 = ("motivation",) if theory_guided else BASELINE
    inter2 = ("interim_motivation",) if theory_guided else Q2_HISTORY
    nr = train.loc[train.high_progress == 0]
    actions2 = stage2_actions(train)
    if len(nr) < 20 or set(nr.stage2_guided.unique()) != set(actions2):
        raise ValueError("Q-learning needs at least 20 nonresponders and all declared stage-2 actions in training. Increase the sample; do not tune the threshold to significance.")
    x2 = _matrix2(nr, inter2, nr.stage2_guided.to_numpy(), actions2)
    beta2, _, rank2, _ = np.linalg.lstsq(x2, nr[OUTCOME].to_numpy(), rcond=None)
    qmax = np.max(np.column_stack([
        _matrix2(train, inter2, np.full(len(train), a), actions2) @ beta2 for a in actions2]), axis=1)
    pseudo_outcome = np.where(train.high_progress == 1, train[OUTCOME], qmax)
    x1 = _matrix(train, BASELINE, inter1, train.stage1_guided.to_numpy())
    beta1, _, rank1, _ = np.linalg.lstsq(x1, pseudo_outcome, rcond=None)
    return QLearningModel(beta1, beta2, inter1, inter2, theory_guided, len(train), int(rank1), int(rank2), actions2)


def _constant(action: int) -> ActionRule:
    return lambda df: np.full(len(df), action, dtype=int)


def fixed_policies(motivation_threshold: float = 0.0, augmentation: bool = False) -> dict[str, tuple[ActionRule, ActionRule]]:
    """Four binary or six augmented embedded regimes; responders maintain."""
    policies = {}
    for first in (0, 1):
        for option in (["continue", "switch", "augment"] if augmentation else ["continue", "switch"]):
            name = f"{'A' if first == 0 else 'B'} first | nonresponders: {option}"
            next_action = first if option == "continue" else 1-first if option == "switch" else 2
            def second(df, first=first, next_action=next_action):
                return np.where(df.high_progress == 1, first, next_action)
            policies[name] = (_constant(first), second)
    return policies


def _actions(rule: ActionRule | np.ndarray | int, df: pd.DataFrame, allowed=(0, 1)) -> np.ndarray:
    result = rule(df) if callable(rule) else rule
    result = np.asarray(result)
    if result.ndim == 0:
        result = np.full(len(df), result.item())
    if result.shape != (len(df),) or not np.isin(result, allowed).all():
        raise ValueError("A policy must return exactly one supported action per row.")
    return result.astype(int)


def _policy_stats(test: pd.DataFrame, d1: ActionRule | np.ndarray | int, d2: ActionRule | np.ndarray | int) -> tuple[dict, np.ndarray]:
    validate_data(test)
    action1, action2 = _actions(d1, test), _actions(d2, test, stage2_actions(test))
    if np.any((test.high_progress.to_numpy() == 1) & (test.stage1_guided.to_numpy() == action1) & (action2 != action1)):
        raise ValueError("A policy must maintain its initial action for responders.")
    matched = (test.stage1_guided.to_numpy() == action1) & (test.stage2_guided.to_numpy() == action2)
    weights = matched / (test.p1_observed.to_numpy() * test.p2_observed.to_numpy())
    n, matched_n = len(test), int(matched.sum())
    base = {"n": n, "matched_n": matched_n, "estimator": "Self-normalized IPW (Hájek)"}
    if matched_n == 0:
        return {**base, "estimate": np.nan, "se": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                "ess": 0.0, "support": "No matching paths; value unavailable"}, np.full(n, np.nan)
    estimate = float(np.sum(weights * test[OUTCOME].to_numpy()) / weights.sum())
    ess = float(weights.sum() ** 2 / np.sum(weights ** 2))
    influence = weights * (test[OUTCOME].to_numpy() - estimate) / weights.mean()
    # One matching path has no empirical information about outcome variability.
    se = float(np.std(influence, ddof=1) / np.sqrt(n)) if matched_n >= 2 and n >= 2 else np.nan
    support = "Adequate descriptive support" if matched_n >= 30 and ess >= 30 else "Limited support; interval may be unreliable"
    return {**base, "estimate": estimate, "se": se,
            "ci_lower": estimate - 1.96 * se, "ci_upper": estimate + 1.96 * se,
            "ess": ess, "support": support}, influence


def evaluate_policy(test: pd.DataFrame, d1: ActionRule | np.ndarray | int, d2: ActionRule | np.ndarray | int) -> dict:
    """Conditional holdout value and asymptotic 95% interval for a fixed policy.

    CI conditions on the trained policy. It does not account for re-selecting
    a winning policy or repeating analysis after looking at holdout results.
    This is the ratio/Hájek estimator, NOT the Horvitz--Thompson estimator.
    """
    return _policy_stats(test, d1, d2)[0]


def evaluate_difference(test: pd.DataFrame, policy_a: tuple[ActionRule, ActionRule], policy_b: tuple[ActionRule, ActionRule]) -> dict:
    """Paired difference using shared test participants and covariance."""
    a, influence_a = _policy_stats(test, *policy_a)
    b, influence_b = _policy_stats(test, *policy_b)
    difference = a["estimate"] - b["estimate"]
    se = (float(np.std(influence_a - influence_b, ddof=1) / np.sqrt(len(test)))
          if np.isfinite(a["se"]) and np.isfinite(b["se"]) else np.nan)
    return {"difference": difference, "se": se, "ci_lower": difference - 1.96 * se,
            "ci_upper": difference + 1.96 * se, "n": len(test)}


def run_analysis(df: pd.DataFrame, test_fraction: float = 0.4, split_seed: int = 2026,
                 motivation_threshold: float = 0.0) -> dict:
    """One reproducible random train/test split, two learned policies, 4 embedded strategies.

    All reported policy values use the same independent test sample. No test
    outcomes influence either fit. Sorting results is descriptive and does not
    turn the largest estimate into a statistically validated best policy.
    """
    validate_data(df, for_fit=True)
    if not 0.2 <= test_fraction <= 0.5:
        raise ValueError("Test fraction must be between 0.20 and 0.50.")
    if not isinstance(split_seed, (int, np.integer)) or split_seed < 0:
        raise ValueError("Split seed must be a nonnegative integer.")
    indices = np.random.default_rng(split_seed).permutation(len(df))
    n_test = max(2, int(round(len(df) * test_fraction)))
    test = df.iloc[indices[:n_test]].copy().reset_index(drop=True)
    train = df.iloc[indices[n_test:]].copy().reset_index(drop=True)
    models = {"unconstrained": fit_q_learning(train, False),
              "theory_guided": fit_q_learning(train, True)}
    policies = fixed_policies(motivation_threshold, augmentation=2 in stage2_actions(df))
    policies["Q-learning: all tailoring features"] = (models["unconstrained"].recommend_stage1, models["unconstrained"].recommend_stage2)
    policies["Q-learning: prespecified theory features"] = (models["theory_guided"].recommend_stage1, models["theory_guided"].recommend_stage2)
    rows = [{"policy": name, **evaluate_policy(test, *policy)} for name, policy in policies.items()]
    results = pd.DataFrame(rows)
    warnings = []
    for key, model in models.items():
        if model.rank1 < len(model.beta1) or model.rank2 < len(model.beta2):
            warnings.append(f"{key}: at least one regression design matrix is rank deficient; coefficients are not uniquely identified.")
    if (results.ess < 30).any():
        warnings.append("Some policies have fewer than 30 effective matching observations; interpret values and intervals cautiously.")
    diagnostics = {
        "train_n": len(train), "test_n": len(test), "seed": int(split_seed),
        "test_fraction": float(test_fraction), "warnings": warnings,
        "train_ids": train.employee_id.tolist() if "employee_id" in train else indices[n_test:].tolist(),
        "test_ids": test.employee_id.tolist() if "employee_id" in test else indices[:n_test].tolist(),
        "stage1_features": list(BASELINE), "stage2_features": list(Q2_HISTORY),
        "interval_scope": "Pointwise approximate 95% intervals conditional on the training fit; no multiple-comparison or policy-selection adjustment.",
        "randomization": "Stage 1 randomized; only nonresponders rerandomized. Responders maintain with p2_observed=1; weights use p1_observed × p2_observed.",
    }
    return {"train": train, "test": test, "models": models, "results": results,
            "diagnostics": diagnostics, "policies": policies}


def pairwise_comparisons(analysis):
    """All unordered contrasts with unadjusted two-sided normal p-values."""
    rows=[]
    for a,b in itertools.combinations(analysis["policies"],2):
        r=evaluate_difference(analysis["test"], analysis["policies"][a], analysis["policies"][b])
        se,delta=r["se"],r["difference"]
        pv=(math.erfc(abs(delta/se)/math.sqrt(2)) if se>0 else
            (1.0 if delta==0 else float("nan"))) if np.isfinite(se) else float("nan")
        rows.append(dict(policy_a=a,policy_b=b,**r,p_raw=pv))
    out=pd.DataFrame(rows)
    out["significance"]=out.p_raw.map(lambda p: "unavailable" if not np.isfinite(p) else "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns")
    return out


def stage2_actions(df):
    declared = df.attrs.get("stage2_actions")
    if declared is not None:
        if tuple(declared) not in ((0, 1), (0, 1, 2)):
            raise ValueError("Unsupported declared stage-2 actions.")
        return tuple(declared)
    return (0, 1, 2) if (df.stage2_guided == 2).any() else (0, 1)


def _matrix2(df, interactions, action, actions):
    # Treatment is categorical: separate indicators, never a numeric 0/1/2 slope.
    blocks = [np.ones(len(df)), df[list(Q2_HISTORY)].to_numpy(dtype=float)]
    for level in actions[1:]:
        a = (np.asarray(action) == level).astype(float)
        blocks += [a, df[list(interactions)].to_numpy(dtype=float) * a[:, None]]
    return np.column_stack(blocks)


def _feature_names2(interactions, actions):
    names = ["intercept", *Q2_HISTORY]
    for level in actions[1:]:
        label = f"stage2_action={level}"
        names += [label, *(f"{label} × {x}" for x in interactions)]
    return names


def simulate_coping(c):
    """Illustrative stress trajectory; no clinical calibration or empirically estimated effects.

    baseline_skill stores baseline stress; interim_skill is baseline minus interim
    stress, so higher reduction means response. OUTCOME is negative final stress.
    All three stress measurements share an arbitrary 0–10 demonstration scale.
    """
    c.validate()
    rng = np.random.default_rng(c.seed); n = c.n
    d = {"theory_aligned":1., "null":0., "reversed":-1.}[c.scenario]
    stress0 = np.clip(5 + rng.normal(0, 1.2, n), 0, 10)
    control0 = rng.normal(size=n)
    a1 = rng.binomial(1, c.p1_guided, n)
    # A's illustrative advantage grows with controllability; B's grows as it falls.
    fit1 = (1 - 2*a1)*c.motivation_moderation*control0
    gain1 = d*(0.6 + c.scaffolding_effect*a1 + fit1)
    stress1 = np.clip(stress0 - gain1 + rng.normal(0, c.noise_sd, n), 0, 10)
    improvement = stress0 - stress1
    control1 = .8*control0 + rng.normal(0, .3, n)
    response = (improvement >= c.response_threshold).astype(int)
    if c.augmentation:
        draw = rng.integers(0, 3, n)
        p2obs = np.where(response == 1, 1., 1/3)
    else:
        draw = rng.binomial(1, c.p2_guided_low, n)
        p2obs = np.where(response == 1, 1., np.where(draw == 1, c.p2_guided_low, 1-c.p2_guided_low))
    a2 = np.where(response == 1, a1, draw)
    g_a = .6 + c.motivation_moderation*control1
    g_b = .6 + c.scaffolding_effect - c.motivation_moderation*control1
    combined = ((g_a+g_b)/2 if c.augmentation_dose == "equal_time" else g_a+g_b) + c.augmentation_effect
    gain2 = np.where(a2 == 0, g_a, np.where(a2 == 1, g_b, combined))
    # Switching may carry a cost/benefit; adding B after A versus A after B retains history.
    gain2 += c.switch_effect*((a2 != a1) & (a2 != 2))
    stress2 = np.clip(stress1 - d*gain2 + rng.normal(0, c.noise_sd, n), 0, 10)
    df = pd.DataFrame({"employee_id":np.arange(1,n+1), "baseline_skill":stress0,
        "motivation":control0, "stage1_guided":a1, "interim_skill":improvement,
        "interim_stress":stress1, "interim_motivation":control1, "high_progress":response,
        "stage2_guided":a2, "p1_observed":np.where(a1 == 1,c.p1_guided,1-c.p1_guided),
        "p2_observed":p2obs, OUTCOME:-stress2})
    df.attrs.update(simulation_config=asdict(c), synthetic=True,
                    stage2_actions=[0,1,2] if c.augmentation else [0,1])
    return df
