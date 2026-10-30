# Methods — coping SMART

## Design and estimands

Stage 1 assigns A or B with P(B)=p1. Responders maintain with p2_observed=1.
Nonresponders receive A, B or A+B with probability 1/3 each when augmentation
is enabled. Otherwise P(B)=p2. Stage-2 treatment is categorical, not ordinal.
Each embedded strategy specifies its first action, maintenance for responders,
and one nonresponder action. Six such strategies share responder observations.
Eight observed paths are not eight independent randomized treatment groups.

All paths end at the same outcome time. The illustrative protocol has baseline
at week 0, interim at week 2, and final at week 4. Response means baseline stress
minus interim stress >= a prespecified threshold, including equality. It does
not mean high raw stress is desirable. Final raw stress is lower-is-better;
internal utility is its negative. Coping, burnout and recovery are not synonyms.

## Executed simulation

All constants below are demonstration assumptions, not estimates from references.
Let d=1,0,-1 for illustrative aligned, null and reversed scenarios, e denote the
B main-effect parameter, m the matching parameter, c the combination parameter
and s the switching parameter. Clip stress to the arbitrary interval [0,10].

- S0 = clip(5 + Normal(0,1.2)); M0 = Normal(0,1).
- A1 ~ Bernoulli(p1).
- G1 = d * [0.6 + e*A1 + (1-2*A1)*m*M0].
- S1 = clip(S0 - G1 + Normal(0,noise_sd)); L = S0-S1.
- M1 = 0.8*M0 + Normal(0,0.3). It is measured before the second decision.
- R = 1[L >= threshold]. A2=A1 for R=1; others are randomized.
- gA=0.6+m*M1; gB=0.6+e-m*M1.
- Equal-time combination: gAB=(gA+gB)/2+c. Added-time: gAB=gA+gB+c.
- g is the chosen component gain, plus s for switching (not augmentation).
- S2=clip(S1-d*g+Normal(0,noise_sd)); Y=-S2.

S0 and S1 are stored so the response calculation can be checked. The coefficient
m illustrates matching to controllability, not a proven interaction. d=0 removes
all effects of assignment, switching, dosage and combination. Changing c or
combination dosage affects only augmented trajectories. The simple model retains
history through S1 but is not a validated psychological model of skill carryover.

## Estimation

Split employees once into training and holdout samples. Fit linear Q2 only among
nonresponders, using separate indicators for B and A+B (A reference) and separate
interactions. Require every declared action in training; missing actions cause
an explicit error rather than silently redefining the study. Optimize Q2 over
eligible actions. Responders' pseudo-outcomes use their observed terminal utility;
nonresponders use the largest fitted Q2. Fit Q1 using baseline information only.
The restricted model uses controllability interactions; the full model includes
all eligible predecision covariates. These are model specifications, not automated
hypothesis or mediation tests. Check rank and effective sample size warnings.

Evaluate each fixed or learned policy on the untouched holdout using the Hajek
ratio: sum(w*Y)/sum(w), where w=I(compatible path)/(p1_observed*p2_observed).
A responder under a fixed first action is compatible with all three matching
complete strategies. Paired policy differences use differences of influence
functions, retaining shared-participant covariance. Pointwise 95% confidence
intervals condition on the fitted policies. Each pair uses an unadjusted two-sided normal-approximation p-value. No family-wise error control is claimed across the displayed comparisons. Selecting a winner after inspecting multiple contrasts remains exploratory.

## Uploads and exports

One row per employee. Use custom labels or internal column names. Stage 1 is 0/1;
stage 2 is 0/1/2 with augmentation, otherwise 0/1. Supply probabilities of the
actual assigned actions, not a binary probability interpreted for three actions.
When supplied, interim_stress must satisfy L=S0-S1. The app derives response from
L and checks an uploaded flag. Confirm measurement order or supply all timestamps.
Legacy internal identifiers are retained for compatibility; measurement maps and
custom-label exports define the current meanings. Exported raw outcomes are
positive stress scores; analysis files contain negative stress utility.

## AI revision and evidence

The assistant receives only whitelisted aggregates, current protocol/settings,
source summaries and recent conversation. It can propose supported field edits;
the researcher chooses a subset before atomic validation and application. The
before/after audit includes the source run. Changes invalidate previous results.
Result-informed revisions are exploratory and require independent confirmation.
Simulation alone does not establish theoretical or empirical findings. No source
establishes our threshold, dose, sequence superiority or combination synergy.

## Pairwise inference update
All pairwise p-values and chart significance annotations are now unadjusted. Confidence intervals are pointwise 95% intervals. These tests do not control family-wise error across the collection of comparisons.
