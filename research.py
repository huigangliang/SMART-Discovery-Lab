"""Researcher-defined constructs over a fixed two-stage numeric demonstration.
No inference from prose to causal coefficients; no invented reference records.
"""
from copy import deepcopy
import json
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from theory import SOURCE_CATALOG

FIELDS = {
    'baseline_skill': ('Baseline outcome measure / covariate', 'Baseline work stress'),
    'motivation': ('Baseline moderator', 'Baseline perceived stressor controllability'),
    'interim_skill': ('Interim response / tailoring variable', 'Interim stress reduction from baseline'),
    'interim_motivation': ('Interim moderator', 'Interim perceived stressor controllability'),
    'independent_competence': ('Dependent variable / final outcome', 'Final work stress'),
}

def defaults():
    return {key: value[1] for key, value in FIELDS.items()} | {
        'action0': 'Problem-focused coping support', 'action1': 'Emotion-focused coping support',
        'direction': 'Lower is better',
        'mechanism': 'Support may work differently depending on perceived stressor controllability; continuation, switching and combination are competing hypotheses.',
        'keywords': 'coping stress appraisal controllability occupational recovery',
    }

def validate(spec):
    s = deepcopy(spec)
    for key in defaults():
        if not isinstance(s.get(key), str) or not s[key].strip():
            raise ValueError(f'{key} requires nonempty text.')
        s[key] = s[key].strip()
        if len(s[key]) > 4000:
            raise ValueError(f'{key} is too long (maximum 4,000 characters).')
    reserved = {'employee_id','stage1_guided','stage2_guided','high_progress','p1_guided','p2_guided','p1_observed','p2_observed','analysis_split','stage 1 action (0/1)','stage 2 action (0/1)','interim response >= threshold'}
    if any(s[k].casefold() in reserved or s[k] in FIELDS and s[k] != k for k in FIELDS):
        raise ValueError('Measurement labels must not collide with technical or action column names.')
    labels = [s[k].casefold() for k in FIELDS]
    if len(set(labels)) != len(labels):
        raise ValueError('Use distinct measurement names, including baseline versus interim timing.')
    if s['action0'].casefold() == s['action1'].casefold():
        raise ValueError('The two interventions must differ.')
    if s['direction'] not in ['Higher is better', 'Lower is better']:
        raise ValueError('Choose an outcome direction.')
    return s

STOP = set('the and for with from into this that before after final initial interim baseline independent variable outcome performance employee employees'.split())
def tokens(text):
    return {w for w in re.findall(r'[a-z]{3,}', text.lower()) if w not in STOP}

def literature_query(spec):
    return ' '.join([spec['keywords'], spec['independent_competence'], spec['action0'], spec['action1']])[:800]

def local_sources(spec):
    query = tokens(' '.join(spec[k] for k in FIELDS) + ' ' + literature_query(spec))
    matches = []
    for source in SOURCE_CATALOG:
        words = tokens(' '.join(str(source.get(k, '')) for k in ['title','supports','design_use']))
        overlap = sorted(query & words)
        if overlap:
            matches.append(deepcopy(source) | {'matched_terms': overlap, 'match_score': len(overlap),
                'match_status': 'Keyword candidate; researcher must review applicability to the selected constructs.'})
    return sorted(matches, key=lambda x: (-x['match_score'], x['id']))

def candidate(spec):
    s = validate(spec)
    y, x, m, r = (s[k] for k in ['independent_competence','baseline_skill','motivation','interim_skill'])
    a, b = s['action1'], s['action0']
    return {
        'title': f'Adaptive {a} versus {b}',
        'research_question': f'When and for whom does {a}, compared with {b}, improve {y} across two decisions?',
        'proposition_learning': f'H1 (proposed): Assignment to {a} versus {b} changes {y}. Prespecify the expected sign and effect size before collecting data. {x} is a measured baseline predictor, not randomized.',
        'proposition_scaffolding': f'H2 (proposed): Among employees below the prespecified {r} response threshold, continuation, switching and (if enabled) augmentation differ in {y}. Proposed mechanism: {s["mechanism"]}',
        'proposition_motivation': f'H3 (proposed): Baseline {m} moderates the stage-1 intervention effect. The interim moderator may be affected by stage 1; it can tailor stage 2 but does not identify causal mediation.',
        'competing_prediction': f'Neither intervention has a stable advantage; {b} may perform better in some prespecified response strata. Compare fixed and adaptive strategies.',
        'falsification': 'Prespecify directional contrasts, meaningful effect sizes, measurement checks, and precision requirements in real randomized data. Null or reversed contrasts may challenge the extension; synthetic results cannot test it.',
        'boundary_conditions': 'Individual employees; two decision stages; responders maintain, nonresponders can continue or switch; continuous numeric predictors, moderators and terminal outcome. Review intervention fidelity and interference.',
    }

def compile_research(base, spec, augmentation=False, dosage="equal_time"):
    s = validate(spec)
    d = deepcopy(base)
    d['schema_version'] = '0.6'
    d['research_variables'] = s
    d['source_catalog'] = local_sources(s)
    d['source_policy'] = {'scope': 'Local keyword-ranked seed catalog; optional online candidate metadata. Neither is a systematic review or automatic evidence synthesis.'}
    d['theories'] = deepcopy(base['theories'])
    d['theories'].append({'label':'Researcher-proposed mechanism', 'researcher_hypothesis':s['mechanism'],
                          'evidence_level':'Untested assumption; keyword matching is not evidence.'})
    for stage in d['stages']:
        stage['actions'] = [{'code': i, 'label': s[f'action{i}'], 'implemented_role': f'Binary action {i}'} for i in (0,1)]
        stage['time'] = 'Before stage 1' if stage['stage'] == 1 else 'After stage 1 and before stage 2'
    d['stages'][1].update(rerandomize_all=False, eligibility='Nonresponders only', responder_rule='Maintain stage-1 action; assignment probability 1')
    d['stages'][1].pop('probability_guided_high',None)
    d['executable_config'].pop('p2_guided_high',None)
    d['executable_config'].pop('motivation_threshold',None)
    if augmentation:
        d['stages'][1]['actions'].append({'code':2,'label':s['action0']+' + '+s['action1'], 'implemented_role':'Retain original support and add the other component'})
        d['stages'][1].pop('probability_guided_low',None)
        d['stages'][1]['nonresponder_probabilities'] = {'A':1/3,'B':1/3,'A+B':1/3}
    d['executable_config'].update(augmentation=augmentation, augmentation_dose=dosage)
    d['inputs']['intervention0'], d['inputs']['intervention1'] = s['action0'], s['action1']
    d['inputs']['outcome'] = s['independent_competence']
    d['hypothesis_tests'].update(main_effect=base['inputs']['proposition_learning'],
        response_dependent=base['inputs']['proposition_scaffolding'], moderator=base['inputs']['proposition_motivation'])
    d['protocol'] = {
        'illustrative_timing':'Week 0 baseline; weeks 1–2 first support; end week 2 response; weeks 3–4 second support; end week 4 final stress.',
        'A':'Problem-focused: identify controllable work stressors, make an action plan and review attempted solutions.',
        'B':'Emotion-focused: practice cognitive reappraisal, emotion regulation and relaxation.',
        'single_support_dose':'Two 30-minute sessions each week, plus recorded practice; illustrative protocol, not a validated clinical program.',
        'combined_dose': 'Same 60 minutes per week, split 30/30 across A and B.' if dosage=='equal_time' else 'Full A plus full B: 120 minutes per week; extra contact time is part of the treatment package.',
        'switch':'Stop scheduled delivery of the initial component and deliver the other; previously acquired skills may persist.',
        'response':'Baseline stress minus interim stress >= prespecified improvement threshold; not raw stress >= threshold.',
        'primary_outcome':'Final work stress; lower is better. No separate burnout or recovery scale is simulated.',
        'interpretation':'A+B superiority alone does not establish psychological synergy. Dosage and sequence must be considered.'}
    d['stages'][1]['response_rule'] = s['interim_skill']+' >= '+str(base['inputs']['response_threshold'])
    d['outcome'].update(label=s['independent_competence'], direction=s['direction'])
    for row in d['measurement_map']:
        if row['variable'] in FIELDS:
            row['construct'] = s[row['variable']]
            row['use'] = FIELDS[row['variable']][0] + '; synthetic continuous slot, measurement validity requires review'
        elif row['variable'] == 'high_progress':
            row['construct'] = 'Interim response at or above prespecified threshold'
    d['measurement_map'] += [{'variable': f'stage{i}_guided', 'when': f'At stage {i}', 'construct': f'0 = {s["action0"]}; 1 = {s["action1"]}' + ('; 2 = A+B' if i==2 and augmentation else ''), 'use': 'Randomized for all at stage 1; only nonresponders randomized at stage 2'} for i in (1,2)]
    d['provenance']['code_scope'] = 'Two-stage numeric SMART template with user-defined construct labels; no automatic causal model construction.'
    d['estimands'][0]['definition'] = 'Mean terminal outcome under a prespecified two-stage regime; policy optimization maximizes utility.'
    d['checks'] = [
        {'severity':'pass','check':'Information timing','detail':'Stage 1 uses baseline measurements only; stage 2 adds observed intermediate history.'},
        {'severity':'pass','check':'Action availability','detail':'Responders maintain; only nonresponders are rerandomized to the configured continuation, switching and optional combination arms.'},
        {'severity':'warning','check':'Measurement model','detail':'Two continuous baseline slots, two continuous interim slots, and one continuous outcome. Additional predictors, categorical measures, mediation and extra stages need model extensions.'},
    ]
    d['limitations'] = [
        'Variable names and mechanisms are researcher-defined. Renaming them does not calibrate the simulation to those constructs.',
        'Simulation coefficients, thresholds and randomization controls alter the executed numerical model. Hypothesis prose does not.',
        'Stored engine column names are retained for compatibility; the measurement map defines their selected meaning.',
        'Restricted Q-learning interaction features remain illustrative restrictions, not automatically justified by a newly entered theory.',
        'Local references are keyword candidates from a small seed catalog. Unmatched topics require online search and researcher assessment.',
        'Lower-is-better outcomes are sign-reversed into utility for optimization; policy tables report utility, where higher is always better.',
    ]
    d['observed_path_count'] = 8 if augmentation else 6
    d['embedded_strategy_count'] = 6 if augmentation else 4
    return d

def display_frame(df, spec):
    mapping = {k: spec[k] for k in FIELDS}
    mapping.update(stage1_guided='Stage 1 action (0/1)', stage2_guided='Stage 2 action (0=A, 1=B, 2=A+B)', high_progress='Interim response >= threshold')
    mapping['interim_stress'] = 'Interim work stress (before stage 2)'
    return df.rename(columns=mapping)

def search_crossref(query, journals='', opener=urlopen):
    """Unauthenticated candidate metadata. Never infer findings from titles."""
    query = query.strip()
    if not query:
        raise ValueError('Enter search terms.')
    params = {'query.bibliographic':query, 'filter':'type:journal-article','rows':100}
    request = Request('https://api.crossref.org/works?' + urlencode(params), headers={'User-Agent':'AdaptiveOrgLab/0.3 (research prototype)', 'Accept':'application/json'})
    with opener(request, timeout=20) as response:
        payload = json.load(response)
    allowed = {x.strip().casefold() for x in journals.splitlines() if x.strip()}
    rows, seen = [], set()
    for item in payload.get('message',{}).get('items',[]):
        doi = item.get('DOI','')
        titles = item.get('title',[])
        outlets = item.get('container-title',[])
        if not doi or not titles or doi.casefold() in seen:
            continue
        if allowed and not any(j.casefold() in allowed for j in outlets):
            continue
        seen.add(doi.casefold())
        date = item.get('published',{}).get('date-parts', [[]])
        rows.append({'title':titles[0], 'journal':'; '.join(outlets), 'doi':doi,
                     'year':date[0][0] if date and date[0] else None,
                     'url':'https://doi.org/'+doi,
                     'status':'Bibliographic candidate only; findings and relevance not reviewed.'})
    return rows


def restore_engine_columns(df, spec):
    """Accept exported custom labels or engine headers, rejecting ambiguous duplicates."""
    import pandas as pd
    from core import REQUIRED
    template = pd.DataFrame(columns=['employee_id', *REQUIRED])
    mapping = dict(zip(display_frame(template,spec).columns,template.columns))
    mapping['Interim work stress (before stage 2)']='interim_stress'
    mapping['Stage 2 action (0/1)']='stage2_guided'
    result=df.rename(columns=mapping)
    if result.columns.duplicated().any():
        raise ValueError('Ambiguous duplicate columns after mapping custom labels. Use either custom or technical headers, not both.')
    return result
