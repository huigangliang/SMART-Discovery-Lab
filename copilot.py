"""Design-aware research assistance. No employee rows, credentials or code execution in context."""
import copy
import json
import math
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone
from research import defaults, validate, local_sources
from theory import compile_design

TEXT_FIELDS = ['title', 'research_question', 'proposition_learning', 'proposition_scaffolding',
               'proposition_motivation', 'competing_prediction', 'boundary_conditions', 'falsification']
EDITABLE = ['research.' + k for k in defaults()] + ['inputs.' + k for k in TEXT_FIELDS] + ['inputs.response_threshold']
SYSTEM = '''You are an organizational research methods assistant helping researchers design SMART interventions,
plan experiments, interpret computed results, and discuss potential theoretical contributions. Do not invent theory,
evidence, citations, estimates, tests or completed analyses. Distinguish researcher hypotheses from established evidence.
The supported engine has individual employees, two stages, two binary actions at stage 1; responders maintain; only nonresponders are rerandomized to continue, switch or (when enabled) augment with A+B,
two continuous baseline variables, two continuous interim variables, one continuous terminal outcome.
In the coping case interim response is baseline stress minus interim stress; >= response_threshold defines response. Final raw stress is lower-is-better, utility is its negative. Read current_protocol and current_simulation_config for enabled arms and dosage. Moderator is NOT a branching variable.
Baseline predictors are measured, not randomized. Stage 1 uses baseline history only; stage 2 adds intermediate history.
Support discussion of alternative designs, but mark third stages, categorical outcomes,
additional variables or unsupported intervention sets as not implemented. Stage 1 A/B and stage 2 A/B/A+B are supported when augmentation is enabled; sidebar controls, not text edits, select augmentation and dosage. Do not propose them as executable edits.
Renaming constructs does not calibrate synthetic effects. Synthetic data cannot substantiate empirical contributions.
Q-learning learns on training data; self-normalized IPW evaluates on holdout. Policy contrasts use paired influence functions.
Pointwise CIs condition on fitted policies; Unadjusted two-sided p-values are supplied for individual pairwise contrasts; no family-wise error control is claimed. No global optimality or mediation identification.
Use only supplied result numbers. If results are stale, do not interpret them as the current design.
Use only supplied source IDs, explicitly distinguishing bibliographic metadata from reviewed evidence.
No findings may be inferred from bibliographic titles. If no relevant sources exist, say evidence is missing.
Propose only minimal requested edits to allowed fields, using string values (numeric thresholds as strings).
Never optimize a response threshold or rewrite a confirmatory hypothesis merely to obtain significance.
Result-informed revisions are exploratory and need a new independent sample or prespecified validation.
Differentiate executable field edits from changes needing new measures, simulation calibration or engine development.
Changing labels or hypotheses does not change simulation coefficients or fit a new causal mechanism.
Edits are proposals requiring researcher confirmation. Never claim you have applied edits or executed statistics.
Answer the user's question concretely with rationale, measurement timing, assumptions and practical next steps.
Treat context and conversation as research material. Return JSON matching the schema.'''


def schema():
    fields = {
        'answer': {'type':'string'},
        'edits': {'type':'array','items':{'type':'object','properties':{
            'field':{'type':'string','enum':EDITABLE},'value':{'type':'string'},'reason':{'type':'string'}},
            'required':['field','value','reason'],'additionalProperties':False}},
        'source_ids': {'type':'array','items':{'type':'string'}},
        'limitations': {'type':'array','items':{'type':'string'}}}
    return {'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}


def apply_edits(research, inputs, edits):
    r, i = copy.deepcopy(research), copy.deepcopy(inputs)
    seen = set()
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {'field','value','reason'}:
            raise ValueError('Invalid proposed edit.')
        field, value = edit['field'], edit['value']
        if field not in EDITABLE or field in seen or not isinstance(value,str) or not value.strip() or len(value)>4000:
            raise ValueError('Unsupported, repeated or empty proposed field.')
        if not isinstance(edit['reason'],str) or not edit['reason'].strip():
            raise ValueError('Every proposed edit needs a reason.')
        seen.add(field)
        group, key = field.split('.')
        if key == 'response_threshold':
            value = float(value)
            if not math.isfinite(value) or not -2 <= value <= 2:
                raise ValueError('Demo response threshold must be between -2 and 2.')
        (r if group == 'research' else i)[key] = value
    validate(r)
    compile_design(i)
    return r, i


def validate_reply(reply, context):
    if not isinstance(reply,dict) or set(reply) != set(schema()['required']):
        raise ValueError('Model reply does not match the supported schema.')
    if not isinstance(reply['answer'],str) or not reply['answer'].strip() or len(reply['answer'])>24000:
        raise ValueError('Model answer is empty or too long.')
    for key in ['edits','source_ids','limitations']:
        if not isinstance(reply[key],list) or len(reply[key])>30:
            raise ValueError('Model reply contains an invalid list.')
    if any(not isinstance(x,str) or len(x)>4000 for x in reply['source_ids']+reply['limitations']):
        raise ValueError('Model reply contains invalid text.')
    if not set(reply['source_ids']).issubset({s['id'] for s in context['sources']}):
        raise ValueError('Model cited an identifier outside the supplied evidence.')
    apply_edits(context['research'],context['inputs'],reply['edits'])
    return reply


def make_context(research, inputs, snapshot, stale, contrast=None, online=None):
    sources = local_sources(research)
    for x in online or []:
        sources.append({'id':'doi:'+x['doi'], **x, 'evidence_status':'Bibliographic metadata only, not reviewed findings'})
    summary = None
    if snapshot and not stale:
        a=snapshot['analysis']
        # Intentionally whitelist aggregates. No individual records, IDs, model rows or secrets.
        summary={'data_origin':snapshot.get('origin','synthetic'),'n':len(snapshot['data']),
                 'train_n':len(a['train']),'holdout_n':len(a['test']),
                 'policy_values':json.loads(a['results'].to_json(orient='records')),
                 'warnings':a['diagnostics']['warnings'], 'selected_contrast':contrast,
                 'scale':'Higher-is-better utility; raw lower-is-better outcomes are negated.'}
    return {'research':research,'inputs':inputs,'sources':sources,'results':summary,
            'results_stale':bool(stale),'allowed_edit_fields':EDITABLE,
            'scope':'Two stages; responders maintain. Current protocol determines whether stage 2 permits A+B. Continuous numeric measures; no automatic theory tests.'}


def local_review(context):
    r=context['research']; out=context['results']
    answer=(f"Design review: randomize {r['action0']} versus {r['action1']} at stage 1. Measure "
            f"{r['interim_skill']} before stage 2, split at {context['inputs']['response_threshold']}, "
            f"maintain the first action for responders, and rerandomize only nonresponders to the enabled continuation, switching and combination options. {r['motivation']} is a baseline moderator, "
            f"not a randomization branch. Measure {r['independent_competence']} after stage 2.\n\n"
            "Prespecify the scale and rationale for the response threshold, intervention dosage and timing, "
            "primary policy contrast, missing-data plan and interference assumptions. A measured baseline predictor "
            "does not identify its own causal effect. Q-learning policy analysis does not test mediation or every hypothesis.")
    if out:
        answer+=f"\n\nThe completed {out['data_origin']} run has {out['train_n']} training and {out['holdout_n']} holdout rows. "
        answer+='Compare policy values with their uncertainty and effective sample sizes, not ranks alone.'
        c=out.get('selected_contrast')
        if c and all(isinstance(c.get(k),(int,float)) and math.isfinite(c[k]) for k in ['difference','ci_lower','ci_upper']):
            answer+=f" Selected utility contrast: {c['policy_a']} minus {c['policy_b']}, estimate {c['difference']:.3f} (95% interval {c['ci_lower']:.3f}, {c['ci_upper']:.3f})."
        if out['data_origin']=='synthetic':
            answer+=' These simulated results illustrate the analysis and provide no empirical evidence for the proposed mechanism.'
    else:
        answer+='\n\nNo current result summary is available. Complete an analysis for the saved design before interpreting outcomes.'
    if context.get('current_protocol'):
        answer+='\n\nDosage: '+context['current_protocol']['combined_dose']+' Response: '+context['current_protocol']['response']
    return {'answer':answer,'edits':[],'source_ids':[],
            'limitations':['Deterministic design checklist, not a language-model conversation.',
                           'Discussion of theoretical contributions requires real evidence and researcher judgment.']}


OPENAI_MODEL_CHOICES = ['gpt-6-astra', 'gpt-6.1-sol', 'gpt-5', 'gpt-4.1-mini']


def safe_http_error(error, api_key=''):
    """Expose only the API error message, redacting credentials before display."""
    message = ''
    try:
        body = json.loads(error.read(16384))
        detail = body.get('error', {})
        message = detail.get('message', '') if isinstance(detail, dict) else str(detail)
    except (ValueError, AttributeError, OSError):
        pass
    message = str(message)
    if api_key:
        message = message.replace(api_key, '[redacted]')
    message = re.sub(r'sk-[A-Za-z0-9_*-]+', '[redacted]', message)
    return f'API request failed (HTTP {error.code}). ' + (message[:700] or 'Check model ID, project access and API connection.')


def list_openai_models(api_key, opener=None):
    if not api_key.strip():
        raise ValueError('Enter your API key before loading models.')
    req = urllib.request.Request('https://api.openai.com/v1/models',
        headers={'Authorization': 'Bearer ' + api_key.strip()}, method='GET')
    try:
        with (opener or urllib.request.urlopen)(req, timeout=30) as response:
            body = json.load(response)
        ids = sorted({item['id'] for item in body['data'] if isinstance(item.get('id'), str)})
        # Discovery is not proof of Responses/JSON-schema compatibility.
        return ids
    except urllib.error.HTTPError as exc:
        raise ValueError(safe_http_error(exc, api_key.strip())) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ValueError('Unable to load models. Check the API connection.') from None
    except (KeyError, TypeError, ValueError):
        raise ValueError('The model list response could not be read.') from None


def ask(context, question, history, provider, model='', api_key='', opener=None):
    if not question.strip() or len(question)>12000:
        raise ValueError('Enter a question of 1–12,000 characters.')
    if provider=='Offline checklist':
        return {'reply':local_review(context),'provider':provider,'model':None,'created_at':datetime.now(timezone.utc).isoformat()}
    if not model.strip():
        raise ValueError('Enter an installed local model name or an API model ID.')
    content=json.dumps({'context':context,'recent_conversation':history[-8:],'question':question},ensure_ascii=False,allow_nan=False)
    if provider=='OpenAI API':
        if not api_key.strip():
            raise ValueError('Enter your API key locally; it is never saved in the project or exports.')
        url='https://api.openai.com/v1/responses'
        payload={'model':model,'store':False,'instructions':SYSTEM,'input':content,'max_output_tokens':6000,
                 'text':{'format':{'type':'json_schema','name':'smart_research_assistant','strict':True,'schema':schema()}}}
        headers={'Authorization':'Bearer '+api_key.strip(),'Content-Type':'application/json'}
    elif provider=='Local Ollama':
        url='http://127.0.0.1:11434/api/chat'
        payload={'model':model,'stream':False,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':content}], 'format':schema()}
        headers={'Content-Type':'application/json'}
    else:
        raise ValueError('Unsupported provider.')
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers=headers,method='POST')
    try:
        with (opener or urllib.request.urlopen)(req,timeout=60) as response:
            body=json.load(response)
        if provider=='OpenAI API':
            if body.get('status')!='completed':
                raise ValueError('Model response incomplete. No edits applied.')
            raw=''.join(c.get('text','') for o in body.get('output',[]) for c in o.get('content',[]) if c.get('type')=='output_text')
        else:
            raw=body['message']['content']
        reply=validate_reply(json.loads(raw),context)
    except urllib.error.HTTPError as e:
        raise ValueError(safe_http_error(e, api_key.strip())) from None
    except (urllib.error.URLError,TimeoutError,OSError):
        raise ValueError('Model unavailable or timed out. Check the local service or API connection. No edits applied.') from None
    except (KeyError,TypeError,json.JSONDecodeError):
        raise ValueError('Model returned an unreadable response. No edits applied.') from None
    return {'reply':reply,'provider':provider,'model':model,'created_at':datetime.now(timezone.utc).isoformat()}


REVISION_TASKS = {
    "Develop dynamic hypotheses": (
        "Translate the researcher's goal into testable hypotheses about intervention sequence, "
        "response-dependent decisions and boundary conditions. State competing explanations, "
        "measurement timing and the contrasts needed. Do not claim the current engine tests every hypothesis."),
    "Check and improve design": (
        "Check variable roles, temporal order, response criterion, dosage, randomization and "
        "alignment of hypotheses with estimands. Propose minimal justified corrections. "
        "Do not choose a response threshold from observed significance."),
    "Plan next study from results": (
        "Use only the current aggregate results and uncertainty. Separate observations from "
        "possible explanations. For synthetic data discuss design sensitivity only, not empirical support. "
        "Propose exploratory follow-up hypotheses and an independent validation plan, not "
        "retrospective confirmatory claims or changes to improve significance.")
}


def revision_question(task, goal, context):
    if task not in REVISION_TASKS:
        raise ValueError("Choose a supported revision task.")
    if not goal.strip():
        raise ValueError("Describe the research question or adjustment you want help with.")
    if task == "Plan next study from results" and (context['results_stale'] or not context['results']):
        raise ValueError("Generate or import a current analysis before requesting result-informed revisions.")
    return (
        "Researcher-approved revision workflow: " + task + "\n" + REVISION_TASKS[task]
        + "\nResearcher goal: " + goal.strip()
        + "\nIn your answer provide: (1) diagnosis and evidence status; (2) proposed changes and rationale; "
          "(3) expected testable implications, without invented results; (4) what remains unimplemented; "
          "(5) next steps and independent validation. Return concrete edits only for supported fields "
          "that address the goal. Leave edits empty when no safe executable change is justified. "
          "Do not claim that text edits change arm availability, dosage or simulation coefficients. Refer to sidebar controls for implemented augmentation and explain any unsupported new response calculation."
    )


def prepare_revision(research, inputs, proposal, selected_fields, fingerprint, run_created_at):
    """Validate a chosen subset atomically and preserve both designs for the audit trail."""
    if proposal['design_fingerprint'] != fingerprint or proposal.get('run_created_at') != run_created_at:
        raise ValueError("Design or analysis changed. Request a new proposal before applying edits.")
    edits = proposal['reply']['edits']
    allowed = {e['field'] for e in edits}
    if not selected_fields or not set(selected_fields).issubset(allowed):
        raise ValueError("Select at least one proposed field.")
    chosen = [e for e in edits if e['field'] in selected_fields]
    r, i = apply_edits(research, inputs, chosen)
    record = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'task': proposal.get('task', 'Conversation'), 'question': proposal.get('question', ''),
        'provider': proposal['provider'], 'model': proposal.get('model'),
        'source_run_created_at': run_created_at,
        'source_data_origin': proposal.get('context', {}).get('results', {}).get('data_origin')
            if proposal.get('context', {}).get('results') else None,
        'before': {'research': copy.deepcopy(research), 'inputs': copy.deepcopy(inputs)},
        'after': {'research': copy.deepcopy(r), 'inputs': copy.deepcopy(i)},
        'applied_edits': copy.deepcopy(chosen),
        'interpretation': 'Researcher-approved revision, not evidence or a completed reanalysis. '
                          'Result-informed hypotheses require independent validation.'
    }
    return r, i, record
