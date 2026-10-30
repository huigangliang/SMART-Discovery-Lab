"""Import the supported complete-case, two-stage randomized data format."""
import numpy as np
import pandas as pd
from core import REQUIRED, validate_data
TIMES=['baseline_time','stage1_time','interim_time','stage2_time','outcome_time']

def import_trial(frame, threshold, direction, timing_confirmed=False, augmentation=None, coping=False):
    d=frame.copy()
    needed=(set(REQUIRED)-{'high_progress'}) | {'employee_id'}
    missing=needed-set(d)
    if missing:
        raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    if len(d)<40:
        raise ValueError('At least 40 complete employee rows are required; this is a computational minimum, not a power justification.')
    if d.employee_id.isna().any() or d.employee_id.astype(str).str.strip().eq('').any() or d.employee_id.duplicated().any():
        raise ValueError('Employee IDs must be nonempty and unique: one row per employee.')
    for col in needed-{'employee_id'}:
        d[col]=pd.to_numeric(d[col],errors='raise')
    if coping and 'interim_stress' in d:
        interim=pd.to_numeric(d.interim_stress,errors='raise')
        if not np.isfinite(interim).all() or not np.allclose(d.baseline_skill-interim,d.interim_skill):
            raise ValueError('Interim reduction must equal baseline stress minus interim stress.')
    response=(d.interim_skill>=threshold).astype(int)
    if 'high_progress' in d and not np.array_equal(pd.to_numeric(d.high_progress).to_numpy(),response.to_numpy()):
        raise ValueError('Uploaded response flag disagrees with the current interim threshold.')
    d['high_progress']=response
    present=[x for x in TIMES if x in d]
    if present and len(present)!=len(TIMES):
        raise ValueError('Include all five timing columns or none.')
    if present:
        times=[pd.to_datetime(d[x],errors='coerce',utc=True) for x in TIMES]
        if any(x.isna().any() for x in times) or any((a>=b).any() for a,b in zip(times,times[1:])):
            raise ValueError('Times must parse and strictly follow baseline < stage 1 < interim < stage 2 < outcome.')
    elif not timing_confirmed:
        raise ValueError('Confirm measurement order when timestamps are absent.')
    for stage in (1,2):
        col=f'p{stage}_guided'
        if col in d:
            if stage == 2 and (d.stage2_guided == 2).any():
                raise ValueError('Three-action data must supply p2_observed, not binary p2_guided.')
            p=pd.to_numeric(d[col],errors='raise').to_numpy()
            expected=np.where(d[f'stage{stage}_guided']==1,p,1-p)
            if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all() or not np.allclose(expected,d[f'p{stage}_observed']):
                raise ValueError(f'{col} is inconsistent with observed-action probabilities.')
    if direction=='Lower is better':
        d['independent_competence']=-d['independent_competence']
    if augmentation is not None:
        d.attrs['stage2_actions']=[0,1,2] if augmentation else [0,1]
    validate_data(d,for_fit=True)
    # Discard unrelated upload fields before fitting/exporting to minimize accidental inclusion.
    return d[['employee_id',*REQUIRED]+(['interim_stress'] if coping and 'interim_stress' in d else [])].copy()
