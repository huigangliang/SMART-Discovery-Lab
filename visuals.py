"""Shared figures for the UI and reproducibility bundle."""
import io
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure


def smart_tree(spec, augmentation=False):
    lines=['digraph SMART {', 'rankdir=TB;',
           'node [shape=box style="rounded,filled" fillcolor="#eef6f7" fontname="Arial"];',
           'R [label="Stage 1: randomize A / B"];']
    count=0
    for first,letter,other in [(0,'A','B'),(1,'B','A')]:
        lines += [f'S{first} [label={json.dumps(letter+": "+spec[f"action{first}"])}];',
                  f'R -> S{first};', f'N{first} [label="Below improvement threshold: randomize again"];',
                  f'S{first} -> N{first} [label="Nonresponder"];']
        outcomes=[('Maintain '+letter,True),('Continue '+letter,False),('Switch to '+other,False)]
        if augmentation: outcomes.append(('Retain '+letter+' + add '+other,False))
        for label,responder in outcomes:
            count+=1
            lines.append(f'P{count} [label={json.dumps(label+"; Path "+str(count))}];')
            source=f'S{first}' if responder else f'N{first}'
            lines.append(f'{source} -> P{count}'+(' [label="Meets threshold"]' if responder else '')+';')
    return '\n'.join(lines+['}'])


def comparison_figure(results, a, b, contrast, spec):
    selected = results.set_index('policy').loc[[a, b]]
    sign = -1 if spec['direction'] == 'Lower is better' else 1
    vals = sign * selected.estimate.to_numpy()
    errs = 1.96 * selected.se.to_numpy()
    fig = Figure(figsize=(7.2, 5.1), layout='constrained', facecolor='white')
    ax = fig.subplots()
    if not np.isfinite(vals).all():
        ax.text(.5, .5, 'Comparison unavailable: insufficient matching paths', ha='center', transform=ax.transAxes)
        return fig
    x = np.array([0., 1.])
    ax.bar(x, vals, width=.29, color=['#496D89', '#91AFA7'], edgecolor='white', linewidth=.7, zorder=3)
    if np.isfinite(errs).all():
        ax.errorbar(x, vals, yerr=errs, fmt='none', color='#293A46', capsize=4, capthick=1.1, lw=1.1, zorder=4)
    upper = vals + np.nan_to_num(errs)
    lower = vals - np.nan_to_num(errs)
    span = max(float(max(0, max(upper)) - min(0, min(lower))), .5)
    top = float(max(upper)) + .17 * span
    h = .035 * span
    ax.plot([0, 0, 1, 1], [top, top+h, top+h, top], color='#293A46', lw=1)
    pval = contrast.get('p_raw', float('nan'))
    ptext = ('p < .001' if pval < .001 else f'p = {pval:.4f}') if np.isfinite(pval) else 'p unavailable'
    ax.text(.5, top+h+.025*span, f"{contrast['significance']}  ({ptext}, unadjusted)", ha='center', va='bottom', fontsize=10, color='#293A46')
    for i, v in enumerate(vals):
        ax.text(i, upper[i]+.035*span, f'{v:.3f}', ha='center', va='bottom', fontsize=10, color='#293A46')
    labels = ['\n'.join(textwrap.wrap(name, 29)) for name in (a, b)]
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_xlim(-.6, 1.6)
    ax.set_ylabel('\n'.join(textwrap.wrap(spec['independent_competence'], 42)) + '\n' + spec['direction'], fontsize=10, labelpad=12)
    ax.set_title('Estimated policy outcomes', loc='left', fontsize=14, fontweight='semibold', pad=20, color='#243643')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#BBC4CA')
    ax.tick_params(axis='both', length=0, colors='#445560', pad=9)
    ax.axhline(0, color='#BBC4CA', lw=.8)
    bottom = 0 if min(lower) >= 0 else float(min(lower))-.06*span
    ax.set_ylim(bottom, top+h+.17*span)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#E9EDF0', lw=.7)
    fig.supxlabel('Bars: policy estimates; error bars: pointwise 95% confidence intervals.\nBracket: unadjusted paired contrast.  * p < .05   ** p < .01   *** p < .001', fontsize=8, color='#52616B')
    return fig


def figure_bytes(fig,fmt='svg'):
    b=io.BytesIO(); fig.savefig(b,format=fmt,dpi=180); return b.getvalue()


def tree_figure(spec, augmentation=False):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig=Figure(figsize=(14,9.5),layout='constrained'); ax=fig.subplots()
    ax.set_xlim(0,1); ax.set_ylim(-.05,1.1); ax.axis('off')
    boxes={'r':(.08,.51,.14,.12,'Stage 1\nRandomize A / B','#177e89','white')}
    links=[]; count=0
    for first,letter,other,top in [(0,'A','B',.92),(1,'B','A',.43)]:
        center=top-.14
        k=str(first)
        boxes['s'+k]=(.30,center,.19,.15,letter+'\n'+'\n'.join(textwrap.wrap(spec[f'action{first}'],23)[:3]),'#e6f2f3','#153743')
        boxes['n'+k]=(.55,top-.23,.19,.12,'Below threshold\nRandomize again','#fff0d8','#6c491b')
        links += [('r','s'+k),('s'+k,'n'+k)]
        outcomes=[('Maintain '+letter,True),('Continue '+letter,False),('Switch to '+other,False)]
        if augmentation: outcomes.append(('Retain '+letter+' + add '+other,False))
        for j,(label,responder) in enumerate(outcomes):
            count+=1; key='p'+str(count); y=top-j*.115
            boxes[key]=(.84,y,.25,.075,str(count)+' · '+label,'#e2f2e7' if responder else '#eef3f7','#153743')
            links.append(('s'+k if responder else 'n'+k,key))
    for source,target in links:
        x,y,w,h,*_=boxes[source];xx,yy,ww,hh,*_=boxes[target]
        ax.add_patch(FancyArrowPatch((x+w/2+.006,y),(xx-ww/2-.006,yy),arrowstyle='-|>',mutation_scale=12,color='#78919c',lw=1.3,zorder=1))
    for x,y,w,h,label,color,fg in boxes.values():
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',facecolor=color,edgecolor='#b7cbd1',zorder=2))
        ax.text(x,y,label,ha='center',va='center',fontsize=10,color=fg,zorder=3)
    ax.text(.01,1.055,'Adaptive coping support: continue, switch'+(' or combine' if augmentation else ''),fontsize=18,fontweight='bold',color='#153743')
    ax.text(.01,1.01,f'{8 if augmentation else 6} observed paths · {6 if augmentation else 4} embedded strategies · responders maintain',fontsize=12,color='#53717d')
    ax.text(.01,-.015,'Response: stress reduction ≥ threshold. Everyone is followed to the same final outcome time.',fontsize=11,color='#53717d')
    return fig
