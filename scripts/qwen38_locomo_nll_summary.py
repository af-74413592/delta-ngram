"""Paired conversation-cluster bootstrap for fixed-reference LoCoMo NLL."""
from __future__ import annotations
import math
from qwen38_locomo_nll_contract import PROTOCOL,CONDITIONS,require

def summarize(rows, *, expected_questions=1540, expected_conversations=10, replicates=10000, seed=20261022):
    import numpy as np
    lookup={(r['qid'],r['condition']):r for r in rows};qids=list(dict.fromkeys(r['qid'] for r in rows))
    require(len(lookup)==len(rows)==expected_questions*3 and len(qids)==expected_questions,'summary requires the complete NLL matrix')
    questions=[]
    for qid in qids:
        triple=[lookup[qid,c] for c in CONDITIONS]
        invariant=('sample_id','category','prompt_sha256','target_sha256','reference_sha256','target_tokens')
        require(all(all(r[k]==triple[0][k] for k in invariant) for r in triple),'NLL reference/input differs across conditions')
        for r in triple:
            require(r['category'] in (1,2,3,4) and r['target_tokens']>0
                and math.isfinite(r['nll_sum']) and math.isfinite(r['nll_token_mean'])
                and abs(r['nll_token_mean']-r['nll_sum']/r['target_tokens'])<=1e-10,'NLL loss/count is invalid')
        questions.append({'qid':qid,'sample_id':triple[0]['sample_id'],'target_tokens':triple[0]['target_tokens'],
            'means':{c:lookup[qid,c]['nll_token_mean'] for c in CONDITIONS},
            'sums':{c:lookup[qid,c]['nll_sum'] for c in CONDITIONS}})
    conversations=sorted({r['sample_id'] for r in questions});require(len(conversations)==expected_conversations,'NLL cluster population differs')
    idx=np.random.default_rng(seed).integers(0,len(conversations),size=(replicates,len(conversations)))
    question_counts=np.array([sum(r['sample_id']==s for r in questions) for s in conversations],dtype=np.float64)
    token_counts=np.array([sum(r['target_tokens'] for r in questions if r['sample_id']==s) for s in conversations],dtype=np.float64)
    metrics={};per_conversation={sid:{'question_count':int(question_counts[i]),'target_token_count':int(token_counts[i])} for i,sid in enumerate(conversations)}
    for name,field,denominator in (('query_mean_nll','means',question_counts),('token_pooled_nll','sums',token_counts)):
        numerators={c:np.array([sum(r[field][c] for r in questions if r['sample_id']==sid) for sid in conversations],dtype=np.float64) for c in CONDITIONS}
        def interval(numerator):
            den=denominator[idx].sum(axis=1);valid=den>0
            estimates=numerator[idx].sum(axis=1)[valid]/den[valid]
            return {'estimate':float(numerator.sum()/denominator.sum()),
                    'cluster_bootstrap_95ci':[float(x) for x in np.percentile(estimates,[2.5,97.5])],
                    'valid_bootstrap_replicates':int(valid.sum())}
        pairs=(('self','empty'),('donor','empty'),('self','donor'))
        metrics[name]={'unit':'nats/token','conditions':{c:interval(numerators[c]) for c in CONDITIONS},
                      'paired_differences':{a+'-'+b:interval(numerators[a]-numerators[b]) for a,b in pairs}}
        for i,sid in enumerate(conversations):per_conversation[sid][name]={c:float(numerators[c][i]/denominator[i]) for c in CONDITIONS}
    return {'protocol':PROTOCOL,'question_count':expected_questions,'evaluation_count':len(rows),'target_token_count_per_condition':int(token_counts.sum()),
            'excluded_category5':446,'metrics':metrics,'per_conversation':per_conversation,
            'negative_paired_difference':'First condition has lower NLL.',
            'bootstrap':{'cluster':'conversation','cluster_count':len(conversations),'replicates':replicates,'seed':seed,
                'paired':True,'percentile':[2.5,97.5],'weighting':'Resample complete conversations; ratio of sums to question counts or token counts.'}}
