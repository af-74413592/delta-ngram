"""Pinned official LoCoMo F1/abstention scoring, with paired conversation bootstrap."""
from __future__ import annotations
from abc import ABCMeta,abstractmethod
import argparse,ast,collections,hashlib,json,re,string
from pathlib import Path
import numpy as np
import regex

HERE=Path(__file__).resolve().parent
CONDITIONS=('self','empty','donor','full_context')
CATEGORIES={1:'multi_hop',2:'temporal',3:'open_domain',4:'single_hop',5:'adversarial'}
EXPECTED_EVAL='8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd'
EXPECTED_PORTER='40797f3ce2d69f3a8d816eaef26d0d9a0cb0b373b77230a7e2de39a6f1e25ea6'
EXPECTED_API='7f20872f9186659ccc9781bea2face969c58c069ca92a81b6e32a1c7d7c698da'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_jsonl(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x]

def official_functions(upstream=HERE/'upstream_v1'):
    """Load only reviewed functions/classes; no upstream CLI, BERT or API client imports."""
    upstream=Path(upstream);deps=upstream/'dependencies/nltk-3.8.1'
    for path,expected in [(upstream/'task_eval/evaluation.py',EXPECTED_EVAL),(deps/'porter.py',EXPECTED_PORTER),(deps/'api.py',EXPECTED_API)]:
        if sha(path)!=expected:raise ValueError(f'pinned source changed: {path.name}')
    scope={'ABCMeta':ABCMeta,'abstractmethod':abstractmethod,'re':re,'regex':regex,
        'string':string,'np':np,'Counter':collections.Counter}
    for path,name in [(deps/'api.py','StemmerI'),(deps/'porter.py','PorterStemmer')]:
        nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name==name]
        if len(nodes)!=1:raise ValueError('expected exactly one reviewed dependency class')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    scope['ps']=scope['PorterStemmer']()
    path=upstream/'task_eval/evaluation.py'
    names={'normalize_answer','f1_score','f1','eval_question_answering'}
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=names:raise ValueError('official scoring functions missing')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope

OFFICIAL=None
def score_prediction(prediction,annotation):
    global OFFICIAL
    if OFFICIAL is None:OFFICIAL=official_functions()
    if not isinstance(prediction,str):raise ValueError('prediction must be raw decoded text')
    category=annotation['category']
    if category==5:
        lower=prediction.lower()
        return float('no information available' in lower or 'not mentioned' in lower)
    if category not in (1,2,3,4):raise ValueError('unknown LoCoMo category')
    if annotation.get('answer') is None:raise ValueError('nonadversarial answer missing')
    answer=str(annotation['answer'])
    if category==3:answer=answer.split(';')[0].strip()
    return float(OFFICIAL['f1' if category==1 else 'f1_score'](prediction,answer))

def bootstrap_summary(scored,*,replicates=10000,seed=20261022):
    """Resample complete conversation clusters; preserve question pairing and weights."""
    conversations=sorted({x['sample_id'] for x in scored})
    if not conversations:raise ValueError('empty scored population')
    idx=np.random.default_rng(seed).integers(0,len(conversations),size=(replicates,len(conversations)))
    metrics={};per_conversation={sid:{} for sid in conversations}
    for group,cats in [('main_f1',(1,2,3,4))]+[(CATEGORIES[c],(c,)) for c in range(1,6)]:
        selected=[x for x in scored if x['category'] in cats]
        counts=np.array([sum(x['sample_id']==sid for x in selected) for sid in conversations],dtype=np.float64)
        if not counts.sum():continue
        sums={condition:np.array([sum(x['scores'][condition] for x in selected if x['sample_id']==sid) for sid in conversations]) for condition in CONDITIONS}
        def interval(numerators):
            totals=numerators[idx].sum(axis=1);denom=counts[idx].sum(axis=1);valid=denom>0
            estimates=totals[valid]/denom[valid]*100
            return {'score_percent':float(numerators.sum()/counts.sum()*100),
                'cluster_bootstrap_95ci_percent':[float(x) for x in np.percentile(estimates,[2.5,97.5])],
                'valid_bootstrap_replicates':int(valid.sum())}
        metrics[group]={'question_count':int(counts.sum()),'conditions':{c:interval(sums[c]) for c in CONDITIONS},
            'paired_differences':{'self-'+c:interval(sums['self']-sums[c]) for c in ('empty','donor','full_context')}}
        for i,sid in enumerate(conversations):
            per_conversation[sid][group]={'question_count':int(counts[i]),
                'score_percent':{c:float(sums[c][i]/counts[i]*100) if counts[i] else None for c in CONDITIONS}}
    return {'metrics':metrics,'per_conversation':per_conversation,'bootstrap':{'cluster':'conversation','cluster_count':len(conversations),
        'replicates':replicates,'seed':seed,'weighting':'question-weighted ratio of score sums to counts after resampling complete conversations',
        'paired':True,'percentile':[2.5,97.5]}}

def score_run(prepared,predictions,output):
    prepared=Path(prepared);predictions=Path(predictions);output=Path(output)
    if output.exists():raise ValueError('scoring output must be fresh')
    manifest=json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    for name in ('scoring/annotations.jsonl','runtime_data/questions.jsonl','protocol.json'):
        if sha(prepared/name)!=manifest['files'][name]['sha256']:raise ValueError('prepared artifact changed')
    annotations=read_jsonl(prepared/'scoring/annotations.jsonl');truth={x['qid']:x for x in annotations}
    questions=read_jsonl(prepared/'runtime_data/questions.jsonl')
    if len(truth)!=len(annotations) or len({x['qid'] for x in questions})!=len(questions):
        raise ValueError('duplicate prepared question ID')
    if set(truth)!={x['qid'] for x in questions}:raise ValueError('annotation/question registry differs')
    rows=read_jsonl(predictions);lookup={}
    for row in rows:
        key=(row['qid'],row['condition'])
        if row['qid'] not in truth or row['condition'] not in CONDITIONS or key in lookup:raise ValueError('unknown or duplicate prediction')
        if not isinstance(row['text'],str):raise ValueError('prediction text must be string')
        lookup[key]=row['text']
    expected={(qid,c) for qid in truth for c in CONDITIONS}
    if set(lookup)!=expected:raise ValueError('incomplete four-condition prediction matrix')
    scored=[{'qid':a['qid'],'sample_id':a['sample_id'],'category':a['category'],
        'scores':{c:score_prediction(lookup[a['qid'],c],a) for c in CONDITIONS}} for a in annotations]
    result=bootstrap_summary(scored)
    rounded=[dict(x,scores={c:round(v,3) for c,v in x['scores'].items()}) for x in scored]
    # The official CLI writes round(score, 3) before its statistics stage.
    # Preserve an explicit point-estimate cross-check without discarding precision.
    rounded_metrics={}
    for group,cats in [('main_f1',(1,2,3,4))]+[(CATEGORIES[c],(c,)) for c in range(1,6)]:
        selected=[x for x in rounded if x['category'] in cats]
        if selected:rounded_metrics[group]={c:100*sum(x['scores'][c] for x in selected)/len(selected) for c in CONDITIONS}
    result.update(protocol='qwen38_locomo_official_f1_v1',llm_judge=False,primary_population='categories 1-4 only; adversarial separate',
        upstream_commit='3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376',official_evaluation_sha256=EXPECTED_EVAL,
        pinned_nltk_version='3.8.1',porter_source_sha256=EXPECTED_PORTER,regex_version=regex.__version__,numpy_version=np.__version__,
        predictions_sha256=sha(predictions),prepared_manifest_sha256=sha(prepared/'manifest.json'),
        official_rounding='raw official scores retained for aggregation and bootstrap; official CLI three-decimal per-question scores and resulting point estimates also recorded',
        official_cli_rounded_point_estimates_percent=rounded_metrics)
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (output/'per_question.jsonl').write_text(''.join(json.dumps(dict(x,official_rounded_scores={c:round(v,3) for c,v in x['scores'].items()}),ensure_ascii=False)+'\n' for x in scored),encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=score_run(args.prepared,args.predictions,args.output)
    print(json.dumps({'output':str(args.output),'main':r['metrics']['main_f1']},indent=2))
