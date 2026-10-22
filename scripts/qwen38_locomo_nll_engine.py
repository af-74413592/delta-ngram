"""Allow exact gold prompt probabilities without modifying LiveEngine generation."""
from __future__ import annotations
import time
from qwen38_locomo_nll_contract import canonical, gold_logprobs, require

class NLLRequestEngine:
    def __init__(self, live_engine):
        self.live=live_engine

    def request(self, arm, target, *, request_id):
        engine=self.live;ids=list(target['prompt_ids'])+list(target['target_ids'])
        require(canonical(ids)==target['input_sha256'],'NLL input binding differs')
        params=engine.params_cls(n=1,temperature=0.,top_p=1.,top_k=-1,seed=17,max_tokens=1,
            ignore_eos=True,stop_token_ids=[],detokenize=False,prompt_logprobs=0,flat_logprobs=False,
            presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.)
        engine.quiescent()
        engine.rpc('delta_live_begin',arm,'query',request_id,engine.epochs[arm],ids,False,None)
        started=time.monotonic()
        responses=engine.llm.generate([{'prompt_token_ids':ids}],params,use_tqdm=False)
        require(len(responses)==1,'NLL response count differs')
        engine.quiescent()
        records=engine.rpc('delta_live_finish')
        require(len(records)==8 and sorted(r['rank'] for r in records)==list(range(8)), 'NLL is missing TP ranks')
        fields=('arm','kind','request_id','epoch','input_sha256','forward_calls','fresh_context',
                'native_ids_sha256','history_sha256','nll_head_rows','nll_one_backbone_forward')
        require(all(all(r[k]==records[0][k] for k in fields) for r in records),'NLL TP request audit disagrees')
        result=records[0]
        require(result['arm']==arm and result['kind']=='query' and result['request_id']==request_id
                and result['epoch']==engine.epochs[arm] and result['input_sha256']==canonical(ids)
                and result['forward_calls']==1 and result['fresh_context'] and result['nll_one_backbone_forward'],
                'NLL request did not use one fresh readonly backbone prefill')
        values=gold_logprobs(responses[0],target['prompt_ids'],target['target_ids'])
        return values,dict(result,elapsed_seconds=time.monotonic()-started)
