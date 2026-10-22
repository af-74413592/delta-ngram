"""Read-only NLL instrumentation layered over the unchanged LoCoMo worker."""
from __future__ import annotations
import types
from qwen38_locomo_delta_worker import LoCoMoDeltaWorker
from qwen38_locomo_nll_contract import validate_head_rows, require, digest

class LoCoMoNLLWorker(LoCoMoDeltaWorker):
    def delta_live_initialize(self, writer_specs):
        result=super().delta_live_initialize(writer_specs)
        model=self.get_model();original=model.compute_logits
        self._nll_head_rows=[]
        def logits(module, hidden_states, *args, **kwargs):
            require(self._live_runtime.store.active is not None, 'unregistered NLL head call')
            self._nll_head_rows.append(int(hidden_states.shape[0]))
            return original(hidden_states,*args,**kwargs)
        model.compute_logits=types.MethodType(logits,model)
        return dict(result,nll_readonly_worker=True,nll_worker_sha256=digest(__file__))

    def delta_live_begin(self,arm,kind,request_id,epoch,input_ids,diagnostics=False,diagnostics_path=None):
        require(kind=='query' and not self._nll_head_rows,'NLL worker only accepts clean readonly queries')
        self._nll_input_tokens=len(input_ids)
        return super().delta_live_begin(arm,kind,request_id,epoch,input_ids,diagnostics,diagnostics_path)

    def delta_live_finish(self):
        heads=list(self._nll_head_rows);diagnostics=self._live_diagnostics
        validate_head_rows(heads,self._nll_input_tokens,diagnostics=diagnostics)
        result=super().delta_live_finish()
        require(result['forward_calls']==1,'NLL invoked the backbone more than once')
        self._nll_head_rows=[]
        return dict(result,nll_head_rows=heads,nll_one_backbone_forward=True)

    def _deny_mutation(self,*args,**kwargs):
        raise ValueError('NLL is read-only: evidence writes, D1, new snapshots and finalization are disabled')
    locomo_reset_conversation=_deny_mutation
    locomo_arm_frame=_deny_mutation
    locomo_finalize_conversation=_deny_mutation
    delta_live_commit=_deny_mutation
    delta_live_checkpoint=_deny_mutation
    delta_live_pad_capacity=_deny_mutation
    delta_live_restore=_deny_mutation
