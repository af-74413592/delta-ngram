"""LoCoMo frame cropping and conversation resets over the sealed live_v3 worker."""
from __future__ import annotations
import os
from pathlib import Path
from qwen38_vllm_live_worker import DeltaNGramLiveWorkerExtension
from qwen38_locomo_delta_contract import (PROTOCOL,WRITER_SHA,WRITER_TENSOR_SHA,canonical,crop_writer_inputs,
    digest,load_amended_runtime,validate_conversation_provenance,require)
from qwen38_locomo_delta_state import ConversationStore

class LoCoMoDeltaWorker(DeltaNGramLiveWorkerExtension):
    def delta_live_initialize(self,writer_specs):
        from delta_ngram import qwen38_vllm_live as live
        require(set(writer_specs)=={'delta'} and writer_specs['delta']['sha256']==WRITER_SHA
            and writer_specs['delta']['tensor_sha256']==WRITER_TENSOR_SHA,'LoCoMo requires only the original formal delta writer')
        result=DeltaNGramLiveWorkerExtension.delta_live_initialize(self,writer_specs)
        binding=live._binding;contract=binding['locomo']
        require(binding['adapter_protocol']==PROTOCOL,'LoCoMo adapter binding differs')
        self._locomo_data=load_amended_runtime(contract['prepared'],contract['parent_execution_plan'],contract['amendment'])
        require(self._locomo_data['amendment_sha256']==contract['amendment_sha256'],'LoCoMo amendment changed on a TP worker')
        self._locomo_root=Path(contract['execution_root']).resolve()
        self._locomo_source=None;self._locomo_query_arm=None
        self._locomo_replace_store()
        self._locomo_check()
        result.update(adapter_protocol=PROTOCOL,adapter_sha256=digest(__file__),
            state_adapter_sha256=digest(Path(__file__).with_name('qwen38_locomo_delta_state.py')),
            empty_state_hashes={a:live.state_hash(s) for a,s in self._live_runtime.store.states.items()})
        return result

    def _locomo_check(self):
        self._live_check()
        for writer,alpha,guard in self._live_writers.values():
            require(not writer.training and not alpha.training and alpha.kind=='fixed' and alpha.initial_alpha==1.,
                    'LoCoMo writer/alpha left the frozen alpha1 condition')

    def _locomo_replace_store(self,*,sample_id=None):
        import torch
        old=self._live_runtime.store
        require(self._live_idle() and old.active is None and old.pending is None and getattr(old,'frame',None) is None,
                'cannot switch conversation during a request/armed frame/D1')
        frames=() if sample_id is None else self._locomo_data['frames'][sample_id]
        with torch.inference_mode(False),torch.no_grad():
            fresh=ConversationStore(old.layout,self.device,sample_id=sample_id,frames=frames)
        fresh.initialized=True;self._live_runtime.store=fresh
        require(all(state.rows.numel()==0 and int(state.block_clock[0])==0 for state in fresh.states.values()),
                'conversation did not start from a newly constructed empty store')

    def locomo_reset_conversation(self,sample_id):
        self._locomo_check();require(sample_id in self._locomo_data['conversations'],'unknown conversation')
        self._locomo_replace_store(sample_id=sample_id)
        self._locomo_source=sample_id;self._locomo_query_arm=None
        return self.locomo_state_status()

    def locomo_arm_frame(self,sample_id,frame_index,frame_sha256):
        self._locomo_check();store=self._live_runtime.store
        require(sample_id==store.sample_id and type(frame_index) is int and 0<=frame_index<len(store.frames),
                'frame owner/index differs')
        frame=store.frames[frame_index];require(frame['frame_sha256']==frame_sha256,'frame identity differs')
        store.arm_frame(frame)
        return {'rank':self.rank,'sample_id':sample_id,'frame_index':frame_index,'frame_sha256':frame_sha256,
                'new_tokens':frame['new_tokens'],'prefix_tokens':frame['prefix_tokens']}

    def delta_live_begin(self,arm,kind,request_id,epoch,input_ids,diagnostics=False,diagnostics_path=None):
        if kind=='query':
            require(arm==self._locomo_query_arm and arm in ('ordinary','delta'),'query did not select a final or empty snapshot')
        else:
            require(kind=='support' and self._locomo_query_arm is None,'evidence write requested while serving frozen queries')
        return DeltaNGramLiveWorkerExtension.delta_live_begin(
            self,arm,kind,request_id,epoch,input_ids,diagnostics,diagnostics_path)

    def delta_live_finish(self):
        from delta_ngram import qwen38_vllm_live as live
        result=DeltaNGramLiveWorkerExtension.delta_live_finish(self)
        if result['kind']=='support':
            frame=self._live_runtime.store.frame
            require(frame is not None,'completed support has no armed LoCoMo frame')
            self._live_hidden,self._live_physical=crop_writer_inputs(frame,self._live_hidden,self._live_physical)
            result.update(sample_id=frame['document_id'],frame_index=frame['frame_index'],frame_sha256=frame['frame_sha256'],
                encoder_valid_tokens=frame['prefix_tokens']+frame['new_tokens'],new_tokens=frame['new_tokens'],
                prefix_tokens_not_written=frame['prefix_tokens'],masked_padding_not_forwarded=frame['padding_tokens'],
                writer_hidden_sha256=live.tensor_hash(self._live_hidden),
                writer_physical_rows_sha256=live.tensor_hash(self._live_physical),current_delta_encoding=True)
        return result

    def locomo_finalize_conversation(self):
        self._locomo_check();require(self._live_idle(),'cannot freeze with an active scheduler')
        result=self._live_runtime.store.freeze()
        return dict(result,rank=self.rank,phase='frozen',adapter_protocol=PROTOCOL)

    def delta_live_checkpoint(self,path,replace=False):
        import torch
        import torch.distributed as dist
        from delta_ngram import qwen38_vllm_live as live
        self._locomo_check()
        store=self._live_runtime.store
        require(store.phase=='frozen' and store.sample_id in self._locomo_data['frames'] and replace is False,
                'only an independently completed conversation may be exported once')
        require(Path(path).resolve()==(self._locomo_root/'states'/(store.sample_id+'.state.pt')).resolve(),
                'snapshot path differs from its bound conversation')
        require(self._live_idle() and store.active is None and store.pending is None and store.frame is None,
                'conversation export requires a completed frozen D1 boundary')
        hashes={arm:live.state_hash(state) for arm,state in store.states.items()}
        provenance={'protocol':PROTOCOL,'sample_id':store.sample_id,'frames':len(store.frames),
            'frame_sequence_sha256':canonical([f['frame_sha256'] for f in store.frames]),
            'written_tokens':store.written_tokens,'writer_tensor_sha256':WRITER_TENSOR_SHA,
            'conversation_token_sha256':self._locomo_data['conversations'][store.sample_id]['token_ids_sha256']}
        reports=[None]*8;dist.all_gather_object(reports,{'hashes':hashes,'locomo':provenance},group=self._live_control)
        require(all(value==reports[0] for value in reports),'TP final snapshot provenance/seven-field mismatch')
        receipt={'state_hashes':hashes,'epochs':store.epochs,'identities':store.identities}
        if self.rank==0:
            target=Path(path);require(not target.exists(),'completed conversation snapshot already exists')
            temporary=target.with_suffix(target.suffix+'.partial')
            torch.save({'format':live.PROTOCOL,'binding_sha256':os.environ[live.ENV_SHA],**receipt,
                'locomo':provenance,'states':{arm:live.state_payload(state) for arm,state in store.states.items()}},temporary)
            with temporary.open('r+b') as stream:os.fsync(stream.fileno())
            os.replace(temporary,target)
            receipt.update(path=str(target),sha256=digest(target))
        return dict(receipt,rank=self.rank)

    def locomo_select_snapshot(self,sample_id,receipt=None):
        import torch
        self._locomo_check();self._locomo_replace_store()
        if sample_id is None:
            require(receipt is None,'empty condition cannot load a checkpoint')
            self._locomo_source=None;self._locomo_query_arm='ordinary'
        else:
            require(sample_id in self._locomo_data['frames'] and isinstance(receipt,dict),'snapshot conversation is unknown')
            path=Path(receipt['path']).resolve()
            require(path==(self._locomo_root/'states'/(sample_id+'.state.pt')).resolve(),
                    'only this LoCoMo run\'s independently written conversation snapshot may be loaded')
            require(digest(path)==receipt['sha256'],'indexed snapshot checksum differs')
            saved=torch.load(path,map_location='cpu',weights_only=True)
            validate_conversation_provenance(saved.get('locomo'),sample_id,self._locomo_data)
            del saved
            result=DeltaNGramLiveWorkerExtension.delta_live_restore(self,str(path),receipt['sha256'])
            require(all(result[key]==receipt[key] for key in ('epochs','identities','state_hashes')),'snapshot receipt differs')
            self._live_runtime.store.adopt_frozen(sample_id,self._locomo_data['frames'][sample_id],receipt['state_hashes'])
            self._locomo_source=sample_id;self._locomo_query_arm='delta'
        return self.locomo_state_status()

    def locomo_state_status(self):
        import resource
        import torch
        from delta_ngram import qwen38_vllm_live as live
        self._locomo_check();store=self._live_runtime.store
        require(self._live_idle() and store.active is None and store.pending is None,'state audit requires a released request')
        for state in store.states.values():live.validate_state(state)
        require(store.states['ordinary'].rows.numel()==0 and store.epochs['ordinary']==0,'empty state was written')
        return {'rank':self.rank,'adapter_protocol':PROTOCOL,'sample_id':store.sample_id,
            'selected_source':self._locomo_source,'query_arm':self._locomo_query_arm,'phase':store.phase,
            'epochs':dict(store.epochs),'identities':dict(store.identities),'written_tokens':store.written_tokens,
            'state_hashes':{arm:live.state_hash(state) for arm,state in store.states.items()},
            'state_fields':list(live.FIELDS),'writer_sha256':WRITER_TENSOR_SHA,'frozen_reader_writer':True,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(self.device),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(self.device),'cpu_max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'binding_sha256':os.environ[live.ENV_SHA]}
