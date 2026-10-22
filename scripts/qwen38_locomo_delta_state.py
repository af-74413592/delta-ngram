"""One independently empty conversation, using the original live FP32 D1 math."""
from __future__ import annotations
import torch
from delta_ngram import qwen38_vllm_live as live
from delta_ngram.qwen38_native import write_native_block
from delta_ngram.qwen38_training import writer_evidence_features
from qwen38_locomo_delta_contract import require,transport_frame_ids

class ConversationStore(live.LiveMemoryStore):
    def __init__(self,layout,device,*,sample_id=None,frames=()):
        super().__init__(layout,device)
        self.sample_id=sample_id;self.frames=tuple(frames);self.frame=None
        self.written_tokens=0;self.phase='writing' if sample_id is not None else 'empty'

    def arm_frame(self,frame):
        require(self.phase=='writing' and self.active is None and self.pending is None and self.frame is None,
                'conversation is not at a clean write boundary')
        index=self.epochs['delta']
        require(index<len(self.frames) and frame==self.frames[index]
            and frame['document_id']==self.sample_id and frame['frame_index']==index,'frame is not the next frozen conversation frame')
        transport_frame_ids(frame);self.frame=frame

    def begin(self,arm,kind,request_id,expected_epoch,input_ids):
        if kind=='support':
            require(arm=='delta' and self.phase=='writing' and self.frame is not None
                and input_ids==transport_frame_ids(self.frame),'only the armed full valid frame may encode evidence')
        else:
            require(kind=='query' and self.phase in ('empty','frozen') and self.frame is None,
                    'queries require a frozen final conversation or an empty store')
        return super().begin(arm,kind,request_id,expected_epoch,input_ids)

    def produce_update(self,writer,alpha,hidden,physical):
        require(self.active is None and self.pending is not None and self.frame is not None,'D1 needs a completed armed frame')
        current=self.pending;arm=current['arm'];new=self.frame['new_tokens']
        require(arm=='delta' and hidden.shape==(1,new,2560) and physical.shape==(1,new,16),
                'D1 must receive only the cropped new-token H and physical rows')
        mask=torch.ones(hidden.shape[:2],dtype=torch.bool,device=hidden.device)
        with torch.inference_mode(False),torch.no_grad():
            features=writer_evidence_features(hidden,mode='rmsnorm',eps=1e-6)
            candidate=write_native_block(writer,self.states[arm],features,physical,mask,alpha)
            return live.make_update(self.states[arm],candidate,physical.flatten(),arm=arm,
                epoch=self.epochs[arm],request_id=current['request_id'],previous_identity=self.identities[arm])

    def commit(self,update):
        require(self.frame is not None,'D1 has no frozen frame binding')
        count=self.frame['new_tokens'];result=super().commit(update)
        self.written_tokens+=count;self.frame=None
        require(self.epochs['ordinary']==0 and int(self.states['ordinary'].block_clock[0])==0
            and self.states['ordinary'].rows.numel()==0,'empty condition acquired evidence')
        return result

    def freeze(self):
        require(self.phase=='writing' and self.active is None and self.pending is None and self.frame is None,
                'cannot freeze an unfinished frame')
        require(self.epochs['delta']==len(self.frames)==int(self.states['delta'].block_clock[0])
            and self.written_tokens==sum(f['new_tokens'] for f in self.frames),'final conversation coverage/clock differs')
        for state in self.states.values():live.validate_state(state)
        self.phase='frozen'
        return {'sample_id':self.sample_id,'frames':len(self.frames),'written_tokens':self.written_tokens,
                'state_hashes':{arm:live.state_hash(state) for arm,state in self.states.items()},'epochs':dict(self.epochs)}

    def adopt_frozen(self,sample_id,frames,expected_hashes):
        require(self.active is None and self.pending is None and self.frame is None,'restored state has an active request')
        require(self.epochs=={'ordinary':0,'delta':len(frames)} and self.states['ordinary'].rows.numel()==0
            and int(self.states['ordinary'].block_clock[0])==0
            and int(self.states['delta'].block_clock[0])==len(frames),'restored state has an unrelated conversation clock')
        require({arm:live.state_hash(state) for arm,state in self.states.items()}==expected_hashes,'restored seven-field state differs')
        for state in self.states.values():
            live.validate_state(state);require(state.row_capacity==self.layout.row_capacity,'restored address capacity differs')
        self.sample_id=sample_id;self.frames=tuple(frames);self.written_tokens=sum(f['new_tokens'] for f in frames);self.phase='frozen'
