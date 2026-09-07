import copy,json
from pathlib import Path
import pytest
import evidence

def test_gate_rejects_failure_stale_sources_and_changed_gpu(tmp_path,monkeypatch):
    current={'packages':{'example':'1'},'source_digest':'current','gpu':{'stdout':'name,uuid,driver\nRTX,GPU-id,595\n'}}
    monkeypatch.setattr(evidence,'environment',lambda:current)
    monkeypatch.setattr(evidence,'digest',lambda:('current',{}))
    path=tmp_path/'gate.json';gate={'status':'passed','environment':copy.deepcopy(current)}
    path.write_text(json.dumps(gate));evidence.require_gate(path)
    for field,value in [('status','failed'),('source_digest','old'),('packages',{'example':'2'}),('gpu',{'stdout':'name,uuid,driver\nRTX,other,595\n'})]:
        bad=copy.deepcopy(gate)
        if field=='status':bad[field]=value
        else:bad['environment'][field]=value
        path.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError):evidence.require_gate(path)

def test_writer_refuses_nonfinite_evidence(tmp_path):
    with pytest.raises(ValueError):evidence.save(tmp_path/'bad.json',{'latency':float('nan')})
