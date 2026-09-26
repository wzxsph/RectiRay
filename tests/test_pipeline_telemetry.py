from ptrn_pipeline.telemetry import Alerts

def snap(t,step=1,progress=0,status='running',alive=True):
 return {'time':t,'training':{'run':'a','status':status,'phase':'evaluation'},'last_step':{'step':step},
 'progress':{'eval':{'evaluated':progress,'elapsed_seconds':t}},'process_alive':alive,'disk_free_bytes':20*2**30,'memory_bytes':2*2**30}

def test_stage_progress_prevents_false_stall_and_deduplicates():
 a=Alerts();assert not a.inspect(snap(0));assert not a.inspect(snap(100,progress=1))
 assert not a.inspect(snap(210,progress=2));events=a.inspect(snap(331,progress=2))
 assert events[0]['kind']=='no_stage_progress_120s';assert not a.inspect(snap(340,progress=2))
 assert a.inspect(snap(350,progress=3))[0]['kind']=='recovered'

def test_exit_is_immediate_and_readonly():
 a=Alerts();a.inspect(snap(0));assert a.inspect(snap(1,alive=False))[0]['kind']=='process_failed'

def test_alert_state_survives_monitor_disconnect(tmp_path):
 from ptrn_pipeline.telemetry import Alerts
 a=Alerts();a.run='run';a.active={'disk_below_15GiB'};a.signature='progress';a.changed=123
 a.save(tmp_path/'state.json');b=Alerts();b.restore(tmp_path/'state.json')
 assert b.active==a.active and b.signature==a.signature and b.changed==123
