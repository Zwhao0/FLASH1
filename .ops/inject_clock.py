#!/usr/bin/env python3
import datetime,json,re,subprocess,time,sys,signal
from pathlib import Path
ROOT=Path('/home/campus/zhangweihao/FLASH1')
name=sys.argv[1] if len(sys.argv)>1 else 'dp-failslow'
assert re.fullmatch(r'[a-zA-Z0-9_-]+', name)
RUN=ROOT/'results'/name
RUN.mkdir(parents=True,exist_ok=True)
GPU='0'; timer='greyhound-clock-reset'
def stop_requested(signum, frame):
    raise SystemExit(128+signum)
signal.signal(signal.SIGTERM, stop_requested)
injected=False; restored=False; first_iteration=None; injection_time=None; plan_time=None

def event(kind,**extra):
    value={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':kind,**extra}
    with (RUN/'injection.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)

def cmd(args):
    result=subprocess.run(args,text=True,capture_output=True)
    event('command',argv=args,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    result.check_returncode()
    return result

def restore(why):
    global restored
    cmd(['sudo','-n','nvidia-smi','-i',GPU,'-rgc'])
    subprocess.run(['sudo','-n','systemctl','stop',timer+'.timer'],capture_output=True)
    restored=True
    event('restore',reason=why,iteration=iteration)

start=time.monotonic(); iteration=0
try:
    while time.monotonic()-start < 1800:
        text='\n'.join(p.read_text(errors='replace') for p in (RUN/'trainlog').glob('*/megatron_output*.log'))
        matches=re.findall(r'iteration\s+(\d+)\s*/',text)
        iteration=int(matches[-1]) if matches else 0
        local='\n'.join(p.read_text(errors='replace') for p in (RUN/'trainlog').glob('*/local_controller*.log'))
        glob='\n'.join(p.read_text(errors='replace') for p in (RUN/'trainlog').glob('*/global_controller*.log'))
        if iteration>0 and first_iteration is None: first_iteration=time.monotonic(); event('first_iteration',iteration=iteration)
        if not injected and iteration>=150 and first_iteration and time.monotonic()-first_iteration>=50 and any(label in local for label in ('Estimated iteration time', 'Esimated iteration time')):
            cmd(['sudo','-n','systemd-run','--collect','--unit='+timer,'--on-active=240s','/usr/bin/nvidia-smi','-i',GPU,'-rgc'])
            cmd(['sudo','-n','nvidia-smi','-i',GPU,'-lgc','210,210'])
            injected=True; injection_time=time.monotonic(); event('inject',gpu=0,graphics_mhz=210,iteration=iteration)
            cmd(['nvidia-smi','--query-gpu=index,clocks.current.graphics,utilization.gpu,memory.used','--format=csv'])
        if injected and not restored:
            has_plan=('new dp plan' in glob.lower() or 'batch_distribution' in glob) and ('Adjust DP' in glob or 'Mitigating fail-slow' in glob)
            if has_plan and plan_time is None: plan_time=time.monotonic(); event('plan_observed',iteration=iteration)
            if plan_time and time.monotonic()-plan_time>=45: restore('DP plan observed; 45 seconds elapsed')
            elif time.monotonic()-injection_time>=220: restore('bounded 220 second injection completed')
        if (RUN/'exit-code.txt').exists(): event('training_finished',iteration=iteration); break
        if restored: break
        time.sleep(2)
finally:
    if injected and not restored: restore('finally cleanup')
