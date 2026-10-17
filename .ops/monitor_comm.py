import subprocess,shlex,pathlib,time,re,json,datetime,sys
name=sys.argv[1]
if not re.fullmatch(r'[a-zA-Z0-9_-]+',name):raise ValueError('Invalid experiment name')
records=pathlib.Path('artifacts/multinode')/f'{name}-comm-injection.jsonl'
root='/home/campus/zhangweihao/FLASH1'
def remote(n,cmd):
 ip=f'10.10.4.{n+1}'
 p=subprocess.run(['ssh','-S',f'/tmp/greyhound-ssh-{ip}',f'test@{ip}',cmd],capture_output=True,text=True,timeout=20)
 if p.returncode:raise RuntimeError(p.stderr)
 return p.stdout
def record(event,**kw):
 obj={'event':event,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw}
 with records.open('a') as f:f.write(json.dumps(obj)+'\n')
 print(json.dumps(obj),flush=True)
def write_delay(ms):
 for n in [0,1]:remote(n,shlex.join(['bash','-c',f'printf "%s\\n" {ms} > {root}/results/{name}/delay-ms']))
start=time.monotonic();first_training=None;injected=False;last_status=0
try:
 record('monitor_started',name=name)
 while time.monotonic()-start<480:
  output=remote(1,shlex.join(['tail','-n','160',f'{root}/results/{name}/console.log']))
  iters=re.findall(r'iteration\s+(\d+)\s*/',output)
  iteration=int(iters[-1]) if iters else 0
  if iteration and first_training is None:first_training=time.monotonic()
  controller=remote(0,shlex.join(['bash','-c',f'cat {root}/results/{name}/trainlog/*/global_controller_*.log 2>/dev/null || true']))
  if not injected and iteration>=150 and first_training and time.monotonic()-first_training>=50:
   if 'Performing pre-check' in controller:
    write_delay(10);record('delay_injected',iteration=iteration,delay_ms=10,pair=[1,2],scope='PP/world communicator size 4, count >=1024');injected=True
  plan=re.findall(r'Apply layer split (\[[^\]]+\]) done',controller)
  if plan:
   record('plan_observed',iteration=iteration,layer_split=json.loads(plan[-1]));break
  if 'FAILED' in output or 'Traceback (most recent call last)' in output:
   record('training_error',iteration=iteration);break
  if time.monotonic()-last_status>=30:
   print(json.dumps({'status':'waiting','iteration':iteration,'injected':injected,'elapsed':round(time.monotonic()-start)}),flush=True);last_status=time.monotonic()
  time.sleep(5)
 else:record('timeout')
finally:
 if injected:write_delay(0);record('delay_restored')
