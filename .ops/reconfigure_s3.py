"""Coordinate two stopped PP4 checkpoints, repartition by the measured plan, copy to both nodes."""
import argparse,subprocess,shlex,json,re,pathlib
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args()
for s in (a.source,a.output):
 if not re.fullmatch(r'[a-zA-Z0-9_-]+',s):raise ValueError(s)
root='/home/campus/zhangweihao/FLASH1'
def ssh(n,cmd):
 ip=f'10.10.4.{n+1}';p=subprocess.run(['ssh','-S',f'/tmp/greyhound-ssh-{ip}',f'test@{ip}',f'cd {root} && '+cmd],text=True,capture_output=True)
 if p.returncode:raise RuntimeError(f'node{n}: {p.stdout}\n{p.stderr}')
 return p.stdout
for n in (0,1):
 code=ssh(n,f'cat results/{a.source}/exit-code.txt').strip()
 if code!='0':raise RuntimeError(f'node{n} training did not stop cleanly: {code}')
log=ssh(0,f'cat results/{a.source}/trainlog/*/global_controller_*.log')
plans=re.findall(r'Apply layer split (\[[^\]]+\]) done',log)
if not plans:raise RuntimeError('No controller-selected plan')
plan=json.loads(plans[-1]);target=','.join(map(str,plan));assert len(plan)==4 and sum(plan)==16 and min(plan)>0
print(json.dumps({'event':'selected_plan','plan':plan}),flush=True)
ssh(1,f'tar -cf artifacts/multinode/{a.source}-worker.tar -C checkpoints/{a.source} . && sha256sum artifacts/multinode/{a.source}-worker.tar > artifacts/multinode/{a.source}-worker.sha256')
ssh(0,f'curl --fail --silent --show-error http://10.10.4.2:18080/{a.source}-worker.tar -o artifacts/multinode/{a.source}-worker.tar && curl --fail --silent --show-error http://10.10.4.2:18080/{a.source}-worker.sha256 -o artifacts/multinode/{a.source}-worker.sha256 && sha256sum -c artifacts/multinode/{a.source}-worker.sha256 && sudo -n chown -R test:test checkpoints/{a.source} && tar --no-same-owner --no-overwrite-dir -xf artifacts/multinode/{a.source}-worker.tar -C checkpoints/{a.source}')
print(json.dumps({'event':'source_shards_assembled'}),flush=True)
argv=['./docker-gh','run','--rm','-e','PYTHONPATH=/workspace/Greyhound/Megatron-LM','-v',root+'/Greyhound:/workspace/Greyhound:ro','-v',root+'/s3_repartition.py:/script.py:ro','-v',root+'/checkpoints:/checkpoints','-v',root+'/results:/repro','--entrypoint','python','tianyuanwu/greyhound:ae','/script.py','--source','/checkpoints/'+a.source,'--output','/checkpoints/'+a.output,'--source-split','4,4,4,4','--target-split',target,'--audit-json','/repro/'+a.output+'-audit.json']
print(ssh(0,shlex.join(argv)),flush=True)
ssh(0,f'sudo -n chown -R test:test checkpoints/{a.output} && tar -cf artifacts/multinode/{a.output}.tar -C checkpoints/{a.output} . && sha256sum artifacts/multinode/{a.output}.tar > artifacts/multinode/{a.output}.sha256')
ssh(1,f'mkdir -p checkpoints/{a.output} && curl --fail --silent --show-error http://10.10.4.1:18080/{a.output}.tar -o artifacts/multinode/{a.output}.tar && curl --fail --silent --show-error http://10.10.4.1:18080/{a.output}.sha256 -o artifacts/multinode/{a.output}.sha256 && sha256sum -c artifacts/multinode/{a.output}.sha256 && tar --no-same-owner -xf artifacts/multinode/{a.output}.tar -C checkpoints/{a.output}')
pathlib.Path('artifacts/multinode/'+a.output+'-plan.json').write_text(json.dumps(plan))
print(json.dumps({'event':'checkpoint_ready_both_nodes','checkpoint':a.output,'plan':plan}),flush=True)
