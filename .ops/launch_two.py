import argparse,json,pathlib,subprocess,shlex,time
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('iters',type=int);p.add_argument('--env',default='{}');a=p.parse_args()
env=json.loads(a.env);out=pathlib.Path('artifacts/multinode');out.mkdir(parents=True,exist_ok=True)
children=[]
for n,ip in enumerate(['10.10.4.1','10.10.4.2']):
    node_env={**env,'MASTER_ADDR':'10.10.4.1','NODE_RANK':str(n)}
    argv=['env',*[f'{k}={v}' for k,v in node_env.items()],'bash','run-multinode.sh',a.name,str(a.iters)]
    cmd='cd /home/campus/zhangweihao/FLASH1 && '+shlex.join(argv)
    handle=(out/f'{a.name}-node{n}-driver.log').open('w')
    children.append((n,subprocess.Popen(['ssh','-S',f'/tmp/greyhound-ssh-{ip}',f'test@{ip}',cmd],stdout=handle,stderr=subprocess.STDOUT),handle))
    print(json.dumps({'event':'started','node':n,'pid':children[-1][1].pid,'name':a.name}),flush=True)
status={}
while children:
    for entry in list(children):
        n,proc,handle=entry
        if proc.poll() is not None:
            status[str(n)]=proc.returncode; handle.close();children.remove(entry)
            print(json.dumps({'event':'completed','node':n,'exit':proc.returncode}),flush=True)
    if children: time.sleep(1)
(out/f'{a.name}-driver-status.json').write_text(json.dumps(status,indent=2))
raise SystemExit(0 if all(v==0 for v in status.values()) else 1)
