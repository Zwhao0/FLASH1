"""Local two-node S3 reproduction driver; source and resume logs remain separate."""
import argparse,json,pathlib,subprocess,sys,time,shlex,re
base=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--iters',type=int,default=1000);a=p.parse_args()
if not re.fullmatch(r'[a-zA-Z0-9_-]+',a.name):raise ValueError('Invalid experiment name')
source=a.name+'-fault';output=a.name+'-checkpoint';resume=a.name+'-resume'
env={'NPROC_PER_NODE':'2','GREYHOUND_GPU_DEVICES':'0,1','GREYHOUND_PP':'4','GREYHOUND_HIDDEN_SIZE':'256','GREYHOUND_NUM_LAYERS':'16','GREYHOUND_SEQ_LENGTH':'64','GREYHOUND_MICRO_BATCH_SIZE':'2','GREYHOUND_GLOBAL_BATCH_SIZE':'16','GREYHOUND_PROBE':'1','GREYHOUND_REPARTITION_METADATA':'1','NCCL_DEBUG':'INFO','NCCL_PATH':'/workspace/Greyhound/detector/injection/libnccl_delay.so','GREYHOUND_EXTRA_ARGS':'--split 1,0,0'}
def launch(name,settings):return [sys.executable,str(base/'launch_two.py'),name,str(a.iters),'--env',json.dumps(settings)]
first=subprocess.Popen(launch(source,{**env,'GREYHOUND_HANDLE_RESTART':'1'}))
try:
 subprocess.run([sys.executable,str(base/'monitor_comm.py'),source],check=True)
 code=first.wait(timeout=120)
 if code:raise RuntimeError('source training failed')
 subprocess.run([sys.executable,str(base/'reconfigure_s3.py'),source,output],check=True)
 plan=json.loads(pathlib.Path('artifacts/multinode/'+output+'-plan.json').read_text())
 for n in (1,2):
  ip=f'10.10.4.{n}'
  subprocess.run(['ssh','-S',f'/tmp/greyhound-ssh-{ip}',f'test@{ip}',f'mkdir -p /home/campus/zhangweihao/FLASH1/results/{resume} && printf "10\\n" > /home/campus/zhangweihao/FLASH1/results/{resume}/delay-ms'],check=True)
 second=launch(resume,{**env,'GREYHOUND_CHECKPOINT_NAME':output,'GREYHOUND_PP_SPLIT':','.join(map(str,plan)),'GREYHOUND_VERIFY_RESTORE':'1','GREYHOUND_EXTRA_ARGS':'--split 1,0,0 --exit-on-missing-checkpoint --use-checkpoint-opt-param-scheduler'})
 subprocess.run(second,check=True)
finally:
 for n in (1,2):
  ip=f'10.10.4.{n}'
  for name in (source,resume):
   subprocess.run(['ssh','-S',f'/tmp/greyhound-ssh-{ip}',f'test@{ip}',f'test ! -d /home/campus/zhangweihao/FLASH1/results/{name} || printf "0\\n" > /home/campus/zhangweihao/FLASH1/results/{name}/delay-ms'],check=False)
