from pathlib import Path
import difflib
repo=Path('/home/campus/zhangweihao/FLASH1/Greyhound')
replacement='''def run_and_log_megatron(megatron_cmd_args, log_file_handle, log_file_dir, distributed_config):
    print(megatron_cmd_args, flush=True)
    my_env = os.environ.copy()
    my_env['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    my_env['OMP_NUM_THREADS'] = '1'
    my_env['LD_PRELOAD'] = '/workspace/Greyhound/detector/build/libncclprobe.so'
    my_env['CONTROL_PLANE_WHL_PATH'] = '/workspace/Greyhound/detector/dist/control_plane-1.0-py3-none-any.whl'
    for key in ('NCCLPROBE_LOG_PATH', 'GLOBAL_CONTROLLER_LOG_PATH', 'LOCAL_CONTROLLER_LOG_PATH'):
        my_env[key] = log_file_dir
    import selectors
    process = subprocess.Popen(megatron_cmd_args, env=my_env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    child_exited_at = None
    try:
        while True:
            for key, _ in selector.select(timeout=0.5):
                raw = os.read(key.fd, 65536)
                if raw:
                    output = raw.decode('utf-8', errors='replace')
                    sys.stdout.write(output)
                    sys.stdout.flush()
                    log_file_handle.write(output)
                    log_file_handle.flush()
            if process.poll() is not None:
                if child_exited_at is None:
                    child_exited_at = time.monotonic()
                if time.monotonic() - child_exited_at >= 2:
                    break
        selector.close()
        process.stdout.close()
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, megatron_cmd_args)
    except BaseException:
        process.terminate()
        process.wait()
        raise


'''
patches=[]
for filename in ('run_training.py','run_training_dp.py'):
    p=repo/filename; original=(p.with_suffix('.py.upstream').read_text() if p.with_suffix('.py.upstream').exists() else p.read_text())
    if not p.with_suffix('.py.upstream').exists(): p.with_suffix('.py.upstream').write_text(original)
    start=original.index('def run_and_log_megatron('); end=original.index('def get_args()',start)
    fixed=original[:start]+replacement+original[end:]
    fixed=fixed.replace('os.mkdir(log_file_dir)','os.makedirs(log_file_dir, exist_ok=True)')
    fixed=fixed.replace('seq_length=32,', 'seq_length=int(os.environ.get("GREYHOUND_SEQ_LENGTH", "32")),')
    if filename == 'run_training_dp.py':
        fixed=fixed.replace('hsize = 1024 #int(1024 * (floor(sqrt(total_gmem / 18) * 2) / 2))', 'hsize = int(os.environ.get("GREYHOUND_HIDDEN_SIZE", "1024"))')
    p.write_text(fixed)
    patches.extend(difflib.unified_diff(original.splitlines(True),fixed.splitlines(True),fromfile=filename+'.upstream',tofile=filename))
(repo.parent/'results/launcher-fixes.patch').write_text(''.join(patches))
