import os
import abc
import sys
import time
import torch
from datetime import datetime
import dataclasses
import redis
from math import sqrt, floor
import subprocess
import argparse
import re
import socket
import shlex


@dataclasses.dataclass
class BaseConfig(abc.ABC):
    def to_config_string(self) -> str:
        config_str = ""
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) and value:
                config_str += f"--{field.name.replace('_', '-')} "
            elif not isinstance(value, bool):
                config_str += f"--{field.name.replace('_', '-')} {shlex.quote(str(value))} "
        return config_str.strip()


@dataclasses.dataclass
class DistributedConfig(BaseConfig):
    nproc_per_node: int
    nnodes: int
    node_rank: int
    master_addr: str
    master_port: str


@dataclasses.dataclass
class DatasetConfig(BaseConfig):
    vocab_file: str
    merge_file: str
    data_path: str
    split: str
    mock_data: bool


@dataclasses.dataclass
class ModelConfig(BaseConfig):
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    seq_length: int
    max_position_embeddings: int
    micro_batch_size: int
    global_batch_size: int
    lr: float
    train_iters: int
    lr_decay_iters: int
    lr_decay_style: str
    min_lr: float
    weight_decay: float
    lr_warmup_fraction: str
    clip_grad: float
    fp16: bool


@dataclasses.dataclass
class TrainConfig(BaseConfig):
    distributed_config: DistributedConfig
    dataset_config: DatasetConfig
    model_config: ModelConfig
    log_interval: int
    save_interval: int
    exit_interval: int
    eval_interval: int
    eval_iters: int
    distributed_backend: str
    save: str   # checkpoint save path
    load: str   # checkpoint load path

    def to_config_string(self, failslow_aware=True) -> str:
        other_config_str = ""
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, BaseConfig):
                if isinstance(value, bool):
                    if value:
                        other_config_str += f"--{field.name.replace('_', '-')} "
                else:
                    other_config_str += (
                        f"--{field.name.replace('_', '-')} {shlex.quote(str(value))} "
                    )
        command = "torchrun {} pretrain_gpt.py {} {} --sequence-parallel".format(
            self.distributed_config.to_config_string(),
            self.model_config.to_config_string(),
            self.dataset_config.to_config_string()
        )
        if other_config_str:
            command += " " + other_config_str.strip()
        if failslow_aware:
            command += " --failslow-aware"
        return command.strip()


def run_and_log_megatron(megatron_cmd_args, log_file_handle, log_file_dir, distributed_config):
    print(megatron_cmd_args, flush=True)
    my_env = os.environ.copy()
    my_env['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    my_env['OMP_NUM_THREADS'] = '1'
    my_env.setdefault('MASTER_ADDR', distributed_config.master_addr)
    my_env.setdefault('REDIS_PORT', os.environ.get('GREYHOUND_REDIS_PORT', '6379'))
    probe_enabled = os.environ.get('GREYHOUND_PROBE', '1').lower() not in ('0', 'false', 'no', 'off')
    if probe_enabled:
        my_env['LD_PRELOAD'] = '/workspace/Greyhound/detector/build/libncclprobe.so'
    else:
        my_env.pop('LD_PRELOAD', None)
    my_env.setdefault('NCCL_SOCKET_IFNAME', 'bond0')
    my_env.setdefault('NCCL_IB_DISABLE', '1')
    my_env.setdefault('GLOO_SOCKET_IFNAME', 'bond0')
    my_env['CONTROL_PLANE_WHL_PATH'] = os.environ.get(
        'CONTROL_PLANE_WHL_PATH',
        '/workspace/Greyhound/detector/dist/control_plane-1.0-py3-none-any.whl'
    )
    for key in ('NCCLPROBE_LOG_PATH', 'GLOBAL_CONTROLLER_LOG_PATH', 'LOCAL_CONTROLLER_LOG_PATH'):
        my_env[key] = log_file_dir
    handle_restart = os.environ.get('GREYHOUND_HANDLE_RESTART', '0').lower() in ('1', 'true', 'yes', 'on')
    restart_client = None
    if handle_restart and distributed_config.node_rank == 0:
        restart_client = redis.Redis(
            host=distributed_config.master_addr,
            port=int(os.environ.get('GREYHOUND_REDIS_PORT', os.environ.get('REDIS_PORT', '6379'))),
            db=0, socket_connect_timeout=0.5, socket_timeout=0.5
        )
    import selectors
    process = subprocess.Popen(megatron_cmd_args, env=my_env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    child_exited_at = None
    restart_signal_sent = False
    try:
        while True:
            if restart_client is not None and not restart_signal_sent:
                try:
                    signal = restart_client.get('terminate_ctl')
                    if signal is not None and signal.decode() in ('123', '321'):
                        restart_client.set('terminate', '2')
                        message = f"RESTART_REQUEST terminate_ctl={signal.decode()} setting terminate=2\n"
                        print(message, end='', flush=True)
                        log_file_handle.write(message)
                        log_file_handle.flush()
                        restart_signal_sent = True
                except redis.RedisError:
                    pass
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
    finally:
        if restart_client is not None:
            close_client = getattr(restart_client, 'close', None)
            if close_client is not None:
                close_client()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default=os.environ.get('GREYHOUND_LOGDIR', '/workspace/Greyhound/trainlog'))
    parser.add_argument('--iter', '--iters', dest='iter', type=int,
                        default=int(os.environ.get('GREYHOUND_ITERS', '10000')))
    parser.add_argument('--master-addr', default=os.environ.get('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master-port', type=int, default=int(os.environ.get('MASTER_PORT', '6000')))
    parser.add_argument('--nnodes', type=int, default=int(os.environ.get('NNODES', '1')))
    parser.add_argument('--node-rank', '--rank', dest='node_rank', type=int,
                        default=int(os.environ.get('NODE_RANK', os.environ.get('RANK', '0'))))
    nproc_default = os.environ.get('NPROC_PER_NODE')
    parser.add_argument('--nproc-per-node', type=int,
                        default=int(nproc_default) if nproc_default is not None else None)

    parser.add_argument('--tp', '--tensor-model-parallel-size', dest='tp', type=int,
                        default=int(os.environ.get('GREYHOUND_TP', '1')))
    parser.add_argument('--pp', '--pipeline-model-parallel-size', dest='pp', type=int,
                        default=int(os.environ.get('GREYHOUND_PP', '1')))
    parser.add_argument('--num-layers', type=int, default=int(os.environ.get('GREYHOUND_NUM_LAYERS', '32')))
    parser.add_argument('--micro-batch-size', type=int,
                        default=int(os.environ.get('GREYHOUND_MICRO_BATCH_SIZE', '4')))
    parser.add_argument('--global-batch-size', type=int,
                        default=int(os.environ.get('GREYHOUND_GLOBAL_BATCH_SIZE', '128')))
    parser.add_argument('--seq-length', type=int,
                        default=int(os.environ.get('GREYHOUND_SEQ_LENGTH', '32')))
    parser.add_argument('--hidden-size', type=int,
                        default=int(os.environ.get('GREYHOUND_HIDDEN_SIZE', '1024')))
    parser.add_argument('--save', default=os.environ.get('GREYHOUND_SAVE', '/workspace/checkpoints'))
    parser.add_argument('--load', default=os.environ.get('GREYHOUND_LOAD', '/workspace/checkpoints'))
    parser.add_argument('--save-interval', type=int,
                        default=int(os.environ.get('GREYHOUND_SAVE_INTERVAL', '10000')))
    parser.add_argument('--exit-interval', type=int,
                        default=int(os.environ.get('GREYHOUND_EXIT_INTERVAL', '0')))
    parser.add_argument('--redis-port', type=int, default=int(os.environ.get('GREYHOUND_REDIS_PORT', '6379')))
    parser.add_argument('--redis-timeout', type=float,
                        default=float(os.environ.get('GREYHOUND_REDIS_TIMEOUT', '120')))
    parser.add_argument('--pp-split', default=os.environ.get('GREYHOUND_PP_SPLIT', ''))
    no_failslow = os.environ.get('GREYHOUND_NO_FAILSLOW', '0').lower() in ('1', 'true', 'yes', 'on')
    aware = os.environ.get('GREYHOUND_FAILSLOW_AWARE', '1').lower() not in ('0', 'false', 'no', 'off')
    parser.add_argument('--no-failslow', action='store_true', default=(no_failslow or not aware),
                        help='disable failslow-aware training (also set GREYHOUND_NO_FAILSLOW=1)')
    args = parser.parse_args()
    args.extra_args = shlex.split(os.environ.get('GREYHOUND_EXTRA_ARGS', ''))
    # The checked-in Megatron argument uses an underscore in this one option.
    args.extra_args = [
        '--use-checkpoint-opt_param-scheduler'
        if arg == '--use-checkpoint-opt-param-scheduler' else arg
        for arg in args.extra_args
    ]
    return args


def parse_pp_split(value, pipeline_stages, total_layers):
    if not value:
        return None
    try:
        layer_counts = [int(part) for part in re.split(r'[\s,]+', value.strip()) if part]
    except ValueError as exc:
        raise ValueError(f'invalid GREYHOUND_PP_SPLIT={value!r}; expected comma-separated layer counts') from exc
    if len(layer_counts) != pipeline_stages:
        raise ValueError(
            f'GREYHOUND_PP_SPLIT has {len(layer_counts)} stages, expected {pipeline_stages}'
        )
    if any(count <= 0 for count in layer_counts) or sum(layer_counts) != total_layers:
        raise ValueError(
            f'GREYHOUND_PP_SPLIT must contain positive counts summing to {total_layers}; got {layer_counts}'
        )
    return layer_counts


def publish_pp_split(client, layer_counts):
    if not layer_counts:
        return
    assignments = {}
    offset = 0
    for stage, count in enumerate(layer_counts):
        assignments[f'pp_offset_{stage}'] = offset
        assignments[f'pp_num_layers_{stage}'] = count
        offset += count
    # MSET publishes every stage's offset and count in one Redis operation.
    client.mset(assignments)


def main():
    args = get_args()
    if args.nnodes < 1:
        raise ValueError('--nnodes must be at least 1')
    if args.node_rank < 0 or args.node_rank >= args.nnodes:
        raise ValueError(f'--node-rank must be in [0, {args.nnodes - 1}]')
    if args.nnodes > 1 and args.master_addr.lower() in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('multi-node runs require --master-addr / MASTER_ADDR to be the master node IP')
    if args.tp < 1 or args.pp < 1:
        raise ValueError('--tp and --pp must be positive')
    if args.num_layers < 1 or args.micro_batch_size < 1 or args.global_batch_size < 1:
        raise ValueError('layer and batch sizes must be positive')
    if args.iter < 1:
        raise ValueError('--iter must be positive')
    if args.redis_timeout <= 0:
        raise ValueError('--redis-timeout must be positive')
    pp_split = parse_pp_split(args.pp_split, args.pp, args.num_layers)

    os.chdir('./Megatron-LM/')
    log_file_dir, iter_1000 = args.logdir, args.iter
    master, nnodes, rank = args.master_addr, args.nnodes, args.node_rank
    redis_proc = None
    client = None
    try:
        num_gpus = torch.cuda.device_count()
        if num_gpus < 1:
            raise RuntimeError('no CUDA GPUs are visible in this container')
        nproc_per_node = args.nproc_per_node if args.nproc_per_node is not None else num_gpus
        if nproc_per_node < 1 or nproc_per_node > num_gpus:
            raise ValueError(f'--nproc-per-node must be between 1 and visible GPU count ({num_gpus})')
        world_size = nproc_per_node * nnodes
        if args.tp * args.pp > world_size or world_size % (args.tp * args.pp) != 0:
            raise ValueError(f'tp*pp ({args.tp * args.pp}) must divide total world size ({world_size})')

        gpu_properties = torch.cuda.get_device_properties("cuda:0")
        gpu_memory = gpu_properties.total_memory / (1024**3)  # GB
        total_gmem = gpu_memory * nproc_per_node
        hostname = socket.gethostname()
        ipaddr = socket.gethostbyname(hostname)
        redis_logstr = ""
        client = redis.Redis(host=master, port=args.redis_port, db=0,
                             socket_connect_timeout=1, socket_timeout=1)

        if rank == 0:
            redis_cmd = [
                "redis-server", "--save", "", "--appendonly", "no",
                "--bind", master, "--protected-mode", "no", "--port", str(args.redis_port)
            ]
            redis_proc = subprocess.Popen(redis_cmd)
            redis_logstr = "Rank 0 starts redis: [" + " ".join(redis_cmd) + "]\n"
            redis_deadline = time.monotonic() + min(args.redis_timeout, 30.0)
            while True:
                if redis_proc.poll() is not None:
                    raise RuntimeError(f'redis-server exited early with code {redis_proc.returncode}')
                try:
                    client.ping()
                    break
                except redis.RedisError:
                    if time.monotonic() >= redis_deadline:
                        raise TimeoutError(f'redis-server did not become ready at {master}:{args.redis_port}')
                    time.sleep(0.2)
            time_str = str(datetime.now()).replace(" ", '_').replace('-', '_').replace(':', '_').replace('.', '_')
            log_file_dir = log_file_dir.rstrip('/') + '/log_' + time_str
            publish_pp_split(client, pp_split)
            # Publish the readiness key last so workers cannot start before a
            # configured PP split is fully visible in Redis.
            client.set("trainlog_dir", log_file_dir)
        else:
            deadline = time.monotonic() + args.redis_timeout
            logdir_ret = None
            last_error = None
            while time.monotonic() < deadline:
                try:
                    logdir_ret = client.get("trainlog_dir")
                    if logdir_ret and len(logdir_ret) > 5:
                        break
                except redis.RedisError as exc:
                    last_error = exc
                time.sleep(0.2)
            if not logdir_ret:
                detail = f': {last_error}' if last_error else ''
                raise TimeoutError(f'timed out waiting for rank 0 Redis logdir at {master}:{args.redis_port}{detail}')
            print("Logdir worker!", logdir_ret, flush=True)
            log_file_dir = logdir_ret.decode()

        # The logdir value is intentionally the same string on all hosts. Each
        # node's /repro mount resolves it to that node's own local results tree.
        log_file_dir += f"_rank{rank}"
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = log_file_dir + f"/megatron_output_{rank}.log"
        info_str = (
            f"***** log={log_file_path}, master={master}, master_port={args.master_port}, "
            f"nnodes={nnodes}, node_rank={rank}, nproc_per_node={nproc_per_node}, "
            f"num_gpus={num_gpus}, gpu_type={gpu_properties.name}, gpu_memory={gpu_memory}, "
            f"total_gmem={total_gmem}, hidden_size={args.hidden_size}, tp={args.tp}, pp={args.pp}, "
            f"[My IP={ipaddr} Master IP={master}]\n"
        )
        print(info_str, flush=True)

        distributed_config = DistributedConfig(
            nproc_per_node=nproc_per_node, nnodes=nnodes, node_rank=rank,
            master_addr=master, master_port=args.master_port
        )
        model_config = ModelConfig(
            tensor_model_parallel_size=args.tp, pipeline_model_parallel_size=args.pp,
            num_layers=args.num_layers, hidden_size=args.hidden_size, num_attention_heads=32,
            seq_length=args.seq_length, max_position_embeddings=max(1024, args.seq_length),
            micro_batch_size=args.micro_batch_size, global_batch_size=args.global_batch_size,
            lr=0.00015, train_iters=int(iter_1000), lr_decay_iters=int(0.64 * iter_1000),
            lr_decay_style='cosine', min_lr=1.0e-5, weight_decay=0.01,
            lr_warmup_fraction='.01', clip_grad=1.0, fp16=True
        )
        dataset_config = DatasetConfig(
            vocab_file='/workspace/dataset/gpt2-vocab.json',
            merge_file='/workspace/dataset/gpt2-merges.txt',
            data_path='/workspace/dataset/gpt2_text_document',
            split='949,50,1', mock_data=True
        )
        train_config = TrainConfig(
            distributed_config=distributed_config, dataset_config=dataset_config,
            model_config=model_config, log_interval=1, save_interval=args.save_interval,
            exit_interval=args.exit_interval, eval_interval=10000, eval_iters=1,
            distributed_backend='nccl', save=args.save, load=args.load
        )

        run_args = shlex.split(train_config.to_config_string(failslow_aware=not args.no_failslow))
        run_args.extend(args.extra_args)
        with open(log_file_path, 'w') as log_file:
            log_file.write(info_str)
            log_file.write(redis_logstr)
            log_file.flush()
            run_and_log_megatron(run_args, log_file, log_file_dir, distributed_config)
    finally:
        if client is not None:
            close_client = getattr(client, 'close', None)
            if close_client is not None:
                close_client()
        if redis_proc is not None:
            try:
                redis_proc.terminate()
            except ProcessLookupError:
                pass
            try:
                redis_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                redis_proc.kill()
                redis_proc.wait()

if __name__ == '__main__':
    main()
