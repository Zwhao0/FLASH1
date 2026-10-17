import ast
import os
import torch.distributed as dist
import redis

from megatron.core import mpu


GLOBAL_BATCH_DISTRIBUTION = None
REDIS_CLIENT = None
DP_MODIFICATION_VER = 0
INITIAL_MB_NUM = 0
CURRENT_MB_NUM = 0

def sync_global_batch_distribution():
    global REDIS_CLIENT, GLOBAL_BATCH_DISTRIBUTION
    content = REDIS_CLIENT.get('batch_distribution')
    if content is not None:
        GLOBAL_BATCH_DISTRIBUTION = ast.literal_eval(content.decode())


def init_batch_distribution():
    global REDIS_CLIENT
    if REDIS_CLIENT is None:
        REDIS_CLIENT = redis.StrictRedis(
            host=os.getenv("MASTER_ADDR", 'localhost'),
            port=int(os.getenv("REDIS_PORT", '6379')),
            db=0
        )
    REDIS_CLIENT.set("dp_version", 0)
    sync_global_batch_distribution()


def set_initial_micro_batch_num(my_dp_rank, my_mb_num):
    global REDIS_CLIENT, CURRENT_MB_NUM, INITIAL_MB_NUM, GLOBAL_BATCH_DISTRIBUTION
    assert REDIS_CLIENT is not None
    CURRENT_MB_NUM = my_mb_num
    INITIAL_MB_NUM = my_mb_num
    GLOBAL_BATCH_DISTRIBUTION = [
        my_mb_num
        for _ in range(mpu.get_data_parallel_world_size(with_context_parallel=False))
    ]
    REDIS_CLIENT.set(f"init_mb_num_{my_dp_rank}", my_mb_num)


def get_my_micro_batch_num():
    global GLOBAL_BATCH_DISTRIBUTION, DP_MODIFICATION_VER, CURRENT_MB_NUM, REDIS_CLIENT
    if mpu.is_unitialized():
        return None
    my_dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
    version_value, distribution_value = REDIS_CLIENT.mget(
        'dp_version', 'batch_distribution')
    new_version = int(version_value or 0)
    # Case 1: My DP batch distribution is the newest version => Do nothing
    if new_version == DP_MODIFICATION_VER:
        return CURRENT_MB_NUM
    # Case 2: My DP batch distribution should be updated => Update it
    if distribution_value is None:
        raise RuntimeError(
            'DP plan version {} has no batch_distribution'.format(new_version))
    GLOBAL_BATCH_DISTRIBUTION = ast.literal_eval(distribution_value.decode())
    _validate_global_batch_distribution(GLOBAL_BATCH_DISTRIBUTION)
    print(f"###### DP changed: from {CURRENT_MB_NUM} to {GLOBAL_BATCH_DISTRIBUTION[my_dp_rank]}")
    CURRENT_MB_NUM = GLOBAL_BATCH_DISTRIBUTION[my_dp_rank]
    DP_MODIFICATION_VER = new_version
    return CURRENT_MB_NUM


def _validate_global_batch_distribution(distribution):
    if not isinstance(distribution, (list, tuple)):
        raise ValueError('DP plan must be a list or tuple: {}'.format(distribution))

    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    if len(distribution) != dp_size or any(int(value) < 1 for value in distribution):
        raise ValueError(
            'DP plan must contain {} positive microbatch counts: {}'.format(
                dp_size, distribution))

    global_batch_size, micro_batch_size = REDIS_CLIENT.mget(
        'global_batch_size', 'micro_batch_size')
    if global_batch_size is None or micro_batch_size is None:
        raise RuntimeError('global_batch_size/micro_batch_size missing in Redis')
    global_batch_size = int(global_batch_size)
    micro_batch_size = int(micro_batch_size)
    allocated_batch_size = sum(int(value) for value in distribution) * micro_batch_size
    if allocated_batch_size != global_batch_size:
        raise ValueError(
            'DP plan allocates {} samples, expected global batch {}: {}'.format(
                allocated_batch_size, global_batch_size, distribution))


def get_synchronized_my_micro_batch_num_dp_only():
    """Read one plan on rank 0 and broadcast it to the DP-only world."""
    global GLOBAL_BATCH_DISTRIBUTION, DP_MODIFICATION_VER, CURRENT_MB_NUM

    if mpu.is_unitialized():
        return None
    if (mpu.get_tensor_model_parallel_world_size() != 1 or
            mpu.get_pipeline_model_parallel_world_size() != 1 or
            dist.get_world_size() !=
            mpu.get_data_parallel_world_size(with_context_parallel=False)):
        raise RuntimeError('DP-only plan broadcast used with non-DP-only topology')

    payload = [None]
    if dist.get_rank() == 0:
        try:
            version_value, distribution_value = REDIS_CLIENT.mget(
                'dp_version', 'batch_distribution')
            new_version = int(version_value or 0)
            if new_version == DP_MODIFICATION_VER:
                distribution = GLOBAL_BATCH_DISTRIBUTION
                if distribution is None:
                    distribution = [INITIAL_MB_NUM] * dist.get_world_size()
            else:
                if distribution_value is None:
                    raise RuntimeError(
                        'DP plan version {} has no batch_distribution'.format(
                            new_version))
                distribution = ast.literal_eval(distribution_value.decode())
                _validate_global_batch_distribution(distribution)
            payload[0] = (True, new_version, distribution)
        except Exception as error:
            payload[0] = (False, repr(error))

    dist.broadcast_object_list(payload, src=0)
    success, *result = payload[0]
    if not success:
        raise RuntimeError('Failed to read synchronized DP plan: {}'.format(result[0]))

    new_version, distribution = result
    my_dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
    if len(distribution) != dist.get_world_size():
        raise ValueError('DP plan has wrong world size: {}'.format(distribution))
    new_micro_batch_num = int(distribution[my_dp_rank])
    version_changed = new_version != DP_MODIFICATION_VER
    count_changed = CURRENT_MB_NUM != new_micro_batch_num
    if version_changed or count_changed:
        print('###### DP changed: from {} to {}'.format(
            CURRENT_MB_NUM, new_micro_batch_num))
    GLOBAL_BATCH_DISTRIBUTION = list(distribution)
    DP_MODIFICATION_VER = new_version
    CURRENT_MB_NUM = new_micro_batch_num
    return CURRENT_MB_NUM


def get_loss_and_grad_weight():
    if INITIAL_MB_NUM <= 0:
        return 1.0
    return CURRENT_MB_NUM / INITIAL_MB_NUM

# mpu.get_data_parallel_group()
