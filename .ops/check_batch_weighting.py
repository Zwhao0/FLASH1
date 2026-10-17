#!/usr/bin/env python3
"""CPU autograd check against the deployed batch-weight helper; run in AE image."""
from types import SimpleNamespace
import json,math
import torch
from megatron.variation_aware import batch_distribution as batching
from megatron.training import _iteration_batch_size

plan=[6,9,8,9]
micro=4
assert sum(plan)*micro==128
original_mpu=batching.mpu
original_redis=batching.REDIS_CLIENT
batching.mpu=SimpleNamespace(get_data_parallel_world_size=lambda **kwargs:4)
batching.REDIS_CLIENT=SimpleNamespace(mget=lambda *args:[b'128',b'4'])
batching._validate_global_batch_distribution(plan)
try:
 batching._validate_global_batch_distribution([6,9,8,8])
except ValueError:
 invalid_rejected=True
else:
 raise AssertionError('Invalid 124-sample plan accepted')
args=SimpleNamespace(failslow_aware=True,rampup_batch_size=None,global_batch_size=128)
assert _iteration_batch_size(args)==128

parameter=torch.tensor(2.0,dtype=torch.float64,requires_grad=True)
gradients=torch.arange(1,129,dtype=torch.float64).square()
weighted_loss=0.0
wrong_gradient=0.0
offset=0
weights=[]
for count in plan:
 n=count*micro
 local=gradients[offset:offset+n]
 batching.INITIAL_MB_NUM=8
 batching.CURRENT_MB_NUM=count
 weight=batching.get_loss_and_grad_weight()
 weights.append(weight)
 weighted_loss=weighted_loss+(parameter*local).mean()*weight/4
 wrong_gradient+=local.mean().item()/4
 offset+=n
weighted_loss.backward()
expected=gradients.mean().item()
actual=parameter.grad.item()
assert math.isclose(actual,expected,rel_tol=1e-12)
assert not math.isclose(wrong_gradient,expected,rel_tol=1e-6)
batching.mpu=original_mpu
batching.REDIS_CLIENT=original_redis
print(json.dumps({'status':'PASS','plan':plan,'global_batch':128,'weights':weights,'weighted_gradient':actual,'reference_global_mean_gradient':expected,'unweighted_local_mean_gradient':wrong_gradient,'invalid_124_sample_plan_rejected':invalid_rejected,'global_iteration_batch_size':128},indent=2))
