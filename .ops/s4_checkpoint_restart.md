# S4 checkpoint-restart audit

The artifact README names S4 “Checkpoint-restart strategy.” The checked-in
restart path is separate from the active experiment launcher:

1. `detector/control_plane/mitigation_plan.py` publishes a new PP layer split
   and sets Redis `terminate_ctl=123`.
2. `Megatron-LM/run_training.py` maps `123` to Redis `terminate=1`, waits for
   Megatron to exit, clears old PP metadata, and launches the same command
   again.
3. `megatron/training.py` handles `terminate=1` by calling
   `save_checkpoint(..., to_mem=True)` before exiting.

The former `to_mem` branch only copied some tensors to CPU and returned. It did
not retain a process-independent checkpoint. The save routine also had an
unconditional early return, so neither normal periodic/final saves nor the
restart request wrote files. The `checkpointing.py` fix now persists both
call paths to `args.save` and updates the iteration tracker atomically. This is
needed because the wrapper starts a new process after the save.

The workspace's active `Greyhound/run_training.py` and
`Greyhound/run_training_dp.py` launchers were previously patched to stream the
child process output and check its exit code. They do not poll `terminate_ctl`
or relaunch Megatron, so the controller-triggered restart chain is not wired
through those launchers. No launcher behavior was changed as part of this
checkpointing fix.

## Validation procedure

Run a same-topology resume smoke test with a checkpoint directory shared by
all participating ranks. Use the same model, TP/PP/DP sizes, data settings, and
optimizer settings in both launches. In a short first run, set
`--save-interval 5 --exit-interval 10 --save <shared-checkpoint-dir>`. Then
launch the same command again with `--load <shared-checkpoint-dir>` and
`--exit-on-missing-checkpoint`; check that it reports loading iteration 10,
completes to the requested final iteration, and advances
`latest_checkpointed_iteration.txt`. Also confirm each expected model shard
and optimizer state file exists before the second launch.

For a real controller-driven restart test, the launcher must monitor
`terminate_ctl`, wait until every rank has finished the checkpoint, and then
relaunch with the same checkpoint directory. The current root launchers do not
implement that control path, so the smoke test above validates checkpoint
save/load only. The ordinary Megatron checkpoint loader also requires matching
TP/PP topology and checkpoint arguments; it does not demonstrate a pipeline
topology change or S3 reconfiguration.

To compare the final resumed run against an uninterrupted run, pass both
checkpoint roots to `.ops/compare_checkpoints.py`; it selects the tracker
iteration, compares every MP shard, checks model/optimizer/scheduler/RNG values
recursively, hashes those groups with SHA256, and compares consumed-sample and
iteration progress. For example:

```bash
python .ops/compare_checkpoints.py checkpoints/uninterrupted checkpoints/resumed \
  --output results/s4-checkpoint-comparison.json
```

The comparison is exact at tensor-byte level and exits nonzero if any required
checkpoint group is absent or differs. It reports `args.iteration` for context
but excludes it from the equality verdict because Megatron keeps that field at
the start iteration of each process; the checkpoint's top-level `iteration`
field is authoritative.

For an exact-comparison failure that needs magnitude context, add
`--numeric-diagnostics`. The JSON then includes per-mismatched-tensor
`max_abs`, `mean_abs`, `num_different`, `numel`, and
`fp32normrelative` (`||left-right||_2 / ||left||_2` computed in FP32) for model
and FP32 optimizer master tensors. These measurements do not change the exact
equality result.
