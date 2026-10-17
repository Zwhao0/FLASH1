# FLASH1 reproduced baseline

This repository snapshots the FLASH1 source tree as of 2026-10-17. The `Greyhound/` source is based on upstream commit `d641b3997c17de700e1ef7d37e353418df7e336d` and includes the ten local source changes present at snapshot time. The source is stored as ordinary files; the upstream checkout metadata remains local and is not included.

`Greyhound/` contains the trainer, detector, and their vendored Megatron-LM tree. See [Greyhound/README.md](Greyhound/README.md) for the upstream setup and dependency notes. The root `.ops/` directory contains offline setup, experiment, checkpoint, and analysis helpers; the main training launchers are `Greyhound/run_training.py` and `Greyhound/run_training_dp.py`.

The archive excludes local server notes, credential-reading SSH helpers, private result reports, PDFs, logs, checkpoints, build outputs, and generated experiment artifacts. See [ARCHIVE.md](ARCHIVE.md) for snapshot details.
