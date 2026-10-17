#!/usr/bin/env python3
"""Compare two Megatron standard checkpoints and emit a JSON report.

The inputs may be checkpoint roots (with a tracker file), iteration
directories, model-parallel shard directories, or model_optim_rng.pt files.
For roots, the tracker selects the checkpoint. All model-parallel shards in
that iteration are compared; DP replicas are represented by the standard
checkpoint's saved shard set.
"""

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


TOPOLOGY_FIELDS = (
    'tensor_model_parallel_size',
    'pipeline_model_parallel_size',
    'virtual_pipeline_model_parallel_size',
    'context_parallel_size',
    'expert_model_parallel_size',
    'data_parallel_size',
    'world_size',
    'num_layers',
    'hidden_size',
    'num_attention_heads',
    'seq_length',
)
_MAX_MISMATCHES = 100


def _stable_key(value):
    return '{}:{}'.format(type(value).__name__, repr(value))


def _tensor_meta(tensor):
    meta = {
        'dtype': str(tensor.dtype),
        'shape': list(tensor.shape),
        'layout': str(tensor.layout),
        'quantized': bool(tensor.is_quantized),
    }
    if tensor.is_quantized:
        qscheme = tensor.qscheme()
        meta['qscheme'] = str(qscheme)
        if qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            meta['scale'] = tensor.q_scale()
            meta['zero_point'] = tensor.q_zero_point()
        else:
            meta['axis'] = tensor.q_per_channel_axis()
            meta['scales'] = tensor.q_per_channel_scales().tolist()
            meta['zero_points'] = tensor.q_per_channel_zero_points().tolist()
    return meta


def _tensor_content(tensor):
    """Return deterministic bytes for a CPU tensor, including sparse layouts."""
    tensor = tensor.detach().cpu()
    if tensor.layout == torch.strided:
        if tensor.is_quantized:
            tensor = tensor.int_repr()
        # Flatten first because dtype reinterpretation of a 0-D tensor is not
        # supported by some PyTorch versions (including the target 2.1 image).
        return tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    if tensor.layout == torch.sparse_coo:
        tensor = tensor.coalesce()
        components = (_tensor_content(tensor.indices()),
                      _tensor_content(tensor.values()))
        return b''.join(len(part).to_bytes(8, 'big') + part for part in components)
    if tensor.layout == torch.sparse_csr:
        components = (_tensor_content(tensor.crow_indices()),
                      _tensor_content(tensor.col_indices()),
                      _tensor_content(tensor.values()))
        return b''.join(len(part).to_bytes(8, 'big') + part for part in components)
    if tensor.layout == torch.sparse_csc:
        components = (_tensor_content(tensor.ccol_indices()),
                      _tensor_content(tensor.row_indices()),
                      _tensor_content(tensor.values()))
        return b''.join(len(part).to_bytes(8, 'big') + part for part in components)
    if tensor.layout == torch.sparse_bsr:
        components = (_tensor_content(tensor.crow_indices()),
                      _tensor_content(tensor.col_indices()),
                      _tensor_content(tensor.values()))
        return b''.join(len(part).to_bytes(8, 'big') + part for part in components)
    if tensor.layout == torch.sparse_bsc:
        components = (_tensor_content(tensor.ccol_indices()),
                      _tensor_content(tensor.row_indices()),
                      _tensor_content(tensor.values()))
        return b''.join(len(part).to_bytes(8, 'big') + part for part in components)
    raise TypeError('unsupported tensor layout {}'.format(tensor.layout))


def _tensor_sha256(tensor):
    digest = hashlib.sha256()
    digest.update(json.dumps(_tensor_meta(tensor), sort_keys=True).encode())
    digest.update(_tensor_content(tensor))
    return digest.hexdigest()


def _canonical_digest(value):
    digest = hashlib.sha256()

    def add(obj):
        if isinstance(obj, torch.Tensor):
            digest.update(b'TENSOR\0')
            digest.update(json.dumps(_tensor_meta(obj), sort_keys=True).encode())
            digest.update(b'\0')
            digest.update(_tensor_content(obj))
        elif isinstance(obj, np.ndarray):
            digest.update(b'NDARRAY\0')
            digest.update(obj.dtype.str.encode())
            digest.update(json.dumps(list(obj.shape)).encode())
            digest.update(np.ascontiguousarray(obj).tobytes())
        elif isinstance(obj, np.generic):
            digest.update(b'NUMPY_SCALAR\0')
            digest.update(obj.dtype.str.encode())
            digest.update(obj.tobytes())
        elif isinstance(obj, Mapping):
            digest.update(b'MAPPING\0')
            for key in sorted(obj, key=_stable_key):
                add(key)
                add(obj[key])
            digest.update(b'END_MAPPING\0')
        elif isinstance(obj, (list, tuple)):
            digest.update(('TUPLE\0' if isinstance(obj, tuple) else 'LIST\0').encode())
            digest.update(str(len(obj)).encode())
            for item in obj:
                add(item)
            digest.update(b'END_SEQUENCE\0')
        elif isinstance(obj, (set, frozenset)):
            digest.update(b'SET\0')
            for item in sorted(obj, key=_stable_key):
                add(item)
            digest.update(b'END_SET\0')
        elif isinstance(obj, bytes):
            digest.update(b'BYTES\0')
            digest.update(obj)
        elif obj is None:
            digest.update(b'NONE\0')
        elif isinstance(obj, bool):
            digest.update(b'BOOL\0' + (b'1' if obj else b'0'))
        elif isinstance(obj, int):
            digest.update(b'INT\0' + str(obj).encode())
        elif isinstance(obj, float):
            digest.update(b'FLOAT\0' + struct.pack('!d', obj))
        elif isinstance(obj, str):
            digest.update(b'STR\0' + obj.encode('utf-8'))
        elif hasattr(obj, '__dict__'):
            digest.update(('OBJECT:' + type(obj).__qualname__ + '\0').encode())
            add(vars(obj))
        else:
            digest.update(('REPR:' + type(obj).__qualname__ + '\0').encode())
            digest.update(repr(obj).encode('utf-8'))

    add(value)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, torch.Tensor):
        return {
            **_tensor_meta(value),
            'sha256': _tensor_sha256(value),
        }
    return repr(value)


class RecursiveComparator:
    def __init__(self, max_mismatches=_MAX_MISMATCHES,
                 numeric_mode=None):
        self.max_mismatches = max_mismatches
        self.numeric_mode = numeric_mode
        self.mismatches = []
        self.mismatch_count = 0
        self.tensor_count = 0
        self.tensor_equal_count = 0
        self.tensor_bytes = 0
        self.ndarray_count = 0
        self.numeric_diagnostic_count = 0

    def _mismatch(self, path, reason, left=None, right=None, details=None):
        self.mismatch_count += 1
        if len(self.mismatches) < self.max_mismatches:
            item = {'path': path, 'reason': reason}
            if left is not None:
                item['left'] = _json_value(left)
            if right is not None:
                item['right'] = _json_value(right)
            if details:
                item.update(details)
            self.mismatches.append(item)

    def _numeric_diagnostic(self, left, right, path):
        if self.numeric_mode is None:
            return None
        if self.numeric_mode == 'optimizer_master' and 'fp32_from_fp16_params' not in path:
            return None
        if (left.layout != torch.strided or right.layout != torch.strided or
                left.is_quantized or right.is_quantized or
                not left.is_floating_point() or not right.is_floating_point()):
            return {'numeric_diagnostics': {
                'available': False,
                'reason': 'requires dense floating-point tensors',
            }}
        if left.shape != right.shape:
            return {'numeric_diagnostics': {
                'available': False,
                'reason': 'tensor shapes differ',
                'left_shape': list(left.shape),
                'right_shape': list(right.shape),
            }}

        left_fp32 = left.detach().cpu().to(torch.float32)
        right_fp32 = right.detach().cpu().to(torch.float32)
        difference = (left_fp32 - right_fp32).abs()
        numel = left.numel()
        max_abs = difference.max().item() if numel else 0.0
        mean_abs = difference.mean().item() if numel else 0.0
        num_different = torch.count_nonzero(
            left.detach().cpu() != right.detach().cpu()).item()
        left_norm = torch.linalg.vector_norm(left_fp32).item()
        difference_norm = torch.linalg.vector_norm(left_fp32 - right_fp32).item()
        if left_norm > 0:
            norm_relative = difference_norm / left_norm
        elif difference_norm == 0:
            norm_relative = 0.0
        else:
            norm_relative = float('inf')
        self.numeric_diagnostic_count += 1
        return {'numeric_diagnostics': {
            'available': True,
            'max_abs': _json_value(float(max_abs)),
            'mean_abs': _json_value(float(mean_abs)),
            'num_different': int(num_different),
            'numel': int(numel),
            'fp32normrelative': _json_value(float(norm_relative)),
            'relative_norm_definition': '||left-right||_2 / ||left||_2, computed in FP32',
        }}

    def compare(self, left, right, path='$'):
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                self._mismatch(path, 'tensor/non-tensor type mismatch', left, right)
                return False
            self.tensor_count += 1
            self.tensor_bytes += left.numel() * left.element_size()
            same_meta = _tensor_meta(left) == _tensor_meta(right)
            same_data = False
            if same_meta:
                try:
                    same_data = _tensor_content(left) == _tensor_content(right)
                except (RuntimeError, TypeError):
                    same_data = False
            if same_meta and same_data:
                self.tensor_equal_count += 1
                return True
            details = self._numeric_diagnostic(left, right, path)
            self._mismatch(path, 'tensor differs', left, right, details)
            return False

        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
                self._mismatch(path, 'array/non-array type mismatch', left, right)
                return False
            self.ndarray_count += 1
            same = (left.dtype == right.dtype and left.shape == right.shape and
                    np.ascontiguousarray(left).tobytes() ==
                    np.ascontiguousarray(right).tobytes())
            if not same:
                self._mismatch(path, 'numpy array differs', left, right)
            return same

        if isinstance(left, np.generic) or isinstance(right, np.generic):
            same = (isinstance(left, np.generic) and isinstance(right, np.generic) and
                    left.dtype == right.dtype and left.tobytes() == right.tobytes())
            if not same:
                self._mismatch(path, 'numpy scalar differs', left, right)
            return same

        if isinstance(left, Mapping) or isinstance(right, Mapping):
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                self._mismatch(path, 'mapping/non-mapping type mismatch', left, right)
                return False
            same = True
            left_keys, right_keys = set(left), set(right)
            for key in sorted(left_keys - right_keys, key=_stable_key):
                self._mismatch('{}.{}'.format(path, key), 'key missing on right', left[key], None)
                same = False
            for key in sorted(right_keys - left_keys, key=_stable_key):
                self._mismatch('{}.{}'.format(path, key), 'key missing on left', None, right[key])
                same = False
            for key in sorted(left_keys & right_keys, key=_stable_key):
                if not self.compare(left[key], right[key], '{}.{}'.format(path, key)):
                    same = False
            return same

        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if type(left) is not type(right):
                self._mismatch(path, 'sequence type mismatch', left, right)
                return False
            if len(left) != len(right):
                self._mismatch(path, 'sequence length mismatch', len(left), len(right))
                return False
            same = True
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                if not self.compare(left_item, right_item, '{}[{}]'.format(path, index)):
                    same = False
            return same

        if isinstance(left, (set, frozenset)) or isinstance(right, (set, frozenset)):
            same = type(left) is type(right) and left == right
            if not same:
                self._mismatch(path, 'set differs', left, right)
            return same

        if type(left) is not type(right):
            self._mismatch(path, 'value type mismatch', left, right)
            return False
        if isinstance(left, float):
            same = struct.pack('!d', left) == struct.pack('!d', right)
        else:
            same = left == right
        if not same:
            self._mismatch(path, 'value differs', left, right)
        return bool(same)


def _get_arg(args, name):
    if isinstance(args, Mapping):
        return args.get(name)
    return getattr(args, name, None)


def _iteration_from_tracker(tracker_path):
    value = tracker_path.read_text().strip()
    if value == 'release':
        return 'release'
    try:
        iteration = int(value)
    except ValueError as exc:
        raise ValueError('invalid checkpoint tracker {}: {!r}'.format(
            tracker_path, value)) from exc
    if iteration <= 0:
        raise ValueError('checkpoint tracker iteration must be positive: {}'.format(value))
    return iteration


def _select_iteration_dir(root):
    tracker = root / 'latest_checkpointed_iteration.txt'
    if tracker.is_file():
        iteration = _iteration_from_tracker(tracker)
        name = 'release' if iteration == 'release' else 'iter_{:07d}'.format(iteration)
        selected = root / name
        if not selected.is_dir():
            raise FileNotFoundError('tracker points to missing directory {}'.format(selected))
        return selected

    if root.name == 'release' or re.fullmatch(r'iter_\d+', root.name):
        return root

    candidates = []
    for child in root.iterdir():
        match = re.fullmatch(r'iter_(\d+)', child.name)
        if child.is_dir() and match:
            candidates.append((int(match.group(1)), child))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return root


def _resolve_checkpoint(input_path):
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        if path.name not in ('model_optim_rng.pt', 'distrib_optim.pt'):
            raise ValueError('expected model_optim_rng.pt or distrib_optim.pt: {}'.format(path))
        relative_name = path.name
        if path.parent.name.startswith('mp_rank_'):
            relative_name = '{}/{}'.format(path.parent.name, path.name)
        files = {relative_name: path}
        sibling = path.parent / ('distrib_optim.pt' if path.name == 'model_optim_rng.pt'
                                 else 'model_optim_rng.pt')
        if sibling.is_file():
            sibling_relative = sibling.name
            if path.parent.name.startswith('mp_rank_'):
                sibling_relative = '{}/{}'.format(path.parent.name, sibling.name)
            files[sibling_relative] = sibling
        return {'input': str(path), 'selected_dir': str(path.parent), 'files': files}

    if not path.is_dir():
        raise ValueError('checkpoint input must be a file or directory: {}'.format(path))

    if path.name.startswith('mp_rank_'):
        selected = path
    else:
        selected = _select_iteration_dir(path)

    files = {}
    model_files = sorted(selected.rglob('model_optim_rng.pt'))
    for file_path in model_files:
        if selected.name.startswith('mp_rank_'):
            relative = '{}/{}'.format(selected.name, file_path.name)
        else:
            relative = file_path.relative_to(selected).as_posix()
        files[relative] = file_path
        distributed_optimizer = file_path.parent / 'distrib_optim.pt'
        if distributed_optimizer.is_file():
            if selected.name.startswith('mp_rank_'):
                optim_relative = '{}/{}'.format(selected.name, distributed_optimizer.name)
            else:
                optim_relative = distributed_optimizer.relative_to(selected).as_posix()
            files[optim_relative] = distributed_optimizer
    if not files:
        raise FileNotFoundError('no model_optim_rng.pt found under {}'.format(selected))

    return {
        'input': str(path),
        'selected_dir': str(selected),
        'files': files,
    }


def _load_checkpoint_file(path):
    # Local artifact checkpoints are trusted and contain argparse.Namespace values.
    return torch.load(str(path), map_location='cpu')


def _extract_groups(checkpoint, file_name):
    if file_name.endswith('distrib_optim.pt'):
        return {
            'model': {'present': False},
            'optimizer': {'present': True, 'distributed_optimizer': checkpoint},
            'scheduler': {'present': False},
            'rng': {'present': False},
            'args_progress': {'present': False},
        }

    model_keys = [key for key in checkpoint
                  if key == 'model' or re.fullmatch(r'model\d+', str(key))]
    model = {key: checkpoint[key] for key in model_keys}
    optimizer_present = 'optimizer' in checkpoint
    optimizer = {'optimizer': checkpoint.get('optimizer')} if optimizer_present else {}
    scheduler_keys = [key for key in ('opt_param_scheduler', 'lr_scheduler')
                      if key in checkpoint]
    scheduler = {key: checkpoint[key] for key in scheduler_keys}
    if 'rng_state' in checkpoint:
        rng = {'rng_state': checkpoint['rng_state']}
    else:
        legacy_rng_keys = (
            'random_rng_state', 'np_rng_state', 'torch_rng_state',
            'cuda_rng_state', 'rng_tracker_states',
        )
        rng = {key: checkpoint[key] for key in legacy_rng_keys if key in checkpoint}

    args = checkpoint.get('args')
    progress = {
        'checkpoint_iteration': checkpoint.get('iteration', checkpoint.get('total_iters')),
        'consumed_train_samples': _get_arg(args, 'consumed_train_samples'),
        'consumed_valid_samples': _get_arg(args, 'consumed_valid_samples'),
        'num_floating_point_operations_so_far': checkpoint.get(
            'num_floating_point_operations_so_far'),
    }
    return {
        'model': {'present': bool(model), 'value': model},
        'optimizer': {'present': optimizer_present, 'value': optimizer},
        'scheduler': {'present': bool(scheduler), 'value': scheduler},
        'rng': {'present': bool(rng), 'value': rng},
        'args_progress': {'present': args is not None, 'value': progress},
    }


def _extract_topology(checkpoint):
    args = checkpoint.get('args')
    values = {name: _get_arg(args, name) for name in TOPOLOGY_FIELDS}
    return {name: value for name, value in values.items() if value is not None}


def _group_report(left, right, path, max_mismatches, numeric_mode=None):
    comparator = RecursiveComparator(max_mismatches, numeric_mode)
    equal = comparator.compare(left, right, path)
    return {
        'equal': equal,
        'sha256_left': _canonical_digest(left),
        'sha256_right': _canonical_digest(right),
        'tensor_count': comparator.tensor_count,
        'tensor_equal_count': comparator.tensor_equal_count,
        'tensor_bytes_left': comparator.tensor_bytes,
        'numpy_array_count': comparator.ndarray_count,
        'numeric_diagnostic_tensor_count': comparator.numeric_diagnostic_count,
        'mismatch_count': comparator.mismatch_count,
        'mismatches': comparator.mismatches,
        'omitted_mismatches': max(0, comparator.mismatch_count - len(comparator.mismatches)),
    }


def compare(left_input, right_input, max_mismatches=_MAX_MISMATCHES,
            numeric_diagnostics=False):
    left_selection = _resolve_checkpoint(left_input)
    right_selection = _resolve_checkpoint(right_input)
    left_files, right_files = left_selection['files'], right_selection['files']
    left_names, right_names = set(left_files), set(right_files)

    report = {
        'left': {
            'input': left_selection['input'],
            'selected_dir': left_selection['selected_dir'],
            'files': sorted(left_names),
        },
        'right': {
            'input': right_selection['input'],
            'selected_dir': right_selection['selected_dir'],
            'files': sorted(right_names),
        },
        'file_sets_equal': left_names == right_names,
        'files': {},
        'overall_equal': False,
        'numeric_diagnostics_enabled': numeric_diagnostics,
    }

    for file_name in sorted(left_names | right_names):
        if file_name not in left_files or file_name not in right_files:
            report['files'][file_name] = {
                'equal': False,
                'reason': 'checkpoint shard missing on one side',
                'left_present': file_name in left_files,
                'right_present': file_name in right_files,
            }
            continue

        left_ckpt = _load_checkpoint_file(left_files[file_name])
        right_ckpt = _load_checkpoint_file(right_files[file_name])
        if not isinstance(left_ckpt, Mapping) or not isinstance(right_ckpt, Mapping):
            raise TypeError('{} must contain a mapping state dict'.format(file_name))

        left_groups = _extract_groups(left_ckpt, file_name)
        right_groups = _extract_groups(right_ckpt, file_name)
        file_report = {
            'left_path': str(left_files[file_name]),
            'right_path': str(right_files[file_name]),
            'groups': {},
        }
        for group_name in ('model', 'optimizer', 'scheduler', 'rng', 'args_progress'):
            numeric_mode = None
            if numeric_diagnostics and group_name == 'model':
                numeric_mode = 'model'
            elif numeric_diagnostics and group_name == 'optimizer':
                numeric_mode = 'optimizer_master'
            file_report['groups'][group_name] = _group_report(
                left_groups[group_name], right_groups[group_name],
                '{}.{}'.format(file_name, group_name), max_mismatches,
                numeric_mode)

        left_topology = _extract_topology(left_ckpt)
        right_topology = _extract_topology(right_ckpt)
        file_report['topology'] = {
            'left': left_topology,
            'right': right_topology,
            'equal': left_topology == right_topology,
        }
        if file_name.endswith('distrib_optim.pt'):
            file_report['required_state_present'] = {
                'distributed_optimizer': bool(left_groups['optimizer']['present'] and
                                              right_groups['optimizer']['present']),
            }
        else:
            left_progress = left_groups['args_progress'].get('value', {})
            right_progress = right_groups['args_progress'].get('value', {})
            required_state_present = {
                'model': bool(left_groups['model']['present'] and
                              right_groups['model']['present']),
                'optimizer': bool(left_groups['optimizer']['present'] and
                                  right_groups['optimizer']['present']),
                'scheduler': bool(left_groups['scheduler']['present'] and
                                  right_groups['scheduler']['present']),
                'rng': bool(left_groups['rng']['present'] and
                            right_groups['rng']['present']),
                'args': bool(left_groups['args_progress']['present'] and
                             right_groups['args_progress']['present']),
            }
            for progress_key in (
                    'checkpoint_iteration', 'consumed_train_samples',
                    'consumed_valid_samples',
                    'num_floating_point_operations_so_far'):
                required_state_present[progress_key] = (
                    left_progress.get(progress_key) is not None and
                    right_progress.get(progress_key) is not None)
            file_report['required_state_present'] = required_state_present
            required_topology = (
                field for field in TOPOLOGY_FIELDS
                if field != 'virtual_pipeline_model_parallel_size'
            )
            file_report['topology_complete'] = all(
                field in left_topology and field in right_topology
                for field in required_topology)
        args_left = left_ckpt.get('args')
        args_right = right_ckpt.get('args')
        # Megatron stores args.iteration at job initialization. After resume it
        # may reflect the restart point, so report it without treating it as the
        # checkpoint's authoritative progress value.
        file_report['args_iteration_at_job_start'] = {
            'left': _get_arg(args_left, 'iteration'),
            'right': _get_arg(args_right, 'iteration'),
            'equal': _get_arg(args_left, 'iteration') == _get_arg(args_right, 'iteration'),
            'compared_for_overall_equal': False,
        }
        file_report['equal'] = (
            all(group['equal'] for group in file_report['groups'].values()) and
            file_report['topology']['equal'] and
            all(file_report['required_state_present'].values()) and
            file_report.get('topology_complete', True))
        report['files'][file_name] = file_report

    all_files_equal = report['file_sets_equal'] and all(
        item.get('equal', False) for item in report['files'].values())
    report['overall_equal'] = all_files_equal
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', help='checkpoint root, iteration/shard directory, or .pt file')
    parser.add_argument('right', help='checkpoint root, iteration/shard directory, or .pt file')
    parser.add_argument('--output', help='write JSON report to this path instead of stdout')
    parser.add_argument('--max-mismatches', type=int, default=_MAX_MISMATCHES,
                        help='maximum detailed differences per group (default: 100)')
    parser.add_argument('--numeric-diagnostics', action='store_true',
                        help=('report per-tensor numeric deltas for differing model '
                              'and FP32 optimizer master tensors'))
    args = parser.parse_args()

    if args.max_mismatches < 0:
        parser.error('--max-mismatches must be non-negative')

    try:
        report = compare(args.left, args.right, args.max_mismatches,
                         args.numeric_diagnostics)
    except Exception as exc:
        error = {'error': type(exc).__name__, 'message': str(exc)}
        rendered = json.dumps(error, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(rendered + '\n')
        else:
            print(rendered)
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + '\n')
    else:
        print(rendered)
    return 0 if report['overall_equal'] else 1


if __name__ == '__main__':
    sys.exit(main())
