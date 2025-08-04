#!/usr/bin/env python
"""
Local multi-GPU smoke test for miner.py with Muon optimizer support.

Runs several inner-loop optimization windows on real data (bins under
DATASET_BINS_PATH) with either AdamW or Muon optimizer. Collects detailed
performance metrics and saves them for analysis.
"""
from __future__ import annotations

import builtins, os, sys, types, argparse, asyncio, json, random, math, time
from pathlib import Path
from tempfile import NamedTemporaryFile
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Import muon module from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#from muon import SingleDeviceMuonWithAuxAdam

# Root-rank printing helper
_ORIG_PRINT = builtins.print

def _is_root_rank() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0

def _print(*a, **kw):
    if _is_root_rank():
        _ORIG_PRINT(*a, **kw)

builtins.print = _print

# Command-line arguments
argp = argparse.ArgumentParser()
argp.add_argument("--device",        default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
argp.add_argument("--amp-dtype",     default="bf16", choices=["bf16", "fp16"])
argp.add_argument("--sequence-length", type=int,     default=2048)
argp.add_argument("--micro-batch-size", type=int,    default=2)
argp.add_argument("--inner-windows", type=int,       default=164, help="Number of Miner.inner_steps() windows to run")
argp.add_argument("--num-workers",   type=int,       default=2, help="DataLoader workers")
argp.add_argument("--wandb-project", default="muon_local_miner_test")
# Optimizer selection
argp.add_argument("--inner-optimizer", default="adamw", choices=["adamw", "muon"], help="Inner optimizer to use")
argp.add_argument("--inner-learning-rate", type=float, default=2e-4, help="Learning rate for inner optimizer")
# Muon-specific hyperparameters
argp.add_argument("--muon-momentum", type=float, default=0.95, help="Momentum for Muon optimizer")
argp.add_argument("--muon-weight-decay", type=float, default=0.01, help="Weight decay for Muon optimizer")
argp.add_argument("--muon-head-lr-scale", type=float, default=0.5, help="LR scale for head params")
argp.add_argument("--muon-embed-lr-scale", type=float, default=0.5, help="LR scale for embedding params")
argp.add_argument("--muon-scalar-lr-scale", type=float, default=0.2, help="LR scale for scalar params")
# Output configuration
argp.add_argument("--output-dir", default=None, help="Directory to save results (auto-generated if not specified)")
# Profiling
argp.add_argument("--enable-profiler", action="store_true", help="Enable torch profiler")
argp.add_argument("--profiler-wait", type=int, default=3, help="Profiler wait steps (default: 3)")
argp.add_argument("--profiler-warmup", type=int, default=1, help="Profiler warmup steps (default: 1)")
argp.add_argument("--profiler-active", type=int, default=2, help="Profiler active steps (default: 2)")
args = argp.parse_args()

# Environment bootstrapping & dummy creds
print("Setting dummy environment variables for local testing")
_DUMMY_VARS = {
    "R2_GRADIENTS_ACCOUNT_ID":          "dummy_id",
    "R2_GRADIENTS_BUCKET_NAME":         "dummy_bucket",
    "R2_GRADIENTS_READ_ACCESS_KEY_ID":  "dummy_key",
    "R2_GRADIENTS_READ_SECRET_ACCESS_KEY": "dummy_secret",
    "R2_GRADIENTS_WRITE_ACCESS_KEY_ID": "dummy_key",
    "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY": "dummy_secret",
    "R2_AGGREGATOR_ACCOUNT_ID":         "dummy_id",
    "R2_AGGREGATOR_BUCKET_NAME":        "dummy_bucket",
    "R2_AGGREGATOR_READ_ACCESS_KEY_ID":    "dummy_key",
    "R2_AGGREGATOR_READ_SECRET_ACCESS_KEY":"dummy_secret",
    "R2_DATASET_ACCOUNT_ID":            "dummy_id",
    "R2_DATASET_BUCKET_NAME":           "dummy_bucket",
    "R2_DATASET_READ_ACCESS_KEY_ID":    "dummy_key",
    "R2_DATASET_READ_SECRET_ACCESS_KEY":"dummy_secret",
    "DATASET_BINS_PATH":                "/workspace/templar/scripts/dataset_prep/test_shard",
}
for k, v in _DUMMY_VARS.items():
    os.environ.setdefault(k, v)

# Ensure single-node defaults if the launcher forgot them
os.environ.setdefault("RANK",        "0")
os.environ.setdefault("WORLD_SIZE",  "1")
os.environ.setdefault("LOCAL_RANK",  os.environ["RANK"])

# Fake/stub out external services before importing Miner
try:
    import bittensor as bt
except ModuleNotFoundError:
    bt = types.ModuleType("bittensor")
    sys.modules["bittensor"] = bt

class _FakeHotkey:
    ss58_address = "fake_hotkey"

class _FakeWallet:
    def __init__(self, *_, **__):
        self.hotkey = _FakeHotkey()
    @staticmethod
    def add_args(_): pass

bt.wallet = _FakeWallet

class _FakeMetagraph:
    def __init__(self, hotkey):
        self.hotkeys = [hotkey]
        self.S       = [1.0]
        self.netuid  = 268

class _FakeSubtensor:
    block = 123_456
    def __init__(self, *_, **__): pass
    @staticmethod
    def add_args(_): pass
    def metagraph(self, _): return _FakeMetagraph(_FakeHotkey().ss58_address)

bt.subtensor = _FakeSubtensor

import importlib
tplr = importlib.import_module("tplr")

class _FakeComms:
    def __init__(self, *_, **__): self.peers = []
    def get_own_bucket(self, *_, **__): return None
    def try_commit(self, *_, **__):     pass
    def start_commitment_fetcher(self): pass

tplr.comms.Comms      = _FakeComms
tplr.initialize_wandb = lambda *_a, **_k: types.SimpleNamespace(log=lambda *_, **__: None)
tplr.metrics          = types.SimpleNamespace(MetricsLogger=lambda *_a, **_k: None)

# TorchTitan Llama-3 8B hparams
_TORCHTITAN_HPARAMS = {
    "spec_version": 5,
    "project": "dough",
    "sequence_length": 2048,
    "micro_batch_size": 1,
    "target_batch_size": 1,
    "batch_size": 128,
    "inner_steps": 30,
    "inner_learning_rate": 2e-4,
    "outer_learning_rate": 0.9,
    "blocks_per_window": 4096,
    "windows_per_weights": 7,
    "momentum_decay": 0.999,
    "topk_compression": 32,
    "target_chunk": 64,
    "use_dct": False,
    "binary_score_ma_alpha": 0.05,
    "moving_average_window": 5,
    "model_size": "8B",
    "weight_decay": 0.1,
    "warmup_steps": 750,
    "alpha_f": 0.1,
    "t_max": 20000,
    "validator_offset": 1,
    "checkpoint_frequency": 5,
    "max_topk_peers": 15,
    "minimum_peers": 5,
    "peer_replacement_frequency": 5,
    "peer_list_window_margin": 1,
    "active_check_interval": 300,
    "recent_windows": 5,
    "power_normalisation": 2.0,
    "validator_sample_micro_bs": 4,
    "gather_peers_slash_threshold": 0.4,
    "uids_per_window": 20,
    "time_window_delta_seconds": 30,
    "reset_inactivity_windows": 10,
    "sync_max_steps_behind": 3,
    "eval_lr_factor": 0.5,
    "openskill_beta": 7,
    "openskill_tau": 0.1,
    "num_evaluation_bins": 5,
    "quantization_bins": 256,
    "quantization_range": 6,
    "burn_rate": 0.8,
    "idx_overlap_threshold": 0.5,
    "torchtitan": {
        "tp_degree": 1, "dp_replicate": 8, "pp_degree": 1, "cp_degree": 1,
        "compile": False,
        "enable_cpu_offload": False,
        "mixed_precision_param": "float32",
        "mixed_precision_reduce": "float32",
        "enable_async_tensor_parallel": False,
        "disable_loss_parallel": True,
        "fsdp_reshard_after_forward": "default",
        "enable_compiled_autograd": False,
        "float8_recipe_name": None,
        "activation_checkpoint": {"mode": "selective", "option": "op"},
    },
}
_hparams_file = NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump(_TORCHTITAN_HPARAMS, _hparams_file)
_hparams_file.flush()

# Override inner learning rate if specified
if args.inner_learning_rate != _TORCHTITAN_HPARAMS["inner_learning_rate"]:
    _TORCHTITAN_HPARAMS["inner_learning_rate"] = args.inner_learning_rate
    _hparams_file.seek(0)
    json.dump(_TORCHTITAN_HPARAMS, _hparams_file)
    _hparams_file.flush()
# Override weight decay if specified
if args.inner_optimizer == 'muon' and args.muon_weight_decay != _TORCHTITAN_HPARAMS["weight_decay"]:
    _TORCHTITAN_HPARAMS["weight_decay"] = args.muon_weight_decay
    _hparams_file.seek(0)
    json.dump(_TORCHTITAN_HPARAMS, _hparams_file)
    _hparams_file.flush()

# Prepare CLI argv for Miner and import it
sys.argv = [
    "local_miner_smoke",
    "--local",
    "--device", args.device,
    "--amp-dtype", args.amp_dtype,
    "--hparams-file", _hparams_file.name,
    "--inner-optimizer", args.inner_optimizer,
    "--muon-momentum", str(args.muon_momentum),
    "--muon-weight-decay", str(args.muon_weight_decay),
    "--muon-head-lr-scale", str(args.muon_head_lr_scale),
    "--muon-embed-lr-scale", str(args.muon_embed_lr_scale),
    "--muon-scalar-lr-scale", str(args.muon_scalar_lr_scale),
    "--project", str(args.wandb_project)
]

print(f"Initialising Miner (Llama-3 8B / TorchTitan) with {args.inner_optimizer} optimizer…")
from neurons.miner import Miner
miner = Miner()
miner.model.eval()
print("Miner initialised.")

# Dataset loader for .bin token shards
class BinShardDataset(Dataset):
    """
    Expects a directory of uint16-encoded `.bin` files containing
    contiguous token IDs. Each item is a (seq_len,) int64 tensor.
    """
    def __init__(self, root: str | Path, seq_len: int):
        self.seq_len = seq_len
        self.files   = sorted(Path(root).glob("*.bin"))
        if not self.files:
            raise FileNotFoundError(f"No .bin files found under {root}")
        self.maps  = [np.memmap(f, dtype=np.uint16, mode="r") for f in self.files]
        self.counts = [len(m) // seq_len for m in self.maps]
        self.cum_counts = np.cumsum([0] + self.counts)

    def __len__(self): return self.cum_counts[-1]

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx = np.searchsorted(self.cum_counts, idx, side="right") - 1
        local_idx = idx - self.cum_counts[file_idx]
        start = local_idx * self.seq_len
        arr = self.maps[file_idx][start : start + self.seq_len]
        return torch.tensor(arr, dtype=torch.long)

_dataset_root = Path(os.environ["DATASET_BINS_PATH"])
dataset = BinShardDataset(_dataset_root, args.sequence_length)
loader  = DataLoader(
    dataset,
    batch_size=args.micro_batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=(args.device.startswith("cuda")),
    drop_last=True,
)

print(f"Dataset loaded from '{_dataset_root}' – {len(dataset):,} sequences")

# Setup output directory and metrics collection
if args.output_dir is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"results/{timestamp}_{args.inner_optimizer}"
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Results will be saved to: {output_dir}")

# Initialize metrics storage
metrics = {
    "optimizer": args.inner_optimizer,
    "config": {
        "device": args.device,
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "inner_windows": args.inner_windows,
        "inner_learning_rate": args.inner_learning_rate,
        "muon_momentum": args.muon_momentum if args.inner_optimizer == "muon" else None,
        "muon_weight_decay": args.muon_weight_decay if args.inner_optimizer == "muon" else None,
        "profiling_enabled": args.enable_profiler,
    },
    "windows": []
}

# Run Miner.inner_steps() for several windows
async def _run():
    global_step = 0
    
    # Setup profiler if requested
    profiler = None
    if args.enable_profiler:
        profiler_dir = output_dir / "profiler_traces"
        profiler_dir.mkdir(exist_ok=True)
        
        def trace_handler(prof):
            trace_file = profiler_dir / f"trace_step_{prof.step_num}.json"
            prof.export_chrome_trace(str(trace_file))
            print(f"Saved profiler trace to {trace_file}")
        
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=args.profiler_wait,
                warmup=args.profiler_warmup,
                active=args.profiler_active,
                repeat=1
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.__enter__()
    
    try:
        for window in range(args.inner_windows):
            window_start_time = time.time()
            
            # Get initial memory stats
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated() / (1024**2)  # MB
            else:
                memory_before = 0
            
            # Run the inner loop (gradients for outer step are produced)
            stats = await miner.inner_steps(loader=loader, step_window=window)
            
            # Apply the outer optimizer (updates model weights)
            miner.outer_optimizer.step()
            miner.outer_optimizer.zero_grad(set_to_none=True)
            
            # Calculate metrics
            window_time = time.time() - window_start_time
            tokens_per_second = stats["batch_tokens"] / window_time
            
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated() / (1024**2)  # MB
                gpu_memory_mb = memory_after
            else:
                gpu_memory_mb = 0
            
            # Store window metrics
            window_metrics = {
                "window": window,
                "global_step": global_step,
                "seconds_per_outer_step": window_time,
                "tokens_per_second": tokens_per_second,
                "memory_mb": gpu_memory_mb,
                "loss": stats.get("window_entry_loss", 0.0),
                "batch_count": stats["batch_count"],
                "batch_tokens": stats["batch_tokens"],
                "timestamp": datetime.now().isoformat(),
            }
            metrics["windows"].append(window_metrics)
            
            # Report metrics
            print(f"\n[Window {window}] metrics:")
            print(f"  Loss: {window_metrics['loss']:.4f}")
            print(f"  Tokens per second: {tokens_per_second:.2f}")
            print(f"  Seconds per outer step: {window_time:.2f}")
            print(f"  Memory usage: {gpu_memory_mb:.2f} MB")
            print(f"  Batch count: {stats['batch_count']}")
            print(f"  Batch tokens: {stats['batch_tokens']}")
            
            # Save metrics after each window
            metrics_file = output_dir / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Step profiler if active
            if profiler:
                profiler.step()
            
            global_step += 1
    
    finally:
        if profiler:
            profiler.__exit__(None, None, None)

print(f"Running {args.inner_windows} inner-windows with {args.inner_optimizer} optimizer…")
asyncio.run(_run())

print("\nFinished.")
print(f"Results saved to: {output_dir}")
print(f"  - Metrics: {output_dir}/metrics.json")
if args.enable_profiler:
    print(f"  - Profiler traces: {output_dir}/profiler_traces/")