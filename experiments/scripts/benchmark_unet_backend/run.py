"""Warm-started Stage-10 U-Net execution benchmark.

This diagnostic is train/validation only. It does not write formal selection
artifacts and does not access the test split.

The benchmark explicitly warms cuDNN/optimizer kernels before timing, then restores
the same initial network weights and optimizer state so first-use autotuning is not
mistaken for steady-state training cost.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common.baselines import unet_small as unet
from experiments.common.utils import experiment_utils as utils
from experiments.configs.formal_experiment import selection_dependency_config


def _make_optimizer(model, learning_rate: float, fused: bool):
    kwargs = {"lr": float(learning_rate)}
    if fused:
        kwargs["fused"] = True
    return torch.optim.Adam(model.parameters(), **kwargs)


def _prepare_batch(pairs, batch_size, patch_size, generator, device, channels_last):
    observed, target = unet._sample_batch(
        pairs, batch_size, patch_size, generator
    )
    observed = observed.to(device)
    target = target.to(device)
    if channels_last:
        observed = observed.contiguous(memory_format=torch.channels_last)
    return observed, target


def _train_steps(
    model,
    optimizer,
    scaler,
    pairs,
    *,
    steps,
    generator,
    device,
    batch_size,
    patch_size,
    amp_enabled,
    channels_last,
):
    model.train()
    last_loss = None
    for _ in range(steps):
        observed, target = _prepare_batch(
            pairs, batch_size, patch_size, generator, device, channels_last
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            prediction = model(observed)
            loss = torch.mean(torch.abs(prediction - target))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        last_loss = loss
    return last_loss


def _fixed_patch(pairs, patch_size):
    observed, target = pairs[0]
    observed = unet._pad_to_patch(observed, patch_size)[:patch_size, :patch_size]
    target = unet._pad_to_patch(target, patch_size)[:patch_size, :patch_size]
    return observed.astype(np.float32), target.astype(np.float32)


def _predict_patch(model, observed_patch, device, channels_last):
    tensor = (
        torch.from_numpy(np.ascontiguousarray(observed_patch))
        .unsqueeze(0).unsqueeze(0).to(device)
    )
    if channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    model.eval()
    with torch.inference_mode():
        prediction = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return prediction.squeeze().float().cpu().numpy()


def _training_variant(
    *,
    name,
    initial_state,
    pairs,
    device,
    warmup_steps,
    timed_steps,
    batch_size,
    patch_size,
    learning_rate,
    base_channels,
    amp_enabled,
    seed,
    channels_last,
    fused_adam,
    fixed_observed,
    fixed_target,
):
    utils.set_seed(seed)
    model = unet.UNetSmall(base_channels=base_channels)
    model.load_state_dict(initial_state)
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)

    try:
        optimizer = _make_optimizer(model, learning_rate, fused_adam)
    except Exception as exc:
        return {"name": name, "supported": False, "error": f"{type(exc).__name__}: {exc}"}

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # Warm cuDNN and optimizer kernels using a distinct deterministic stream.
    warm_generator = torch.Generator().manual_seed(seed + 987654)
    try:
        _train_steps(
            model, optimizer, scaler, pairs,
            steps=warmup_steps,
            generator=warm_generator,
            device=device,
            batch_size=batch_size,
            patch_size=patch_size,
            amp_enabled=amp_enabled,
            channels_last=channels_last,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except Exception as exc:
        return {"name": name, "supported": False, "error": f"{type(exc).__name__}: {exc}"}

    # Restore the exact common initial state and a fresh optimizer after warm-up.
    model.load_state_dict(initial_state)
    optimizer = _make_optimizer(model, learning_rate, fused_adam)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    generator = torch.Generator().manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    last_loss = _train_steps(
        model, optimizer, scaler, pairs,
        steps=timed_steps,
        generator=generator,
        device=device,
        batch_size=batch_size,
        patch_size=patch_size,
        amp_enabled=amp_enabled,
        channels_last=channels_last,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    patch_prediction = _predict_patch(
        model, fixed_observed, device, channels_last
    )
    patch_mae = float(np.mean(np.abs(patch_prediction - fixed_target)))
    peak_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
        if device.type == "cuda" else 0.0
    )

    result = {
        "name": name,
        "supported": True,
        "channels_last": bool(channels_last),
        "fused_adam": bool(fused_adam),
        "warmup_steps": int(warmup_steps),
        "timed_steps": int(timed_steps),
        "runtime_s": float(elapsed),
        "seconds_per_step": float(elapsed / max(timed_steps, 1)),
        "steps_per_second": float(timed_steps / max(elapsed, 1e-12)),
        "final_training_loss": float(last_loss.detach().cpu()),
        "fixed_validation_patch_mae": patch_mae,
        "peak_allocated_vram_mb": peak_mb,
        "_patch_prediction": patch_prediction,
    }

    del optimizer, scaler, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _predict_tiled_candidate(
    model,
    observed,
    device,
    *,
    tile_size,
    overlap,
    tile_batch_size,
    channels_last,
):
    observed = np.asarray(observed, dtype=np.float32)
    original_height, original_width = observed.shape
    padded = unet._pad_to_patch(observed, tile_size)
    prediction_sum = np.zeros_like(padded, dtype=np.float32)
    prediction_count = np.zeros_like(padded, dtype=np.float32)
    stride = tile_size - overlap
    positions = [
        (top, left)
        for top in _tile_starts(padded.shape[0], tile_size, stride)
        for left in _tile_starts(padded.shape[1], tile_size, stride)
    ]

    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(positions), tile_batch_size):
            group = positions[offset : offset + tile_batch_size]
            patches = np.stack(
                [padded[t:t+tile_size, l:l+tile_size] for t, l in group],
                axis=0,
            )
            tensor = torch.from_numpy(np.ascontiguousarray(patches)).unsqueeze(1).to(device)
            if channels_last:
                tensor = tensor.contiguous(memory_format=torch.channels_last)
            preds = model(tensor).squeeze(1).float().cpu().numpy()
            for pred, (top, left) in zip(preds, group):
                prediction_sum[top:top+tile_size, left:left+tile_size] += pred
                prediction_count[top:top+tile_size, left:left+tile_size] += 1.0

    prediction = prediction_sum / np.maximum(prediction_count, 1.0)
    return prediction[:original_height, :original_width], len(positions)


def _warm_inference(model, observed, device, tile_size, channels_last, batch_size):
    padded = unet._pad_to_patch(np.asarray(observed, dtype=np.float32), tile_size)
    count = max(batch_size, 1)
    patches = np.stack([padded[:tile_size, :tile_size]] * count, axis=0)
    tensor = torch.from_numpy(np.ascontiguousarray(patches)).unsqueeze(1).to(device)
    if channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    model.eval()
    with torch.inference_mode():
        for _ in range(3):
            _ = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _inference_variant(
    *,
    name,
    initial_state,
    observed,
    device,
    base_channels,
    tile_size,
    overlap,
    tile_batch_size,
    channels_last,
):
    model = unet.UNetSmall(base_channels=base_channels)
    model.load_state_dict(initial_state)
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)

    _warm_inference(model, observed, device, tile_size, channels_last, tile_batch_size)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    started = time.perf_counter()
    prediction, tile_count = _predict_tiled_candidate(
        model, observed, device,
        tile_size=tile_size,
        overlap=overlap,
        tile_batch_size=tile_batch_size,
        channels_last=channels_last,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
        if device.type == "cuda" else 0.0
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": name,
        "channels_last": bool(channels_last),
        "tile_batch_size": int(tile_batch_size),
        "tile_count": int(tile_count),
        "runtime_s": float(elapsed),
        "tiles_per_second": float(tile_count / max(elapsed, 1e-12)),
        "peak_allocated_vram_mb": peak_mb,
        "_prediction": prediction,
    }


def main(
    *,
    device_name="cuda:0",
    warmup_steps=10,
    timed_steps=100,
    output_json=None,
):
    config = selection_dependency_config("unet_small")
    training = config["training"]
    inference = config["inference"]

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    train_rows = utils.load_manifest(utils.SYNTHETIC_MANIFEST, split="train")
    validation_rows = utils.load_manifest(utils.SYNTHETIC_MANIFEST, split="validation")
    if any(row.get("split") == "test" for row in train_rows + validation_rows):
        raise AssertionError("test row entered benchmark")

    print("Loading train/validation arrays once ...")
    train_pairs = unet._load_pairs(train_rows)
    validation_pairs = unet._load_pairs(validation_rows)

    seed = int(training["seeds"][0])
    utils.set_seed(seed)
    initial_model = unet.UNetSmall(base_channels=int(training["base_channels"]))
    initial_state = {
        k: v.detach().cpu().clone() for k, v in initial_model.state_dict().items()
    }
    del initial_model

    fixed_observed, fixed_target = _fixed_patch(
        validation_pairs, int(training["patch_size"])
    )

    common = dict(
        initial_state=initial_state,
        pairs=train_pairs,
        device=device,
        warmup_steps=int(warmup_steps),
        timed_steps=int(timed_steps),
        batch_size=int(training["batch_size"]),
        patch_size=int(training["patch_size"]),
        learning_rate=float(training["learning_rate"]),
        base_channels=int(training["base_channels"]),
        amp_enabled=bool(training["amp_requested"] and device.type == "cuda"),
        seed=seed,
        fixed_observed=fixed_observed,
        fixed_target=fixed_target,
    )
    training_variants = [
        _training_variant(
            name="baseline_nchw_default_adam",
            channels_last=False,
            fused_adam=False,
            **common,
        ),
        _training_variant(
            name="channels_last_default_adam",
            channels_last=True,
            fused_adam=False,
            **common,
        ),
        _training_variant(
            name="channels_last_fused_adam",
            channels_last=True,
            fused_adam=True,
            **common,
        ),
    ]

    baseline = next(
        row for row in training_variants
        if row["name"] == "baseline_nchw_default_adam" and row.get("supported")
    )
    base_step = baseline["seconds_per_step"]
    base_pred = baseline["_patch_prediction"]
    for row in training_variants:
        if row.get("supported"):
            row["speedup_vs_baseline"] = float(
                base_step / max(row["seconds_per_step"], 1e-12)
            )
            diff = np.abs(
                row["_patch_prediction"].astype(np.float64)
                - base_pred.astype(np.float64)
            )
            row["patch_prediction_max_abs_diff_vs_baseline"] = float(
                diff.max(initial=0.0)
            )
            row["patch_prediction_mean_abs_diff_vs_baseline"] = float(
                diff.mean()
            )
        row.pop("_patch_prediction", None)

    observed = validation_pairs[0][0]
    inference_variants = []
    for channels_last, batch_size, name in [
        (False, 1, "nchw_tile_batch_1"),
        (False, 2, "nchw_tile_batch_2"),
        (True, 1, "channels_last_tile_batch_1"),
        (True, 2, "channels_last_tile_batch_2"),
        (True, 4, "channels_last_tile_batch_4"),
    ]:
        try:
            inference_variants.append(
                _inference_variant(
                    name=name,
                    initial_state=initial_state,
                    observed=observed,
                    device=device,
                    base_channels=int(training["base_channels"]),
                    tile_size=int(inference["tile_size"]),
                    overlap=int(inference["overlap"]),
                    tile_batch_size=batch_size,
                    channels_last=channels_last,
                )
            )
        except RuntimeError as exc:
            inference_variants.append(
                {"name": name, "supported": False, "error": str(exc)}
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    ref = next(
        row for row in inference_variants
        if row["name"] == "nchw_tile_batch_1" and row.get("supported", True)
    )
    ref_time = ref["runtime_s"]
    ref_pred = ref["_prediction"]
    for row in inference_variants:
        if row.get("supported", True):
            row["speedup_vs_nchw_batch1"] = float(
                ref_time / max(row["runtime_s"], 1e-12)
            )
            diff = np.abs(
                row["_prediction"].astype(np.float64)
                - ref_pred.astype(np.float64)
            )
            row["prediction_max_abs_diff_vs_reference"] = float(
                diff.max(initial=0.0)
            )
            row["prediction_mean_abs_diff_vs_reference"] = float(diff.mean())
        row.pop("_prediction", None)

    report = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else "CPU"
        ),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "formal_training_config": dict(training),
        "formal_inference_config": dict(inference),
        "warmup_steps": int(warmup_steps),
        "timed_steps": int(timed_steps),
        "training_variants": training_variants,
        "inference_variants": inference_variants,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved: {path}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        device_name=args.device,
        warmup_steps=args.warmup_steps,
        timed_steps=args.steps,
        output_json=args.output_json,
    )
