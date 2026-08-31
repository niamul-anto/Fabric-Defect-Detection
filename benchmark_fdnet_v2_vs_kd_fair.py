
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import io
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch

from fdnet_v2_kd import FDNetV2 as CanonicalFDNetV2


# ============================================================
# FAIR BENCHMARK SETTINGS
# ============================================================
SEED = 42
IMAGE_SIZE = 224
NUM_CLASSES = 4
USE_AMP = False
WARMUP_RUNS = 100
PEAK_MEMORY_RUNS = 100
LATENCY_REPEATS = 5
RUNS_PER_REPEAT = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parent

BASELINE_CHECKPOINT = (
    ROOT_DIR / "results" / "fdnet_v2_classification" / "best_fdnet_v2.pt"
)
KD_CHECKPOINT = (
    ROOT_DIR / "results" / "fdnet_v2_kd_T4_A07" / "fdnet_v2_kd_best.pth"
)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# CHECKPOINT HELPERS
# ============================================================
def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Extract a model state_dict from common PyTorch checkpoint layouts."""
    if isinstance(checkpoint, Mapping):
        for key in (
            "model_state_dict",
            "state_dict",
            "best_model_state_dict",
            "model_state",
            "model",
            "student_state_dict",
        ):
            value = checkpoint.get(key)
            if isinstance(value, Mapping) and value and all(
                torch.is_tensor(v) for v in value.values()
            ):
                return value

        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

    raise ValueError(
        "Could not locate a model state_dict in the checkpoint."
    )


def clean_state_dict(
    state_dict: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    """Strip common wrapper prefixes without changing key order."""
    cleaned: OrderedDict[str, torch.Tensor] = OrderedDict()

    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod.", "model.", "student."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value

    return cleaned


def load_checkpoint_cpu(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# ============================================================
# LEGACY -> CANONICAL BASELINE CONVERSION
# ============================================================
def map_state_dict_by_order(
    source_state: Mapping[str, torch.Tensor],
    target_template: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], list[dict[str, str]]]:
    """
    Map a legacy FD-Net V2 state_dict to the canonical FD-Net V2 key layout.

    The user's legacy baseline and the KD student contain the same ordered
    trainable/buffer tensors, but their module names differ (for example
    stage1 vs standard_stage and ConvBNAct.block vs Sequential indices).

    This function refuses to map if tensor counts or shapes differ.
    """
    source_items = list(source_state.items())
    target_items = list(target_template.items())

    if len(source_items) != len(target_items):
        raise ValueError(
            "State-dict length mismatch: "
            f"source={len(source_items)}, target={len(target_items)}"
        )

    mapped: OrderedDict[str, torch.Tensor] = OrderedDict()
    audit: list[dict[str, str]] = []

    for index, ((source_key, source_tensor), (target_key, target_tensor)) in enumerate(
        zip(source_items, target_items)
    ):
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(
                "State-dict shape mismatch at position "
                f"{index}: {source_key} {tuple(source_tensor.shape)} -> "
                f"{target_key} {tuple(target_tensor.shape)}"
            )

        mapped[target_key] = source_tensor.detach().clone()
        audit.append(
            {
                "index": str(index),
                "source_key": source_key,
                "target_key": target_key,
                "shape": str(tuple(source_tensor.shape)),
            }
        )

    return mapped, audit


def load_legacy_fdnet_class():
    """
    Load the exact legacy FDNetV2 class from the user's existing
    benchmark_fdnet_v2.py. It is used only to validate/load the old
    checkpoint layout; benchmarking is performed with CanonicalFDNetV2.
    """
    legacy_path = ROOT_DIR / "benchmark_fdnet_v2.py"
    if not legacy_path.exists():
        raise FileNotFoundError(
            "benchmark_fdnet_v2.py is required in the project root so the "
            "legacy baseline checkpoint can be validated before conversion."
        )

    spec = importlib.util.spec_from_file_location(
        "_legacy_fdnet_v2_benchmark", legacy_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import legacy benchmark: {legacy_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "FDNetV2"):
        raise RuntimeError(
            "benchmark_fdnet_v2.py does not define FDNetV2."
        )

    return module.FDNetV2


def load_baseline_as_canonical():
    checkpoint = load_checkpoint_cpu(BASELINE_CHECKPOINT)
    source_state = clean_state_dict(extract_state_dict(checkpoint))

    LegacyFDNetV2 = load_legacy_fdnet_class()
    legacy_model = LegacyFDNetV2(num_classes=NUM_CLASSES)
    legacy_model.load_state_dict(source_state, strict=True)
    legacy_model.eval()

    canonical_model = CanonicalFDNetV2(num_classes=NUM_CLASSES)
    mapped_state, mapping_audit = map_state_dict_by_order(
        legacy_model.state_dict(),
        canonical_model.state_dict(),
    )
    canonical_model.load_state_dict(mapped_state, strict=True)
    canonical_model.eval()

    # Numerical equivalence check before benchmarking.
    # This proves the key conversion preserves the legacy baseline function
    # within floating-point tolerance, despite the different module names.
    torch.manual_seed(SEED)
    audit_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.inference_mode():
        legacy_output = legacy_model(audit_input)
        canonical_output = canonical_model(audit_input)

    max_abs_diff = (
        legacy_output - canonical_output
    ).abs().max().item()

    if max_abs_diff > 1e-5:
        raise RuntimeError(
            "Legacy-to-canonical conversion failed numerical equivalence "
            f"check (max abs difference={max_abs_diff:.8g})."
        )

    del legacy_model, audit_input, legacy_output, canonical_output
    return canonical_model, checkpoint, mapping_audit, max_abs_diff


def load_kd_as_canonical():
    checkpoint = load_checkpoint_cpu(KD_CHECKPOINT)
    state = clean_state_dict(extract_state_dict(checkpoint))

    model = CanonicalFDNetV2(num_classes=NUM_CLASSES)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def clone_canonical_from_state(
    state_dict: Mapping[str, torch.Tensor]
) -> CanonicalFDNetV2:
    """Build an independent CPU CanonicalFDNetV2 from a CPU state_dict."""
    model = CanonicalFDNetV2(num_classes=NUM_CLASSES)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ============================================================
# BENCHMARK HELPERS
# ============================================================
def forward_once(model, x):
    if DEVICE.type == "cuda" and USE_AMP:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(x)
    return model(x)


def get_parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total, trainable


def get_model_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 ** 2)


def get_flops_cpu(model):
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:
        raise ImportError(
            "fvcore is not installed. Run: python -m pip install fvcore"
        ) from exc

    cpu_model = model.cpu().eval()
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    try:
        analysis = FlopCountAnalysis(cpu_model, dummy)
        total_flops = analysis.total()
        unsupported = dict(analysis.unsupported_ops())
        return total_flops / 1e9, unsupported
    except Exception as error:
        print("FLOPs calculation failed:", error)
        return None, {}


def warm_up(model, dummy_input):
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            _ = forward_once(model, dummy_input)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def get_peak_gpu_memory_mb(model, dummy_input):
    if DEVICE.type != "cuda":
        return None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    with torch.inference_mode():
        for _ in range(PEAK_MEMORY_RUNS):
            _ = forward_once(model, dummy_input)

    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated(DEVICE)
    return peak_bytes / (1024 ** 2)


def get_latency_statistics(model, dummy_input):
    repeat_latencies = []

    if DEVICE.type == "cuda":
        for _ in range(LATENCY_REPEATS):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            starter.record()

            with torch.inference_mode():
                for _ in range(RUNS_PER_REPEAT):
                    _ = forward_once(model, dummy_input)

            ender.record()
            torch.cuda.synchronize()

            total_ms = starter.elapsed_time(ender)
            repeat_latencies.append(
                total_ms / RUNS_PER_REPEAT
            )
    else:
        import time

        for _ in range(LATENCY_REPEATS):
            start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(RUNS_PER_REPEAT):
                    _ = forward_once(model, dummy_input)
            end = time.perf_counter()

            repeat_latencies.append(
                ((end - start) * 1000)
                / RUNS_PER_REPEAT
            )

    mean_latency = statistics.mean(repeat_latencies)
    std_latency = (
        statistics.stdev(repeat_latencies)
        if len(repeat_latencies) > 1
        else 0.0
    )
    throughput = 1000.0 / mean_latency

    return (
        mean_latency,
        std_latency,
        throughput,
        repeat_latencies,
    )


def release_cuda(model=None, dummy_input=None):
    if dummy_input is not None:
        del dummy_input
    if model is not None:
        del model

    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def benchmark_one(label, model):
    """
    Benchmark a model using the exact same canonical FD-Net V2 graph.
    """
    if not isinstance(model, CanonicalFDNetV2):
        raise TypeError(
            "Fair benchmark requires CanonicalFDNetV2 for both models."
        )

    total_params, trainable_params = get_parameter_counts(model)
    model_size_mb = get_model_size_mb(model)

    print(f"\n[{label}] Calculating FLOPs on CPU...")
    gflops, unsupported_ops = get_flops_cpu(model)

    release_cuda()
    model = model.to(DEVICE).eval()
    dummy_input = torch.randn(
        1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE
    )

    with torch.inference_mode():
        output = forward_once(model, dummy_input)

    if tuple(output.shape) != (1, NUM_CLASSES):
        raise RuntimeError(
            f"{label}: unexpected output shape {tuple(output.shape)}"
        )

    print(f"[{label}] Running {WARMUP_RUNS} warm-up inferences...")
    warm_up(model, dummy_input)

    print(f"[{label}] Measuring peak GPU memory...")
    peak_memory_mb = get_peak_gpu_memory_mb(
        model, dummy_input
    )

    print(
        f"[{label}] Measuring latency "
        f"({LATENCY_REPEATS} x {RUNS_PER_REPEAT})..."
    )
    (
        latency_mean,
        latency_std,
        throughput,
        latency_repeats,
    ) = get_latency_statistics(model, dummy_input)

    result = {
        "Model": label,
        "Device": str(DEVICE),
        "GPU": (
            torch.cuda.get_device_name(0)
            if DEVICE.type == "cuda"
            else "CPU"
        ),
        "CUDA": (
            torch.version.cuda
            if DEVICE.type == "cuda"
            else "N/A"
        ),
        "Precision_Mode": (
            "AMP/FP16"
            if USE_AMP and DEVICE.type == "cuda"
            else "FP32"
        ),
        "Inference_Batch_Size": 1,
        "Input_Size": f"1x3x{IMAGE_SIZE}x{IMAGE_SIZE}",
        "Total_Parameters": total_params,
        "Trainable_Parameters": trainable_params,
        "Parameters_Million": round(
            total_params / 1e6, 6
        ),
        "Model_Weight_Size_MB": round(
            model_size_mb, 4
        ),
        "GFLOPs_Estimate": (
            round(gflops, 6)
            if gflops is not None
            else "N/A"
        ),
        "Latency_Mean_ms_per_image": round(
            latency_mean, 6
        ),
        "Latency_STD_ms": round(
            latency_std, 6
        ),
        "Throughput_images_per_sec": round(
            throughput, 4
        ),
        "Peak_GPU_Memory_MB": (
            round(peak_memory_mb, 4)
            if peak_memory_mb is not None
            else "N/A"
        ),
        "Latency_Repeats_ms": ";".join(
            f"{x:.6f}" for x in latency_repeats
        ),
    }

    print(f"\n{label} results")
    print("-" * 60)
    print(f"Parameters      : {total_params:,}")
    print(f"Model size      : {model_size_mb:.2f} MB")
    if gflops is not None:
        print(f"GFLOPs          : {gflops:.3f}")
    print(
        f"Latency         : "
        f"{latency_mean:.3f} +/- {latency_std:.3f} ms/image"
    )
    print(
        f"Throughput      : "
        f"{throughput:.2f} images/sec"
    )
    if peak_memory_mb is not None:
        print(
            f"Peak GPU memory : "
            f"{peak_memory_mb:.2f} MB"
        )

    release_cuda(model, dummy_input)
    return result, unsupported_ops


# ============================================================
# OUTPUT
# ============================================================
def save_mapping_audit(
    audit_rows: list[dict[str, str]],
    path: Path,
):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "source_key",
                "target_key",
                "shape",
            ],
        )
        writer.writeheader()
        writer.writerows(audit_rows)


def save_results(
    results: list[dict[str, Any]],
    output_dir: Path,
    order_name: str,
    baseline_equivalence_diff: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        output_dir
        / f"fdnet_v2_vs_kd_fair_{order_name}.csv"
    )
    txt_path = (
        output_dir
        / f"fdnet_v2_vs_kd_fair_{order_name}.txt"
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(
            "FD-NET V2 vs FD-NET V2-KD FAIR BENCHMARK\n"
        )
        f.write("=" * 72 + "\n")
        f.write(
            "Both checkpoints are benchmarked with the SAME "
            "CanonicalFDNetV2 implementation and forward graph.\n"
        )
        f.write(
            "The ViT-B/16 teacher is NOT used during KD-student inference.\n"
        )
        f.write(
            "Legacy baseline -> canonical numerical equivalence "
            f"max abs diff: {baseline_equivalence_diff:.10g}\n"
        )
        f.write(
            f"Order: {order_name}\n\n"
        )

        for result in results:
            f.write(result["Model"] + "\n")
            f.write("-" * 72 + "\n")
            for key, value in result.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

    return csv_path, txt_path


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fair same-implementation benchmark for the baseline "
            "FD-Net V2 and the FD-Net V2-KD student."
        )
    )
    parser.add_argument(
        "--order",
        choices=("baseline-first", "kd-first"),
        default="baseline-first",
        help=(
            "Benchmark order. For publication robustness, run once "
            "with each order and check that conclusions are stable."
        ),
    )
    args = parser.parse_args()

    output_dir = (
        ROOT_DIR
        / "results"
        / "fdnet_v2_vs_kd_fair_benchmark"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FD-NET V2 vs FD-NET V2-KD FAIR BENCHMARK")
    print("=" * 72)
    print("Device              :", DEVICE)
    print(
        "Precision           :",
        "AMP/FP16"
        if USE_AMP and DEVICE.type == "cuda"
        else "FP32",
    )
    print(
        f"Input               : "
        f"1 x 3 x {IMAGE_SIZE} x {IMAGE_SIZE}"
    )
    print("Inference batch size: 1")
    print("Warm-up runs        :", WARMUP_RUNS)
    print(
        "Timed runs          :",
        f"{LATENCY_REPEATS} x {RUNS_PER_REPEAT}",
    )
    print("Baseline checkpoint :", BASELINE_CHECKPOINT)
    print("KD checkpoint       :", KD_CHECKPOINT)
    print("Benchmark order     :", args.order)

    if DEVICE.type == "cuda":
        print(
            "GPU                 :",
            torch.cuda.get_device_name(0),
        )
        print("CUDA version        :", torch.version.cuda)

    print(
        "\nLoading baseline and converting legacy key layout "
        "to the canonical FD-Net V2 graph..."
    )
    (
        baseline_model,
        baseline_checkpoint,
        mapping_audit,
        equivalence_diff,
    ) = load_baseline_as_canonical()

    print(
        "Baseline conversion verified. "
        f"Max abs output difference: {equivalence_diff:.10g}"
    )

    mapping_path = (
        output_dir
        / "baseline_legacy_to_canonical_key_mapping.csv"
    )
    save_mapping_audit(mapping_audit, mapping_path)

    print("\nLoading KD student with the same canonical graph...")
    kd_model, kd_checkpoint = load_kd_as_canonical()

    baseline_params = sum(
        p.numel() for p in baseline_model.parameters()
    )
    kd_params = sum(
        p.numel() for p in kd_model.parameters()
    )

    if baseline_params != kd_params:
        raise RuntimeError(
            "Baseline and KD canonical models do not have the "
            "same parameter count."
        )

    if baseline_params != 3_068_448:
        raise RuntimeError(
            "Unexpected FD-Net V2 parameter count: "
            f"{baseline_params:,}"
        )

    print(
        "Canonical architecture check passed: "
        f"{baseline_params:,} parameters for both models."
    )

    # Keep only CPU state_dict copies between benchmark runs.
    # This prevents the first benchmarked model from remaining resident on
    # the GPU and contaminating the second model's peak-memory measurement.
    baseline_state = OrderedDict(
        (key, value.detach().cpu().clone())
        for key, value in baseline_model.state_dict().items()
    )
    kd_state = OrderedDict(
        (key, value.detach().cpu().clone())
        for key, value in kd_model.state_dict().items()
    )

    del baseline_model, kd_model, baseline_checkpoint, kd_checkpoint
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    loaders = {
        "FD-Net V2": lambda: clone_canonical_from_state(baseline_state),
        "FD-Net V2-KD": lambda: clone_canonical_from_state(kd_state),
    }

    order = (
        ["FD-Net V2", "FD-Net V2-KD"]
        if args.order == "baseline-first"
        else ["FD-Net V2-KD", "FD-Net V2"]
    )

    results = []
    unsupported_by_model = {}

    # Each object is benchmarked exactly once in the requested order.
    for label in order:
        model = loaders[label]()
        result, unsupported = benchmark_one(label, model)
        results.append(result)
        unsupported_by_model[label] = unsupported

    # Always write comparison rows in the logical baseline -> KD order.
    results.sort(
        key=lambda row: 0
        if row["Model"] == "FD-Net V2"
        else 1
    )

    csv_path, txt_path = save_results(
        results,
        output_dir,
        args.order,
        equivalence_diff,
    )

    print("\n" + "=" * 72)
    print("FAIR COMPARISON SUMMARY")
    print("=" * 72)

    baseline_result = results[0]
    kd_result = results[1]

    for key, title in (
        ("Total_Parameters", "Parameters"),
        ("Model_Weight_Size_MB", "Model size (MB)"),
        ("GFLOPs_Estimate", "GFLOPs"),
        (
            "Latency_Mean_ms_per_image",
            "Latency (ms/image)",
        ),
        (
            "Throughput_images_per_sec",
            "Throughput (images/s)",
        ),
        (
            "Peak_GPU_Memory_MB",
            "Peak GPU memory (MB)",
        ),
    ):
        print(
            f"{title:24s}: "
            f"baseline={baseline_result[key]} | "
            f"KD={kd_result[key]}"
        )

    print("\nInterpretation rule:")
    print(
        "- Parameters, serialized size, and GFLOPs should be "
        "identical because knowledge distillation does not alter "
        "the student architecture."
    )
    print(
        "- Treat latency/throughput differences as runtime "
        "measurement variation unless they remain stable across "
        "repeated runs and both benchmark orders."
    )
    print(
        "- Do not attribute a speed or memory change causally to "
        "knowledge distillation when the inference graph is identical."
    )

    print("\nSaved:")
    print("CSV    :", csv_path)
    print("TXT    :", txt_path)
    print("Mapping:", mapping_path)

    print(
        "\nFor publication robustness, also run:\n"
        "python benchmark_fdnet_v2_vs_kd_fair.py --order kd-first"
    )


if __name__ == "__main__":
    main()
