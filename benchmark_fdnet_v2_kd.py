from pathlib import Path
import io
import csv
import statistics
import torch

from fdnet_v2_kd import FDNetV2

# =========================
# SETTINGS
# =========================
SEED = 42
IMAGE_SIZE = 224
NUM_CLASSES = 4
USE_AMP = False          # Keep SAME as the baseline benchmark (FP32)
WARMUP_RUNS = 100
LATENCY_REPEATS = 5
RUNS_PER_REPEAT = 200    # 5 x 200 = 1000 timed inferences
PEAK_MEMORY_RUNS = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    ROOT_DIR
    / "results"
    / "fdnet_v2_kd_T4_A07"
    / "fdnet_v2_kd_best.pth"
)
OUTPUT_DIR = ROOT_DIR / "results" / "fdnet_v2_kd_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "fdnet_v2_kd_benchmark.csv"
TXT_PATH = OUTPUT_DIR / "fdnet_v2_kd_benchmark.txt"
PURE_WEIGHTS_PATH = OUTPUT_DIR / "fdnet_v2_kd_state_dict.pt"


torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# HELPERS
# =========================
def forward_once(model, x):
    if DEVICE.type == "cuda" and USE_AMP:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(x)
    return model(x)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "best_model_state_dict",
            "model",
        ):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        # Plain state_dict saved directly as a dictionary.
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

    raise RuntimeError(
        "Unable to locate the student model state_dict in the checkpoint."
    )


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model.", "student."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{MODEL_PATH}\n"
            "Confirm that the selected T=4, alpha=0.7 checkpoint exists."
        )

    model = FDNetV2(num_classes=NUM_CLASSES)

    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    state_dict = clean_state_dict(extract_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model, checkpoint


def get_parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(model):
    """Return pure student state_dict size, excluding optimizer/checkpoint metadata."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 ** 2)


def save_pure_student_weights(model):
    """Save only the deployable student weights for an auditable model footprint."""
    torch.save(model.state_dict(), PURE_WEIGHTS_PATH)


def get_flops(model, dummy_input):
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:
        raise ImportError(
            "fvcore is not installed. Run: python -m pip install fvcore"
        ) from exc

    try:
        analysis = FlopCountAnalysis(model, dummy_input)
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
            repeat_latencies.append(total_ms / RUNS_PER_REPEAT)
    else:
        import time

        for _ in range(LATENCY_REPEATS):
            start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(RUNS_PER_REPEAT):
                    _ = forward_once(model, dummy_input)
            end = time.perf_counter()
            repeat_latencies.append(
                ((end - start) * 1000) / RUNS_PER_REPEAT
            )

    mean_latency = statistics.mean(repeat_latencies)
    std_latency = (
        statistics.stdev(repeat_latencies)
        if len(repeat_latencies) > 1
        else 0.0
    )
    throughput = 1000.0 / mean_latency
    return mean_latency, std_latency, throughput, repeat_latencies


def save_results(result):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    with open(TXT_PATH, "w", encoding="utf-8") as f:
        f.write("FD-NET V2-KD BENCHMARK RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(
            "NOTE: These measurements benchmark ONLY the distilled FD-Net V2 "
            "student. The ViT-B/16 teacher is not loaded during inference.\n\n"
        )
        for key, value in result.items():
            f.write(f"{key}: {value}\n")


# =========================
# MAIN
# =========================
def main():
    print("=" * 70)
    print("FD-NET V2-KD BENCHMARK")
    print("=" * 70)
    print("Device              :", DEVICE)
    print(
        "Precision           :",
        "AMP/FP16" if USE_AMP and DEVICE.type == "cuda" else "FP32",
    )
    print(f"Input size          : 1 x 3 x {IMAGE_SIZE} x {IMAGE_SIZE}")
    print("Inference batch size: 1 (benchmarking only)")
    print("Teacher at inference: NO")
    if DEVICE.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
        print("CUDA version        :", torch.version.cuda)
    print("Checkpoint          :", MODEL_PATH)

    print("\nLoading distilled student model...")
    model, checkpoint = load_trained_model()
    print("FD-Net V2-KD student loaded successfully.")

    best_epoch = None
    best_val_f1 = None
    temperature = None
    alpha = None

    if isinstance(checkpoint, dict):
        best_epoch = checkpoint.get("epoch")
        best_val_f1 = checkpoint.get("best_val_f1")
        temperature = checkpoint.get("temperature")
        alpha = checkpoint.get("alpha")

    if best_epoch is not None:
        print("Best epoch          :", best_epoch)
    if best_val_f1 is not None:
        print(f"Best validation F1  : {float(best_val_f1):.6f}")
    if temperature is not None:
        print("KD temperature      :", temperature)
    if alpha is not None:
        print("KD alpha (CE weight):", alpha)

    dummy_input = torch.randn(
        1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE
    )

    with torch.inference_mode():
        output = forward_once(model, dummy_input)
    print("Output shape        :", tuple(output.shape))

    if tuple(output.shape) != (1, NUM_CLASSES):
        raise RuntimeError("Unexpected model output shape.")

    total_params, trainable_params = get_parameter_counts(model)
    model_size_mb = get_model_size_mb(model)
    save_pure_student_weights(model)

    print("\nCalculating FLOPs...")
    gflops, unsupported_ops = get_flops(model, dummy_input)

    print("Running warm-up...")
    warm_up(model, dummy_input)

    print("Measuring peak GPU memory...")
    peak_memory_mb = get_peak_gpu_memory_mb(model, dummy_input)

    print("Measuring inference latency...")
    latency_mean, latency_std, throughput, latency_repeats = (
        get_latency_statistics(model, dummy_input)
    )

    precision_mode = (
        "AMP/FP16"
        if USE_AMP and DEVICE.type == "cuda"
        else "FP32"
    )

    result = {
        "Model": "FD-Net V2-KD",
        "KD_Temperature": temperature if temperature is not None else "N/A",
        "KD_Alpha_CE": alpha if alpha is not None else "N/A",
        "Best_Validation_Macro_F1": (
            round(float(best_val_f1), 6)
            if best_val_f1 is not None
            else "N/A"
        ),
        "Best_Epoch": best_epoch if best_epoch is not None else "N/A",
        "Device": str(DEVICE),
        "GPU": (
            torch.cuda.get_device_name(0)
            if DEVICE.type == "cuda"
            else "CPU"
        ),
        "Precision_Mode": precision_mode,
        "Inference_Batch_Size": 1,
        "Input_Size": f"1x3x{IMAGE_SIZE}x{IMAGE_SIZE}",
        "Teacher_Used_At_Inference": "No",
        "Total_Parameters": total_params,
        "Trainable_Parameters": trainable_params,
        "Parameters_Million": round(total_params / 1e6, 6),
        "Model_Weight_Size_MB": round(model_size_mb, 4),
        "GFLOPs_Estimate": (
            round(gflops, 6) if gflops is not None else "N/A"
        ),
        "Latency_Mean_ms_per_image": round(latency_mean, 6),
        "Latency_STD_ms": round(latency_std, 6),
        "Throughput_images_per_sec": round(throughput, 4),
        "Peak_GPU_Memory_MB": (
            round(peak_memory_mb, 4)
            if peak_memory_mb is not None
            else "N/A"
        ),
    }

    print("\n" + "=" * 70)
    print("FD-NET V2-KD BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Parameters     : {total_params:,}")
    print(f"Parameters (Million) : {total_params / 1e6:.3f} M")
    print(f"Model Weight Size    : {model_size_mb:.2f} MB")
    if gflops is not None:
        print(f"FLOPs Estimate       : {gflops:.3f} GFLOPs")
    else:
        print("FLOPs Estimate       : N/A")
    print(
        f"Latency              : {latency_mean:.3f} +/- "
        f"{latency_std:.3f} ms/image"
    )
    print(f"Throughput           : {throughput:.2f} images/sec")
    if peak_memory_mb is not None:
        print(f"Peak GPU Memory      : {peak_memory_mb:.2f} MB")
    else:
        print("Peak GPU Memory      : N/A")
    print(f"Precision Mode       : {precision_mode}")
    print("Teacher at inference : No")
    print("=" * 70)

    if unsupported_ops:
        print("\nIMPORTANT FLOPs NOTE")
        print("fvcore did not count these operations:")
        for op, count in unsupported_ops.items():
            print(f"  {op}: {count}")
        print("Use the SAME FLOPs tool for every compared model.")

    print("\nLatency repeats (ms/image):")
    for i, value in enumerate(latency_repeats, start=1):
        print(f"Run {i}: {value:.4f}")

    save_results(result)
    print("\nResults saved to:")
    print("CSV         :", CSV_PATH)
    print("TXT         :", TXT_PATH)
    print("Pure weights:", PURE_WEIGHTS_PATH)
    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
