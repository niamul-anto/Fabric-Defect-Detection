# ============================================================
# EFFICIENTNET-B0 BENCHMARKING SCRIPT
# Loads the already-trained best checkpoint. NO retraining.
#
# Metrics:
#   1) Total / trainable parameters
#   2) Model weight size (MB)
#   3) FLOPs estimate (GFLOPs)
#   4) Peak GPU memory (MB)
#   5) Inference latency (ms/image)
#   6) Throughput (images/second)
#   7) Latency standard deviation across repeated runs
#
# Install once in PowerShell:
#   python -m pip install fvcore
# ============================================================

from pathlib import Path
import io
import csv
import statistics

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError as exc:
    raise ImportError(
        "fvcore is not installed.\n"
        "Open PowerShell with your virtual environment active and run:\n"
        "python -m pip install fvcore"
    ) from exc


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
IMAGE_SIZE = 224
NUM_CLASSES = 4

# IMPORTANT:
# Keep this SAME for every model in your comparison.
# False = FP32
# True  = AMP/FP16 autocast on CUDA
USE_AMP = False

WARMUP_RUNS = 100

# 5 x 200 = 1000 timed forward passes
LATENCY_REPEATS = 5
RUNS_PER_REPEAT = 200

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ROOT_DIR = Path(__file__).resolve().parent

# This matches your original EfficientNet-B0 training script.
MODEL_PATH = (
    ROOT_DIR
    / "results"
    / "efficientnet_b0_classification"
    / "best_efficientnet_b0.pt"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "results"
    / "efficientnet_b0_benchmark"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_PATH = OUTPUT_DIR / "efficientnet_b0_benchmark.csv"
TXT_PATH = OUTPUT_DIR / "efficientnet_b0_benchmark.txt"


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HELPER: OPTIONAL AMP FORWARD
# ============================================================

def forward_once(model, x):

    if (
        DEVICE.type == "cuda"
        and USE_AMP
    ):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            return model(x)

    return model(x)


# ============================================================
# CREATE + LOAD TRAINED EFFICIENTNET-B0
# ============================================================

def load_trained_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "\nBest EfficientNet-B0 checkpoint was not found at:\n"
            f"{MODEL_PATH}\n\n"
            "If best_efficientnet_b0.pt is stored elsewhere, "
            "change MODEL_PATH near the top of this script."
        )

    # No need to download ImageNet weights again during benchmarking.
    # The trained checkpoint already contains the final learned weights.
    model = efficientnet_b0(
        weights=None
    )

    # Recreate the 4-class classifier used in your training script.
    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    model = model.to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # Your training code saved a checkpoint dictionary
    # containing "model_state_dict".
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        # Fallback only if the file happens to contain raw state_dict.
        model.load_state_dict(
            checkpoint
        )

    model.eval()

    return model, checkpoint


# ============================================================
# PARAMETERS
# ============================================================

def get_parameter_counts(model):

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# ============================================================
# MODEL WEIGHT SIZE
# ============================================================

def get_model_size_mb(model):

    buffer = io.BytesIO()

    torch.save(
        model.state_dict(),
        buffer
    )

    size_bytes = (
        buffer.getbuffer().nbytes
    )

    return (
        size_bytes
        / (1024 ** 2)
    )


# ============================================================
# FLOPs
# ============================================================

def get_flops(
    model,
    dummy_input
):

    try:

        flop_analysis = FlopCountAnalysis(
            model,
            dummy_input
        )

        total_flops = (
            flop_analysis.total()
        )

        unsupported = (
            flop_analysis.unsupported_ops()
        )

        gflops = (
            total_flops / 1e9
        )

        return (
            gflops,
            dict(unsupported)
        )

    except Exception as error:

        print(
            "\nFLOPs calculation failed:"
        )

        print(error)

        return None, {}


# ============================================================
# WARM-UP
# ============================================================

def warm_up(
    model,
    dummy_input
):

    with torch.inference_mode():

        for _ in range(
            WARMUP_RUNS
        ):

            _ = forward_once(
                model,
                dummy_input
            )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ============================================================
# PEAK GPU MEMORY
# ============================================================

def get_peak_gpu_memory_mb(
    model,
    dummy_input
):

    if DEVICE.type != "cuda":
        return None

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(
        DEVICE
    )

    with torch.inference_mode():

        for _ in range(100):

            _ = forward_once(
                model,
                dummy_input
            )

    torch.cuda.synchronize()

    peak_memory_bytes = (
        torch.cuda.max_memory_allocated(
            DEVICE
        )
    )

    return (
        peak_memory_bytes
        / (1024 ** 2)
    )


# ============================================================
# LATENCY
# ============================================================

def get_latency_statistics(
    model,
    dummy_input
):

    repeat_latencies = []

    if DEVICE.type == "cuda":

        for _ in range(
            LATENCY_REPEATS
        ):

            starter = torch.cuda.Event(
                enable_timing=True
            )

            ender = torch.cuda.Event(
                enable_timing=True
            )

            torch.cuda.synchronize()

            starter.record()

            with torch.inference_mode():

                for _ in range(
                    RUNS_PER_REPEAT
                ):

                    _ = forward_once(
                        model,
                        dummy_input
                    )

            ender.record()

            torch.cuda.synchronize()

            total_time_ms = (
                starter.elapsed_time(
                    ender
                )
            )

            average_ms = (
                total_time_ms
                / RUNS_PER_REPEAT
            )

            repeat_latencies.append(
                average_ms
            )

    else:

        import time

        for _ in range(
            LATENCY_REPEATS
        ):

            start_time = (
                time.perf_counter()
            )

            with torch.inference_mode():

                for _ in range(
                    RUNS_PER_REPEAT
                ):

                    _ = forward_once(
                        model,
                        dummy_input
                    )

            end_time = (
                time.perf_counter()
            )

            total_time_ms = (
                (end_time - start_time)
                * 1000
            )

            average_ms = (
                total_time_ms
                / RUNS_PER_REPEAT
            )

            repeat_latencies.append(
                average_ms
            )

    mean_latency = (
        statistics.mean(
            repeat_latencies
        )
    )

    if len(repeat_latencies) > 1:

        std_latency = (
            statistics.stdev(
                repeat_latencies
            )
        )

    else:

        std_latency = 0.0

    throughput = (
        1000.0
        / mean_latency
    )

    return (
        mean_latency,
        std_latency,
        throughput,
        repeat_latencies
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(result):

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=result.keys()
        )

        writer.writeheader()
        writer.writerow(result)

    with open(
        TXT_PATH,
        "w",
        encoding="utf-8"
    ) as text_file:

        text_file.write(
            "EFFICIENTNET-B0 BENCHMARK RESULTS\n"
        )

        text_file.write(
            "=" * 60 + "\n"
        )

        for key, value in result.items():

            text_file.write(
                f"{key}: {value}\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EFFICIENTNET-B0 BENCHMARK")
    print("=" * 70)

    print(
        f"Device        : {DEVICE}"
    )

    print(
        f"Precision     : "
        f"{'AMP/FP16' if USE_AMP and DEVICE.type == 'cuda' else 'FP32'}"
    )

    print(
        f"Input size    : "
        f"1 x 3 x {IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    if DEVICE.type == "cuda":

        print(
            "GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            "CUDA version  : "
            f"{torch.version.cuda}"
        )

    print(
        f"Checkpoint    : {MODEL_PATH}"
    )

    print(
        "\nLoading trained EfficientNet-B0..."
    )

    model, checkpoint = (
        load_trained_model()
    )

    print(
        "Best trained EfficientNet-B0 loaded successfully."
    )

    if (
        isinstance(checkpoint, dict)
        and "epoch" in checkpoint
    ):

        print(
            f"Best epoch    : "
            f"{checkpoint['epoch']}"
        )

    if (
        isinstance(checkpoint, dict)
        and "best_valid_f1" in checkpoint
    ):

        print(
            f"Best valid F1 : "
            f"{checkpoint['best_valid_f1']:.6f}"
        )

    dummy_input = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE
    )

    # Quick output-shape check
    with torch.inference_mode():

        output = forward_once(
            model,
            dummy_input
        )

    print(
        f"Output shape  : "
        f"{tuple(output.shape)}"
    )

    if tuple(output.shape) != (
        1,
        NUM_CLASSES
    ):

        raise RuntimeError(
            "Unexpected EfficientNet-B0 output shape."
        )

    # --------------------------------------------------------
    # PARAMETERS + SIZE
    # --------------------------------------------------------

    print(
        "\nCalculating parameter count..."
    )

    total_params, trainable_params = (
        get_parameter_counts(
            model
        )
    )

    model_size_mb = (
        get_model_size_mb(
            model
        )
    )

    # --------------------------------------------------------
    # FLOPs
    # --------------------------------------------------------

    print(
        "\nCalculating FLOPs..."
    )

    gflops, unsupported_ops = (
        get_flops(
            model,
            dummy_input
        )
    )

    # --------------------------------------------------------
    # WARM-UP
    # --------------------------------------------------------

    print(
        "\nRunning warm-up..."
    )

    warm_up(
        model,
        dummy_input
    )

    print(
        "Warm-up completed."
    )

    # --------------------------------------------------------
    # PEAK GPU MEMORY
    # --------------------------------------------------------

    print(
        "\nMeasuring peak GPU memory..."
    )

    peak_memory_mb = (
        get_peak_gpu_memory_mb(
            model,
            dummy_input
        )
    )

    # --------------------------------------------------------
    # LATENCY
    # --------------------------------------------------------

    print(
        "\nMeasuring inference latency..."
    )

    (
        latency_mean,
        latency_std,
        throughput,
        latency_repeats
    ) = get_latency_statistics(
        model,
        dummy_input
    )

    precision_mode = (
        "AMP/FP16"
        if (
            USE_AMP
            and DEVICE.type == "cuda"
        )
        else "FP32"
    )

    # --------------------------------------------------------
    # RESULT DICTIONARY
    # --------------------------------------------------------

    result = {

        "Model": "EfficientNet-B0",

        "Device": str(DEVICE),

        "GPU": (
            torch.cuda.get_device_name(0)
            if DEVICE.type == "cuda"
            else "CPU"
        ),

        "Precision_Mode":
            precision_mode,

        "Input_Size":
            f"1x3x{IMAGE_SIZE}x{IMAGE_SIZE}",

        "Total_Parameters":
            total_params,

        "Trainable_Parameters":
            trainable_params,

        "Parameters_Million":
            round(
                total_params / 1e6,
                6
            ),

        "Model_Weight_Size_MB":
            round(
                model_size_mb,
                4
            ),

        "GFLOPs_Estimate":
            (
                round(
                    gflops,
                    6
                )
                if gflops is not None
                else "N/A"
            ),

        "Latency_Mean_ms_per_image":
            round(
                latency_mean,
                6
            ),

        "Latency_STD_ms":
            round(
                latency_std,
                6
            ),

        "Throughput_images_per_sec":
            round(
                throughput,
                4
            ),

        "Peak_GPU_Memory_MB":
            (
                round(
                    peak_memory_mb,
                    4
                )
                if peak_memory_mb is not None
                else "N/A"
            )
    }

    # --------------------------------------------------------
    # PRINT FINAL RESULTS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EFFICIENTNET-B0 BENCHMARK RESULTS")
    print("=" * 70)

    print(
        f"Total Parameters     : "
        f"{total_params:,}"
    )

    print(
        f"Parameters (Million) : "
        f"{total_params / 1e6:.3f} M"
    )

    print(
        f"Model Weight Size    : "
        f"{model_size_mb:.2f} MB"
    )

    if gflops is not None:

        print(
            f"FLOPs Estimate       : "
            f"{gflops:.3f} GFLOPs"
        )

    else:

        print(
            "FLOPs Estimate       : N/A"
        )

    print(
        f"Latency              : "
        f"{latency_mean:.3f} "
        f"+/- {latency_std:.3f} ms/image"
    )

    print(
        f"Throughput           : "
        f"{throughput:.2f} images/sec"
    )

    if peak_memory_mb is not None:

        print(
            f"Peak GPU Memory      : "
            f"{peak_memory_mb:.2f} MB"
        )

    else:

        print(
            "Peak GPU Memory      : N/A"
        )

    print(
        f"Precision Mode       : "
        f"{precision_mode}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # FLOPs WARNINGS
    # --------------------------------------------------------

    if unsupported_ops:

        print(
            "\nIMPORTANT FLOPs NOTE"
        )

        print(
            "fvcore did not count these operations:"
        )

        for operation, count in (
            unsupported_ops.items()
        ):

            print(
                f"{operation}: {count}"
            )

        print(
            "\nUse the SAME FLOPs tool for every compared model."
        )

    # --------------------------------------------------------
    # LATENCY REPEATS
    # --------------------------------------------------------

    print(
        "\nLatency repeats (ms/image):"
    )

    for index, value in enumerate(
        latency_repeats,
        start=1
    ):

        print(
            f"Run {index}: {value:.4f}"
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    save_results(
        result
    )

    print(
        "\nResults saved to:"
    )

    print(
        f"CSV: {CSV_PATH}"
    )

    print(
        f"TXT: {TXT_PATH}"
    )

    print(
        "\nBenchmark completed successfully."
    )


if __name__ == "__main__":
    main()
