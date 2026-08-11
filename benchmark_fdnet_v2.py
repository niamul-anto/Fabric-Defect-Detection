from pathlib import Path
import io
import csv
import statistics
import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError as exc:
    raise ImportError(
        "fvcore is not installed. Run: python -m pip install fvcore"
    ) from exc

# =========================
# SETTINGS
# =========================
SEED = 42
IMAGE_SIZE = 224
NUM_CLASSES = 4
USE_AMP = False          # Keep SAME for every model comparison
WARMUP_RUNS = 100
LATENCY_REPEATS = 5
RUNS_PER_REPEAT = 200    # 5 x 200 = 1000 timed inferences

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "results" / "fdnet_v2_classification" / "best_fdnet_v2.pt"
OUTPUT_DIR = ROOT_DIR / "results" / "fdnet_v2_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "fdnet_v2_benchmark.csv"
TXT_PATH = OUTPUT_DIR / "fdnet_v2_benchmark.txt"

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# MODEL DEFINITION
# =========================
class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, activation=True):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_channels = max(channels // reduction, 16)
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.maximum_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_network = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_attention = self.shared_network(self.average_pool(x))
        max_attention = self.shared_network(self.maximum_pool(x))
        attention = self.sigmoid(avg_attention + max_attention)
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        average_map = torch.mean(x, dim=1, keepdim=True)
        maximum_map, _ = torch.max(x, dim=1, keepdim=True)
        combined_map = torch.cat([average_map, maximum_map], dim=1)
        attention = self.sigmoid(self.conv(combined_map))
        return x * attention


class CombinedAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4,
                 use_attention=False, dropout_rate=0.0):
        super().__init__()
        hidden_channels = in_channels * expansion

        self.expand = ConvBNAct(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.depthwise = ConvBNAct(hidden_channels, hidden_channels, kernel_size=3,
                                   stride=stride, groups=hidden_channels)
        self.attention = CombinedAttention(hidden_channels) if use_attention else nn.Identity()
        self.project = ConvBNAct(hidden_channels, out_channels, kernel_size=1,
                                 stride=1, activation=False)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBNAct(in_channels, out_channels, kernel_size=1,
                                      stride=stride, activation=False)
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        output = self.expand(x)
        output = self.depthwise(output)
        output = self.attention(output)
        output = self.project(output)
        output = self.dropout(output)
        output = output + identity
        output = self.activation(output)
        return output


class FDNetV2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNAct(3, 48, kernel_size=3, stride=2),
            ConvBNAct(48, 48, kernel_size=3, stride=1)
        )

        self.stage1 = nn.Sequential(
            ConvBNAct(48, 64, kernel_size=3, stride=2),
            ConvBNAct(64, 64, kernel_size=3, stride=1)
        )

        self.stage2 = nn.Sequential(
            InvertedResidualBlock(64, 96, stride=2, expansion=3, use_attention=False),
            InvertedResidualBlock(96, 96, stride=1, expansion=3, use_attention=False),
            InvertedResidualBlock(96, 96, stride=1, expansion=3, use_attention=False)
        )

        self.stage3 = nn.Sequential(
            InvertedResidualBlock(96, 192, stride=2, expansion=4,
                                  use_attention=True, dropout_rate=0.03),
            InvertedResidualBlock(192, 192, stride=1, expansion=4,
                                  use_attention=True, dropout_rate=0.03),
            InvertedResidualBlock(192, 192, stride=1, expansion=4,
                                  use_attention=True, dropout_rate=0.03)
        )

        self.stage4 = nn.Sequential(
            InvertedResidualBlock(192, 320, stride=2, expansion=4,
                                  use_attention=True, dropout_rate=0.05),
            InvertedResidualBlock(320, 320, stride=1, expansion=4,
                                  use_attention=True, dropout_rate=0.05)
        )

        self.final_features = nn.Sequential(
            ConvBNAct(320, 512, kernel_size=1, stride=1),
            CombinedAttention(512)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.final_features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

# =========================
# HELPERS
# =========================
def forward_once(model, x):
    if DEVICE.type == "cuda" and USE_AMP:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(x)
    return model(x)


def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{MODEL_PATH}\n"
            "Change MODEL_PATH at the top if your checkpoint is elsewhere."
        )

    model = FDNetV2(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, checkpoint


def get_parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 ** 2)


def get_flops(model, dummy_input):
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
        for _ in range(100):
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
            repeat_latencies.append(((end - start) * 1000) / RUNS_PER_REPEAT)

    mean_latency = statistics.mean(repeat_latencies)
    std_latency = statistics.stdev(repeat_latencies) if len(repeat_latencies) > 1 else 0.0
    throughput = 1000.0 / mean_latency
    return mean_latency, std_latency, throughput, repeat_latencies


def save_results(result):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    with open(TXT_PATH, "w", encoding="utf-8") as f:
        f.write("FD-NET V2 BENCHMARK RESULTS\n")
        f.write("=" * 60 + "\n")
        for key, value in result.items():
            f.write(f"{key}: {value}\n")

# =========================
# MAIN
# =========================
def main():
    print("=" * 70)
    print("FD-NET V2 BENCHMARK")
    print("=" * 70)
    print("Device        :", DEVICE)
    print("Precision     :", "AMP/FP16" if USE_AMP and DEVICE.type == "cuda" else "FP32")
    print(f"Input size    : 1 x 3 x {IMAGE_SIZE} x {IMAGE_SIZE}")
    if DEVICE.type == "cuda":
        print("GPU           :", torch.cuda.get_device_name(0))
        print("CUDA version  :", torch.version.cuda)
    print("Checkpoint    :", MODEL_PATH)

    print("\nLoading trained model...")
    model, checkpoint = load_trained_model()
    print("Best trained model loaded successfully.")

    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print("Best epoch    :", checkpoint["epoch"])
    if isinstance(checkpoint, dict) and "best_valid_f1" in checkpoint:
        print(f"Best valid F1 : {checkpoint['best_valid_f1']:.6f}")

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

    with torch.inference_mode():
        output = forward_once(model, dummy_input)
    print("Output shape  :", tuple(output.shape))

    if tuple(output.shape) != (1, NUM_CLASSES):
        raise RuntimeError("Unexpected model output shape.")

    total_params, trainable_params = get_parameter_counts(model)
    model_size_mb = get_model_size_mb(model)

    print("\nCalculating FLOPs...")
    gflops, unsupported_ops = get_flops(model, dummy_input)

    print("Running warm-up...")
    warm_up(model, dummy_input)

    print("Measuring peak GPU memory...")
    peak_memory_mb = get_peak_gpu_memory_mb(model, dummy_input)

    print("Measuring inference latency...")
    latency_mean, latency_std, throughput, latency_repeats = get_latency_statistics(model, dummy_input)

    precision_mode = "AMP/FP16" if USE_AMP and DEVICE.type == "cuda" else "FP32"

    result = {
        "Model": "FD-Net V2",
        "Device": str(DEVICE),
        "GPU": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU",
        "Precision_Mode": precision_mode,
        "Input_Size": f"1x3x{IMAGE_SIZE}x{IMAGE_SIZE}",
        "Total_Parameters": total_params,
        "Trainable_Parameters": trainable_params,
        "Parameters_Million": round(total_params / 1e6, 6),
        "Model_Weight_Size_MB": round(model_size_mb, 4),
        "GFLOPs_Estimate": round(gflops, 6) if gflops is not None else "N/A",
        "Latency_Mean_ms_per_image": round(latency_mean, 6),
        "Latency_STD_ms": round(latency_std, 6),
        "Throughput_images_per_sec": round(throughput, 4),
        "Peak_GPU_Memory_MB": round(peak_memory_mb, 4) if peak_memory_mb is not None else "N/A"
    }

    print("\n" + "=" * 70)
    print("FD-NET V2 BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Parameters     : {total_params:,}")
    print(f"Parameters (Million) : {total_params / 1e6:.3f} M")
    print(f"Model Weight Size    : {model_size_mb:.2f} MB")
    print(f"FLOPs Estimate       : {gflops:.3f} GFLOPs" if gflops is not None else "FLOPs Estimate       : N/A")
    print(f"Latency              : {latency_mean:.3f} +/- {latency_std:.3f} ms/image")
    print(f"Throughput           : {throughput:.2f} images/sec")
    print(f"Peak GPU Memory      : {peak_memory_mb:.2f} MB" if peak_memory_mb is not None else "Peak GPU Memory      : N/A")
    print(f"Precision Mode       : {precision_mode}")
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
    print("CSV:", CSV_PATH)
    print("TXT:", TXT_PATH)
    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
