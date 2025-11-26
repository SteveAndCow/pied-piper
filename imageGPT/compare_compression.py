"""
Comparison script for standard BGPT vs RAG-enhanced BGPT compression
"""
import os

# Allow FAISS + PyTorch (both linked against OpenMP) to coexist on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import logging
import shutil
from glob import glob
from pathlib import Path

import torch
from PIL import Image

# Add paths - ensure we import from the correct locations
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
raglm_path = os.path.join(project_root, 'RAGLMCompress')

# Import using importlib to avoid conflicts with RAGLMCompress/BGPTCompress.py
import importlib.util

# Import standard BGPT compression from imageGPT directory
bgpt_compress_path = os.path.join(current_dir, 'BGPTCompress.py')
spec = importlib.util.spec_from_file_location("BGPTCompress", bgpt_compress_path)
bgpt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bgpt_module)
CompressionConfig = bgpt_module.CompressionConfig
load_bgpt_model = bgpt_module.load_bgpt_model
test_bmp_compression = bgpt_module.test_bmp_compression
Metric = bgpt_module.Metric

# Import RAG module from local imageGPT directory
rag_module_path = os.path.join(current_dir, 'RAG_BGPTCompress.py')
spec = importlib.util.spec_from_file_location("RAG_BGPTCompress", rag_module_path)
rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_module)
RAGBGPTConfig = rag_module.RAGBGPTConfig
run_rag_bgpt_compression = rag_module.run_rag_bgpt_compression

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('compression_comparison.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def run_standard_compression():
    """Run standard BGPT compression without RAG"""
    print("\n" + "=" * 80)
    print("TEST 1: STANDARD BGPT COMPRESSION (NO RAG)")
    print("=" * 80)
    
    device = torch.device(CompressionConfig.DEVICE)
    
    # Load model
    model_checkpoint = CompressionConfig.get_model_checkpoint()
    model = load_bgpt_model(model_checkpoint, device)
    
    # Run compression (skip decompression for speed)
    test_bmp_compression(
        model=model,
        device=device,
        test=True,
        temp_folder="temp_img_standard",
        output_folder="output_img_standard",
        patch_size=CompressionConfig.BMP_PATCH_SIZE,
        skip_decompression=True  # Skip decompression to save time
    )
    
    # Archive standard log before RAG run overwrites it
    try:
        shutil.copyfile("compression.log", "compression_standard.log")
        logger.info("Saved standard compression log to compression_standard.log")
    except FileNotFoundError:
        logger.warning("compression.log not found after standard run; cannot archive log.")
    
    print("=" * 80)
    print("STANDARD COMPRESSION COMPLETED")
    print("=" * 80)
    
    return model, device


def run_rag_compression():
    """Run RAG-enhanced BGPT compression"""
    print("\n" + "=" * 80)
    print("TEST 2: RAG-ENHANCED BGPT COMPRESSION")
    print("=" * 80)
    
    # Run RAG compression (it will handle model loading internally, skip decompression for speed)
    run_rag_bgpt_compression(
        test=True,
        temp_folder="temp_img_rag",
        output_folder="output_img_rag",
        patch_size=RAGBGPTConfig.PATCH_SIZE,
        skip_decompression=True  # Skip decompression to save time
    )
    
    # Archive the log produced during RAG run (BGPT logger reuses compression.log)
    try:
        shutil.copyfile("compression.log", "compression_rag.log")
        logger.info("Saved RAG compression log to compression_rag.log")
    except FileNotFoundError:
        logger.warning("compression.log not found after RAG run; cannot archive log.")
    
    print("=" * 80)
    print("RAG COMPRESSION COMPLETED")
    print("=" * 80)


def write_builtin_log(log_path, label, metrics):
    with open(log_path, "w") as log_file:
        log_file.write(f"{label} compression results\n")
        log_file.write(f"Final compression ratio: {metrics['ratio']:.6f}\n")
        log_file.write(f"Final compression rate: {metrics['rate']:.6f}\n")
        log_file.write(f"Total compressed size: {metrics['compressed_size']:,} bytes\n")
        log_file.write(f"Total original size: {metrics['original_size']:,} bytes\n")


def run_builtin_image_compression(format_name: str, save_kwargs: dict | None = None):
    """Compress BMP dataset into another image format (JPEG/PNG) and report metrics."""
    format_upper = format_name.upper()
    format_lower = format_name.lower()
    save_kwargs = save_kwargs or {}
    
    dataset_path = CompressionConfig.get_dataset_path(test=True)
    image_paths = glob(dataset_path)
    if not image_paths:
        logger.warning(f"No BMP files found for builtin {format_upper} compression in {dataset_path}")
        return None
    
    output_folder = os.path.join("output_img_builtin", format_lower)
    os.makedirs(output_folder, exist_ok=True)
    
    total_original = 0
    total_compressed = 0
    
    print("\n" + "=" * 80)
    print(f"TEST {format_upper}: BUILTIN {format_upper} COMPRESSION")
    print("=" * 80)
    
    for image_path in image_paths:
        original_size = os.path.getsize(image_path)
        total_original += original_size
        
        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_folder, f"{filename}.{format_lower}")
        
        with Image.open(image_path) as img:
            # Convert to appropriate mode for each format
            if format_upper == "PNG":
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")
            img.save(output_path, format=format_upper, **save_kwargs)
        
        compressed_size = os.path.getsize(output_path)
        total_compressed += compressed_size
        
        logger.info(
            f"{format_upper} | {filename} | original: {original_size} bytes | "
            f"compressed: {compressed_size} bytes"
        )
    
    if total_original == 0 or total_compressed == 0:
        logger.warning(f"{format_upper} compression produced zero-length metrics.")
        return None
    
    ratio = total_compressed / total_original
    rate = total_original / total_compressed
    metrics = {
        "ratio": ratio,
        "rate": rate,
        "compressed_size": total_compressed,
        "original_size": total_original,
    }
    
    print(f"\n{format_upper} compression finished")
    print(f"  Overall compression ratio: {ratio:.6f}")
    print(f"  Overall compression rate: {rate:.6f}x")
    print(f"  Total original size: {total_original:,} bytes")
    print(f"  Total compressed size: {total_compressed:,} bytes")
    
    log_path = f"compression_{format_lower}.log"
    write_builtin_log(log_path, f"{format_upper}", metrics)
    logger.info(f"Saved {format_upper} compression log to {log_path}")
    
    return metrics


def extract_compression_metrics_from_log(log_file):
    """Extract compression metrics from log file"""
    metrics = {}
    try:
        if not os.path.exists(log_file):
            return metrics
            
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                import re
                # Extract compression ratio (look for "Final compression ratio: 0.123456")
                if "Final compression ratio:" in line:
                    match = re.search(r'Final compression ratio:\s*([\d.]+)', line)
                    if match:
                        metrics['ratio'] = float(match.group(1))
                elif "compression ratio:" in line and "Final" not in line:
                    match = re.search(r'compression ratio:\s*([\d.]+)', line)
                    if match and 'ratio' not in metrics:
                        metrics['ratio'] = float(match.group(1))
                
                # Extract compression rate (look for "Final compression rate: 2.123456")
                if "Final compression rate:" in line:
                    match = re.search(r'Final compression rate:\s*([\d.]+)', line)
                    if match:
                        metrics['rate'] = float(match.group(1))
                elif "compression rate:" in line and "Final" not in line:
                    match = re.search(r'compression rate:\s*([\d.]+)', line)
                    if match and 'rate' not in metrics:
                        metrics['rate'] = float(match.group(1))
                
                # Extract compressed size (look for "Total compressed size: 1,234 bytes")
                if "Total compressed size:" in line:
                    match = re.search(r'Total compressed size:\s*([\d,]+)', line)
                    if match:
                        metrics['compressed_size'] = int(match.group(1).replace(',', ''))
                elif "compressed size:" in line and "Total" not in line:
                    match = re.search(r'compressed size:\s*([\d,]+)', line)
                    if match and 'compressed_size' not in metrics:
                        metrics['compressed_size'] = int(match.group(1).replace(',', ''))
                
                # Extract original size (look for "Total original size: 1,234 bytes")
                if "Total original size:" in line:
                    match = re.search(r'Total original size:\s*([\d,]+)', line)
                    if match:
                        metrics['original_size'] = int(match.group(1).replace(',', ''))
                elif "original size:" in line and "Total" not in line:
                    match = re.search(r'original size:\s*([\d,]+)', line)
                    if match and 'original_size' not in metrics:
                        metrics['original_size'] = int(match.group(1).replace(',', ''))
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Error reading log file {log_file}: {e}")
    
    return metrics


def compare_results():
    """Compare compression results from both methods"""
    print("\n" + "=" * 80)
    print("COMPRESSION COMPARISON RESULTS")
    print("=" * 80)
    
    configs = [
        ("Standard BGPT", "compression_standard.log"),
        ("RAG BGPT", "compression_rag.log"),
        ("JPEG (builtin)", "compression_jpeg.log"),
        ("PNG (builtin)", "compression_png.log"),
    ]
    
    metrics_by_label = {}
    for label, log_path in configs:
        metrics = extract_compression_metrics_from_log(log_path)
        if metrics:
            metrics_by_label[label] = metrics
            logger.info(f"Extracted metrics from {label} ({log_path})")
        else:
            logger.warning(f"No metrics found for {label} ({log_path})")
    
    if not metrics_by_label:
        print("\n⚠ Could not extract metrics from any log files.")
        print("Please ensure the comparison script ran successfully.")
        print("=" * 80)
        return
    
    print("\n" + "=" * 80)
    print("COMPRESSION COMPARISON SUMMARY")
    print("=" * 80)
    header = (
        f"\n{'Method':<22}"
        f"{'Ratio':>12}"
        f"{'Rate':>12}"
        f"{'Compressed':>15}"
        f"{'Original':>15}"
        f"{'Δ Ratio vs Std':>16}"
        f"{'Δ Bytes vs Std':>18}"
    )
    print(header)
    print("-" * len(header))
    
    standard_metrics = metrics_by_label.get("Standard BGPT")
    for label, _ in configs:
        metrics = metrics_by_label.get(label)
        if not metrics:
            continue
        ratio = metrics.get("ratio", 0)
        rate = metrics.get("rate", 0)
        compressed = metrics.get("compressed_size", 0)
        original = metrics.get("original_size", 0)
        
        if standard_metrics and label != "Standard BGPT":
            ratio_diff = ratio - standard_metrics.get("ratio", 0)
            size_diff = compressed - standard_metrics.get("compressed_size", 0)
            ratio_diff_str = f"{ratio_diff:+.6f}"
            size_diff_str = f"{size_diff:+,}"
        else:
            ratio_diff_str = "-"
            size_diff_str = "-"
        
        print(
            f"{label:<22}"
            f"{ratio:>12.6f}"
            f"{rate:>12.6f}"
            f"{compressed:>15,}"
            f"{original:>15,}"
            f"{ratio_diff_str:>16}"
            f"{size_diff_str:>18}"
        )
    
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    if standard_metrics:
        std_ratio = standard_metrics.get("ratio", 0)
        best_method = min(
            metrics_by_label.items(),
            key=lambda item: item[1].get("ratio", 1e9)
        )
        print(
            f"Baseline (Standard BGPT) ratio: {std_ratio:.6f} "
            f"({standard_metrics.get('compressed_size', 0):,} bytes)"
        )
        print(
            f"Best ratio achieved by {best_method[0]}: "
            f"{best_method[1]['ratio']:.6f} "
            f"({best_method[1]['compressed_size']:,} bytes)"
        )
    else:
        print("Standard BGPT metrics missing; shown values are absolute.")
    
    print("=" * 80)


def main():
    """Main function to run both compression tests and compare"""
    logger.info("\n" + "=" * 80)
    logger.info("STARTING COMPRESSION COMPARISON")
    logger.info("=" * 80)
    logger.info("This will run:")
    logger.info("  1. Standard BGPT compression (no RAG)")
    logger.info("  2. RAG-enhanced BGPT compression")
    logger.info("  3. Comparison of results")
    logger.info("=" * 80)
    
    # Run standard compression
    try:
        run_standard_compression()
    except Exception as e:
        logger.error(f"Error in standard compression: {e}", exc_info=True)
    
    # Run RAG compression
    try:
        run_rag_compression()
    except Exception as e:
        logger.error(f"Error in RAG compression: {e}", exc_info=True)
    
    # Run builtin JPEG/PNG compression for comparison
    try:
        run_builtin_image_compression("JPEG", {"quality": 85, "optimize": True})
        run_builtin_image_compression("PNG", {"optimize": True, "compress_level": 9})
    except Exception as e:
        logger.error(f"Error in builtin compression: {e}", exc_info=True)
    
    # Compare results
    compare_results()
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

