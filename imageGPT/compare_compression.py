"""
Comparison script for standard BGPT vs RAG-enhanced BGPT compression
"""
import sys
import os
import logging
import torch
from pathlib import Path

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
    
    # Run compression
    test_bmp_compression(
        model=model,
        device=device,
        test=True,
        temp_folder="temp_img_standard",
        output_folder="output_img_standard",
        patch_size=CompressionConfig.BMP_PATCH_SIZE
    )
    
    print("=" * 80)
    print("STANDARD COMPRESSION COMPLETED")
    print("=" * 80)
    
    return model, device


def run_rag_compression():
    """Run RAG-enhanced BGPT compression"""
    print("\n" + "=" * 80)
    print("TEST 2: RAG-ENHANCED BGPT COMPRESSION")
    print("=" * 80)
    
    # Run RAG compression (it will handle model loading internally)
    run_rag_bgpt_compression(
        test=True,
        temp_folder="temp_img_rag",
        output_folder="output_img_rag",
        patch_size=RAGBGPTConfig.PATCH_SIZE
    )
    
    print("=" * 80)
    print("RAG COMPRESSION COMPLETED")
    print("=" * 80)


def extract_compression_metrics_from_log(log_file):
    """Extract compression metrics from log file"""
    metrics = {}
    try:
        if not os.path.exists(log_file):
            return metrics
            
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Extract compression ratio
                if "Final compression ratio:" in line or "compression ratio:" in line:
                    # Look for pattern like "ratio: 0.123456"
                    import re
                    match = re.search(r'ratio:\s*([\d.]+)', line)
                    if match:
                        metrics['ratio'] = float(match.group(1))
                
                # Extract compression rate
                if "Final compression rate:" in line or "compression rate:" in line:
                    import re
                    match = re.search(r'rate:\s*([\d.]+)', line)
                    if match:
                        metrics['rate'] = float(match.group(1))
                
                # Extract compressed size
                if "Total compressed size:" in line or "compressed size:" in line:
                    import re
                    match = re.search(r'size:\s*([\d,]+)', line)
                    if match:
                        metrics['compressed_size'] = int(match.group(1).replace(',', ''))
                
                # Extract original size
                if "Total original size:" in line or "original size:" in line:
                    import re
                    match = re.search(r'size:\s*([\d,]+)', line)
                    if match:
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
    
    # Try to extract metrics from log files
    standard_log = "compression.log"
    rag_log = "rag_bgpt_compression.log"
    comparison_log = "compression_comparison.log"
    
    # Read all log files and extract metrics
    all_logs = [standard_log, rag_log, comparison_log]
    standard_metrics = {}
    rag_metrics = {}
    
    for log_file in all_logs:
        if os.path.exists(log_file):
            metrics = extract_compression_metrics_from_log(log_file)
            # Determine which test this is from by checking context
            with open(log_file, 'r') as f:
                content = f.read()
                if "STANDARD BGPT" in content or "BMP COMPRESSION TEST" in content:
                    standard_metrics.update(metrics)
                elif "RAG" in content or "RAG-ENHANCED" in content:
                    rag_metrics.update(metrics)
    
    print("\n" + "=" * 80)
    print("COMPRESSION COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'Standard BGPT':<20} {'RAG BGPT':<20} {'Difference':<20}")
    print("-" * 90)
    
    if standard_metrics and rag_metrics:
        # Compression ratio (higher is better)
        std_ratio = standard_metrics.get('ratio', 0)
        rag_ratio = rag_metrics.get('ratio', 0)
        ratio_diff = rag_ratio - std_ratio
        ratio_pct = (ratio_diff / std_ratio * 100) if std_ratio > 0 else 0
        print(f"{'Compression Ratio':<30} {std_ratio:<20.6f} {rag_ratio:<20.6f} {ratio_diff:+.6f} ({ratio_pct:+.2f}%)")
        
        # Compression rate (lower is better - percentage of original)
        std_rate = standard_metrics.get('rate', 0)
        rag_rate = rag_metrics.get('rate', 0)
        rate_diff = rag_rate - std_rate
        rate_pct = (rate_diff / std_rate * 100) if std_rate > 0 else 0
        print(f"{'Compression Rate':<30} {std_rate:<20.6f} {rag_rate:<20.6f} {rate_diff:+.6f} ({rate_pct:+.2f}%)")
        
        # Compressed size
        std_compressed = standard_metrics.get('compressed_size', 0)
        rag_compressed = rag_metrics.get('compressed_size', 0)
        size_diff = rag_compressed - std_compressed
        size_pct = (size_diff / std_compressed * 100) if std_compressed > 0 else 0
        print(f"{'Compressed Size (bytes)':<30} {std_compressed:<20,} {rag_compressed:<20,} {size_diff:+,} ({size_pct:+.2f}%)")
        
        # Original size (should be same)
        std_original = standard_metrics.get('original_size', 0)
        rag_original = rag_metrics.get('original_size', 0)
        print(f"{'Original Size (bytes)':<30} {std_original:<20,} {rag_original:<20,}")
        
        print("\n" + "=" * 80)
        print("INTERPRETATION:")
        print("=" * 80)
        if ratio_diff > 0:
            print(f"✓ RAG compression achieved {ratio_diff:.6f} better compression ratio ({ratio_pct:+.2f}% improvement)")
            print(f"  This means RAG compressed {abs(ratio_diff):.6f}x more effectively")
        elif ratio_diff < 0:
            print(f"✗ RAG compression achieved {abs(ratio_diff):.6f} worse compression ratio ({abs(ratio_pct):.2f}% worse)")
            print(f"  Standard compression was {abs(ratio_diff):.6f}x more effective")
        else:
            print("= Both methods achieved similar compression ratios")
        
        if size_diff < 0:
            print(f"✓ RAG produced {abs(size_diff):,} fewer bytes ({abs(size_pct):.2f}% smaller)")
        elif size_diff > 0:
            print(f"✗ RAG produced {size_diff:,} more bytes ({size_pct:.2f}% larger)")
        else:
            print("= Both methods produced the same compressed size")
    else:
        print("\n⚠ Could not extract metrics from log files.")
        print("Please check the log files manually:")
        print(f"  - Standard: {standard_log}")
        print(f"  - RAG: {rag_log}")
        print(f"  - Comparison: {comparison_log}")
    
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
    
    # Compare results
    compare_results()
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

