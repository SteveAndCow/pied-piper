# ImageGPT Image Compression

Image compression system using OpenAI's ImageGPT model and arithmetic coding for lossless compression.

## Overview

This system compresses images using ImageGPT with two modes:

### Lossless Mode (Default)
1. **Channel Splitting**: Splits RGB image into separate R, G, B channels
2. **Lossless Mapping**: Maps channel values 0-255 directly to vocab tokens 0-255 (no quantization)
3. **Parallel Compression**: Compresses each channel separately (optionally in parallel)
4. **Decompression**: Reconstructs original image with perfect pixel accuracy

### Quantized Mode (Original)
1. **Preprocessing**: Resizes images to 24×24 pixels and applies color quantization (16.7M colors → 512 clusters)
2. **Tokenization**: Converts quantized pixels to token sequences using ImageGPT's vocabulary
3. **Compression**: Uses ImageGPT's probability predictions with arithmetic coding to compress tokens
4. **Decompression**: Reverses the process to reconstruct the original quantized image

## Pipeline

```
Original Image (any size)
    ↓
Resize to 24×24
    ↓
Color Quantization (512 clusters)
    ↓
Token Sequence (576 tokens)
    ↓
ImageGPT Probabilities
    ↓
Arithmetic Coding
    ↓
compressed.bin
    ↓
Decompression
    ↓
Reconstructed Image (24×24)
```

## Files

- `compress_image.py` - Main compression/decompression script
- `test/pixel.png` - Test image (24×24 pixels recommended)
- `requirements.txt` - Python dependencies

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Note: Requires arithmetic_coder module (from RAGLMCompress/arithmetic_coder/)
# Copy the arithmetic_coder directory to this folder if needed
```

## Usage

### Lossless Mode (Default - No Quantization)

Compress images by splitting RGB channels and compressing each separately. This preserves all 256 values per channel (lossless):

```bash
# Basic usage (lossless mode, default)
python compress_image.py

# Specify custom image
python compress_image.py --image path/to/your/image.png

# Disable parallel processing (sequential)
python compress_image.py --no-parallel

# Custom output filename
python compress_image.py --image test.png --output my_compressed
```

### Quantized Mode (Original ImageGPT Approach)

Uses ImageGPT's color quantization (512 clusters):

```bash
# Quantized mode with default 24x24 size
python compress_image.py --mode quantized

# Custom image size
python compress_image.py --mode quantized --size 32 32
```

### Command-Line Arguments

- `--mode`: Compression mode - `lossless` (default) or `quantized`
- `--image`: Path to input image (default: `test/pixel.png`)
- `--output`: Output filename prefix (default: based on input image name)
- `--no-parallel`: Disable parallel processing of channels (lossless mode only)
- `--size HEIGHT WIDTH`: Target image size (default: `24 24`, only used in quantized mode)

### Output

- `compressed.bin` - Compressed binary file
- `compressed-pixel.png` - Decompressed/reconstructed image
- `compression_comparison.png` - Side-by-side comparison (if matplotlib enabled)

## How It Works

### 1. Image Preprocessing

```python
# Resize to 24×24 and quantize colors
inputs = image_processor(images=image, return_tensors="pt", size={"height": 24, "width": 24})
input_ids = inputs["input_ids"]  # Token sequence with SOS token
```

- **Target Size**: 24×24 pixels (576 pixels total)
- **Color Quantization**: ImageGPT uses k-means clustering to reduce colors from 16.7M to 512 clusters
- **Token Format**: Each pixel becomes 1 token (value 0-511), plus 1 SOS token

### 2. Compression

```python
# Get probability distributions from ImageGPT
logits = model(input_ids).logits[:, :-1]
probs = logits.softmax(dim=-1)

# Compress using arithmetic coding
compressed_bytes = arithmetic_encode(tokens, probs)
```

**Key Steps**:
- Forward pass through ImageGPT to get probability distributions
- For each token position, get `P(token | previous tokens)`
- Use arithmetic coding with these probabilities to compress
- Compute theoretical compression: `bits = -log2(P(actual_token))`

**File Format** (`compressed.bin`):
```
[1 byte: padding_bits] [2 bytes: sequence_length] [compressed_data]
```

### 3. Decompression

```python
# Decode using stored probability distributions
decompressed_tokens = arithmetic_decode(compressed_bytes, stored_probs)
```

**Critical**: Uses **stored probability distributions** from encoding (not recomputed) for lossless reconstruction.

### 4. Image Reconstruction

```python
# Convert tokens back to RGB pixels using color clusters
for each token:
    rgb = clusters[token]  # Look up RGB value from cluster
    image[i, j] = rgb
```

## Compression Metrics

### Bits Calculation

- **Original bits**: `H × W × 3 × 8 = 24 × 24 × 3 × 8 = 13,824 bits`
- **Compressed bits**: Sum of `-log2(P(token))` for all tokens
- **Compression ratio**: `compressed_bits / original_bits`
- **Compression rate**: `original_bits / compressed_bits` (e.g., 5.26x means 5.26× smaller)

### Example Output

```
Compression ratio: 0.1900 (19.00% of original)
Compression rate: 5.26x
Reconstruction: ✓ Perfect match
```

## Key Design Choices

### Why 24×24?

- ImageGPT's context window: 1024 tokens
- 24×24 = 576 pixels + 1 SOS = 577 tokens < 1024 ✓
- Balances image detail vs. computational efficiency

### Why Color Quantization?

- Without quantization: Each pixel = 3 bytes (RGB) = 24 bits
- With quantization: Each pixel = 1 token (512 options) = 9 bits
- Reduces sequence length while maintaining reasonable quality

### Lossless vs. Lossy

- **Lossless token compression**: Arithmetic coding ensures perfect token reconstruction
- **Lossy preprocessing**: Color quantization and resizing reduce image quality before compression
- Final image matches the **quantized original**, not the full-resolution original

## Code Structure

### Main Functions

- `compress_image()` - Compresses token sequence using ImageGPT probabilities
- `decode_image()` - Decompresses tokens back to sequence
- `tokens_to_image()` - Converts token sequence to PIL Image
- `write_padded_bytes()` / `read_padded_bytes()` - File I/O with metadata

### Classes

- `Metric` - Tracks compression statistics (original bits, compressed bits, ratios)

## Limitations

1. **Small Image Size**: Only 24×24 pixels due to model constraints
2. **Color Limitation**: Maximum 512 colors (quantization)
3. **Lossy Preprocessing**: Original image quality is reduced before compression
4. **Model Dependency**: Requires ImageGPT model weights (~350MB download)

## Dependencies

- `torch` - PyTorch for model inference
- `transformers` - Hugging Face transformers (ImageGPT)
- `Pillow` - Image processing
- `numpy` - Numerical operations
- `arithmetic_coder` - Custom arithmetic coding module

## References

- [ImageGPT Paper](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf) - Generative Pretraining from Pixels
- [Hugging Face ImageGPT](https://huggingface.co/docs/transformers/model_doc/imagegpt) - Model documentation

## Example Workflow

```python
# 1. Load and preprocess
image = Image.open("test/pixel.png")
inputs = image_processor(images=image, size={"height": 24, "width": 24})
tokens = inputs["input_ids"]  # Shape: [1, 577]

# 2. Get model predictions
logits = model(tokens).logits  # Shape: [1, 577, 512]

# 3. Compress
compressed_bytes = compress_image(tokens, logits, ...)

# 4. Decompress
decompressed_tokens = decode_image(compressed_bytes, ...)

# 5. Reconstruct image
image = tokens_to_image(decompressed_tokens, clusters, 24, 24)
```

## Troubleshooting

**CUDA out of memory**
- Model runs on CPU by default, should work fine
- For GPU: Ensure sufficient VRAM for ImageGPT-small model

**Image quality issues**
- Expected: Output is 24×24 with 512 colors
- This is a limitation of ImageGPT's design, not a bug

