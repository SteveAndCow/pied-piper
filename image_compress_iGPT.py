# image_compress_iGPT.py
"""
Image compression baseline using an ImageGPT-style causal transformer
and arithmetic coding.

Requirements (pip):
  pip install torch torchvision scikit-learn pillow numpy transformers

Notes:
- This script attempts to use Hugging Face ImageGPT model "openai/imagegpt-small".
  If your environment uses a different image-GPT checkpoint, change MODEL_NAME.
- The arithmetic_coder and ac_utils modules are assumed to be available (same as your text code).
  If not available, replace with any arithmetic coding library that uses:
    Encoder(base, precision, output_fn), Decoder(base, precision, input_fn),
    ac_utils.normalize_pdf_for_arithmetic_coding(pdf, dtype),
    ac_utils.bits_to_bytes(bitstring) -> (bytes, num_padded_bits),
    ac_utils.bytes_to_bits(bytes, num_padded_bits=num_padded_bits) -> generator of '0'/'1'.

- This follows the iGPT approach: create a k=512 color palette (kmeans) across input images,
  map each pixel to a palette index (0..511), feed the index sequence to the model,
  use model logits to arithmetic encode the sequence (autoregressive).
"""

import os
import argparse
import time
from glob import glob
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torchvision import transforms
try:
    from transformers import ImageGPTForCausalLM, ImageGPTConfig
except ImportError:
    from transformers import AutoModelForCausalLM as ImageGPTForCausalLM
    ImageGPTConfig = None

# replace these with your arithmetic coder module paths/names
import arithmetic_coder
from arithmetic_coder import ac_utils

# -----------------------------
# Utilities
# -----------------------------
def load_images_from_folder(folder: str, limit: int = 10, ext=("png","jpg","jpeg")) -> List[Image.Image]:
    paths = []
    for e in ext:
        paths.extend(glob(os.path.join(folder, f"**/*.{e}"), recursive=True))
    paths = sorted(paths)[:limit]
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append((p, img))
    return imgs

def fit_palette(images: List[Image.Image], k: int = 512, sample_pixels: int = 200000) -> np.ndarray:
    """Fit kmeans palette on pixels sampled across images."""
    all_pixels = []
    for _, img in images:
        arr = np.asarray(img).reshape(-1, 3)
        all_pixels.append(arr)
    all_pixels = np.vstack(all_pixels)
    rng = np.random.default_rng(1234)
    if all_pixels.shape[0] > sample_pixels:
        idx = rng.choice(all_pixels.shape[0], sample_pixels, replace=False)
        sample = all_pixels[idx]
    else:
        sample = all_pixels
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(sample.astype(np.float32))
    palette = kmeans.cluster_centers_.astype(np.uint8)  # shape (k,3)
    return palette, kmeans

def quantize_image_to_palette(img: Image.Image, kmeans) -> np.ndarray:
    """Return a 2D array (H,W) with palette indices [0..k-1]."""
    arr = np.asarray(img).astype(np.float32).reshape(-1, 3)
    labels = kmeans.predict(arr)
    return labels.reshape(img.height, img.width)

def image_to_sequence_indices(index_map: np.ndarray) -> np.ndarray:
    """Flatten HxW -> sequence of length H*W"""
    return index_map.reshape(-1).astype(np.int64)

# -----------------------------
# Compression / Decompression functions (adapted from your text code)
# -----------------------------
class Metric:
    def __init__(self):
        self.total_length = 0
        self.compressed_length = 0

    def compute_ratio(self):
        if self.total_length != 0 and self.compressed_length != 0:
            return (self.total_length / self.compressed_length, self.compressed_length / self.total_length)
        else:
            return 0, 0

    def accumulate(self, compressed, original):
        # compressed in bytes length + padded bits
        if isinstance(compressed, int):
            self.compressed_length += compressed
        else:
            self.compressed_length += len(compressed)
        if isinstance(original, int):
            self.total_length += original
        else:
            self.total_length += len(original)

def compress_sequence(seq: np.ndarray, model_logits: np.ndarray, metric: Metric, vocab_size: int = 512):
    """
    seq: 1D np array of token indices (length L)
    model_logits: np.array of shape (L, vocab_size) giving p(token_t | prefix) for each position t
        Note: for autoregressive compression we need, for each t, the model's predicted pdf over tokens
        when conditioning on previous tokens.
    This function performs arithmetic encoding and returns bytes, padded_bits.
    """
    L = seq.shape[0]
    output_bits = []
    encoder = arithmetic_coder.Encoder(base=2, precision=64, output_fn=output_bits.append)

    # model_logits expected to be probabilities (not logits)
    # if not, softmax externally before calling

    for symbol, probs in zip(seq, model_logits):
        # normalize & convert to required dtype
        pdf = ac_utils.normalize_pdf_for_arithmetic_coding(probs.astype(np.float32), np.float32)
        encoder.encode(pdf, int(symbol))
    encoder.terminate()
    bits_str = "".join(map(str, output_bits))
    compressed_bytes, num_padded = ac_utils.bits_to_bytes(bits_str)
    metric.accumulate(len(compressed_bytes) + num_padded, len(seq))
    return compressed_bytes, num_padded

def decode_sequence_to_image(
    compressed_file: str,
    kmeans,
    model_wrapper,
    device,
    out_path: str,
    image_size: Tuple[int, int],
):
    """
    Decodes a compressed .imgc file back into an image using the same model and palette.
    """
    # ---- Step 1: read file header ----
    with open(compressed_file, "rb") as f:
        pad_bits = int.from_bytes(f.read(1), "big")
        original_len = int.from_bytes(f.read(2), "big")
        data = f.read()

    print(f"Decoding {compressed_file}: length={original_len}, pad_bits={pad_bits}")

    # ---- Step 2: bit iterator for decoder ----
    bit_iter = iter(ac_utils.bytes_to_bits(data, num_padded_bits=pad_bits))

    def _input_fn(bit_seq=bit_iter):
        try:
            return int(next(bit_seq))
        except StopIteration:
            return None

    decoder = arithmetic_coder.Decoder(base=2, precision=64, input_fn=_input_fn)

    # ---- Step 3: iterative decode ----
    generated = [0]  # start token (you used 0 as start)
    vocab_size = model_wrapper.vocab_size

    print(f"Decoding {original_len} tokens (this may take a while, ~{original_len} model inferences)...")
    for t in range(original_len):
        if (t + 1) % 100 == 0 or t == 0:
            print(f"  Progress: {t+1}/{original_len} tokens ({100*(t+1)/original_len:.1f}%)", end='\r')
        prefix = torch.tensor([generated], dtype=torch.long, device=device)
        with torch.no_grad():
            probs_full = model_wrapper.get_prefix_conditioned_probs(prefix)
        probs = probs_full[0, -1]  # next-token probabilities

        pdf = ac_utils.normalize_pdf_for_arithmetic_coding(probs.astype(np.float32), np.float32)
        sym = decoder.decode(pdf)
        generated.append(int(sym))
    
    print(f"\n✅ Completed decoding {original_len} tokens")
    sequence = np.array(generated[1:], dtype=np.int64)  # remove start token
    print("Decoded sequence length:", len(sequence))

    # ---- Step 4: reconstruct image from token sequence ----
    palette = kmeans.cluster_centers_.astype(np.uint8)
    H, W = image_size
    img_array = palette[sequence].reshape(H, W, 3)
    img = Image.fromarray(img_array)
    img.save(out_path)
    print(f"✅ Reconstructed image saved to {out_path}")


def decode_sequence(compressed_bytes: bytes, num_padded_bits: int, model_fn, start_seq: np.ndarray, seq_len: int, device, vocab_size: int = 512):
    """
    Decode using arithmetic decoder and model_fn which returns pdf for next token given prefix.
    model_fn(prefix_tensor) -> numpy array of shape (prefix_len, vocab_size) giving pdf per position
    start_seq is the initial prefix used (maybe empty)
    """
    data_iter = iter(ac_utils.bytes_to_bits(compressed_bytes, num_padded_bits=num_padded_bits))

    def _input_fn(bit_seq=data_iter):
        try:
            return int(next(bit_seq))
        except StopIteration:
            return None

    decoder = arithmetic_coder.Decoder(base=2, precision=64, input_fn=_input_fn)

    generated = list(start_seq.flatten())  # prefix tokens
    # decode tokens sequentially
    for t in range(seq_len - len(start_seq)):
        # obtain pdf for position len(generated) (model should return probability of next token given prefix)
        prefix = torch.tensor([generated], dtype=torch.long, device=device)  # (1, prefix_len)
        with torch.no_grad():
            logits = model_fn(prefix)  # should return (prefix_len, vocab_size) or just last-step probs
        # We expect logits to be (1, prefix_len, vocab_size) - use last row
        if isinstance(logits, np.ndarray):
            if logits.ndim == 3:
                probs = logits[0, -1]
            elif logits.ndim == 2:
                probs = logits[-1]
        else:
            raise RuntimeError("model_fn must return numpy logits/probs")
        probs = probs.astype(np.float32)
        pdf = ac_utils.normalize_pdf_for_arithmetic_coding(probs, np.float32)
        sym = decoder.decode(pdf)
        generated.append(int(sym))
    return np.array(generated, dtype=np.int64)

# -----------------------------
# Model wrapper
# -----------------------------
from transformers import AutoConfig, ImageGPTModel
try:
    from transformers import ImageGPTForCausalLM
except ImportError:
    # Fallback if ImageGPTForCausalLM doesn't exist
    ImageGPTForCausalLM = None

class ImageGPTWrapper:
    def __init__(self, model_name="openai/imagegpt-small", device="cpu"):
        self.device = torch.device(device)
        try:
            # Try to use ImageGPTForCausalLM first (has LM head)
            if ImageGPTForCausalLM is not None:
                try:
                    self.model = ImageGPTForCausalLM.from_pretrained(model_name).to(self.device)
                    self.model.eval()
                    self.vocab_size = self.model.config.vocab_size
                    self.has_lm_head = True
                    print(f"✅ Loaded ImageGPTForCausalLM ({model_name}) with vocab size {self.vocab_size}")
                except Exception:
                    # Fallback to ImageGPTModel if ForCausalLM fails
                    self.model = ImageGPTModel.from_pretrained(model_name).to(self.device)
                    self.model.eval()
                    self.vocab_size = self.model.config.vocab_size
                    self.has_lm_head = False
                    # Create a projection layer for hidden states -> vocab logits
                    hidden_size = self.model.config.n_embd if hasattr(self.model.config, 'n_embd') else self.model.config.hidden_size
                    self.lm_head = torch.nn.Linear(hidden_size, self.vocab_size).to(self.device)
                    # Initialize with reasonable weights
                    torch.nn.init.normal_(self.lm_head.weight, std=0.02)
                    print(f"✅ Loaded ImageGPTModel ({model_name}) with vocab size {self.vocab_size}, using manual projection")
            else:
                # Fallback when ImageGPTForCausalLM import fails
                self.model = ImageGPTModel.from_pretrained(model_name).to(self.device)
                self.model.eval()
                self.vocab_size = self.model.config.vocab_size
                self.has_lm_head = False
                # Create a projection layer for hidden states -> vocab logits
                hidden_size = self.model.config.n_embd if hasattr(self.model.config, 'n_embd') else self.model.config.hidden_size
                self.lm_head = torch.nn.Linear(hidden_size, self.vocab_size).to(self.device)
                torch.nn.init.normal_(self.lm_head.weight, std=0.02)
                print(f"✅ Loaded ImageGPTModel ({model_name}) with vocab size {self.vocab_size}, using manual projection")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load ImageGPT model '{model_name}'.") from e

    def get_prefix_conditioned_probs(self, prefix_tensor: torch.LongTensor) -> np.ndarray:
        with torch.no_grad():
            if self.has_lm_head:
                # Use the built-in language modeling head
                outputs = self.model(prefix_tensor)
                logits = outputs.logits  # shape (1, seq_len, vocab_size)
            else:
                # Manually project hidden states to vocabulary logits
                outputs = self.model(prefix_tensor)
                hidden_states = outputs.last_hidden_state  # shape (1, seq_len, hidden_size)
                logits = self.lm_head(hidden_states)  # shape (1, seq_len, vocab_size)
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)  # shape (1, seq_len, vocab_size)
        return probs.detach().cpu().numpy()


# -----------------------------
# Main: compress N images and report metrics
# -----------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_wrapper = ImageGPTWrapper(model_name=args.model_name, device=device)

    imgs = load_images_from_folder(args.image_folder, limit=args.limit)
    if len(imgs) == 0:
        raise RuntimeError(f"No images found in {args.image_folder}")

    print(f"Found {len(imgs)} images. Fitting palette (k={args.palette_k}) on them...")
    palette, kmeans = fit_palette(imgs, k=args.palette_k, sample_pixels=args.sample_pixels)
    
    # Save palette for later decoding (CRITICAL: must use same palette for decode)
    if args.save_palette:
        palette_path = args.save_palette
        np.save(palette_path, kmeans.cluster_centers_)
        print(f"✅ Saved palette to {palette_path} (k={kmeans.cluster_centers_.shape[0]})")
        print("   NOTE: You MUST use this same palette file for decoding!")

    metric = Metric()
    results = []

    for idx, (path, pil_img) in enumerate(imgs):
        start_time = time.time()
        # optionally resize to target resolution (paper works with 32 or 48 or 64). We'll resize to args.res
        if args.res is not None:
            pil_img = pil_img.resize((args.res, args.res), Image.BILINEAR)

        index_map = quantize_image_to_palette(pil_img, kmeans)  # H x W of ints in [0,k-1]
        seq = image_to_sequence_indices(index_map)  # length H*W

        # Now produce model-conditioned probabilities for each token in autoregressive order.
        # We'll compress the whole sequence; for autoreg, we must feed prefixes and request pdfs for next token.
        # For speed, produce prefix probabilities incrementally but batched.
        L = seq.shape[0]
        probs_per_position = np.zeros((L, model_wrapper.vocab_size), dtype=np.float32)

        # We will feed the model with growing prefixes. For efficiency, feed in sliding batches:
        # to keep it simple: generate probs for each prefix length 0..L-1 (costly) OR use teacher-forcing:
        # For compression we need p(x_t | x_<t), so we can compute logits by feeding full sequence shifted right.
        # Create input tensor = [start_token, seq[:-1]] where start_token=some token (e.g., 0).
        # Here we choose 0 as a start token and assume model was trained on same vocabulary mapping.
        start_token = 0
        input_seq = np.concatenate([[start_token], seq[:-1]])  # length L
        input_tensor = torch.tensor(input_seq, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)

        # Get model probs for all positions in one forward pass (if memory permits)
        probs_all = model_wrapper.get_prefix_conditioned_probs(input_tensor)  # shape (1, L, V)
        probs_all = probs_all[0]  # (L, V)

        # Now compress: we encode true seq tokens given probs_all at each position
        compressed_bytes, num_padded = compress_sequence(seq, probs_all, metric, vocab_size=model_wrapper.vocab_size)

        # write compressed to file
        base = os.path.splitext(os.path.basename(path))[0]
        out_fname = os.path.join(args.out_dir, f"{base}.imgc")
        # file format: [1 byte padded_bits][2 bytes original_length][payload bytes]
        with open(out_fname, "wb") as f:
            f.write(num_padded.to_bytes(1, "big"))
            f.write((L).to_bytes(2, "big"))
            f.write(compressed_bytes)

        elapsed = time.time() - start_time
        print(f"[{idx+1}/{len(imgs)}] {path} -> {out_fname}  seq_len={L}  bytes={len(compressed_bytes)} padded_bits={num_padded} time={elapsed:.2f}s")
        results.append((path, L, len(compressed_bytes), num_padded, elapsed))

    # Report summary
    compress_rate, compress_ratio = metric.compute_ratio()
    print("=== SUMMARY ===")
    print(f"Total raw tokens: {metric.total_length}")
    print(f"Total compressed bytes+pad: {metric.compressed_length}")
    print(f"Compression rate (raw tokens / compressed length): {compress_rate:.6f}")
    print(f"Compression ratio (compressed / original): {compress_ratio:.6f}")

    for r in results:
        print(r)

def decode_main(args):
    """Decode a compressed .imgc file back to an image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the same model used for compression
    model_wrapper = ImageGPTWrapper(model_name=args.model_name, device=device)
    
    # Load the same palette (CRITICAL: must match the palette used during compression)
    if args.palette_file:
        # Load saved palette
        print(f"Loading palette from {args.palette_file}...")
        centers = np.load(args.palette_file)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=centers.shape[0])
        kmeans.cluster_centers_ = centers
        print(f"✅ Loaded palette with k={centers.shape[0]} colors")
    elif args.image_folder:
        # Recreate palette from images (may not match exactly due to randomness)
        print(f"⚠️  WARNING: Recreating palette from images - may not match compression palette exactly!")
        print(f"   Consider using --palette_file with a saved palette instead.")
        print(f"Loading images from {args.image_folder} to fit palette...")
        imgs = load_images_from_folder(args.image_folder, limit=args.limit)
        if len(imgs) == 0:
            raise RuntimeError(f"No images found in {args.image_folder}")
        palette, kmeans = fit_palette(imgs, k=args.palette_k, sample_pixels=args.sample_pixels)
    else:
        raise RuntimeError("Either --palette_file or --image_folder must be provided for decoding")
    
    # Determine image size from args.res
    if args.res and args.res > 0:
        image_size = (args.res, args.res)
    else:
        # Try to infer from file or use default
        image_size = (32, 32)
        print(f"Warning: --res not specified, using default {image_size}")
    
    # Decode the file
    compressed_file = args.compressed_file
    out_img = args.output_image
    
    os.makedirs(os.path.dirname(out_img) if os.path.dirname(out_img) else ".", exist_ok=True)
    
    decode_sequence_to_image(
        compressed_file=compressed_file,
        kmeans=kmeans,
        model_wrapper=model_wrapper,
        device=device,
        out_path=out_img,
        image_size=image_size,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image compression/decompression using ImageGPT")
    
    # Common arguments
    parser.add_argument("--model_name", type=str, default="openai/imagegpt-small", help="HF ImageGPT model name")
    parser.add_argument("--palette_k", type=int, default=512, help="kmeans palette size")
    parser.add_argument("--res", type=int, default=32, help="resize images to square resolution (e.g., 32, 48, 64). Set 0 or None to keep original.")
    parser.add_argument("--sample_pixels", type=int, default=200000, help="pixels sampled for kmeans")
    
    # Mode selection
    parser.add_argument("--decode", action="store_true", help="Enable decode mode")
    
    # Compression arguments
    parser.add_argument("--image_folder", type=str, help="folder with images (recursive scan). Required for compression.")
    parser.add_argument("--limit", type=int, default=10, help="number of images to compress")
    parser.add_argument("--out_dir", type=str, default="compressed_out", help="output directory for compressed files")
    parser.add_argument("--save_palette", type=str, help="Save palette to .npy file (recommended for later decoding)")
    
    # Decode arguments
    parser.add_argument("--compressed_file", type=str, help="Path to compressed .imgc file (required for decode)")
    parser.add_argument("--output_image", type=str, help="Path for decoded output image (required for decode)")
    parser.add_argument("--palette_file", type=str, help="Path to saved palette .npy file (recommended for decode). If not provided, will recreate from --image_folder (may not match exactly).")
    
    args = parser.parse_args()
    
    if args.decode:
        # Decode mode
        if not args.compressed_file:
            parser.error("--compressed_file is required when --decode is used")
        if not args.output_image:
            parser.error("--output_image is required when --decode is used")
        if not args.palette_file and not args.image_folder:
            parser.error("Either --palette_file or --image_folder must be provided for decode mode (--palette_file is recommended)")
        decode_main(args)
    else:
        # Compression mode (default)
        if not args.image_folder:
            parser.error("--image_folder is required for compression mode")
        os.makedirs(args.out_dir, exist_ok=True)
        main(args)
        
    # ============================================
    # EXAMPLE: How to decode a compressed file
    # ============================================
    # 
    # RECOMMENDED WORKFLOW:
    # 
    # 1. Compress images and save palette:
    #    python image_compress_iGPT.py \\
    #      --image_folder ./images \\
    #      --limit 10 \\
    #      --res 32 \\
    #      --save_palette palette.npy
    # 
    # 2. Decode using saved palette:
    #    python image_compress_iGPT.py --decode \\
    #      --compressed_file "compressed_out/images.imgc" \\
    #      --output_image "decoded_out/images_decoded.png" \\
    #      --palette_file "palette.npy" \\
    #      --res 32
    #
    # ============================================
    # NOTES & GOTCHAS
    # ============================================
    # 
    # 1. SAME PALETTE (CRITICAL):
    #    You MUST reuse the same k-means model (kmeans) from encoding —
    #    otherwise the token→color mapping changes and decoding will fail.
    #    
    #    Always use --save_palette during compression and --palette_file during decode.
    #    The palette file stores kmeans.cluster_centers_ as a .npy file.
    # 
    # 2. MODEL CONSISTENCY:
    #    The decoder must use the same ImageGPT model checkpoint 
    #    (default: openai/imagegpt-small) as the encoder.
    #    Use --model_name to specify if using a different model.
    # 
    # 3. START TOKEN:
    #    This code assumes start_token = 0 was used during compression.
    #    Both encode and decode use the same start token.
    # 
    # 4. IMAGE SIZE:
    #    You must use the same --res value during decode as during compression.
    #    The decoder needs to know the original image dimensions to reshape the sequence.
    # 
    # 5. VERIFY RESULTS:
    #    After decoding, compare the decoded image (*_decoded.png) with the original.
    #    They should look visually identical if arithmetic coding was lossless.
    #    (Note: quantization to palette colors will cause some loss, but arithmetic
    #     coding itself is lossless.)
    #
    # ============================================
