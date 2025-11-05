"""
Image Compression using ImageGPT
Compresses images using ImageGPT model and arithmetic coding.
"""

import logging
import numpy as np
import torch
from transformers import AutoImageProcessor, ImageGPTForCausalImageModeling
from PIL import Image
from typing import Iterator, Tuple, List
from arithmetic_coder import arithmetic_coder, ac_utils
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compress_channel_lossless(
    channel_values: np.ndarray,
    model,
    device,
    channel_name: str = "channel",
    vocab_size: int = 512
) -> Tuple[bytes, int, torch.Tensor, np.ndarray, torch.Tensor]:
    """
    Compress a single channel (R, G, or B) losslessly.
    Maps channel values 0-255 directly to vocab tokens 0-255.
    
    :param channel_values: 1D array of channel values (0-255), shape (H*W,)
    :param model: ImageGPT model
    :param device: device to run model on
    :param channel_name: name of channel for logging
    :param vocab_size: vocabulary size (512 for ImageGPT)
    :return: (compressed_bytes, num_padded_bits, start_symbol, sequence_array, probs)
    """
    # Map channel values 0-255 directly to tokens 0-255 (lossless)
    # Pad tokens 256-511 with zero probability if needed
    tokens = channel_values.astype(np.int64).copy()
    
    # Ensure tokens are in valid range [0, 255] for vocab mapping
    tokens = np.clip(tokens, 0, 255)
    
    # Add SOS token (use 0 as SOS for simplicity, or vocab_size-1)
    sos_token = np.array([vocab_size - 1])  # Use last vocab token as SOS
    token_sequence = np.concatenate([sos_token, tokens])
    
    # Convert to tensor
    input_ids = torch.tensor(token_sequence, dtype=torch.long, device=device).unsqueeze(0)
    
    # Get model predictions
    with torch.no_grad():
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
        model_outputs = model(
            input_ids=input_ids,
            use_cache=False,
            position_ids=position_ids,
            past_key_values=None
        )
        # ImageGPT predicts next token, so we need logits[:-1] to match input_ids[1:]
        logits = model_outputs.logits[:, :-1].to(torch.float32)
    
    # Get probabilities
    probs = logits.softmax(dim=-1).to(torch.float32)
    
    # Create probability distribution for actual tokens (0-255)
    # We'll restrict probabilities to first 256 tokens
    probs_restricted = probs[:, :, :256]  # Only use first 256 tokens
    probs_restricted_sum = probs_restricted.sum(dim=-1, keepdim=True)
    probs_normalized = probs_restricted / (probs_restricted_sum + 1e-10)
    
    # Get actual sequence (excluding SOS)
    sequence_array = tokens
    sequence_tensor = input_ids[:, 1:]  # Exclude SOS
    
    # Get probability of true tokens
    pd = torch.gather(probs_normalized, dim=-1, index=sequence_tensor.unsqueeze(-1)).squeeze(-1)
    pd = pd.squeeze()
    
    # Compute theoretical compressed bits
    compressed_bits_theoretical = -torch.sum(torch.log2(torch.clamp(pd, min=1e-10))).item()
    
    # Compress using arithmetic coding
    output = []
    encoder = arithmetic_coder.Encoder(
        base=2,
        precision=64,
        output_fn=output.append,
    )
    
    probs_np = probs_normalized.squeeze(0).detach().cpu().numpy()  # [seq_len, 256]
    if probs_np.ndim == 1:
        probs_np = probs_np.reshape(1, -1)
    
    for i, (symbol, prob) in enumerate(zip(sequence_array, probs_np)):
        normalized_prob = ac_utils.normalize_pdf_for_arithmetic_coding(prob, np.float32)
        encoder.encode(normalized_prob, int(symbol))
    
    encoder.terminate()
    
    # Convert to bytes
    compressed_bits_str = "".join(map(str, output))
    compressed_bytes, num_padded_bits = ac_utils.bits_to_bytes(compressed_bits_str)
    
    start_symbol = input_ids[:, :1]  # SOS token
    
    logger.info(f"{channel_name}: {len(tokens)} tokens, "
                f"compressed: {compressed_bits_theoretical:.2f} bits, "
                f"actual: {len(compressed_bytes)*8+num_padded_bits:.2f} bits")
    
    return compressed_bytes, num_padded_bits, start_symbol, sequence_array, probs_normalized


def decompress_channel_lossless(
    compressed_bytes: bytes,
    num_padded_bits: int,
    model,
    start_symbol: torch.Tensor,
    device: torch.device,
    original_seq_len: int,
    stored_probs: torch.Tensor = None,
    vocab_size: int = 512
) -> np.ndarray:
    """
    Decompress a single channel losslessly.
    
    :param compressed_bytes: compressed data
    :param num_padded_bits: padded bits
    :param model: ImageGPT model
    :param start_symbol: SOS token
    :param device: device to run model on
    :param original_seq_len: original sequence length (excluding SOS)
    :param stored_probs: stored probability distributions from encoding
    :param vocab_size: vocabulary size (512 for ImageGPT)
    :return: decompressed channel values (0-255)
    """
    # Convert bytes back to bit stream
    data_iter = iter(
        ac_utils.bytes_to_bits(compressed_bytes, num_padded_bits=num_padded_bits)
    )
    
    def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
        try:
            return int(next(bit_sequence))
        except StopIteration:
            return None
    
    decoder = arithmetic_coder.Decoder(
        base=2,
        precision=64,
        input_fn=_input_fn,
    )
    
    # Start with SOS token
    sequence_array_de = start_symbol.squeeze(0).detach().cpu().numpy().copy()
    sequence_array_de_input = start_symbol.to(device)
    
    # Prepare stored probability distributions
    stored_probs_np = None
    if stored_probs is not None:
        stored_probs_np = stored_probs.detach().cpu().numpy()
        if stored_probs_np.ndim == 3:
            stored_probs_np = stored_probs_np[0]  # Remove batch dimension
        if stored_probs_np.ndim == 1:
            stored_probs_np = stored_probs_np.reshape(1, -1)
    
    # Decode tokens one by one
    for i in range(original_seq_len):
        try:
            if stored_probs_np is not None and i < stored_probs_np.shape[0]:
                prob_at_pos = stored_probs_np[i, :256]  # Only first 256 tokens
            else:
                # Fallback: compute from model
                with torch.no_grad():
                    current_seq_len = sequence_array_de_input.shape[1]
                    position_ids = torch.arange(0, current_seq_len, dtype=torch.long, device=device).unsqueeze(0)
                    model_outputs = model(
                        input_ids=sequence_array_de_input,
                        use_cache=False,
                        position_ids=position_ids,
                        past_key_values=None
                    )
                    logits = model_outputs.logits.to(torch.float32)
                
                prob_de = logits.softmax(dim=-1).detach().cpu().numpy()
                if prob_de.ndim == 3:
                    prob_at_pos = prob_de[0, -1, :256]  # Last position, first 256 tokens
                elif prob_de.ndim == 2:
                    prob_at_pos = prob_de[-1, :256]
                else:
                    prob_at_pos = prob_de[:256]
                
                # Normalize to sum to 1
                prob_sum = prob_at_pos.sum()
                if prob_sum > 0:
                    prob_at_pos = prob_at_pos / prob_sum
            
            # Normalize and decode
            normalized_prob = ac_utils.normalize_pdf_for_arithmetic_coding(prob_at_pos, np.float32)
            de_token = decoder.decode(normalized_prob)
            
            # Append to sequence
            sequence_array_de = np.append(sequence_array_de, de_token)
            
            # Update input for next iteration
            if stored_probs_np is None or i >= stored_probs_np.shape[0]:
                max_seq_len = min(len(sequence_array_de), model.config.n_positions)
                sequence_array_de_input = torch.tensor(
                    sequence_array_de[-max_seq_len:], 
                    dtype=torch.long, device=device
                ).unsqueeze(0)
            
        except StopIteration:
            logger.warning(f"Ran out of compressed bits at position {i}/{original_seq_len}")
            # Use model's top prediction for remaining tokens
            remaining = original_seq_len - i
            with torch.no_grad():
                for j in range(remaining):
                    position_ids = torch.arange(0, sequence_array_de_input.shape[1], dtype=torch.long, device=device).unsqueeze(0)
                    model_outputs = model(
                        input_ids=sequence_array_de_input,
                        use_cache=False,
                        position_ids=position_ids,
                        past_key_values=None
                    )
                    logits = model_outputs.logits.to(torch.float32)
                    prob_de = logits.softmax(dim=-1).detach().cpu().numpy().squeeze(0)
                    if prob_de.ndim == 2:
                        de_token = prob_de[-1, :256].argmax()
                    else:
                        de_token = prob_de[:256].argmax()
                    sequence_array_de = np.append(sequence_array_de, de_token)
                    
                    sequence_array_de_input = torch.tensor(
                        sequence_array_de[-min(len(sequence_array_de), model.config.n_positions):], 
                        dtype=torch.long, device=device
                    ).unsqueeze(0)
            break
    
    # Return channel values (excluding SOS token)
    return sequence_array_de[1:].astype(np.uint8)


def compress_image_lossless_channels(
    image: Image.Image,
    model,
    device: torch.device,
    parallel: bool = True,
    vocab_size: int = 512
) -> Tuple[List[bytes], List[int], List[torch.Tensor], List[np.ndarray], List[torch.Tensor], Tuple[int, int]]:
    """
    Compress image by splitting RGB channels and compressing each separately.
    Lossless compression - preserves all 256 values per channel.
    
    :param image: PIL Image
    :param model: ImageGPT model
    :param device: device to run model on
    :param parallel: whether to compress channels in parallel
    :param vocab_size: vocabulary size (512 for ImageGPT)
    :return: (compressed_bytes_list, num_padded_bits_list, start_symbols_list, 
              sequences_list, probs_list, (height, width))
    """
    # Convert image to numpy array
    img_array = np.array(image.convert('RGB'))
    height, width = img_array.shape[0], img_array.shape[1]
    
    # Split into R, G, B channels
    r_channel = img_array[:, :, 0].flatten()
    g_channel = img_array[:, :, 1].flatten()
    b_channel = img_array[:, :, 2].flatten()
    
    channels = {
        'R': r_channel,
        'G': g_channel,
        'B': b_channel
    }
    
    if parallel:
        # Compress channels in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(compress_channel_lossless, ch, model, device, name, vocab_size): name
                for name, ch in channels.items()
            }
            
            results = {}
            for future in as_completed(futures):
                channel_name = futures[future]
                results[channel_name] = future.result()
    else:
        # Compress channels sequentially
        results = {}
        for name, ch in channels.items():
            results[name] = compress_channel_lossless(ch, model, device, name, vocab_size)
    
    # Extract results in R, G, B order
    compressed_bytes_list = [results['R'][0], results['G'][0], results['B'][0]]
    num_padded_bits_list = [results['R'][1], results['G'][1], results['B'][1]]
    start_symbols_list = [results['R'][2], results['G'][2], results['B'][2]]
    sequences_list = [results['R'][3], results['G'][3], results['B'][3]]
    probs_list = [results['R'][4], results['G'][4], results['B'][4]]
    
    return compressed_bytes_list, num_padded_bits_list, start_symbols_list, sequences_list, probs_list, (height, width)


def decompress_image_lossless_channels(
    compressed_bytes_list: List[bytes],
    num_padded_bits_list: List[int],
    start_symbols_list: List[torch.Tensor],
    model,
    device: torch.device,
    original_seq_lens: List[int],
    stored_probs_list: List[torch.Tensor] = None,
    height: int = None,
    width: int = None,
    vocab_size: int = 512,
    parallel: bool = True
) -> Image.Image:
    """
    Decompress image by decompressing each channel separately.
    
    :param compressed_bytes_list: list of compressed data for R, G, B channels
    :param num_padded_bits_list: list of padded bits for each channel
    :param start_symbols_list: list of SOS tokens for each channel
    :param model: ImageGPT model
    :param device: device to run model on
    :param original_seq_lens: list of original sequence lengths for each channel
    :param stored_probs_list: list of stored probability distributions
    :param height: image height
    :param width: image width
    :param vocab_size: vocabulary size (512 for ImageGPT)
    :param parallel: whether to decompress channels in parallel
    :return: PIL Image
    """
    if stored_probs_list is None:
        stored_probs_list = [None, None, None]
    
    if parallel:
        # Decompress channels in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    decompress_channel_lossless,
                    compressed_bytes_list[i],
                    num_padded_bits_list[i],
                    model,
                    start_symbols_list[i],
                    device,
                    original_seq_lens[i],
                    stored_probs_list[i],
                    vocab_size
                ): i
                for i in range(3)
            }
            
            channel_results = {}
            for future in as_completed(futures):
                channel_idx = futures[future]
                channel_results[channel_idx] = future.result()
            
            # Reorder to R, G, B
            r_channel = channel_results[0]
            g_channel = channel_results[1]
            b_channel = channel_results[2]
    else:
        # Decompress channels sequentially
        r_channel = decompress_channel_lossless(
            compressed_bytes_list[0], num_padded_bits_list[0],
            model, start_symbols_list[0], device, original_seq_lens[0],
            stored_probs_list[0], vocab_size
        )
        g_channel = decompress_channel_lossless(
            compressed_bytes_list[1], num_padded_bits_list[1],
            model, start_symbols_list[1], device, original_seq_lens[1],
            stored_probs_list[1], vocab_size
        )
        b_channel = decompress_channel_lossless(
            compressed_bytes_list[2], num_padded_bits_list[2],
            model, start_symbols_list[2], device, original_seq_lens[2],
            stored_probs_list[2], vocab_size
        )
    
    # Reconstruct image
    if height is None or width is None:
        # Infer from channel length
        total_pixels = len(r_channel)
        height = width = int(np.sqrt(total_pixels))
    
    r_channel = r_channel.reshape(height, width)
    g_channel = g_channel.reshape(height, width)
    b_channel = b_channel.reshape(height, width)
    
    img_array = np.stack([r_channel, g_channel, b_channel], axis=2)
    return Image.fromarray(img_array, mode='RGB')


def write_channel_data(filename: str, compressed_bytes_list: List[bytes], 
                       num_padded_bits_list: List[int], original_seq_lens: List[int],
                       height: int, width: int):
    """
    Write compressed channel data to file.
    Format:
    - 2 bytes: height
    - 2 bytes: width
    - For each channel (R, G, B):
      - 1 byte: num_padded_bits
      - 2 bytes: original_seq_len
      - 4 bytes: data_size (size of compressed_data)
      - compressed_data
    """
    with open(filename, 'wb') as f:
        # Write image dimensions
        f.write(height.to_bytes(2, 'big'))
        f.write(width.to_bytes(2, 'big'))
        
        # Write each channel
        for compressed_bytes, num_padded_bits, original_len in zip(
            compressed_bytes_list, num_padded_bits_list, original_seq_lens
        ):
            f.write(num_padded_bits.to_bytes(1, 'big'))
            f.write(original_len.to_bytes(2, 'big'))
            f.write(len(compressed_bytes).to_bytes(4, 'big'))  # Size of compressed data
            f.write(compressed_bytes)


def read_channel_data(filename: str) -> Tuple[List[bytes], List[int], List[int], int, int]:
    """
    Read compressed channel data from file.
    Returns: (compressed_bytes_list, num_padded_bits_list, original_seq_lens, height, width)
    """
    with open(filename, 'rb') as f:
        # Read image dimensions
        height = int.from_bytes(f.read(2), 'big')
        width = int.from_bytes(f.read(2), 'big')
        
        compressed_bytes_list = []
        num_padded_bits_list = []
        original_seq_lens = []
        
        # Read each channel (R, G, B)
        for _ in range(3):
            num_padded_bits = int.from_bytes(f.read(1), 'big')
            original_len = int.from_bytes(f.read(2), 'big')
            data_size = int.from_bytes(f.read(4), 'big')  # Size of compressed data
            compressed_data = f.read(data_size)
            
            num_padded_bits_list.append(num_padded_bits)
            original_seq_lens.append(original_len)
            compressed_bytes_list.append(compressed_data)
    
    return compressed_bytes_list, num_padded_bits_list, original_seq_lens, height, width


def main():
    """Main compression and decompression pipeline (lossless, 24x24)."""
    parser = argparse.ArgumentParser(
        description="Image compression using ImageGPT in lossless RGB channel-split mode"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="test/pixel.png",
        help="Path to input image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename prefix (default: based on input image name)"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing of channels"
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[24, 24],
        metavar=("HEIGHT", "WIDTH"),
        help="Size the image will be resized to BEFORE compression (default: 24 24)"
    )

    args = parser.parse_args()
    
    # Verify GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Load ImageGPT model
    print("\nLoading ImageGPT model...")
    model = ImageGPTForCausalImageModeling.from_pretrained("openai/imagegpt-small")
    model.eval()
    model.to(device)
    vocab_size = model.config.vocab_size
    print(f"Model vocabulary size: {vocab_size}")

    # Load image
    print(f"\nLoading image from {args.image}...")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    
    image = Image.open(args.image).convert("RGB")
    print(f"✓ Successfully loaded image")
    print(f"  Original size: {image.size} (width x height)")

    # Resize image BEFORE compression (this is the "true" image for the pipeline)
    target_height, target_width = args.size
    print(f"\nResizing image to {target_width}x{target_height} for compression...")
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    print(f"  New size for compression: {image.size} (width x height)")

    # Set output filenames (based on resized size)
    suffix = f"{target_width}x{target_height}"
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        compressed_file = f"{base_name}_{suffix}_compressed.bin"
        original_resized_file = f"{base_name}_{suffix}_original.png"
        decompressed_file = f"{base_name}_{suffix}_decompressed.png"
    else:
        prefix = f"{args.output}_{suffix}"
        compressed_file = f"{prefix}_compressed.bin"
        original_resized_file = f"{prefix}_original.png"
        decompressed_file = f"{prefix}_decompressed.png"

    # Save the resized "original 24x24" image
    image.save(original_resized_file)
    print(f"\n✓ Saved resized original image (for reference) to: {original_resized_file}")

    # LOSSLESS MODE (always)
    print("\n" + "="*60)
    print("=== LOSSLESS MODE (Channel-Split) ===")
    print("="*60)
    print("Compressing RGB channels separately (no quantization)")

    # Use the resized image size for lossless compression
    width, height = image.size
    original_bits = height * width * 3 * 8
    print(f"Original bits: {original_bits} (H={height}, W={width}, C=3, bits_per_pixel=8)")

    # COMPRESSION
    print("\n" + "="*60)
    print("=== COMPRESSION ===")
    print("="*60)
    compression_start_time = time.time()

    parallel = not args.no_parallel
    (
        compressed_bytes_list,
        num_padded_bits_list,
        start_symbols_list,
        sequences_list,
        probs_list,
        (img_height, img_width),
    ) = compress_image_lossless_channels(
        image, model, device, parallel=parallel, vocab_size=vocab_size
    )

    compression_end_time = time.time()
    compression_time = compression_end_time - compression_start_time

    # Calculate total compressed bits
    total_compressed_bits = 0
    for i, (comp_bytes, pad_bits) in enumerate(zip(compressed_bytes_list, num_padded_bits_list)):
        channel_bits = len(comp_bytes) * 8 + pad_bits
        total_compressed_bits += channel_bits
        print(f"Channel {['R','G','B'][i]}: {channel_bits:.2f} bits")

    compression_ratio = total_compressed_bits / original_bits
    compression_rate = original_bits / total_compressed_bits if total_compressed_bits > 0 else 0

    print(f"\nTotal compressed bits: {total_compressed_bits:.2f}")
    print(f"Compression ratio: {compression_ratio:.6f} ({compression_ratio*100:.2f}% of original)")
    print(f"Compression rate: {compression_rate:.6f}x")
    print(f"Compression time: {compression_time:.2f} seconds")

    # Save compressed data for all channels
    original_seq_lens = [len(seq) for seq in sequences_list]
    write_channel_data(
        compressed_file,
        compressed_bytes_list,
        num_padded_bits_list,
        original_seq_lens,
        img_height,
        img_width
    )
    print(f"\n✓ Saved compressed data to {compressed_file}")
    total_compressed_size = sum(len(b) for b in compressed_bytes_list)
    print(f"  Total compressed raw size (all channels): {total_compressed_size} bytes ({total_compressed_size / 1024:.2f} KB)")

    # DECOMPRESSION
    print("\n" + "="*60)
    print("=== DECOMPRESSION ===")
    print("="*60)
    decompression_start_time = time.time()

    compressed_bytes_read, num_padded_bits_read, original_seq_lens_read, height_read, width_read = read_channel_data(compressed_file)

    reconstructed_image = decompress_image_lossless_channels(
        compressed_bytes_read,
        num_padded_bits_read,
        start_symbols_list,
        model,
        device,
        original_seq_lens_read,
        stored_probs_list=probs_list,
        height=height_read,
        width=width_read,
        vocab_size=vocab_size,
        parallel=parallel
    )

    decompression_end_time = time.time()
    decompression_time = decompression_end_time - decompression_start_time
    print(f"✓ Decompression completed in {decompression_time:.2f} seconds")

    # VERIFICATION (both 24x24)
    original_array = np.array(image)
    reconstructed_array = np.array(reconstructed_image)

    print("\n" + "="*60)
    print("=== VERIFICATION (24x24) ===")
    print("="*60)
    if original_array.shape == reconstructed_array.shape:
        matches = np.array_equal(original_array, reconstructed_array)
        print(f"Images match exactly: {matches}")
        if not matches:
            mismatch_count = np.sum(original_array != reconstructed_array)
            total_pixels = original_array.size
            accuracy = (1 - mismatch_count/total_pixels) * 100
            print(f"  Mismatched pixels: {mismatch_count}/{total_pixels}")
            print(f"  Accuracy: {accuracy:.2f}%")
    else:
        print(f"Image shape mismatch: {original_array.shape} vs {reconstructed_array.shape}")

    # Save decompressed 24x24 image
    reconstructed_image.save(decompressed_file)
    print(f"\n✓ Saved decompressed 24x24 image to: {decompressed_file}")

    print("\n" + "="*60)
    print("✓ SUCCESS! Lossless 24x24 pipeline completed.")
    print("="*60)
    print(f"  - Compressed file: {compressed_file}")
    print(f"  - Original 24x24 image: {original_resized_file}")
    print(f"  - Decompressed 24x24 image: {decompressed_file}")

if __name__ == "__main__":
    main()

