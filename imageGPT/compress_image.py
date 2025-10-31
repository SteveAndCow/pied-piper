"""
Image Compression using ImageGPT
Compresses images using ImageGPT model and arithmetic coding.
"""

import logging
import numpy as np
import torch
from transformers import AutoImageProcessor, ImageGPTForCausalImageModeling
from PIL import Image
from typing import Iterator
from arithmetic_coder import arithmetic_coder, ac_utils
import matplotlib.pyplot as plt
import os
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Metric:
    def __init__(self):
        self.total_bits = 0.0
        self.compressed_bits = 0.0

    def compute_ratio(self):
        if self.total_bits != 0 and self.compressed_bits != 0:
            return (
                self.total_bits / self.compressed_bits,
                self.compressed_bits / self.total_bits,
            )
        else:
            return 0, 0

    def accumulate(self, compressed_bits, original_bits):
        self.compressed_bits += compressed_bits
        self.total_bits += original_bits


def compress_image(compress_input, logits, metric, original_bits):
    """
    Compress image tokens using probability distribution from ImageGPT model.
    
    :param compress_input: image token sequence (with SOS token at start)
    :param logits: model logits for next token prediction
    :param metric: compression metrics
    :param original_bits: original image bits (H × W × 3 × 8)
    :return: compressed data and metadata
    """
    output = []
    encoder = arithmetic_coder.Encoder(
        base=2,
        precision=64,
        output_fn=output.append,
    )
    
    # Save the first symbol (SOS token) for decoding
    start_symbol = compress_input[:, :1]
    
    # Get probabilities for the actual sequence (excluding SOS)
    probs = logits.softmax(dim=-1).to(torch.float32)
    sequence_array = compress_input[:, 1:].detach().cpu().numpy().reshape(-1)
    
    # Get probability of true tokens
    pd = torch.gather(probs, dim=-1, index=compress_input[:, 1:].unsqueeze(-1)).squeeze(-1)
    probs_np = np.vstack(probs.detach().cpu().numpy().squeeze())
    pd = pd.squeeze()
    
    # Compute theoretical compressed bits using -log2(p_true)
    compressed_bits_theoretical = -torch.sum(torch.log2(torch.clamp(pd, min=1e-10))).item()
    
    # Also use arithmetic coding for actual compression
    for symbol, prob in zip(sequence_array, probs_np):
        encoder.encode(
            ac_utils.normalize_pdf_for_arithmetic_coding(prob, np.float32), symbol
        )
    encoder.terminate()

    # Convert to bytes
    compressed_bits_str = "".join(map(str, output))
    compressed_bytes, num_padded_bits = ac_utils.bits_to_bytes(compressed_bits_str)
    compressed_bits_actual = len(compressed_bytes) * 8 + num_padded_bits
    
    # Use theoretical bits for metric (as per instructions)
    metric.accumulate(compressed_bits_theoretical, original_bits)

    compress_rate, compress_ratio = metric.compute_ratio()
    logger.info(f"Original bits: {original_bits:.2f}")
    logger.info(f"Compressed bits (theoretical): {compressed_bits_theoretical:.2f}")
    logger.info(f"Compressed bits (actual): {compressed_bits_actual:.2f}")
    logger.info(f"Compression ratio: {compress_ratio:.6f}")
    logger.info(f"Compression rate: {compress_rate:.6f}")

    return compressed_bytes, num_padded_bits, start_symbol, sequence_array, pd, probs, compressed_bits_theoretical


def decode_image(
    compressed_bytes,
    num_padded_bits,
    model,
    start_symbol,
    device,
    original_seq_len,
    original_sequence=None,
    pd=None,
    probs=None,
    do_test=True,
):
    """
    Decode compressed image tokens back to token sequence.
    
    :param compressed_bytes: compressed data
    :param num_padded_bits: padded bits
    :param model: ImageGPT model
    :param start_symbol: SOS token to start decoding
    :param device: device to run model on
    :param original_seq_len: original sequence length (excluding SOS)
    :param original_sequence: original token sequence for testing
    :param pd: probability distribution (for testing)
    :param probs: probability arrays (for testing)
    :param do_test: whether to print test information
    :return: decoded token sequence including SOS token
    """
    # Convert bytes back to bit stream
    data_iter = iter(
        ac_utils.bytes_to_bits(compressed_bytes, num_padded_bits=num_padded_bits)
    )

    # Utils function to read bits
    def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
        try:
            return int(next(bit_sequence))
        except StopIteration:
            return None

    # Initialize a Decoder Object
    decoder = arithmetic_coder.Decoder(
        base=2,
        precision=64,
        input_fn=_input_fn,
    )

    # Start with SOS token
    sequence_array_de = start_symbol.squeeze(0).detach().cpu().numpy().copy()
    sequence_array_de_input = start_symbol.to(device)

    # Prepare stored probability distributions if available
    stored_probs = None
    if probs is not None:
        # Convert to numpy if it's a torch tensor
        if isinstance(probs, torch.Tensor):
            stored_probs = probs.detach().cpu().numpy()
            if stored_probs.ndim == 3:  # [batch, seq_len, vocab]
                stored_probs = stored_probs[0]  # Remove batch dimension
        else:
            stored_probs = probs
        if stored_probs.ndim == 1:
            stored_probs = stored_probs.reshape(1, -1)  # Ensure 2D

    # Decode tokens one by one
    for i in range(original_seq_len):
        try:
            # Use stored probability distribution if available, otherwise compute from model
            if stored_probs is not None and i < stored_probs.shape[0]:
                # Use the stored probability distribution from encoding
                prob_at_pos = stored_probs[i]
            else:
                # Fallback: compute from model (shouldn't happen if probs were passed)
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
                
                # Get probability distribution for the last position
                prob_de = logits.softmax(dim=-1).detach().cpu().numpy()
                if prob_de.ndim == 3:  # [batch, seq_len, vocab]
                    prob_at_pos = prob_de[0, -1, :]  # Last position of first batch
                elif prob_de.ndim == 2:  # [seq_len, vocab]
                    prob_at_pos = prob_de[-1, :]  # Last position
                else:  # [vocab]
                    prob_at_pos = prob_de
            
            # Normalize and decode
            normalized_prob = ac_utils.normalize_pdf_for_arithmetic_coding(prob_at_pos, np.float32)
            de_token = decoder.decode(normalized_prob)
            
            # Append to the generated sequence
            sequence_array_de = np.append(sequence_array_de, de_token)
            
            # Update input for next iteration - only needed if we're computing probs from model
            # If using stored probs, we don't need to update the input
            if stored_probs is None or i >= stored_probs.shape[0]:
                max_seq_len = min(len(sequence_array_de), model.config.n_positions)
                sequence_array_de_input = torch.tensor(
                    sequence_array_de[:max_seq_len], 
                    dtype=torch.long, device=device
                ).unsqueeze(0)
            
        except StopIteration:
            # If we run out of bits, use model's top prediction for remaining tokens
            logger.warning(f"Ran out of compressed bits at position {i}/{original_seq_len}")
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
                        de_token = prob_de[-1].argmax()
                    else:
                        de_token = prob_de.argmax()
                    sequence_array_de = np.append(sequence_array_de, de_token)
                    
                    # Update input
                    sequence_array_de_input = torch.tensor(
                        sequence_array_de[-min(len(sequence_array_de), model.config.n_positions):], 
                        dtype=torch.long, device=device
                    ).unsqueeze(0)
            break
    
    # Convert back to tensor format
    sequence_array_de_input = torch.tensor(
        sequence_array_de, dtype=torch.long, device=device
    ).unsqueeze(0)
    
    return sequence_array_de_input


def write_padded_bytes(filename: str, data: bytes, num_padded_bits: int, original_length: int):
    """
    file format:
    - first byte: number of padded bit
    - second and third byte: original length (usually, llm context will not exceed 65535)
    - subsequent bytes: actual bytes data

    :param filename: output file name
    :param data: bytes data to write
    :param num_padded_bits: number of padded bits (must be between 0 and 7)
    :param original_length: original length of the uncompressed data (in tokens)
    """
    if not 0 <= num_padded_bits <= 7:
        raise ValueError("num_padded_bits must be between 0 and 7.")

    if not 0 <= original_length <= 65535:
        raise ValueError("original_length must be between 0 and 65535.")

    if not isinstance(data, bytes):
        raise TypeError("data must be of bytes type.")

    with open(filename, 'wb') as f:
        padding_byte = num_padded_bits.to_bytes(1, 'big')
        f.write(padding_byte)
        f.write(original_length.to_bytes(2, 'big'))
        f.write(data)


def read_padded_bytes(filename: str) -> tuple[bytes, int, int]:
    """
    Read data and padding bits from a file.

    :param filename: The name of the file to read.
    :return: A tuple containing (bytes data, number of padded bits, original_length).
    """
    with open(filename, 'rb') as f:
        padding_byte = f.read(1)
        if not padding_byte:
            raise EOFError("File is empty or improperly formatted: unable to read padding bits byte.")

        original_length_bytes = f.read(2)
        if not original_length_bytes:
            raise EOFError("File is empty or improperly formatted: unable to read original length bytes.")
    
        padding_bits = int.from_bytes(padding_byte, 'big')
        original_length = int.from_bytes(original_length_bytes, 'big')
        data = f.read()
        
        return data, padding_bits, original_length


def tokens_to_image(tokens, clusters, height, width):
    """
    Convert token sequence to PIL Image.
    
    :param tokens: token sequence (numpy array, excluding SOS)
    :param clusters: color clusters from image processor
    :param height: image height
    :param width: image width
    :return: PIL Image
    """
    # Ensure we have enough tokens
    expected_tokens = height * width
    if len(tokens) < expected_tokens:
        # Pad with zeros if needed
        tokens = np.pad(tokens, (0, expected_tokens - len(tokens)), constant_values=0)
    elif len(tokens) > expected_tokens:
        # Truncate if too many
        tokens = tokens[:expected_tokens]
    
    # Reshape to image dimensions
    token_indices = tokens.reshape(height, width)
    
    # Convert cluster indices to RGB values
    reconstructed_img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            token_idx = int(token_indices[i, j])
            if 0 <= token_idx < len(clusters):
                # Convert from [-1, 1] to [0, 255]
                rgb = np.rint(127.5 * (clusters[token_idx] + 1.0)).astype(np.uint8)
                reconstructed_img_array[i, j] = rgb
    
    return Image.fromarray(reconstructed_img_array, mode='RGB')


def main():
    """Main compression and decompression pipeline."""
    # Verify GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Load ImageGPT model and processor
    print("\nLoading ImageGPT model...")
    image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
    model = ImageGPTForCausalImageModeling.from_pretrained("openai/imagegpt-small")
    model.eval()
    model.to(device)

    # Load image
    print("\nLoading sample image...")
    image_path = "test/pixel.png"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    print(f"✓ Successfully loaded image from {image_path}")
    print(f"  Original size: {image.size} (width x height)")

    # Preprocess image: resize to 24x24
    target_size = {"height": 24, "width": 24}
    print(f"\nPreprocessing image to {target_size}...")
    inputs = image_processor(
        images=image, 
        return_tensors="pt",
        size=target_size
    )
    input_ids = inputs["input_ids"].to(device)
    print(f"Input token shape: {input_ids.shape}")
    print(f"Sequence length: {input_ids.shape[1]} (including SOS token)")

    # Compute original bits: H × W × 3 × 8
    height, width = target_size["height"], target_size["width"]
    original_bits = height * width * 3 * 8
    print(f"Original bits: {original_bits} (H={height}, W={width}, C=3, bits_per_pixel=8)")

    # Compression workflow
    print("\n" + "="*60)
    print("=== COMPRESSION ===")
    print("="*60)
    compression_start_time = time.time()

    metric = Metric()
    with torch.no_grad():
        # Get logits for all positions
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
        model_outputs = model(
            input_ids=input_ids,
            use_cache=False,
            position_ids=position_ids,
            past_key_values=None
        )
        # ImageGPT predicts next token, so we need logits[:-1] to match input_ids[1:]
        logits = model_outputs.logits[:, :-1].to(torch.float32)

    compressed_bytes, num_padded_bits, start_symbol, sequence_array, pd, probs, compressed_bits = compress_image(
        input_ids, logits, metric, original_bits
    )

    compression_end_time = time.time()
    compression_time = compression_end_time - compression_start_time
    print(f"\nCompression completed in {compression_time:.2f} seconds")

    # Save compressed data
    original_length = input_ids.shape[1] - 1  # Exclude SOS token
    bin_filename = "compressed.bin"
    write_padded_bytes(bin_filename, compressed_bytes, num_padded_bits, original_length)
    print(f"✓ Saved compressed data to {bin_filename}")
    print(f"  Compressed file size: {len(compressed_bytes)} bytes ({len(compressed_bytes) / 1024:.2f} KB)")

    # Test reading back
    compressed_bytes_read, num_padded_bits_read, original_length_read = read_padded_bytes(bin_filename)
    print(f"✓ Read back: original_length={original_length_read}, num_padded_bits={num_padded_bits_read}")

    # Decompression workflow
    print("\n" + "="*60)
    print("=== DECOMPRESSION ===")
    print("="*60)
    decompression_start_time = time.time()

    decompressed = decode_image(
        compressed_bytes_read,
        num_padded_bits_read,
        model,
        start_symbol,
        device,
        original_length_read,
        sequence_array,
        pd,
        probs,
        do_test=False,  # Disable test output for cleaner logs
    )

    decompression_end_time = time.time()
    decompression_time = decompression_end_time - decompression_start_time
    print(f"✓ Decompression completed in {decompression_time:.2f} seconds")

    # Verify reconstruction
    print("\n" + "="*60)
    print("=== VERIFICATION ===")
    print("="*60)
    decompressed_tokens = decompressed.squeeze(0).cpu().numpy()
    original_tokens = input_ids.squeeze(0).cpu().numpy()
    
    print(f"Original tokens shape: {original_tokens.shape}")
    print(f"Decoded tokens shape: {decompressed_tokens.shape}")
    
    # Compare (only compare the non-SOS tokens)
    if len(decompressed_tokens) == len(original_tokens):
        matches = np.array_equal(original_tokens, decompressed_tokens)
        print(f"Tokens match exactly: {matches}")
        if not matches:
            mismatch_count = np.sum(original_tokens != decompressed_tokens)
            accuracy = (1 - mismatch_count/len(original_tokens)) * 100
            print(f"  Mismatched tokens: {mismatch_count}/{len(original_tokens)}")
            print(f"  Accuracy: {accuracy:.2f}%")
    else:
        min_len = min(len(original_tokens), len(decompressed_tokens))
        matches = np.array_equal(original_tokens[:min_len], decompressed_tokens[:min_len])
        print(f"Token length mismatch: {len(original_tokens)} vs {len(decompressed_tokens)}")
        print(f"First {min_len} tokens match: {matches}")

    # Final metrics
    compress_rate, compress_ratio = metric.compute_ratio()
    print("\n" + "="*60)
    print("=== FINAL METRICS ===")
    print("="*60)
    print(f"Original bits: {metric.total_bits:.2f}")
    print(f"Compressed bits: {metric.compressed_bits:.2f}")
    print(f"Compression ratio: {compress_ratio:.6f} ({compress_ratio*100:.2f}% of original)")
    print(f"Compression rate: {compress_rate:.6f}x")
    print(f"Compression time: {compression_time:.2f} seconds")
    print(f"Decompression time: {decompression_time:.2f} seconds")

    # Reconstruct and save image
    print("\n" + "="*60)
    print("=== IMAGE RECONSTRUCTION ===")
    print("="*60)
    clusters = image_processor.clusters
    
    # Get decompressed tokens (excluding SOS token)
    decompressed_tokens_no_sos = decompressed_tokens[1:] if len(decompressed_tokens) > 1 else decompressed_tokens
    print(f"Decompressed tokens (excluding SOS): {len(decompressed_tokens_no_sos)}")
    print(f"Expected image size: {height}×{width} = {height * width} pixels")

    # Convert tokens back to image
    reconstructed_image = tokens_to_image(decompressed_tokens_no_sos, clusters, height, width)
    
    # Save the decompressed image
    output_image = "compressed-pixel.png"
    reconstructed_image.save(output_image)
    print(f"✓ Saved decompressed image to: {output_image}")
    
    file_size = os.path.getsize(output_image) / 1024
    print(f"  File size: {file_size:.2f} KB")
    
    # Display comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original (quantized)
    original_tokens_no_sos = original_tokens[1:]
    original_processed = tokens_to_image(original_tokens_no_sos, clusters, height, width)
    axes[0].imshow(original_processed)
    axes[0].set_title('Original (Quantized)\n24×24 pixels', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Reconstructed
    axes[1].imshow(reconstructed_image)
    axes[1].set_title(f'Decompressed\nCompression ratio: {compress_ratio:.3f}', 
                     fontsize=12, fontweight='bold', color='green')
    axes[1].axis('off')
    
    plt.tight_layout()
    comparison_filename = "compression_comparison.png"
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison figure to: {comparison_filename}")
    
    print("\n" + "="*60)
    print("✓ SUCCESS! Pipeline completed.")
    print("="*60)
    print(f"  - Compressed file: {bin_filename}")
    print(f"  - Decompressed image: {output_image}")
    print(f"  - Comparison figure: {comparison_filename}")


if __name__ == "__main__":
    main()

