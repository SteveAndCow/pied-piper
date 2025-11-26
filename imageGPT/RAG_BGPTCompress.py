import gc
import torch
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from glob import glob
from tqdm import tqdm
import logging
import sys

# Add parent directory to path to import from imageGPT
imagegpt_path = os.path.join(os.path.dirname(__file__), '..', 'imageGPT')
sys.path.insert(0, imagegpt_path)
from BGPTCompress import (
    CompressionConfig, load_bgpt_model, pad_input_for_bgpt, 
    bgpt_compress, bgpt_decode, read_bytes, write_bytes,
    write_padded_bytes, read_padded_bytes, Metric
)
from bmp_utils import split_bmp_to_patches

logger = logging.getLogger(__name__)

# ==================== Configuration ====================
class RAGBGPTConfig:
    """Configuration for RAG-enhanced bGPT compression"""
    # Dataset paths for RAG indexing
    RAG_DATASET_PATH = "datasets/clic-2024/bmp/*.bmp"  # Images to index for RAG
    TEST_COMPRESSION_DATASET = "datasets/simple/bmp/*.bmp"  # Images to compress
    
    # Storage paths
    RETRIEVER_STORAGE_PATH = "retriever_cache/image-patches-storage"
    
    # Model paths
    MODEL_CHECKPOINT_IMAGE = "./pretrained/bgpt/weights-image.pth"
    
    # Retrieval parameters
    NUM_PATCHES_TO_INDEX = 10000  # Number of patches to index from RAG dataset
    TOP_K_RETRIEVAL = 3  # Number of similar patches to retrieve as context
    PATCH_SIZE = 32  # Size of square patches (should match CompressionConfig.BMP_PATCH_SIZE)
    
    # Compression parameters
    NUM_IMAGES_TO_COMPRESS = 1
    VERBOSE_THRESHOLD = 5  # Print retrieval results if num_images <= this
    
    # Embedding method
    EMBEDDING_METHOD = "raw_bytes"  # Options: "raw_bytes", "bgpt_features" (future)
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== Image Patch RAG Retriever ====================
class ImagePatchRetriever:
    """RAG retriever for image patches using FAISS"""
    
    def __init__(
        self,
        persist_path: str = None,
        patch_size: int = None,
        embedding_method: str = None
    ):
        """
        Initialize image patch retriever
        
        :param persist_path: path to persist/load index (default from config)
        :param patch_size: size of square patches (default from config)
        :param embedding_method: method for creating embeddings (default from config)
        """
        if persist_path is None:
            persist_path = RAGBGPTConfig.RETRIEVER_STORAGE_PATH
        if patch_size is None:
            patch_size = RAGBGPTConfig.PATCH_SIZE
        if embedding_method is None:
            embedding_method = RAGBGPTConfig.EMBEDDING_METHOD
        
        self.persist_path = persist_path
        self.patch_size = patch_size
        self.embedding_method = embedding_method
        
        self.index_file = os.path.join(persist_path, "faiss_index.bin")
        self.patch_store_file = os.path.join(persist_path, "patch_store.pkl")
        
        # Try to load existing index
        if os.path.exists(self.index_file) and os.path.exists(self.patch_store_file):
            logger.info(f"Found existing index at '{self.persist_path}'")
            self._load_index()
            logger.info(f"Index loaded: {self.index.ntotal} patches indexed")
        else:
            logger.info(f"No existing index found at '{self.persist_path}'")
            self.patch_store = {}  # Maps patch_id -> patch_bytes
            self.next_id = 0
            self.index = None
            self.embedding_dim = None
    
    def _create_embedding(self, patch_bytes: List[int]) -> np.ndarray:
        """
        Create embedding for a patch
        
        :param patch_bytes: list of bytes representing the patch
        :return: embedding vector
        """
        if self.embedding_method == "raw_bytes":
            # Simple approach: use patch bytes directly as embedding
            # Normalize to unit vector for cosine similarity
            embedding = np.array(patch_bytes, dtype=np.float32)
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        else:
            raise ValueError(f"Unknown embedding method: {self.embedding_method}")
    
    def _save_index(self):
        """Save the FAISS index and patch store to disk"""
        if not self.persist_path:
            return
        
        logger.info(f"Saving index to '{self.persist_path}'...")
        os.makedirs(self.persist_path, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, self.index_file)
        
        # Save patch store
        with open(self.patch_store_file, "wb") as f:
            pickle.dump({
                "patch_store": self.patch_store,
                "next_id": self.next_id,
                "embedding_dim": self.embedding_dim
            }, f)
        
        logger.info(f"Index saved successfully")
    
    def _load_index(self):
        """Load the FAISS index and patch store from disk"""
        # Load FAISS index
        self.index = faiss.read_index(self.index_file)
        
        # Load patch store
        with open(self.patch_store_file, "rb") as f:
            data = pickle.load(f)
            self.patch_store = data["patch_store"]
            self.next_id = data["next_id"]
            self.embedding_dim = data.get("embedding_dim")
    
    def index_patches_from_images(
        self,
        image_paths: List[str],
        max_patches: int = None
    ):
        """
        Index patches from a list of images
        
        :param image_paths: list of paths to BMP images
        :param max_patches: maximum number of patches to index (default from config)
        """
        if max_patches is None:
            max_patches = RAGBGPTConfig.NUM_PATCHES_TO_INDEX
        
        logger.info("=" * 80)
        logger.info("Indexing Image Patches for RAG")
        logger.info("=" * 80)
        logger.info(f"Number of images: {len(image_paths)}")
        logger.info(f"Max patches to index: {max_patches}")
        
        all_patches = []
        start_id = self.next_id
        
        # Extract patches from all images
        temp_split_folder = os.path.join(self.persist_path, "temp_split")
        os.makedirs(temp_split_folder, exist_ok=True)
        
        for img_idx, img_path in enumerate(tqdm(image_paths, desc="Processing images")):
            if self.next_id - start_id >= max_patches:
                break
            
            # Split image into patches
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_split_folder = os.path.join(temp_split_folder, img_name)
            os.makedirs(img_split_folder, exist_ok=True)
            
            # Use existing split function
            split_bmp_to_patches(
                source_folder=os.path.dirname(img_path),
                output_folder=temp_split_folder,
                patch_size=self.patch_size
            )
            
            # Read all patches from this image
            patch_files = sorted(glob(os.path.join(img_split_folder, "*.bmp")))
            
            for patch_file in patch_files:
                if self.next_id - start_id >= max_patches:
                    break
                
                # Read patch bytes
                patch_bytes, _ = read_bytes(patch_file)
                
                # Store patch
                patch_id = self.next_id
                self.patch_store[patch_id] = patch_bytes
                all_patches.append(patch_bytes)
                self.next_id += 1
        
        if not all_patches:
            logger.warning("No patches extracted")
            return
        
        logger.info(f"Extracted {len(all_patches)} patches")
        
        # Create embeddings
        logger.info(f"Creating embeddings using method: {self.embedding_method}")
        embeddings = []
        for patch_bytes in tqdm(all_patches, desc="Creating embeddings"):
            embedding = self._create_embedding(patch_bytes)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Set embedding dimension
        if self.embedding_dim is None:
            self.embedding_dim = embeddings.shape[1]
        
        # Build or update FAISS index
        logger.info("Building FAISS index...")
        if self.index is None:
            # Create new FAISS index (using Inner Product for normalized embeddings)
            base_index = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIDMap(base_index)
            logger.info(f"Created new index with dimension {self.embedding_dim}")
        
        # Add embeddings to index
        ids_to_add = np.arange(start_id, self.next_id, dtype=np.int64)
        self.index.add_with_ids(embeddings, ids_to_add)
        
        logger.info(f"Indexing complete: {self.index.ntotal} patches indexed")
        logger.info("=" * 80)
        
        # Save index
        self._save_index()
        
        # Clean up temp folder
        import shutil
        if os.path.exists(temp_split_folder):
            shutil.rmtree(temp_split_folder)
    
    def retrieve_similar_patches(
        self,
        query_patch_bytes: List[int],
        k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar patches
        
        :param query_patch_bytes: bytes of the query patch
        :param k: number of results to return (default from config)
        :return: list of retrieval results with id, patch_bytes, and score
        """
        if k is None:
            k = RAGBGPTConfig.TOP_K_RETRIEVAL
        
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError(
                "Index is empty. Please call index_patches_from_images() first."
            )
        
        # Create query embedding
        query_embedding = self._create_embedding(query_patch_bytes)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        retrieved_ids = indices[0]
        retrieved_scores = distances[0]
        
        for i in range(len(retrieved_ids)):
            patch_id = retrieved_ids[i]
            if patch_id != -1:
                score = retrieved_scores[i]
                patch_bytes = self.patch_store.get(patch_id)
                if patch_bytes is not None:
                    results.append({
                        "id": int(patch_id),
                        "patch_bytes": patch_bytes,
                        "score": float(score)
                    })
        
        return results


# ==================== Setup Functions ====================
def setup_image_retriever(
    rag_dataset_path: str = None,
    num_patches_to_index: int = None,
    persist_path: str = None
) -> ImagePatchRetriever:
    """
    Setup image patch retriever, either by loading existing index or building new one
    
    :param rag_dataset_path: path to images for indexing (default from config)
    :param num_patches_to_index: number of patches to index (default from config)
    :param persist_path: path to persist/load index (default from config)
    :return: initialized retriever
    """
    if rag_dataset_path is None:
        rag_dataset_path = RAGBGPTConfig.RAG_DATASET_PATH
    if num_patches_to_index is None:
        num_patches_to_index = RAGBGPTConfig.NUM_PATCHES_TO_INDEX
    if persist_path is None:
        persist_path = RAGBGPTConfig.RETRIEVER_STORAGE_PATH
    
    logger.info("\n=== Setting up Image Patch Retriever ===")
    retriever = ImagePatchRetriever(persist_path=persist_path)
    
    if retriever.index is None or retriever.index.ntotal == 0:
        logger.info("Index is empty. Building new index...")
        logger.info(f"Loading images from: {rag_dataset_path}")
        
        image_paths = glob(rag_dataset_path)
        if not image_paths:
            logger.warning(f"No images found at {rag_dataset_path}")
            logger.info("Retriever initialized but not indexed. You can index later.")
        else:
            logger.info(f"Found {len(image_paths)} images")
            retriever.index_patches_from_images(image_paths, max_patches=num_patches_to_index)
    else:
        logger.info("Index loaded from disk. Skipping indexing.")
    
    logger.info("--- Retriever is ready ---\n")
    return retriever


# ==================== RAG Compression Functions ====================
def compress_patch_with_rag_context(
    patch_bytes: List[int],
    ext: List[int],
    retriever: ImagePatchRetriever,
    model: Any,
    device: torch.device,
    metric: Metric,
    top_k: int = None,
    verbose: bool = False
) -> Tuple:
    """
    Compress a single patch using RAG context
    
    :param patch_bytes: bytes of the patch to compress
    :param ext: extension bytes
    :param retriever: image patch retriever
    :param model: bGPT model
    :param device: torch device
    :param metric: compression metric tracker
    :param top_k: number of similar patches to retrieve (default from config)
    :param verbose: whether to print retrieval results
    :return: tuple of compression results
    """
    if top_k is None:
        top_k = RAGBGPTConfig.TOP_K_RETRIEVAL
    
    # Retrieve similar patches
    similar_patches = retriever.retrieve_similar_patches(patch_bytes, k=top_k)
    
    if verbose:
        logger.info(f"\n--- Top {top_k} Retrieval Results ---")
        for i, result in enumerate(similar_patches, 1):
            logger.info(f"Result {i}: Score={result['score']:.4f}, Patch size={len(result['patch_bytes'])} bytes")
    
    # Prepare context patches (concatenate retrieved patches)
    context_patches = []
    if similar_patches:
        for result in similar_patches:
            context_patches.extend(result['patch_bytes'])
    
    # Prepare input: context + target patch
    if context_patches:
        # Pad context patches for bGPT
        context_padded = pad_input_for_bgpt([context_patches], [ext], device)
        context_input_ids = context_padded["patches"]
        context_length = context_input_ids.shape[1]
    else:
        context_input_ids = None
        context_length = 0
    
    # Prepare target patch
    target_padded = pad_input_for_bgpt([patch_bytes], [ext], device)
    target_input_ids = target_padded["patches"]
    target_masks = target_padded["masks"]
    
    # Concatenate context and target
    if context_input_ids is not None:
        full_input_ids = torch.cat([context_input_ids, target_input_ids], dim=1)
        # Extend masks for context
        context_masks = torch.ones((1, context_length), dtype=torch.long, device=device)
        full_masks = torch.cat([context_masks, target_masks], dim=1)
    else:
        full_input_ids = target_input_ids
        full_masks = target_masks
    
    # Generate logits
    with torch.inference_mode():
        output = model(patches=full_input_ids, masks=full_masks)
        logits = output.logits
        
        # Process logits (same as in BGPTCompress)
        logits = logits[:-1, :-1, :]
        logits = logits.reshape(1, -1, 257)
        
        # Extract only the target patch portion (after context)
        start_patch = full_input_ids[:, :CompressionConfig.PATCH_SIZE].squeeze(0)
        target_start_idx = CompressionConfig.PATCH_SIZE + context_length
        target_end_idx = full_input_ids.shape[1] - CompressionConfig.PATCH_SIZE
        target_input_ids = full_input_ids[:, target_start_idx:target_end_idx]
        
        # Add start token
        target_input_ids = torch.cat(
            [torch.tensor([[256]], device=device), target_input_ids], dim=1
        )
        
        # Extract corresponding logits for target patch
        target_logits_start = context_length
        target_logits = logits[:, target_logits_start:, :]
    
    # Compress using the target portion
    compression_results = bgpt_compress(
        target_input_ids,
        target_logits,
        metric,
        prefix_length=CompressionConfig.PREFIX_LENGTH
    )
    
    return compression_results, start_patch


# ==================== Main Workflow ====================
def run_rag_bgpt_compression(
    test: bool = True,
    temp_folder: str = "temp_img_rag",
    output_folder: str = "output_img_rag",
    patch_size: int = None
):
    """
    Main workflow for RAG-enhanced bGPT image compression
    
    :param test: whether to use test dataset
    :param temp_folder: temporary folder for intermediate files
    :param output_folder: folder for final output
    :param patch_size: size of square patches (default from config)
    """
    if patch_size is None:
        patch_size = RAGBGPTConfig.PATCH_SIZE
    
    device = torch.device(RAGBGPTConfig.DEVICE)
    
    # Setup retriever
    retriever = setup_image_retriever()
    
    # Load bGPT model
    logger.info("\n=== Loading bGPT Model ===")
    model = load_bgpt_model(RAGBGPTConfig.MODEL_CHECKPOINT_IMAGE, device)
    
    # Get dataset path
    if test:
        dataset_path = RAGBGPTConfig.TEST_COMPRESSION_DATASET
    else:
        dataset_path = RAGBGPTConfig.RAG_DATASET_PATH
    
    # Find images to compress
    image_paths = glob(dataset_path)
    if not image_paths:
        logger.error(f"No images found at {dataset_path}")
        return
    
    # Limit number of images
    image_paths = image_paths[:RAGBGPTConfig.NUM_IMAGES_TO_COMPRESS]
    
    logger.info(f"\n=== Starting RAG Compression ===")
    logger.info(f"Number of images to compress: {len(image_paths)}")
    
    # Create folders
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    split_folder = os.path.join(temp_folder, "split")
    compressed_folder = os.path.join(temp_folder, "compressed")
    decompressed_folder = os.path.join(temp_folder, "decompressed")
    
    os.makedirs(split_folder, exist_ok=True)
    os.makedirs(compressed_folder, exist_ok=True)
    os.makedirs(decompressed_folder, exist_ok=True)
    
    total_metric = Metric()
    verbose = len(image_paths) <= RAGBGPTConfig.VERBOSE_THRESHOLD
    
    for img_idx, img_path in enumerate(image_paths, 1):
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Processing image {img_idx}/{len(image_paths)}: {os.path.basename(img_path)}")
        logger.info("=" * 80)
        
        filename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Split image into patches
        logger.info(f"Splitting image into {patch_size}x{patch_size} patches...")
        split_bmp_to_patches(
            source_folder=os.path.dirname(img_path),
            output_folder=split_folder,
            patch_size=patch_size
        )
        
        patches_subfolder = os.path.join(split_folder, name_without_ext)
        patch_files = sorted(glob(os.path.join(patches_subfolder, "*.bmp")))
        logger.info(f"Created {len(patch_files)} patches")
        
        # Compress each patch with RAG
        logger.info(f"Compressing {len(patch_files)} patches with RAG context...")
        compressed_info = {}
        
        for patch_idx, patch_file in enumerate(tqdm(patch_files, desc="Compressing patches"), 1):
            patch_name = os.path.basename(patch_file)
            patch_id = os.path.splitext(patch_name)[0]
            
            # Read patch bytes
            patch_bytes, ext = read_bytes(patch_file)
            
            # Compress with RAG context
            metric = Metric()
            compression_results, start_patch = compress_patch_with_rag_context(
                patch_bytes=patch_bytes,
                ext=ext,
                retriever=retriever,
                model=model,
                device=device,
                metric=metric,
                verbose=verbose and patch_idx <= 3  # Only verbose for first few patches
            )
            
            compressed_bytes, num_padded_bits, _, sequence_array, pd, probs = compression_results
            
            # Save compressed data
            compressed_path = os.path.join(compressed_folder, f"{name_without_ext}_{patch_id}.bin")
            original_length = len(patch_bytes)
            write_padded_bytes(compressed_path, compressed_bytes, num_padded_bits, original_length)
            
            # Store info for decompression
            compressed_info[patch_id] = {
                'compressed_path': compressed_path,
                'start_patch': start_patch,
                'ext': ext,
                'original_length': original_length,
            }
            
            total_metric.accumulate(metric.compressed_length, metric.total_length)
        
        compress_rate, compress_ratio = total_metric.compute_ratio()
        logger.info(f"\nCompression completed")
        logger.info(f"  Overall compression ratio: {compress_ratio:.6f}")
        logger.info(f"  Overall compression rate: {compress_rate:.6f}x")
        
        # Decompress patches (simplified - would need RAG context during decompression too)
        logger.info(f"Decompressing patches...")
        decompressed_subfolder = os.path.join(decompressed_folder, name_without_ext)
        os.makedirs(decompressed_subfolder, exist_ok=True)
        
        for patch_id, info in tqdm(compressed_info.items(), desc="Decompressing patches"):
            # Read compressed data
            compressed_bytes, num_padded_bits, original_length = read_padded_bytes(
                info['compressed_path']
            )
            
            # Decompress (Note: This is simplified - full RAG decompression would need context)
            decompressed_tensor = bgpt_decode(
                compressed_bytes,
                num_padded_bits,
                model,
                info['start_patch'],
                info['ext'],
                device,
                original_length,
                do_test=False,
            )
            
            # Save decompressed patch
            decompressed_bytes = decompressed_tensor.squeeze(0).cpu().numpy().tolist()
            decompressed_path = os.path.join(decompressed_subfolder, f"{patch_id}.bmp")
            write_bytes(decompressed_path, decompressed_bytes)
        
        logger.info("=" * 80)
    
    logger.info("\n" + "=" * 80)
    logger.info("RAG BGPT COMPRESSION COMPLETED!")
    logger.info("=" * 80)
    final_rate, final_ratio = total_metric.compute_ratio()
    logger.info(f"Final compression ratio: {final_ratio:.6f}")
    logger.info(f"Final compression rate: {final_rate:.6f}x")


# ==================== Main ====================
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_bgpt_compression.log', mode='w')
        ]
    )
    
    # Run RAG compression
    run_rag_bgpt_compression(
        test=True,
        temp_folder="temp_img_rag",
        output_folder="output_img_rag",
        patch_size=RAGBGPTConfig.PATCH_SIZE
    )

