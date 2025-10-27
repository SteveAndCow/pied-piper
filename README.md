# Pied Piper Research Group

## Image Compression System

### Goal

To modify the current text compression system to an image compression system, using the ImageGPT model. We should create a successfully running pipeline that can report the compression ratio when compressing 10 random images.

### Resources

Original Paper for ImageGPT: [Generative Pretraining from Pixels (V2)](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)

HuggingFace Model Documentation for ImageGPT: [Transformers docs - ImageGPT](https://huggingface.co/docs/transformers/model_doc/imagegpt)


## One-Day Blitz Plan

**Duration:** 8:00 AM – 5:00 PM  
**Team Size:** 4  
**Goal:** Build a working demo of an **Image Compression System** using **ImageGPT**, adapted from our text-compression system.  
By the end of the day we should have:

- A working preprocessing → model → evaluation pipeline.
- Compression ratios printed for 10 test images.
- A simple visualization and README write-up.

---

### 🕒 Schedule Overview

| Time | Phase | Objective | Deliverable |
|------|--------|------------|--------------|
| **8:00 – 8:30 AM** | **Kickoff & Setup** | Align goals, assign initial roles, clone repo, install dependencies. | Verified environment, working project directory. |
| **8:30 – 9:45 AM** | **Phase 1 – Model Sanity Check** | Load pretrained ImageGPT, test dummy forward pass, confirm logits shape. | Screenshot/log of successful model forward. |
| **10:00 – 11:00 AM** | _(Meeting break)_ | Pause coding. Optional doc writing or reading ImageGPT paper. | — |
| **11:00 – 12:30 PM** | **Phase 2 – Preprocessing Pipeline** | Load/resize 32×32 images, flatten into sequences, compute original bits. | Working `preprocess_image()` function. |
| **12:30 – 1:30 PM** | _(Lunch + meeting)_ | Optional async doc updates. | — |
| **1:30 – 3:00 PM** | **Phase 3 – Integration Sprint** | Feed preprocessed tokens into model, compute per-token log-probabilities and total bits. | End-to-end pipeline on 1–3 images. |
| **3:00 – 3:15 PM** | **Sync + Role Rotation** | Quick regroup: assign new driver/navigator, list bugs. | Updated live task list. |
| **3:15 – 4:15 PM** | **Phase 4 – Evaluation + Visualization** | Run on 10 images, compute compression ratios, plot results. | CSV + bar chart + average compression ratio. |
| **4:15 – 5:00 PM** | **Phase 5 – Polish + Wrap-Up** | Clean code, finalize README, summarize learnings, optional slide prep. | Complete README + demo screenshot. |

---

### 🧩 Core Task Checklist (MVP)

#### **Environment & Setup**
- [ ] Install: `torch`, `transformers`, `Pillow`, `matplotlib`.
- [ ] Verify GPU availability (`torch.cuda.is_available()`).
- [ ] Load `openai/imagegpt-small` successfully.
- [ ] Run dummy input test: check logits shape `[1, seq_len, vocab_size]`.

#### **Preprocessing**
- [ ] Load 10 sample images (e.g. CIFAR-10 or local folder).
- [ ] Resize to 24×24 RGB (≈ 1728 tokens < 2048 context limit).
- [ ] Flatten into single token sequence.
- [ ] Compute `orig_bits = H × W × 3 × 8`.
- [ ] Verify tensor shape matches model input.

#### **Model + Integration**
- [ ] Forward tokens through model (`outputs = model(tokens)`).
- [ ] Compute softmax → probabilities → `p_true`.
- [ ] Compute `bits = -log2(p_true)` per token.
- [ ] Sum to get total `compressed_bits`.
- [ ] Compare `orig_bits` vs. `compressed_bits` for 1–3 images.

#### **Evaluation + Visualization**
- [ ] Loop over all 10 images, store `(orig_bits, compressed_bits, ratio)`.
- [ ] Compute averages + std deviation.
- [ ] Plot bar chart with horizontal line at 1.0.
- [ ] Save CSV and figure.

#### **Documentation**
- [ ] Clean up code into: `preprocess.py`, `model_infer.py`, `run_demo.py`.
- [ ] Update `README.md`:
  - Pipeline overview diagram.
  - Bits/compression formulas.
  - Key design choices (token layout, image size, etc.).
  - Limitations and next steps.
- [ ] Add results table and sample plot.

---

### 🚀 Stretch Goals (Optional if Time Allows)

- [ ] **Quantization Experiment:** Use ImageGPT’s 512-color quantizer instead of raw RGB.
- [ ] **Grayscale Comparison:** Compare 1-channel vs. 3-channel compression efficiency.
- [ ] **Tiling Larger Images:** Process 64×64 images by dividing into 24×24 patches.
- [ ] **Entropy-Coding Simulation:** Convert predicted probs into actual byte lengths with arithmetic-coding formula.
- [ ] **Reconstruction Demo (Pseudo):** Greedily sample pixels from model to reconstruct a rough image.
- [ ] **Performance Benchmark:** Compare model-based bits/pixel to PNG/JPEG file size.

*(If you run short on time, skip all stretch goals.)*

---

### 👥 Pair-Work Fallback Plan (for Parallel Work)

If time runs tight, split into **two pairs** to parallelize:

| Pair | Focus Area | Core Deliverables |
|------|-------------|-------------------|
| **Pair A – Model + Math** | - Load ImageGPT<br>- Test forward pass<br>- Compute per-token log-probs and bits | `model_infer.py` working for single image |
| **Pair B – Data + Evaluation** | - Implement preprocessing<br>- Build 10-image loop<br>- Generate visualization and README | `preprocess.py`, plot, and table |

**Sync Points:**
- *11:00 AM* – Agree on token format (tensor shape, dtype).  
- *3:00 PM* – Integrate Pair A and B code paths and run first full pipeline test.  
- *4:15 PM* – Finalize evaluation + documentation together.

---

### ✅ Success Criteria (End-of-Day)

| Metric | Target |
|---------|--------|
| Pipeline runs end-to-end | ✅ on at least 10 images |
| Compression ratio computed | ✅ (> 0, even if < 1) |
| Plot generated | ✅ saved as `.png` |
| README updated | ✅ includes design + results |
| Code < 150 lines core logic | ✅ clean, readable |
| Everyone understands each step | ✅ team learning goal met |

---

### 📈 Notes on Scope and Feasibility

- This is a **conceptual compression demo**, not a production compressor.  
- We expect total code ≈ 100 lines + glue.  
- Avoid scope creep (no training, no reconstruction unless extra time).  
- Focus on *pipeline correctness + understanding*.

---
