# Pied Piper Research Group

## Image Compression System

### Goal

To modify the current text compression system to an image compression system, using the ImageGPT model. We should create a successfully running pipeline that can report the compression ratio when compressing 10 random images.

### Resources

Original Paper for ImageGPT: [Generative Pretraining from Pixels (V2)](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)

HuggingFace Model Documentation for ImageGPT: [Transformers docs - ImageGPT](https://huggingface.co/docs/transformers/model_doc/imagegpt)


# Task Breakdown

Let’s treat this like a mini engineering sprint with four people. I’ll break it into 4 workstreams you can assign (A–D), list the concrete tasks in each stream, and call out dependencies / handoffs.

High-level pipeline we’re building

1. Load an image → preprocess into model tokens (per-channel sequences).
2. Run ImageGPT to get predicted distributions over pixels.
3. Use those predictions to compress (e.g. entropy code / bit length estimate).
4. Report compression ratio across 10 random images.

We need all four of those to exist by the end.

---

## Workstream A. Model + Research Owner (“Model / Arch lead”)

Goal: Understand and lock in how we’re using ImageGPT for compression.

Tasks

1. Read + summarize model behavior

   * From the original paper and HuggingFace docs, confirm:

     * What is ImageGPT expecting as input? (Tokenized pixel values? Quantized color codes? Shape `[seq_len]`?)
     * Does it accept raw pixel intensities 0–255 or does it require clustering / codebook IDs first? (ImageGPT in the paper actually uses color quantization to 512 color codes in some variants, not raw RGB; you need to verify which checkpoint you’ll load. This affects preprocessing and sequence length.)
     * What is the max context length actually supported by the checkpoint you’ll use (2048, per hint)?
   * Output of this task: a short spec doc that others can depend on.

2. Define compression formulation

   * Decide how we estimate bits:

     * We can treat ImageGPT as a probabilistic model and compute ideal code length = (-\sum \log_2 p(\text{pixel}_t | \text{pixels}<t)). That gives us an effective bits-per-pixel. This mirrors text compression with language models.
     * Then Compression Ratio = (Uncompressed size in bits) / (Model-estimated code length in bits).
   * Finalize formula and share it with Workstream D for reporting.

3. Define sequence layout

   * Are we doing channel-major? (All R flattened, then all G, then all B.)
   * Or interleaved RGBRGB...?
   * You’ll choose and write it down, because Workstream B must match it and Workstream C must reconstruct it.

4. Context window policy

   * For images whose flattened sequence > 2048 tokens:

     * Option 1: restrict dataset to small images (e.g. 32×32, 64×64) so seq_len ≤ 2048.
     * Option 2: tile the image into 32×32 crops and treat each tile independently.
   * Make that call early, because it drives data prep (B) and evaluation batching (D).

Deliverables / handoffs

* A spec doc with:

  * Input tokenization scheme
  * Sequence ordering
  * Image size constraint / tiling rule
  * Bit-length math
* Deadline: this spec unblocks literally everyone else.

Blocking / dependencies

* This has to happen first. Workstream B and C cannot start coding without this.

Recommended person

* Someone comfortable reading the ImageGPT paper + HF docs and making “we’re doing it this way” calls.

---

## Workstream B. Data + Preprocessing Owner (“Data engineer”)

Goal: Turn raw images into valid ImageGPT input tensors the model will accept.

Tasks

1. Image loader + dataset sampler

   * Pick a source of ~10 random test images (CIFAR-10-style 32×32? Tiny ImageNet? personal folder?). Keep them local so we don’t need to train, just evaluate.
   * Write code to:

     * Load each image
     * Resize / center-crop to the agreed target size from A
     * Convert to the model’s expected representation (quantized color indices or pixel intensities depending on A).

2. Channel sequencing

   * Split RGB into 3 channels if that’s what the model wants (per mentor hint).
   * Flatten each channel to a 1D sequence.
   * Concatenate sequences according to the layout from A.
   * Produce final token sequence of length L ≤ context window.

3. Batching & padding

   * Prepare PyTorch tensors shaped like what HuggingFace ImageGPT forward() wants.
   * Handle padding if some tiles are shorter than max length (and create attention masks if required).

4. Metadata logging

   * For each image/patch:

     * Store original shape (H, W, 3)
     * Store token sequence length
     * Store raw size in bits (H×W×3×8).
   * You’ll hand this to D.

Deliverables / handoffs

* `preprocess.py` with a function like:

  ```python
  def preprocess_image(img_path) -> {
      "tokens": torch.LongTensor[1, L],
      "orig_bits": int,
      "H": int, "W": int
  }
  ```
* A small script that loops over 10 images and dumps a list of these dicts to disk (pickle / JSON + npy).

Blocking / dependencies

* Needs Workstream A’s spec first (tokenization, tiling, ordering).
* Unblocks Workstream C (model inference) and D (compression ratio calc).

Recommended person

* Someone comfortable with PIL / torchvision / numpy / PyTorch tensor wrangling.

---

## Workstream C. Inference + Likelihood Owner (“Model runtime engineer”)

Goal: Given a preprocessed sequence of tokens, run ImageGPT and compute per-token probabilities.

Tasks

1. Model load

   * Using HuggingFace `AutoModelForCausalLM` or `ImageGPTForCausalLM` (depending on the exact class in docs), load the chosen pretrained ImageGPT checkpoint from HF.
   * Put model in eval mode, no grad.

2. Forward pass over sequence

   * For each tokenized sequence from B:

     * Feed into the model.
     * Get logits for each position t.
     * Convert logits[t-1] → probs for token[t] using softmax.

       * Store (p_t = P(x_t | x_{<t})).

   Note: first token doesn’t have a history; define how to handle it (A should specify, but usually you skip t=0 in the sum).

3. Bits calculation per image

   * Compute bits_t = -log2(p_t).
   * Sum over t to get total_bits_model.
   * Return that number.

4. API / function boundary

   * Expose a function like:

     ```python
     def evaluate_bits(tokens, model) -> float:
         # returns compressed_bits for that image
     ```
   * This output + `orig_bits` from B goes to D.

5. Efficiency considerations

   * Watch out for long loops in Python. You may want to batch positions if possible (causal LM logits give you all next-token dists in one forward).
   * GPU vs CPU: try GPU if you have it, but keep code CPU-safe for demo.

Deliverables / handoffs

* `inference.py` that:

  * loads the model once,
  * takes all preprocessed sequences,
  * returns `compressed_bits`.

Blocking / dependencies

* Needs Workstream B’s output format (`tokens` tensor shape).
* Needs Workstream A’s definition of how to treat padding, BOS tokens, etc.

Recommended person

* Someone comfy with PyTorch, softmax, log probs, GPU debug.

---

## Workstream D. Metrics + Demo Owner (“Evaluation / Demo engineer”)

Goal: Tie it together and produce the final deliverable we show.

Tasks

1. Compression ratio calculator

   * For each image i:

     * Get `orig_bits_i` from B.
     * Get `model_bits_i` from C.
     * Compute:

       * bits_per_pixel_original = orig_bits_i / (H_i * W_i)
       * bits_per_pixel_model = model_bits_i / (H_i * W_i)
       * compression_ratio = orig_bits_i / model_bits_i
   * Store all of that in a table.

2. Aggregate stats

   * Compute averages across the 10 images:

     * avg bits/pixel (raw vs model)
     * avg compression ratio
   * This is the headline number we report.

3. Sanity checks / plotting

   * Flag any images where model_bits > orig_bits (i.e. “compression ratio < 1”). That means “the model can’t beat raw PNG-style naive 8 bits/channel”, which is useful to talk about.
   * Optional: simple matplotlib bar chart:

     * x-axis = image index 1..10
     * y-axis = compression ratio
     * mark >1 good (<1 bad)
   * Optional: dump a CSV of the table for slides.

4. Demo script / README

   * Write `run_demo.py` that:

     * loads images list
     * calls preprocess (B)
     * calls inference (C)
     * prints a nice table and averages
   * Write `README.md` explaining:

     * “How we tokenized”
     * “How we estimated bits”
     * “Final avg compression ratio on 10 random images”

Deliverables / handoffs

* Final report numbers
* Demo script
* README

Blocking / dependencies

* Needs outputs from B and C, plus formulas from A.

Recommended person

* Someone who’s good at presentation / storytelling and okay writing glue code.

---

## Cross-cutting considerations (everyone should know)

1. Header / format definition

   * Mentors said “need to define our own head of the compression format.”
   * For this sprint demo, we don’t actually have to write a binary file format, but we *should* define in words:

     * What metadata must be stored alongside the compressed stream? (image size, tiling info, quantization palette/codebook, maybe a checksum)
     * How we’d decode (greedy sampling or arithmetic decoding using the same model probs).
   * This explanation belongs in README (D) but depends on design choices from A.

2. Decoding story

   * Even if we don’t fully implement decompression, we should describe how it *would* work:

     * Given the model and arithmetic coding, you could reconstruct pixel tokens sequentially because at each step the model predicts the next token distribution.
   * That pitch makes the project feel “real compression,” not just math.

3. Reproducibility

   * Fix random seeds for image sampling so results are stable.
   * Record model checkpoint version string.

4. Division of labor sync

   * A finishes spec first.
   * Then B builds preprocessing and hands sample tensors to C to test forward().
   * C returns per-image bit counts back to D.
   * D runs final demo and writes README.
   * While B/C/D are coding, A can work on the “format header / decoding story / limitations” writeup and review README.

---

## TL;DR Task Assignment Board

**Person A — Modeling Spec / Compression Math**

* Lock in tokenization, sequence layout, tiling policy, BOS/pad handling, bits formula.
* Write mini-design doc + decoding story.

**Person B — Preprocessing / Dataloader**

* Implement `preprocess_image` to generate token sequences + metadata for 10 images.

**Person C — Inference / Bit Calculator**

* Load ImageGPT from HF.
* For each token sequence, compute per-token probs and total compressed_bits.

**Person D — Metrics / Demo / README**

* Compute compression ratios.
* Produce table, plot, averages.
* Write `run_demo.py` and README with final numbers.

That division should let all 4 people work mostly in parallel after A delivers the spec, and it ends with something demoable: “Here’s the average compression ratio of ImageGPT on 10 images, here’s how we did it.”
