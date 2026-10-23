# ProductLens

**Multimodal product retrieval with a fine-tuned CLIP model: search 121,549 products by text description or by photo.**

[![Live demo](https://img.shields.io/badge/demo-Hugging%20Face%20Space-FFD21E.svg)](https://huggingface.co/spaces/ranjeet258/ProductLens)
[![ONNX models](https://img.shields.io/badge/models-ONNX%20INT8-005CED.svg)](https://huggingface.co/ranjeet258/ProductLens-onnx)
[![Text to image R@1](https://img.shields.io/badge/text%E2%86%92image%20R%401-84.3%25-2E7D32.svg)](#results)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

| | |
|---|---|
| **Live demo** | [huggingface.co/spaces/ranjeet258/ProductLens](https://huggingface.co/spaces/ranjeet258/ProductLens) |
| **Models** | [huggingface.co/ranjeet258/ProductLens-onnx](https://huggingface.co/ranjeet258/ProductLens-onnx) |
| **Dataset** | [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) (CC BY 4.0) |

> ProductLens is a retrieval system built on a public research dataset. It is not a store: no products are sold and no prices are shown.

---

## Overview

ProductLens retrieves products from the Amazon Berkeley Objects (ABO) catalog given either a natural-language query (for example, "black leather office chair") or a photograph.

General-purpose CLIP models handle everyday images and captions well, but product listings are a different domain: titles dense with brand names, model numbers and dimensions, and studio photographs on plain backgrounds. ProductLens adapts OpenCLIP ViT-B/32 to this domain with contrastive fine-tuning on the catalog's own image-title pairs, which **doubles top-1 retrieval accuracy** over the zero-shot baseline.

The system is deployed in two forms:

1. **REST API.** A FastAPI service in Docker, backed by a FAISS IVF-PQ index.
2. **Serverless web app.** A static page in which the INT8-quantized encoder and the compressed index run entirely in the browser through ONNX Runtime Web. Queries and photos never leave the user's device.

---

## Results

Evaluated on a held-out test split. The split is made by product, so no product in the test set was seen during training.

| Metric | Zero-shot CLIP | ProductLens |
|---|---:|---:|
| Text → image Recall@1 | 41.6% | **84.3%** |
| Text → image Recall@5 | 65.8% | **96.7%** |
| Text → image Recall@10 | 74.2% | **98.2%** |
| Image → text Recall@1 | 48.7% | **83.8%** |
| Image → text Recall@5 | 73.5% | **96.5%** |
| Image → text Recall@10 | 80.9% | **98.2%** |
| Mean recall | 64.1 | **92.9** |

Recall@K is the percentage of queries for which the correct product appears in the top K results.

**Full-catalog retrieval.** The same held-out queries searched against all 121,549 products through the deployed IVF-PQ index:

| Text encoder | Size | Latency (1 CPU thread) | Recall@1 | Recall@10 |
|---|---:|---:|---:|---:|
| PyTorch fp32 | 242 MB | 126 ms | 28.2% | 72.3% |
| ONNX INT8 (browser) | 70 MB | 45 ms | 27.5% | 72.0% |

Exact-item recall drops on the full catalog because ABO lists many variants separately, for example one phone-case design for dozens of phone models, with identical photos. Error analysis shows that 74% of top-1 misses are such near-duplicate variants, and 61% of misses still have the exact item in the top 10. See the [analysis notebook](notebooks/02_results_analysis.ipynb) for details.

**Training progression (mean validation recall)**

| Zero-shot | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
|---:|---:|---:|---:|---:|---:|
| 63.5 | 89.1 | 91.0 | 92.1 | 93.0 | 93.2 |

**System characteristics**

| Property | Value |
|---|---|
| Catalog size | 121,549 products |
| Embedding dimension | 512 |
| FAISS IVF-PQ index | 7.1 MB (1,024 lists, 32 × 8-bit codes) |
| Catalog embedding throughput (T4) | 671 images/s |
| Index search, 16 of 1,024 lists | 0.30 ms per query (1 CPU thread) |
| Browser text encoder | 70 MB INT8, cosine 0.997 vs fp32 |
| Training time (T4) | about 39 minutes per epoch, 5 epochs |

---

## Method

### 1. Data preparation
`scripts/prepare_abo.py` parses the ABO listings, keeps English titles, pairs each title with the product's photographs, and splits the data by product into train, validation and test sets.

### 2. Contrastive fine-tuning
`src/train.py` fine-tunes OpenCLIP ViT-B/32 (LAION-2B pretrained weights) with a symmetric InfoNCE loss and a learnable temperature.

- Optimizer: AdamW (lr 1e-5, weight decay 0.1, no decay on biases, norms and temperature)
- Schedule: linear warmup followed by cosine decay
- Mixed-precision training with gradient clipping
- In-batch negatives: every other pair in a batch serves as a negative example

### 3. Evaluation
`src/evaluate.py` reports Recall@1/5/10 in both retrieval directions, comparing the zero-shot and fine-tuned models on the held-out test set.

### 4. Indexing
`src/build_index.py` embeds every catalog photograph and builds a FAISS IVF-PQ index. Each product is stored as 32 bytes of PQ codes plus a list id, roughly 35 times smaller than the raw float32 vectors.

### 5. Deployment
- **API:** `serve/app.py` loads the model and index from a local directory or from S3 (via an IAM role, so no credentials are stored on the host) and serves search endpoints from a Docker container.
- **Browser:** `scripts/export_static.py` exports both encoders to ONNX, applies dynamic INT8 quantization, verifies the result against PyTorch (cosine similarity above 0.97), and flattens the index into binary arrays for client-side search.

---

## Architecture

```
 Training (Kaggle GPU)                          Serving
 ---------------------                          -------
 ABO listings + photos                          A) FastAPI in Docker
   | prepare_abo.py: English titles,               model.pt, index.faiss, meta.parquet
   | split by product                              local folder or S3 via IAM role
   v                                               probes 16 of 1,024 lists
 train.py: symmetric InfoNCE,
 learnable temperature                          B) Static site on Hugging Face
   v                                               export_static.py: encoders -> ONNX -> INT8,
 build_index.py: embed catalog                     IVF-PQ index -> binary arrays
 photos -> FAISS IVF-PQ                            browser: ONNX Runtime Web + JavaScript
 (1024 lists, 32 x 8-bit codes)                    scan of every PQ code
```

## Implementation notes

- **Exhaustive client-side search.** The browser scores every product with the same asymmetric distance computation FAISS uses (coarse centroid plus PQ lookup tables), across all 1,024 inverted lists.
- **Similar-item search without an image model.** A product's vector is reconstructed from its PQ code and used directly as the query.
- **Lazy loading.** The image encoder is downloaded only when a user first searches by photo.
- **Native tokenizer.** OpenCLIP's byte-level BPE tokenizer is reimplemented in JavaScript and validated against token ids produced in Python during export.
- **Mixed-precision quantization.** After fine-tuning, the text encoder's first transformer block develops activation outliers that dynamic INT8 quantization cannot represent well (cosine 0.91 against PyTorch). Keeping that single block in fp32 restores a cosine of 0.998 at a cost of about 9 MB.

---

## Comparison

| Capability | ProductLens | Zero-shot CLIP | Keyword search |
|---|---|---|---|
| Search by photo | Yes | Yes | No |
| Understands product titles | Yes | Partially | Yes |
| Matches without shared words | Yes | Yes | No |
| Runs without a server | Yes | No | No |
| Text → image Recall@1 (this catalog) | 84.3% | 41.6% | Not measured |

---

## Getting started

### Requirements
Python 3.10 or later. A single NVIDIA T4 (16 GB) is sufficient for training. Indexing, export and serving run on CPU.

```bash
pip install -r requirements.txt
```

### 1. Download and prepare the data (about 3.5 GB)

```bash
mkdir abo && cd abo
wget https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar
wget https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar
tar xf abo-listings.tar && tar xf abo-images-small.tar && cd ..
python scripts/prepare_abo.py --abo-root abo --out-dir data --use-other-images
```

### 2. Fine-tune and evaluate

```bash
python src/train.py --train-csv data/train.csv --val-csv data/val.csv --image-root abo/images/small \
    --epochs 5 --batch-size 128 --lr 1e-5 --out runs/clip_abo
python src/evaluate.py --test-csv data/test.csv --image-root abo/images/small \
    --checkpoint runs/clip_abo/best.pt --out runs/clip_abo/test_metrics.json
```

If the GPU runs out of memory, reduce `--batch-size` or add `--unlocked-image-groups 4`.

No local GPU? Steps 1 to 3 run end to end on a free Kaggle T4:

```bash
pip install kaggle && kaggle auth login
python scripts/kaggle/submit_pipeline.py --user <kaggle-username>
```

### 3. Build the index

```bash
python src/build_index.py --catalog-csv data/catalog.csv --image-root abo/images/small \
    --checkpoint runs/clip_abo/best.pt --index-type ivfpq --out artifacts
cp runs/clip_abo/best.pt artifacts/model.pt
```

### 4a. Run the API

```bash
docker build -t productlens .
docker run -p 8000:8000 -v $PWD/artifacts:/app/artifacts productlens
curl "localhost:8000/search?q=wooden%20coffee%20table&k=5"
curl -X POST -F "file=@some_product.jpg" "localhost:8000/search_by_image?k=5"
```

To load artifacts from S3, set `-e ARTIFACTS_S3_URI=s3://<bucket>/productlens/` and attach an instance role with read access to that prefix.

### 4b. Publish the browser version

```bash
export HF_TOKEN=hf_...   # Hugging Face token with write access
python scripts/export_static.py --artifacts artifacts --out static_site --push <hf-username>
```

This creates the model repository `<hf-username>/ProductLens-onnx` and a static Space `<hf-username>/ProductLens`.

### Smoke test
Runs every stage end to end on a synthetic catalog with a randomly initialized model. No GPU and no downloads are required.

```bash
bash tests/smoke_test.sh
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Search interface |
| GET | `/health` | Health check (used by the Docker `HEALTHCHECK`) |
| GET | `/search?q=<text>&k=10` | Text query; returns the top-k products |
| POST | `/search_by_image?k=10` | Image upload; returns the top-k products |
| GET | `/similar` | Products similar to a given product |

## Project structure

```
ProductLens/
├── src/                          core library
│   ├── train.py                  contrastive fine-tuning
│   ├── evaluate.py               Recall@1/5/10, zero-shot vs fine-tuned
│   ├── build_index.py            Flat or IVF-PQ FAISS index, optional S3 upload
│   └── common.py                 model loading, dataset, metrics
├── scripts/                      pipeline entry points
│   ├── prepare_abo.py            ABO listings to image-text pairs, split by product
│   ├── export_static.py          ONNX export, accuracy-gated INT8 quantization, browser index
│   ├── bench_latency.py          p50 / p95 / p99 latency against the running API
│   ├── make_synthetic.py         synthetic catalog for the smoke test
│   ├── publish_static.sh         one-command publish of the browser version
│   └── kaggle/
│       └── submit_pipeline.py    submit the full GPU pipeline to Kaggle
├── notebooks/
│   ├── 01_training_pipeline_kaggle.ipynb   step-by-step training on a Kaggle T4
│   └── 02_results_analysis.ipynb           measured analysis, figures and comparisons
├── results/                      measured metrics (JSON) behind every number in this README
├── docs/
│   ├── ProductLens_Technical_Report.pdf    full technical report
│   ├── build_report.py           rebuilds the report from results/
│   └── figures/                  charts and architecture diagram
├── serve/                        FastAPI service and search page
├── static/                       browser application and Space configuration
├── tests/smoke_test.sh           end-to-end test on a synthetic catalog
├── Dockerfile                    CPU serving image
└── requirements.txt
```

Local-only folders (git-ignored): `artifacts/` (model, index, metadata), `static_site/` (built browser app), `data/`, `abo/`.

## Documentation

- **[Technical report (PDF)](docs/ProductLens_Technical_Report.pdf)**: problem, dataset, architecture, training, evaluation, quantization study, engineering challenges and comparison with alternative techniques.
- **[Training notebook](notebooks/01_training_pipeline_kaggle.ipynb)**: the full GPU pipeline, step by step.
- **[Analysis notebook](notebooks/02_results_analysis.ipynb)**: full-catalog evaluation, IVF probe sweep, INT8 vs fp32 encoder, storage footprint.

---

## Limitations

- The index is built from product photographs, so attributes that are not visible (warranty, battery life, materials) are matched poorly.
- Only products in the ABO catalog can be retrieved.
- Near-duplicate variants (same photo, different model or size) limit exact-item recall on the full catalog to 28.2% at rank 1 and 72.3% at rank 10. A text re-ranking stage over the top results is the natural next step.
- The browser version downloads the text encoder on the first visit. It is cached afterwards.
- Product images are loaded from the public ABO bucket. If that bucket becomes unavailable, search still works without images.

## Acknowledgements

- Based on the original multimodal product search project by Mahendra Meena (MIT License).
- Amazon Berkeley Objects dataset, CC BY 4.0. Collins et al., "ABO: Dataset and Benchmarks for Real-World 3D Object Understanding", CVPR 2022.
- [OpenCLIP](https://github.com/mlfoundations/open_clip), [FAISS](https://github.com/facebookresearch/faiss) and [ONNX Runtime Web](https://onnxruntime.ai).

## License

The code is released under the MIT License; see [LICENSE](LICENSE). The ABO dataset is distributed under its own CC BY 4.0 license.

---

Maintained by [Ranjeet Gupta](https://github.com/ranjeet258).
