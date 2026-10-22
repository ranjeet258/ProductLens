# ProductLens

**Describe it or photograph it. ProductLens finds it among 121,549 products.**

A fine-tuned CLIP model that searches Amazon product listings by text or by photo. It ships two ways: a FastAPI service in Docker, and a website that runs the whole search inside your browser, with no server.

[![Live demo](https://img.shields.io/badge/demo-Hugging%20Face%20Space-FFD21E.svg)](https://huggingface.co/spaces/ranjeet258/ProductLens)
[![ONNX models](https://img.shields.io/badge/models-ONNX%20INT8-005CED.svg)](https://huggingface.co/ranjeet258/ProductLens-onnx)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Text to image R@1](https://img.shields.io/badge/text%E2%86%92image%20R%401-84.3%25-2E7D32.svg)](#results)
[![No backend](https://img.shields.io/badge/browser%20demo-no%20server-555.svg)](static/index.html)

**Try it:** [huggingface.co/spaces/ranjeet258/ProductLens](https://huggingface.co/spaces/ranjeet258/ProductLens)
**Models:** [ranjeet258/ProductLens-onnx](https://huggingface.co/ranjeet258/ProductLens-onnx)

> **Not a shopping site.** Nothing is for sale and no prices are shown. ProductLens is a retrieval system trained and measured on a public research dataset, Amazon Berkeley Objects (ABO).

---

## The idea in one line

Type "black leather office chair" or upload a photo of one, and get the closest matching products back in milliseconds.

## Why fine-tune CLIP at all?

Off-the-shelf CLIP understands images and text in general. It does not understand product listings: titles packed with brand names, model numbers and sizes, shot on plain white backgrounds. Fine-tuning on the catalog's own photo and title pairs teaches it that language: the right product comes first **twice as often**.

| Held-out test set | Zero-shot CLIP | ProductLens (fine-tuned) |
|---|---|---|
| Text → image Recall@1 | 41.6% | **84.3%** |
| Text → image Recall@5 | 65.8% | **96.7%** |
| Text → image Recall@10 | 74.2% | **98.2%** |
| Image → text Recall@1 | 48.7% | **83.8%** |
| Image → text Recall@10 | 80.9% | **98.2%** |

Recall@1 is how often the exact product described comes first. The split is by product, so no product appears in both train and test.

---

## How ProductLens works

### 1. Listings become training pairs
`scripts/prepare_abo.py` reads the ABO listings, keeps English titles, pairs each one with its product photos, and splits by product into train, validation and test.

### 2. Contrastive fine-tuning
`src/train.py` trains OpenCLIP ViT-B/32 (LAION-2B weights) with a symmetric InfoNCE loss and a learnable temperature. It uses AdamW, warmup followed by a cosine schedule, and mixed precision. Every other item in a batch acts as a negative. Five epochs on a Kaggle T4 GPU, about 39 minutes each. Mean validation recall: 63.5 (zero-shot), 89.1, 91.0, 92.1, 93.0, 93.2.

### 3. Honest evaluation
`src/evaluate.py` reports Recall@1/5/10 in both directions, zero-shot against fine-tuned, on products the model never saw.

### 4. A compact index
`src/build_index.py` embeds every catalog photo into a 512-d vector and builds a FAISS IVF-PQ index (1,024 lists, 32 × 8-bit codes). Each product costs 32 bytes plus a list id.

### 5a. Served as an API
`serve/app.py` loads the model and index from a folder or from S3 through an IAM role, so no keys live on the machine. It runs in Docker.

### 5b. Served with no server at all
`scripts/export_static.py` exports both encoders to ONNX, quantizes them to INT8, and flattens the index into plain arrays. The page runs the text encoder with ONNX Runtime Web and scans the PQ codes in JavaScript.

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
 learnable temperature                          B) Static site on Hugging Face, no server
   v                                               export_static.py: encoders -> ONNX -> INT8,
 build_index.py: embed catalog                     IVF-PQ index -> plain arrays
 photos -> FAISS IVF-PQ                            browser: ONNX Runtime Web + a JavaScript
 (1024 lists, 32 x 8-bit codes)                    scan of every PQ code
```

## Under the hood

- **The browser scans everything.** It scores every product with the same IVF-PQ formula FAISS uses (coarse centroid plus PQ lookup tables), across all 1,024 lists.
- **"More like this" needs no image model.** A product's vector is rebuilt from its PQ code and used as the query.
- **Photo search loads lazily.** The image encoder downloads only when someone uploads a photo.
- **A hand-written tokenizer.** open_clip's byte-level BPE is reimplemented in JavaScript. `export_static.py` writes test strings with the Python token ids, and the page reproduces them exactly.
- **Private by design.** In the browser version your photo is encoded on your own device and never uploaded.

---

## Results

| | Value |
|---|---|
| Text → image Recall@1, held out | 84.3% (zero-shot 41.6%) |
| Image → text Recall@1, held out | 83.8% (zero-shot 48.7%) |
| Mean recall, held out | 92.9 (zero-shot 64.1) |
| Catalog size | 121,549 products |
| FAISS IVF-PQ index | 7.1 MB |
| Catalog embedding speed, T4 | 671 images/s |

The browser page prints its own timing on every search, split into model time and index scan time.

## ProductLens vs the alternatives

| | ProductLens | Zero-shot CLIP | Keyword search |
|---|---|---|---|
| Search by photo | ✓ | ✓ | ✗ |
| Understands product titles | ✓ | partly | ✓ |
| Finds a match with no shared words | ✓ | ✓ | ✗ |
| Runs with no server | ✓ | ✗ | ✗ |
| Text → image Recall@1 on this catalog | 84.3% | 41.6% | not measured |

---

## Quick start

**1. Get the data (about 3.5 GB).**

```bash
mkdir abo && cd abo
wget https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar
wget https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar
tar xf abo-listings.tar && tar xf abo-images-small.tar && cd ..
pip install -r requirements.txt
python scripts/prepare_abo.py --abo-root abo --out-dir data --use-other-images
```

**2. Fine-tune and evaluate (one T4 is enough).**

```bash
python src/train.py --train-csv data/train.csv --val-csv data/val.csv --image-root abo/images/small \
    --epochs 5 --batch-size 128 --lr 1e-5 --out runs/clip_abo
python src/evaluate.py --test-csv data/test.csv --image-root abo/images/small \
    --checkpoint runs/clip_abo/best.pt --out runs/clip_abo/test_metrics.json
```

Out of GPU memory? Lower `--batch-size` or add `--unlocked-image-groups 4`.

**3. Build the index.**

```bash
python src/build_index.py --catalog-csv data/catalog.csv --image-root abo/images/small \
    --checkpoint runs/clip_abo/best.pt --index-type ivfpq --out artifacts
cp runs/clip_abo/best.pt artifacts/model.pt
```

**4a. Run the API.**

```bash
docker build -t productlens .
docker run -p 8000:8000 -v $PWD/artifacts:/app/artifacts productlens
curl "localhost:8000/search?q=wooden%20coffee%20table&k=5"
curl -X POST -F "file=@some_product.jpg" "localhost:8000/search_by_image?k=5"
```

To load artifacts from S3, add `-e ARTIFACTS_S3_URI=s3://<bucket>/productlens/` and give the instance a role that can read that prefix.

**4b. Publish the browser version.**

```bash
export HF_TOKEN=hf_...   # a Hugging Face write token
python scripts/export_static.py --artifacts artifacts --out static_site --push <hf-username>
```

This creates `<hf-username>/ProductLens-onnx` for the encoders and a free static Space `<hf-username>/ProductLens` for the page.

**Smoke test, no GPU and no downloads.** Runs every script end to end on a synthetic catalog with a random model:

```bash
bash tests/smoke_test.sh
```

---

## API reference

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Search page |
| GET | `/health` | Health check, also used by the Docker `HEALTHCHECK` |
| GET | `/search?q=...&k=10` | Text query, returns the closest products |
| POST | `/search_by_image?k=10` | Upload a photo, returns the closest products |
| GET | `/similar` | More products like a given one |

## Project layout

```
ProductLens/
├── scripts/
│   ├── prepare_abo.py        ABO listings to English image-text pairs, split by product
│   ├── export_static.py      INT8 ONNX encoders, index arrays, product info, page
│   ├── bench_latency.py      p50 / p95 / p99 against the running API
│   ├── make_synthetic.py     tiny fake catalog for the smoke test
│   └── publish_static.sh
├── src/
│   ├── train.py              contrastive fine-tuning
│   ├── evaluate.py           Recall@1/5/10, zero-shot vs fine-tuned
│   ├── build_index.py        Flat or IVF-PQ FAISS index, optional upload to S3
│   └── common.py
├── serve/                    FastAPI service and its page
├── static/                   browser page and the Hugging Face Space card
├── tests/smoke_test.sh
├── Dockerfile                CPU image, base weights baked in
└── requirements.txt
```

---

## Known limits

- Questions a photo cannot answer (warranty, battery life, materials you can't see) match poorly, because the index is built from photos.
- Only products in the ABO catalog are searchable.
- Recall is measured on the held-out test products, not against the full catalog, where there are more look-alikes.
- The first visit downloads the text encoder (tens of MB), cached afterwards.
- Product photos load from the public ABO bucket. If that bucket moves, results still work, without pictures.

## Credits

- Data: Amazon Berkeley Objects, CC BY 4.0 (Collins et al., 2022).
- Built with [OpenCLIP](https://github.com/mlfoundations/open_clip), [FAISS](https://github.com/facebookresearch/faiss) and [ONNX Runtime Web](https://onnxruntime.ai).

## License

Code: MIT, see [LICENSE](LICENSE). The dataset keeps its own CC BY 4.0 license.

---

*ProductLens is maintained by [Ranjeet Gupta](https://github.com/ranjeet258).*
