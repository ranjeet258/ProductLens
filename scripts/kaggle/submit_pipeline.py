"""Submit the full ProductLens training pipeline to Kaggle as a GPU script kernel.

The repository's code is bundled into a single self-contained script, so nothing has to be published first.
The kernel downloads ABO, prepares pairs, fine-tunes, evaluates, builds the index and exports the browser site.

    pip install kaggle && kaggle auth login
    python scripts/kaggle/submit_pipeline.py --user <kaggle-username>
    kaggle kernels status <kaggle-username>/productlens-train
    kaggle kernels output <kaggle-username>/productlens-train -p kaggle_output

Outputs (in /kaggle/working): run/metrics.json, run/test_metrics.json, artifacts/, static_site/.
"""
import argparse
import base64
import io
import json
import os
import subprocess
import tempfile
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PIPELINE = r'''
import base64, glob, io, os, shutil, subprocess, time, zipfile

def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    subprocess.run(cmd, shell=True, check=True)
    print(f"[{time.time() - t:.0f}s]", flush=True)

zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))).extractall("/tmp/code")
os.chdir("/tmp/code")
W = "/kaggle/working"
sh("nvidia-smi")
sh("pip install -q open_clip_torch faiss-cpu onnx onnxruntime onnxscript")

ABO = "/tmp/abo"
os.makedirs(ABO, exist_ok=True)
for name in ("abo-listings.tar", "abo-images-small.tar"):
    sh(f"cd {ABO} && wget -q https://amazon-berkeley-objects.s3.amazonaws.com/archives/{name} && tar xf {name} && rm {name}")
img_meta = glob.glob(f"{ABO}/**/images.csv.gz", recursive=True)[0]
SMALL = os.path.join(os.path.dirname(os.path.dirname(img_meta)), "small")

sh(f"python scripts/prepare_abo.py --abo-root {ABO} --out-dir /tmp/data --use-other-images")
sh(f"python src/train.py --train-csv /tmp/data/train.csv --val-csv /tmp/data/val.csv --image-root {SMALL} "
   f"--epochs {EPOCHS} --batch-size {BATCH} --lr 1e-5 --workers 4 --out {W}/run")
sh(f"python src/evaluate.py --test-csv /tmp/data/test.csv --image-root {SMALL} "
   f"--checkpoint {W}/run/best.pt --workers 4 --out {W}/run/test_metrics.json")
sh(f"python src/build_index.py --catalog-csv /tmp/data/catalog.csv --image-root {SMALL} "
   f"--checkpoint {W}/run/best.pt --index-type ivfpq --workers 4 --out {W}/artifacts")
os.remove(f"{W}/run/last.pt")
shutil.move(f"{W}/run/best.pt", f"{W}/artifacts/model.pt")
sh(f"python scripts/export_static.py --artifacts {W}/artifacts --out {W}/static_site")
print("ALL DONE", flush=True)
'''


def bundle():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for d in ("src", "scripts", "static", "serve"):
            for root, dirs, files in os.walk(os.path.join(REPO, d)):
                dirs[:] = [x for x in dirs if x != "__pycache__"]
                for f in files:
                    p = os.path.join(root, f)
                    z.write(p, os.path.relpath(p, REPO))
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Kaggle username that owns the kernel")
    ap.add_argument("--slug", default="productlens-train")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--dry-run", action="store_true", help="write the kernel folder and print its path only")
    args = ap.parse_args()

    out = tempfile.mkdtemp(prefix="productlens_kernel_")
    with open(os.path.join(out, "run.py"), "w", encoding="utf-8") as f:
        f.write(f'CODE_B64 = "{bundle()}"\nEPOCHS = {args.epochs}\nBATCH = {args.batch_size}\n' + PIPELINE)
    meta = {"id": f"{args.user}/{args.slug}", "title": args.slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    with open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"kernel folder: {out}")
    if not args.dry_run:
        subprocess.run(["kaggle", "kernels", "push", "-p", out], check=True)


if __name__ == "__main__":
    main()
