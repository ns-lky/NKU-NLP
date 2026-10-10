"""Download the assignment's official pretrained assets inside this repository."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HF_HOME = ROOT / "artifacts" / "huggingface"
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
os.environ["HF_XET_CACHE"] = str(HF_HOME / "xet")
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
GLOVE_SHA256 = "95dde4dfd627ab26608d33e76d1195ec059734bd29089ea52cadb08d07c64544"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_glove() -> None:
    target = ROOT / "artifacts" / "glove.6B.100d.txt"
    if target.exists() and sha256(target) == GLOVE_SHA256:
        print(f"GloVe is already present and verified: {target}", flush=True)
        return
    archive = ROOT / "artifacts" / "downloads" / "glove.6B.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".zip.part")
    expected_size = 862182613
    if not archive.exists():
        for attempt in range(1, 9):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                request = urllib.request.Request(
                    "https://nlp.stanford.edu/data/glove.6B.zip",
                    headers={"User-Agent": "NKU-NLP/1.0", "Range": f"bytes={offset}-"},
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    resume = offset > 0 and response.status == 206
                    if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned an incorrect resume offset")
                    if not resume:
                        offset = 0
                    with partial.open("ab" if resume else "wb") as output:
                        next_progress = offset + 64 * 1024 * 1024
                        for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                            output.write(chunk)
                            offset += len(chunk)
                            if offset >= next_progress:
                                print(f"GloVe: {offset / 1024**2:.0f}/{expected_size / 1024**2:.0f} MiB", flush=True)
                                next_progress = offset + 64 * 1024 * 1024
                if partial.stat().st_size != expected_size:
                    raise ValueError("GloVe archive size does not match the official file")
                partial.replace(archive)
                break
            except (OSError, ValueError) as error:
                if attempt == 8:
                    raise
                print(f"Download attempt {attempt} failed: {error}; retrying from the saved offset", flush=True)
                time.sleep(2)
    if archive.stat().st_size != expected_size:
        raise ValueError("Unexpected GloVe archive size")
    # Reading the complete member verifies its ZIP CRC automatically.
    with zipfile.ZipFile(archive) as zipped:
        with zipped.open("glove.6B.100d.txt") as source, target.open("wb") as output:
            shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    if sha256(target) != GLOVE_SHA256:
        raise ValueError("Extracted GloVe file failed SHA-256 verification")
    print(f"GloVe extracted and verified: {target}", flush=True)


def download_bert() -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    options = dict(
        repo_id="google-bert/bert-base-uncased", revision=BERT_REVISION,
        cache_dir=HF_HOME / "hub",
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"],
    )
    try:
        snapshot = Path(snapshot_download(**options, local_files_only=True))
        if not all((snapshot / name).exists() for name in options["allow_patterns"]):
            raise LocalEntryNotFoundError("The local BERT snapshot is incomplete")
    except LocalEntryNotFoundError:
        snapshot = Path(snapshot_download(**options))
    print(f"Official BERT snapshot ready: {snapshot}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=["all", "glove", "bert"], default="all")
    args = parser.parse_args()
    if args.asset in ("all", "glove"):
        download_glove()
    if args.asset in ("all", "bert"):
        download_bert()
