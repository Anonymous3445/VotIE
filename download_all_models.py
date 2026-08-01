"""
Pre-download every pretrained weight used by the VotIE pipeline configs:
the HuggingFace encoders AND the Portuguese FastText vectors the BiLSTM baseline
needs.

Run this ONCE on a machine with internet access (e.g. the Deucalion login node)
before submitting run_all_configs.slurm, run_lomo_camera_ready.slurm or
run_standard_camera_ready.slurm. Compute nodes run with HF_HUB_OFFLINE=1 and no
route to the internet, so anything missing here fails there.

Everything lands under one cache root, which the SLURM scripts export as HF_HOME:

    <cache-dir>/hub/models--microsoft--mdeberta-v3-base/...   encoders
    <cache-dir>/fasttext/cc.pt.300.bin                        BiLSTM embeddings

Usage:
    # Default location (matches every SLURM script's HF_CACHE_DIR):
    python download_all_models.py

    # Custom location:
    python download_all_models.py --cache-dir /projects/F202600030AIVLABDEUCALION/<USER>/hf_cache

    # Subset of models:
    python download_all_models.py --models microsoft/mdeberta-v3-base

    # Encoders only / FastText only (it is a ~4.5 GB download, ~7.2 GB unpacked):
    python download_all_models.py --skip-fasttext
    python download_all_models.py --only-fasttext
"""

import argparse
import gzip
import logging
import os
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.embeddings_cache import (  # noqa: E402
    FASTTEXT_APPROX_BIN_BYTES,
    FASTTEXT_APPROX_GZ_BYTES,
    FASTTEXT_PATH_ENV,
    FASTTEXT_URL,
    fasttext_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("download_all_models")

# Every HuggingFace model referenced by configs/*.yaml and
# configs/municipality_experiments/*.yaml.
MODELS = [
    "microsoft/mdeberta-v3-base",       # deberta_crf.yaml, deberta_linear.yaml, deberta_crf_M0X.yaml
    "FacebookAI/xlm-roberta-base",      # xlmr_crf.yaml, xlmr_linear.yaml, xlmr_crf_M0X.yaml
    "neuralmind/bert-large-portuguese-cased",  # bert_crf.yaml, bert_linear.yaml
    "PORTULAN/gervasio-8b-portuguese-ptpt-decoder",  # GERVASIO LLM baseline, served locally by run_gervasio.slurm
]

# Default cache dir mirrors the HF_HOME used in the SLURM scripts:
#   <repo_parent>/hf_cache  (one level up from this file's directory)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "hf_cache"


def download(model_id: str, cache_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    # cache_dir is the HF_HOME root. snapshot_download (via HF_HOME) will
    # place files in <cache_dir>/hub/models--owner--name/, which is exactly
    # where transformers.from_pretrained looks when HF_HOME=<cache_dir>.
    log.info("Fetching %s into %s", model_id, cache_dir)
    path = snapshot_download(repo_id=model_id)
    log.info("  → %s", path)


def _human(n: float) -> str:
    return f"{n / 1024 ** 3:.1f} GB"


def _report_progress(label: str, expected: int):
    """urlretrieve reporthook that logs every ~5%, so batch logs stay readable."""
    state = {"last": -1}

    def hook(block_num: int, block_size: int, total_size: int) -> None:
        total = total_size if total_size > 0 else expected
        done = block_num * block_size
        pct = min(int(100 * done / total), 100) if total else 0
        if pct >= state["last"] + 5:
            state["last"] = pct
            log.info("  %s %3d%% (%s / %s)", label, pct, _human(done), _human(total))

    return hook


def download_fasttext(cache_dir: Path, force: bool = False) -> Path:
    """Fetch and decompress Facebook's Portuguese FastText vectors.

    Downloads to a .part file and decompresses to a .tmp file, renaming only on
    success, so an interrupted run can never leave a truncated binary that
    fasttext would later fail to load halfway through a queued job.
    """
    target = fasttext_path(cache_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        size = target.stat().st_size
        if size < FASTTEXT_APPROX_BIN_BYTES * 0.9:
            raise RuntimeError(
                f"{target} exists but is only {_human(size)}; expected about "
                f"{_human(FASTTEXT_APPROX_BIN_BYTES)}. It is probably a truncated "
                f"download — delete it or re-run with --force-fasttext."
            )
        log.info("FastText already present: %s (%s)", target, _human(size))
        return target

    # Both files coexist during decompression, so the peak requirement is the
    # sum. Checking now avoids discovering it after a 4.5 GB download.
    needed = FASTTEXT_APPROX_GZ_BYTES + FASTTEXT_APPROX_BIN_BYTES
    free = shutil.disk_usage(target.parent).free
    if free < needed:
        raise RuntimeError(
            f"Not enough space in {target.parent}: {_human(free)} free, "
            f"about {_human(needed)} needed (gzip + decompressed, both present "
            f"at once). Point --cache-dir at a larger filesystem."
        )

    archive = target.with_suffix(".bin.gz.part")
    staging = target.with_suffix(".bin.tmp")
    log.info("Downloading FastText vectors (%s) from %s",
             _human(FASTTEXT_APPROX_GZ_BYTES), FASTTEXT_URL)
    try:
        urllib.request.urlretrieve(
            FASTTEXT_URL, archive,
            reporthook=_report_progress("download", FASTTEXT_APPROX_GZ_BYTES),
        )
        log.info("Decompressing to %s (%s)", target, _human(FASTTEXT_APPROX_BIN_BYTES))
        with gzip.open(archive, "rb") as src, staging.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=32 * 1024 * 1024)
        staging.replace(target)
    finally:
        for leftover in (archive, staging):
            if leftover.exists():
                leftover.unlink()

    log.info("  → %s (%s)", target, _human(target.stat().st_size))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="HF cache directory (default: %(default)s). "
             "Point HF_HOME / TRANSFORMERS_CACHE here in your SLURM job.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="HuggingFace model IDs to download (default: every model used by the configs).",
    )
    parser.add_argument(
        "--skip-fasttext",
        action="store_true",
        help="Skip the FastText vectors (~4.5 GB download, ~7.2 GB unpacked). "
             "The BiLSTM-CRF baseline will not run without them.",
    )
    parser.add_argument(
        "--only-fasttext",
        action="store_true",
        help="Fetch only the FastText vectors, skipping the HuggingFace encoders.",
    )
    parser.add_argument(
        "--force-fasttext",
        action="store_true",
        help="Re-download the FastText vectors even if they are already present.",
    )
    args = parser.parse_args()

    if args.skip_fasttext and args.only_fasttext:
        parser.error("--skip-fasttext and --only-fasttext are mutually exclusive")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Point huggingface_hub at the same HF_HOME the SLURM jobs will use, so
    # snapshot_download writes files to <cache_dir>/hub/ — the layout that
    # transformers.from_pretrained reads from in offline mode.
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "hub")

    log.info("Cache dir: %s", cache_dir)
    log.info("  encoders -> %s/hub", cache_dir)
    log.info("  fasttext -> %s", fasttext_path(cache_dir))

    failed = []

    if not args.only_fasttext:
        log.info("Models to fetch: %s", ", ".join(args.models))
        for model_id in args.models:
            try:
                download(model_id, cache_dir)
            except Exception as exc:
                log.error("Failed to download %s: %s", model_id, exc)
                failed.append(model_id)

    if not args.skip_fasttext:
        try:
            download_fasttext(cache_dir, force=args.force_fasttext)
        except Exception as exc:
            log.error("Failed to download FastText vectors: %s", exc)
            failed.append("fasttext/cc.pt.300.bin")

    if failed:
        log.error("Done with errors. Failed: %s", ", ".join(failed))
        sys.exit(1)

    log.info("All weights cached. In your SLURM job, set:")
    log.info("  export HF_HOME=%s", cache_dir)
    log.info("  export HF_HUB_CACHE=%s/hub", cache_dir)
    log.info("  export TRANSFORMERS_CACHE=%s/hub", cache_dir)
    log.info("  export TRANSFORMERS_OFFLINE=1")
    log.info("  export HF_HUB_OFFLINE=1")
    if not args.skip_fasttext:
        # Resolution also works from HF_HOME alone; exporting the explicit path
        # keeps it working if a job ever points HF_HOME somewhere else.
        log.info("  export %s=%s", FASTTEXT_PATH_ENV, fasttext_path(cache_dir))


if __name__ == "__main__":
    main()
