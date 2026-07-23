"""RAF-DB placement instructions + structure verification.

RAF-DB requires a signed license/agreement request from the authors — it is
NOT a free open download. This script does NOT auto-download anything; it prints
placement instructions and verifies the expected on-disk structure, failing
loudly if files are missing.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/download_data.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_data")

INSTRUCTIONS_OFFICIAL = """
================================================================================
RAF-DB — official layout (data.format: rafdb_official)
================================================================================
RAF-DB is released under a license agreement and cannot be auto-downloaded.

1. Request access from the authors:
   http://www.whdeng.cn/RAF/model1.html
2. After approval, download the *basic* emotions subset.
3. Place the files so the layout matches (relative to the data root):

   {root}/
   +-- basic/
       +-- Image/
       |   +-- aligned/                 # pre-aligned face crops (used for training)
       |       +-- train_00001_aligned.jpg
       |       +-- ...
       +-- EmoLabel/
           +-- list_patition_label.txt  # "<image_name> <label_id>" per line

   Label file lines look like:  train_00001.jpg 5
   (space-separated: filename, 1-indexed label). Aligned crops add the
   "_aligned" suffix that the label-file names omit.
================================================================================
"""

INSTRUCTIONS_IMAGEFOLDER = """
================================================================================
RAF-DB — ImageFolder layout (data.format: imagefolder, e.g. Kaggle mirror)
================================================================================
The Kaggle mirrors redistribute the (license-gated) RAF-DB. Use for personal /
educational / research purposes.

1. Download a mirror, e.g.:
   kaggle datasets download -d shuvoalok/raf-db-dataset -p {root} --unzip
2. Ensure the layout matches (relative to the data root):

   {root}/
   +-- {train_dir}/
   |   +-- 1/  2/  3/  4/  5/  6/  7/    # folder name = RAF-DB 1-indexed label id
   |       +-- *.jpg
   +-- {test_dir}/
       +-- 1/  2/  3/  4/  5/  6/  7/
           +-- *.jpg

   Class folders may be RAF-DB ids (1..7) or emotion names (happy, sad, ...).
   Adjust data.root / data.train_dir / data.test_dir in the config to match
   wherever your extracted files actually live.
================================================================================
"""


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def verify_official(root: Path, aligned_dir: str, label_file: str) -> bool:
    """Verify the official RAF-DB structure exists under ``root``."""
    ok = True
    aligned = root / aligned_dir
    labels = root / label_file

    for path, desc in [
        (root, "data root"),
        (root / "basic", "basic/ directory"),
        (aligned, "aligned images directory"),
        (labels, "label/partition file"),
    ]:
        if path.exists():
            logger.info("OK: %s exists (%s)", desc, path)
        else:
            logger.error("MISSING: %s not found (%s)", desc, path)
            ok = False

    if aligned.is_dir():
        n_imgs = sum(1 for _ in aligned.glob("*_aligned.*"))
        if n_imgs == 0:
            logger.error("No '*_aligned.*' images found under %s", aligned)
            ok = False
        else:
            logger.info("Found %d aligned images under %s", n_imgs, aligned)
    return ok


def _count_images(split_root: Path) -> int:
    """Count image files nested under an ImageFolder split directory."""
    if not split_root.is_dir():
        return 0
    return sum(
        1 for p in split_root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def verify_imagefolder(root: Path, train_dir: str, test_dir: str) -> bool:
    """Verify the ImageFolder (Kaggle) layout exists under ``root``."""
    ok = True
    train_root = root / train_dir
    test_root = root / test_dir

    for path, desc in [
        (root, "data root"),
        (train_root, "train split directory"),
        (test_root, "test split directory"),
    ]:
        if path.exists():
            logger.info("OK: %s exists (%s)", desc, path)
        else:
            logger.error("MISSING: %s not found (%s)", desc, path)
            ok = False

    for split_root, name in [(train_root, "train"), (test_root, "test")]:
        n = _count_images(split_root)
        if n == 0:
            logger.error("No images found under %s split (%s)", name, split_root)
            ok = False
        else:
            n_classes = sum(1 for p in split_root.iterdir() if p.is_dir()) if split_root.is_dir() else 0
            logger.info("Found %d images across %d class folders in %s split", n, n_classes, name)
    return ok


INSTRUCTIONS_AFFECTNET = """
================================================================================
AffectNet (auxiliary training data) — manual acquisition required
================================================================================
AffectNet is license-gated: request access from the authors at
http://mohammadmahoor.com/affectnet/ . Unofficial Kaggle mirrors also exist
(e.g. "noamsegal/affectnet-training-data", "mouadriali/affectnetsample").

Place it ImageFolder-style; class folders may be emotion names or numeric ids.
Then point an entry in config `data.extra_train_datasets` at it, e.g.:

  data:
    extra_train_datasets:
      - name: affectnet
        root: data/affectnet
        train_dir: train
        class_alias:            # map AffectNet's names -> our 7 classes
          anger: angry          # AffectNet says "anger", we use "angry"
          contempt: drop        # 8th class -> excluded (RAF-DB has 7)
          # happy/sad/surprise/fear/disgust/neutral match by name automatically

AffectNet's official numeric ids are:
  0 neutral  1 happy  2 sad  3 surprise  4 fear  5 disgust  6 anger  7 contempt
(so with numeric folders use: {'6': angry, '7': drop} and the rest map by id).

Note: AffectNet is large (100k-280k+ images). Set train.samples_per_epoch to
keep epochs a reasonable length, and train.balanced_sampler: true so the pooled
classes stay balanced.
================================================================================
"""


def verify_extra_datasets(cfg) -> bool:
    """Verify each configured auxiliary training dataset exists and has images."""
    ok = True
    for e in cfg.data.extra_train_datasets:
        print(INSTRUCTIONS_AFFECTNET if e.name.lower() == "affectnet" else "")
        split_root = Path(e.root) / e.train_dir
        n = _count_images(split_root)
        if n == 0:
            logger.error("Extra dataset '%s': no images under %s", e.name, split_root)
            ok = False
        else:
            n_classes = sum(1 for p in split_root.iterdir() if p.is_dir())
            logger.info(
                "Extra dataset '%s': found %d images across %d class folders (%s)",
                e.name, n, n_classes, split_root,
            )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="RAF-DB placement + verification")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.data.root)

    if cfg.data.format == "imagefolder":
        print(INSTRUCTIONS_IMAGEFOLDER.format(
            root=root, train_dir=cfg.data.train_dir, test_dir=cfg.data.test_dir
        ))
        ok = verify_imagefolder(root, cfg.data.train_dir, cfg.data.test_dir)
    else:
        print(INSTRUCTIONS_OFFICIAL.format(root=root))
        ok = verify_official(root, cfg.data.aligned_dir, cfg.data.label_file)

    if cfg.data.extra_train_datasets:
        ok = verify_extra_datasets(cfg) and ok

    if ok:
        logger.info("Dataset structure verified successfully (format=%s).", cfg.data.format)
    else:
        logger.error("Dataset structure verification FAILED. See instructions above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
