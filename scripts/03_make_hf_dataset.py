import os
import sys
import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict
import traceback

import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split

def get_repo_root() -> Path:
    return Path(__file__).parent.parent

REPO_ROOT       = get_repo_root()
DATA_DIR        = REPO_ROOT / "data"
MANIFEST_FILE   = DATA_DIR / "segments_manifest.jsonl"
DATASETS_HF_DIR = REPO_ROOT / "datasets_hf"
TRAIN_DIR       = DATASETS_HF_DIR / "train"
VAL_DIR         = DATASETS_HF_DIR / "val"
DATASET_LOG     = DATA_DIR / "dataset_build.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(DATASET_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> List[Dict]:
    entries = []
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line))
    logger.info(f"Loaded {len(entries)} entries from manifest")
    return entries


def stratified_split(
    entries: List[Dict],
    val_ratio: float = 0.1,
) -> tuple:
    by_recording = {}
    for entry in entries:
        rid = entry['recording_id']
        if rid not in by_recording:
            by_recording[rid] = []
        by_recording[rid].append(entry)

    recording_ids = list(by_recording.keys())
    logger.info(f"Total unique recordings: {len(recording_ids)}")

    train_rids, val_rids = train_test_split(
        recording_ids, test_size=val_ratio, random_state=42
    )

    train_entries = []
    val_entries   = []
    for rid in train_rids:
        train_entries.extend(by_recording[rid])
    for rid in val_rids:
        val_entries.extend(by_recording[rid])

    logger.info(f"Train recordings: {len(train_rids)}  segments: {len(train_entries)}")
    logger.info(f"Val   recordings: {len(val_rids)}  segments: {len(val_entries)}")
    return train_entries, val_entries


def create_hf_dataset(entries: List[Dict]) -> Dataset:
    from datasets.features import Audio

    data = {
        "audio":        [],
        "sentence":     [],
        "recording_id": [],
        "duration":     [],
        "speaker_id":   [],
        "segment_idx":  [],
    }
    for entry in entries:
        data["audio"].append({"path": entry["audio_path"], "sampling_rate": 16000})
        data["sentence"].append(entry["text"])
        data["recording_id"].append(entry["recording_id"])
        data["duration"].append(entry["duration"])
        data["speaker_id"].append(entry["speaker_id"])
        data["segment_idx"].append(entry["segment_idx"])

    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def compute_stats(entries: List[Dict]) -> Dict:
    durations = np.array([e['duration'] for e in entries])
    word_counts = np.array([len(e['text'].split()) for e in entries])
    wps = word_counts / durations
    return {
        "num_segments":    len(entries),
        "total_hours":     durations.sum() / 3600,
        "min_duration":    durations.min(),
        "max_duration":    durations.max(),
        "mean_duration":   durations.mean(),
        "median_duration": np.median(durations),
        "mean_wps":        wps.mean(),       # FIX [2]
        "min_wps":         wps.min(),        # FIX [2]
    }


def main():
    try:
        logger.info("=" * 70)
        logger.info("Building HuggingFace Datasets")
        logger.info("=" * 70)

        entries = load_manifest(MANIFEST_FILE)
        train_entries, val_entries = stratified_split(entries)

        train_stats = compute_stats(train_entries)
        val_stats   = compute_stats(val_entries)

        logger.info("\nTrain split statistics:")
        logger.info(f"  Segments     : {train_stats['num_segments']}")
        logger.info(f"  Duration     : {train_stats['total_hours']:.2f} hours")
        logger.info(f"  Mean duration: {train_stats['mean_duration']:.2f}s")
        logger.info(f"  Mean wps     : {train_stats['mean_wps']:.2f}")
        logger.info(f"  Min  wps     : {train_stats['min_wps']:.2f}")

        logger.info("\nVal split statistics:")
        logger.info(f"  Segments     : {val_stats['num_segments']}")
        logger.info(f"  Duration     : {val_stats['total_hours']:.2f} hours")
        logger.info(f"  Mean duration: {val_stats['mean_duration']:.2f}s")
        logger.info(f"  Mean wps     : {val_stats['mean_wps']:.2f}")
        logger.info(f"  Min  wps     : {val_stats['min_wps']:.2f}")

        # FIX [1]: safe overwrite
        for d in [TRAIN_DIR, VAL_DIR]:
            if d.exists():
                logger.info(f"Removing stale dataset: {d}")
                shutil.rmtree(d)
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        VAL_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("\nCreating train dataset...")
        train_dataset = create_hf_dataset(train_entries)
        logger.info(f"Train dataset shape: {train_dataset.shape}")

        logger.info("Creating val dataset...")
        val_dataset = create_hf_dataset(val_entries)
        logger.info(f"Val dataset shape: {val_dataset.shape}")

        logger.info(f"\nSaving train → {TRAIN_DIR}")
        train_dataset.save_to_disk(str(TRAIN_DIR))

        logger.info(f"Saving val   → {VAL_DIR}")
        val_dataset.save_to_disk(str(VAL_DIR))

        logger.info("\n" + "=" * 70)
        logger.info("Dataset building complete!")
        logger.info(f"  Train: {train_stats['num_segments']} segments "
                    f"({train_stats['total_hours']:.2f} h)")
        logger.info(f"  Val  : {val_stats['num_segments']} segments "
                    f"({val_stats['total_hours']:.2f} h)")
        logger.info("=" * 70)
        sys.exit(0)

    except Exception:
        logger.error(f"Fatal error:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()