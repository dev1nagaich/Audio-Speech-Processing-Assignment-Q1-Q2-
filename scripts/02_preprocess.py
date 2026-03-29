#!/usr/bin/env python3
"""
Preprocess downloaded audio and transcriptions.

FIXES vs previous version:
  [1] words_per_second filter added: segments with < 0.8 words/sec are
      discarded. These are backchannel/acknowledgment segments ("जी हां" in
      5.8s, "हम्म" in 4.8s) that contain mostly silence or crosstalk bleed.
      The model hallucinates full sentences on these, destroying WER.

  [2] Safe overwrite: SEGMENTS_DIR is wiped and recreated, MANIFEST_FILE
      is deleted before processing so re-runs start clean without stale data.

  [3] Filler word filter tightened: removed the 2s duration exception.

  [4] MIN_TEXT_LENGTH raised from 3 to 5 Devanagari code-points.

  [5] Unicode NFC normalisation applied before length check.

  [6] count_devanagari() helper used for meaningful content length.

  [7] Punctuation-only filter.

  [8] Segment-level snr_check() via RMS to skip near-silent segments.
"""

import os
import sys
import json
import logging
import unicodedata
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm


# ============================================================================
# Setup
# ============================================================================

def get_repo_root() -> Path:
    return Path(__file__).parent.parent


REPO_ROOT          = get_repo_root()
DATA_DIR           = REPO_ROOT / "data"
TRANSCRIPTIONS_DIR = DATA_DIR / "transcriptions"
RAW_AUDIO_DIR      = DATA_DIR / "raw_audio"
SEGMENTS_DIR       = DATA_DIR / "segments"
MANIFEST_FILE      = DATA_DIR / "segments_manifest.jsonl"
PREPROCESS_LOG     = DATA_DIR / "preprocess.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(PREPROCESS_LOG),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

TARGET_SR             = 16_000
MIN_SEGMENT_DURATION  = 1.0
MAX_SEGMENT_DURATION  = 30.0
MIN_DEVANAGARI_CHARS  = 5
MIN_RMS               = 0.002

# FIX [1]: minimum words per second to filter silence-heavy segments
# "जी हां" (2 words) in 5.8s = 0.34 wps → filtered
# "बिल्कुल" (1 word) in 5.2s = 0.19 wps → filtered
# Normal speech: 2-4 words/sec → passes
MIN_WORDS_PER_SECOND  = 0.8


# ============================================================================
# Audio helpers
# ============================================================================

def verify_and_resample_audio(audio_path: Path) -> Tuple[bool, int]:
    try:
        audio_data, sr = librosa.load(str(audio_path), sr=None, mono=True)
        if sr != TARGET_SR:
            logger.info(f"Resampling {audio_path.name}: {sr} Hz → {TARGET_SR} Hz")
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=TARGET_SR)
            sf.write(str(audio_path), audio_data, TARGET_SR)
        return True, TARGET_SR
    except Exception as exc:
        logger.error(f"Error processing {audio_path.name}: {exc}")
        return False, 0


# ============================================================================
# Text helpers
# ============================================================================

def normalize_text(text: str) -> str:
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def count_devanagari(text: str) -> int:
    return sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)


def is_punctuation_only(text: str) -> bool:
    return not any(c.isalpha() or c.isdigit() for c in text)


FILLERS = {
    "हाँ", "हूं", "जी", "नहीं",
    "हाँ।", "हूं।", "जी।", "नहीं।",
    "अ", "ह", "म्म", "ओह", "आं", "हम्म",
    "uh", "um", "hmm",
}


def is_filler(text: str) -> bool:
    return text.strip().rstrip("।.") in {f.rstrip("।.") for f in FILLERS}


def snr_check(audio: np.ndarray) -> bool:
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    return rms >= MIN_RMS


def should_include_segment(
    start: float, end: float, text: str, audio: Optional[np.ndarray] = None
) -> Tuple[bool, Optional[str]]:
    duration = end - start

    if duration < MIN_SEGMENT_DURATION:
        return False, f"too_short_{duration:.2f}s"
    if duration > MAX_SEGMENT_DURATION:
        return False, f"too_long_{duration:.2f}s"

    stripped = text.strip()

    if is_punctuation_only(stripped):
        return False, "punctuation_only"

    if is_filler(stripped):
        return False, "filler_word"

    if count_devanagari(stripped) < MIN_DEVANAGARI_CHARS:
        return False, f"insufficient_devanagari_{count_devanagari(stripped)}"

    # FIX [1]: words-per-second check
    word_count = len(stripped.split())
    wps = word_count / duration
    if wps < MIN_WORDS_PER_SECOND:
        return False, f"sparse_wps_{wps:.2f}"

    if audio is not None and not snr_check(audio):
        return False, "near_silent"

    return True, None


# ============================================================================
# Segment processing
# ============================================================================

def process_recording(
    recording_id: int,
    audio_path: Path,
    transcription_path: Path,
) -> Tuple[int, int, float]:
    try:
        audio_data, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)

        with open(transcription_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        kept = filtered = 0
        total_duration  = 0.0

        for seg_idx, seg in enumerate(segments):
            start      = seg["start"]
            end        = seg["end"]
            text       = seg["text"]
            speaker_id = seg.get("speaker_id")

            text = normalize_text(text)

            s_sample  = int(start * TARGET_SR)
            e_sample  = int(end   * TARGET_SR)
            seg_audio = audio_data[s_sample:e_sample]

            include, skip_reason = should_include_segment(start, end, text, seg_audio)
            if not include:
                filtered += 1
                continue

            duration = len(seg_audio) / TARGET_SR

            seg_name = f"{recording_id}_{seg_idx:04d}"
            seg_path = SEGMENTS_DIR / f"{seg_name}.wav"
            sf.write(str(seg_path), seg_audio, TARGET_SR)

            entry = {
                "audio_path":   str(seg_path.relative_to(REPO_ROOT)),
                "text":         text,
                "recording_id": recording_id,
                "duration":     round(duration, 3),
                "speaker_id":   speaker_id,
                "segment_idx":  seg_idx,
            }
            with open(MANIFEST_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            kept           += 1
            total_duration += duration

        return kept, filtered, total_duration

    except Exception:
        logger.error(f"Error processing recording {recording_id}:\n{traceback.format_exc()}")
        return 0, 0, 0.0


# ============================================================================
# Main
# ============================================================================

def preprocess_all() -> None:
    # FIX [2]: safe overwrite — wipe stale segments and manifest
    if SEGMENTS_DIR.exists():
        logger.info(f"Removing stale segments directory: {SEGMENTS_DIR}")
        shutil.rmtree(SEGMENTS_DIR)
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    if MANIFEST_FILE.exists():
        logger.info(f"Removing stale manifest: {MANIFEST_FILE}")
        MANIFEST_FILE.unlink()

    audio_files = sorted(RAW_AUDIO_DIR.glob("*_audio.wav"))
    logger.info(f"Found {len(audio_files)} audio files")

    if not audio_files:
        logger.warning("No audio files found. Run 01_download_data.py first.")
        sys.exit(1)

    # Step 1: Resample
    logger.info("=" * 70)
    logger.info("STEP 1: Audio verification & resampling")
    logger.info("=" * 70)
    for ap in tqdm(audio_files, desc="Verifying audio"):
        verify_and_resample_audio(ap)

    # Step 2: Segment & manifest
    logger.info("=" * 70)
    logger.info("STEP 2: Segment cutting & manifest creation")
    logger.info("=" * 70)

    # Track filter reasons
    filter_reasons: Dict[str, int] = {}
    total_before = total_after = 0
    total_dur    = 0.0

    for ap in tqdm(audio_files, desc="Processing segments"):
        rid = int(ap.stem.replace("_audio", ""))
        tp  = TRANSCRIPTIONS_DIR / f"{rid}_transcription.json"
        if not tp.exists():
            logger.warning(f"Transcription missing for recording {rid}")
            continue

        with open(tp, "r", encoding="utf-8") as f:
            segs = json.load(f)
        total_before += len(segs)

        kept, filtered, dur = process_recording(rid, ap, tp)
        total_after += kept
        total_dur   += dur

    # Stats
    logger.info("=" * 70)
    logger.info("PREPROCESSING STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Segments before filtering : {total_before}")
    logger.info(f"Segments after  filtering : {total_after}")
    logger.info(f"Removed                   : {total_before - total_after}")
    if total_before:
        logger.info(f"Retention rate            : {100 * total_after / total_before:.1f}%")
    logger.info(f"Total duration            : {total_dur / 3600:.2f} h")

    if MANIFEST_FILE.exists():
        durations = []
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                durations.append(json.loads(line)["duration"])
        d = np.array(durations)
        logger.info(
            f"Duration — min={d.min():.2f}s  max={d.max():.2f}s  "
            f"mean={d.mean():.2f}s  median={np.median(d):.2f}s"
        )
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Hindi conversational speech dataset"
    )
    parser.parse_args()
    try:
        preprocess_all()
        logger.info("Preprocessing complete!")
        sys.exit(0)
    except Exception:
        logger.error(f"Fatal error:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()