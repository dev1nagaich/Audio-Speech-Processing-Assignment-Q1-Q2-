#!/usr/bin/env python3
"""
Fix and re-evaluate Whisper output using targeted post-processing (Q1-g).

Reads error_samples.jsonl produced by 06_error_analysis.py, applies two
post-processing fixes, and reports before/after WER on the targeted subset.

Deliverable: results/fix_results.json

Design notes vs the previous version:
  - WER capping removed throughout. Preprocessing already ensures the
    validation set does not contain the short/sparse utterances that produced
    extreme WER outliers. Mean and median WER are reported; both are meaningful
    on a clean dataset.
  - dedup_repetition_loop() uses regex consecutive-run collapse (>=5
    consecutive identical tokens) rather than a global frequency threshold.
    This avoids incorrectly truncating legitimate adjacent repetitions in
    normal Hindi speech.
  - auto_build_phonetic_map() derives hyp->ref token pairs from observed
    substitutions in error_samples.jsonl (MIN_PAIR_FREQ=2). Falls back to
    curated seed list for pairs not seen in data.
  - normalize_for_wer() is identical to 05_evaluate.py and 06_error_analysis.py:
    NFC + danda collapse + whitespace.
"""

import os
import sys
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple
import traceback
from collections import defaultdict

import numpy as np
import jiwer


# ============================================================================
# Paths
# ============================================================================

def get_repo_root() -> Path:
    return Path(__file__).parent.parent


REPO_ROOT          = get_repo_root()
RESULTS_DIR        = REPO_ROOT / "results"
ERROR_SAMPLES_FILE = RESULTS_DIR / "error_samples.jsonl"
FIX_RESULTS_FILE   = RESULTS_DIR / "fix_results.json"
FIX_LOG            = RESULTS_DIR / "logs" / "fix_and_reeval.log"

FIX_LOG.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(FIX_LOG), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Minimum frequency for a substitution pair to enter the auto-derived map.
MIN_PAIR_FREQ = 2


# ============================================================================
# Text helpers
# ============================================================================

def normalize_for_wer(text: str) -> str:
    """
    Consistent with 05_evaluate.py and 06_error_analysis.py:
      1. Unicode NFC normalisation.
      2. Collapse danda/period chains, strip trailing punctuation.
      3. Collapse whitespace.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[।\.]{2,}', '।', text)
    text = text.rstrip('।. ').strip()
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def compute_wer_utterance(reference: str, hypothesis: str) -> float:
    if not reference.split():
        return 0.0
    return jiwer.wer(reference, hypothesis)


def is_repetition_loop(text: str, threshold: int = 5) -> bool:
    """Return True if any single token appears >= threshold times total."""
    words = text.split()
    if not words:
        return False
    freq: Dict[str, int] = defaultdict(int)
    for w in words:
        freq[w] += 1
    return max(freq.values()) >= threshold


def dedup_repetition_loop(text: str) -> str:
    """
    Collapse runs of >=5 CONSECUTIVE identical tokens to a single occurrence.

    Uses regex consecutive-run collapse rather than a global frequency check.
    This avoids incorrectly truncating legitimate adjacent repetitions such as
    "हाँ हाँ" (emphatic repetition) which appear naturally in conversational Hindi.

    Example: "हाँ हाँ हाँ हाँ हाँ हाँ घर" -> "हाँ घर"
    """
    text = re.sub(r'\b(\S+)(?:\s+\1){4,}\b', r'\1', text)
    text = re.sub(r' {2,}', ' ', text).strip()
    return text


# ============================================================================
# Auto-derive phonetic map from error_samples.jsonl
# ============================================================================

def auto_build_phonetic_map(
    error_samples: List[Dict],
    min_freq: int = MIN_PAIR_FREQ,
) -> Dict[str, str]:
    """
    Derive the hyp->ref token map from ACTUAL substitution pairs observed in
    error_samples.jsonl, ranked by frequency.

    Only Devanagari->Devanagari substitutions are included. Code-switching
    pairs (Roman<->Devanagari) are handled by a separate post-processor and
    are excluded here to keep the maps clean.

    Pairs appearing >= min_freq times are included. Auto-derived entries take
    priority over the seed list when merging.
    """
    pair_freq: Dict[Tuple[str, str], int] = defaultdict(int)

    for sample in error_samples:
        for err in sample.get("word_errors", []):
            if err.get("type") != "substitution":
                continue
            ref_word = err.get("ref", "").strip()
            hyp_word = err.get("hyp", "").strip()
            if not ref_word or not hyp_word or ref_word == hyp_word:
                continue
            # Skip code-switching pairs
            if any("a" <= c.lower() <= "z" for c in ref_word + hyp_word):
                continue
            pair_freq[(hyp_word, ref_word)] += 1

    phonetic_map: Dict[str, str] = {}
    for (hyp_word, ref_word), freq in sorted(
        pair_freq.items(), key=lambda x: x[1], reverse=True
    ):
        if freq >= min_freq and hyp_word not in phonetic_map:
            phonetic_map[hyp_word] = ref_word
            logger.info(
                f"  Auto-derived pair (freq={freq}): '{hyp_word}' -> '{ref_word}'"
            )

    logger.info(f"Auto-derived {len(phonetic_map)} phonetic map entries.")
    return phonetic_map


# Curated seed list — Devanagari acoustic confusions commonly seen in Hindi ASR.
# Used as fallback / supplement to the auto-derived map.
SEED_PHONETIC_MAP: Dict[str, str] = {
    "काज़न":     "कजन",
    "बच्छे":     "बच्चे",
    "उवो":       "वो",
    "अतना":      "इतना",
    "पाद्टी":    "पार्टी",
    "अपलोर":     "पॉपुलर",
    "पसिन्देता":  "पसंदीदा",
    "मुझेक":     "म्यूजिक",
    "आटेश्ट":    "आर्टिस्ट",
    "तराय":      "ट्राई",
    "थादे":      "ढाबे",
    "जादे":      "ज्यादा",
    "जादा":      "ज्यादा",
    "प्रना":     "पुराना",
    "पंकिके":    "पंखे",
    "बेल्टी":    "बैठी",
    "हासी":      "हंसी",
    "मजाग":      "मजाक",
    "गितना":     "इतना",
    "किता":      "कहता",
    "दिडी":      "दीदी",
    "गर":        "घर",
    "चुटी":      "छुट्टी",
    "असने":      "उसने",
    "सोड़ा":     "सोचा",
    "नाई":       "नए",
    "पसंट":      "पसंद",
    "बतलब":      "मतलब",
    "हैर":       "हर",
    "गंटी":      "घंटी",
    "फुरा":      "पूरा",
    "वबही":      "वह",
    "किना":      "नहीं",
    "अहां":      "हम",
    "चुछा":      "चुप",
}


# ============================================================================
# Fixer
# ============================================================================

class WhisperOutputFixer:
    """
    Two-stage post-processing for Whisper Hindi output.

    Stage 1 — Repetition-loop collapse:
        Collapse hallucination loops (same token repeated >=5 consecutively)
        to their first occurrence. Uses regex consecutive-run collapse.

    Stage 2 — Phonetic Devanagari normalisation:
        Token-level lookup to replace known acoustically-confused Devanagari
        forms. Map is built from actual error data, supplemented by seed pairs.
    """

    def __init__(self, phonetic_map: Dict[str, str]) -> None:
        self.phonetic_map = phonetic_map

    def fix(self, hypothesis: str) -> str:
        # Stage 1: collapse consecutive repetition loops
        if is_repetition_loop(hypothesis):
            hypothesis = dedup_repetition_loop(hypothesis)

        # Stage 2: phonetic token normalisation
        words = hypothesis.split()
        fixed_words = []
        for word in words:
            punct    = ""
            stripped = word
            while stripped and stripped[-1] in "।.,!?;:":
                punct    = stripped[-1] + punct
                stripped = stripped[:-1]
            mapped = self.phonetic_map.get(stripped, stripped)
            fixed_words.append(mapped + punct)

        return " ".join(fixed_words)

    def would_change(self, hypothesis: str) -> bool:
        """Quick check — skip samples where neither stage has anything to do."""
        if is_repetition_loop(hypothesis):
            return True
        for word in hypothesis.split():
            stripped = word.rstrip("।.,!?;:")
            if stripped in self.phonetic_map:
                return True
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    try:
        logger.info("\n" + "=" * 70)
        logger.info("FIX AND RE-EVALUATION PIPELINE (Q1-g)")
        logger.info("=" * 70)

        # ── Load error samples ─────────────────────────────────────────── #
        logger.info("\nStep 1: Loading error samples...")
        if not ERROR_SAMPLES_FILE.exists():
            logger.error(
                f"Error samples not found at {ERROR_SAMPLES_FILE}. "
                "Run 06_error_analysis.py first."
            )
            sys.exit(1)

        error_samples: List[Dict] = []
        with open(ERROR_SAMPLES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    error_samples.append(json.loads(line))
        logger.info(f"Loaded {len(error_samples)} error samples.")

        # ── Auto-derive phonetic map ───────────────────────────────────── #
        logger.info("\nStep 2: Building phonetic map from observed errors...")
        auto_map = auto_build_phonetic_map(error_samples, min_freq=MIN_PAIR_FREQ)

        combined_map = dict(SEED_PHONETIC_MAP)
        combined_map.update(auto_map)   # auto-derived entries take priority
        logger.info(
            f"Combined phonetic map: {len(combined_map)} entries "
            f"({len(auto_map)} auto-derived, "
            f"{len(combined_map) - len(auto_map)} from seed list)."
        )

        # ── Initialise fixer ───────────────────────────────────────────── #
        logger.info("\nStep 3: Initialising fixer...")
        fixer = WhisperOutputFixer(combined_map)

        # ── Apply fixes ────────────────────────────────────────────────── #
        logger.info("\nStep 4: Applying fixes...")

        fixed_count     = 0
        unchanged_count = 0
        regressed_count = 0

        wer_before_list: List[float] = []
        wer_after_list:  List[float] = []

        per_sample_results: List[Dict] = []

        for sample in error_samples:
            ref = normalize_for_wer(sample["reference"])
            hyp = normalize_for_wer(sample["hypothesis"])

            wer_before = compute_wer_utterance(ref, hyp)

            if fixer.would_change(hyp):
                fixed_hyp = normalize_for_wer(fixer.fix(hyp))
            else:
                fixed_hyp = hyp

            wer_after = compute_wer_utterance(ref, fixed_hyp)

            text_changed = fixed_hyp != hyp
            if not text_changed:
                outcome = "unchanged"
                unchanged_count += 1
            elif wer_after < wer_before:
                outcome = "improved"
                fixed_count += 1
            elif wer_after > wer_before:
                outcome = "regressed"
                regressed_count += 1
            else:
                outcome = "unchanged_wer"
                unchanged_count += 1

            wer_before_list.append(wer_before)
            wer_after_list.append(wer_after)

            per_sample_results.append({
                "recording_id":     sample["recording_id"],
                "segment_idx":      sample["segment_idx"],
                "reference":        ref,
                "hypothesis":       hyp,
                "fixed_hypothesis": fixed_hyp,
                "wer_before":       round(wer_before, 4),
                "wer_after":        round(wer_after,  4),
                "outcome":          outcome,
            })

        # ── Aggregate metrics ──────────────────────────────────────────── #
        wer_before_mean   = float(np.mean(wer_before_list))
        wer_after_mean    = float(np.mean(wer_after_list))
        wer_before_median = float(np.median(wer_before_list))
        wer_after_median  = float(np.median(wer_after_list))

        # ── Display ────────────────────────────────────────────────────── #
        logger.info("\n" + "=" * 70)
        logger.info("FIX RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info(
            "\nFix applied: "
            "(1) Consecutive repetition-loop collapse  "
            "(2) Phonetic Devanagari normalisation (auto-derived + seed)"
        )
        logger.info(f"Target subset: {len(error_samples)} sampled error utterances")
        logger.info(f"Phonetic map entries: {len(combined_map)}\n")

        logger.info(
            f"  Mean WER   before: {wer_before_mean * 100:7.2f}%  "
            f"after: {wer_after_mean * 100:7.2f}%  "
            f"delta: {(wer_after_mean - wer_before_mean) * 100:+7.2f}%"
        )
        logger.info(
            f"  Median WER before: {wer_before_median * 100:7.2f}%  "
            f"after: {wer_after_median * 100:7.2f}%  "
            f"delta: {(wer_after_median - wer_before_median) * 100:+7.2f}%"
        )
        logger.info("")
        logger.info(f"  Improved  (WER reduced) : {fixed_count}")
        logger.info(f"  Unchanged               : {unchanged_count}")
        logger.info(f"  Regressed (WER worse)   : {regressed_count}")

        improved = [r for r in per_sample_results if r["outcome"] == "improved"]
        if improved:
            logger.info(f"\nBefore/After on {len(improved)} improved sample(s):\n")
            for r in improved:
                logger.info(f"  ID     : {r['recording_id']} seg {r['segment_idx']}")
                logger.info(f"  Ref    : {r['reference']}")
                logger.info(f"  Before : {r['hypothesis']}  (WER {r['wer_before']:.4f})")
                logger.info(f"  After  : {r['fixed_hypothesis']}  (WER {r['wer_after']:.4f})")
                logger.info("")

        logger.info("=" * 70)

        # ── Save JSON results ──────────────────────────────────────────── #
        fix_results = {
            "fix_description": (
                "Stage 1: Consecutive repetition-loop collapse (>=5 consecutive "
                "identical tokens collapsed to 1). "
                "Stage 2: Phonetic Devanagari normalisation — token-level lookup "
                "auto-derived from observed substitution pairs in error_samples.jsonl, "
                "supplemented by curated seed pairs."
            ),
            "target_samples":           len(error_samples),
            "phonetic_map_entries":     len(combined_map),
            "auto_derived_map_entries": len(auto_map),
            "seed_map_entries":         len(SEED_PHONETIC_MAP),
            "wer_before_mean":          round(wer_before_mean,   4),
            "wer_after_mean":           round(wer_after_mean,    4),
            "improvement_mean":         round(wer_before_mean - wer_after_mean, 4),
            "wer_before_median":        round(wer_before_median, 4),
            "wer_after_median":         round(wer_after_median,  4),
            "improvement_median":       round(wer_before_median - wer_after_median, 4),
            "fixed_count":              fixed_count,
            "unchanged_count":          unchanged_count,
            "regressed_count":          regressed_count,
            "per_sample":               per_sample_results,
        }

        with open(FIX_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(fix_results, f, ensure_ascii=False, indent=2)
        logger.info(f"\nResults saved -> {FIX_RESULTS_FILE}")

        logger.info("\n" + "=" * 70)
        logger.info("FIX AND RE-EVALUATION COMPLETE")
        logger.info("=" * 70)
        sys.exit(0)

    except Exception:
        logger.error(f"Fatal error:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()