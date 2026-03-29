#!/usr/bin/env python3
"""
Error analysis for fine-tuned Whisper model (Q1-d and Q1-e).

Runs the fine-tuned model over the validation set, systematically samples
>=25 error utterances, builds a data-driven error taxonomy with 3-5 examples
per category, and writes two deliverables:
  results/error_samples.jsonl   — sampled error records
  results/error_taxonomy.md     — taxonomy report with recommendations

Notes on design decisions vs the original version:
  - WER capping removed: preprocessing already filters short/sparse/silent
    segments, so the repetition-loop outlier problem that drove the capping
    logic is no longer present in the validation set.
  - is_repetition_loop threshold raised to 8 (was 5). With clean data, a
    token appearing 5 times is more plausible as legitimate speech repetition
    (e.g. "एक एक एक एक एक" in a countdown). 8 consecutive is clearly a loop.
  - normalize_for_wer() is consistent with 05_evaluate.py: NFC + danda
    collapse + whitespace only.
  - All other logic (systematic_error_sampling, get_word_errors, taxonomy
    classifier, generate_taxonomy_report) is unchanged — the categories
    emerge from real substitution/deletion/insertion patterns in the data.
"""

import os
import sys
import json
import logging
import argparse
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Any
import traceback

import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_from_disk
import jiwer
from tqdm import tqdm
from collections import defaultdict


# ============================================================================
# Paths
# ============================================================================

def get_repo_root() -> Path:
    return Path(__file__).parent.parent


REPO_ROOT           = get_repo_root()
VAL_DIR             = REPO_ROOT / "datasets_hf" / "val"
MODEL_DIR           = REPO_ROOT / "models" / "whisper-small-hindi"
RESULTS_DIR         = REPO_ROOT / "results"
ERROR_SAMPLES_FILE  = RESULTS_DIR / "error_samples.jsonl"
ERROR_TAXONOMY_FILE = RESULTS_DIR / "error_taxonomy.md"
ERROR_ANALYSIS_LOG  = RESULTS_DIR / "logs" / "error_analysis.log"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ERROR_ANALYSIS_LOG.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(ERROR_ANALYSIS_LOG),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# GPU helpers
# ============================================================================

def setup_gpu(gpu_id: int = 0) -> None:
    if torch.cuda.is_available():
        total = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= total:
            logger.warning(f"gpu_id={gpu_id} invalid for {total} GPU(s). Using GPU 0.")
            gpu_id = 0
        props = torch.cuda.get_device_properties(gpu_id)
        logger.info(f"Using GPU {gpu_id}: {props.name}  {props.total_memory / 1e9:.1f} GB")
    else:
        logger.warning("CUDA not available, using CPU.")


def get_device(gpu_id: int = 0) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    total = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= total:
        gpu_id = 0
    return torch.device(f"cuda:{gpu_id}")


# ============================================================================
# Text helpers
# ============================================================================

def normalize_for_wer(text: str) -> str:
    """
    Consistent with 05_evaluate.py:
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


def get_word_errors(reference: str, hypothesis: str) -> List[Dict]:
    """
    Word-level error list using SequenceMatcher.

    Inside the 'replace' opcode, emits the correct type when one side of
    the aligned span is shorter than the other:
      - Both sides present → substitution
      - Ref exhausted      → insertion
      - Hyp exhausted      → deletion
    """
    from difflib import SequenceMatcher

    ref_words = reference.split()
    hyp_words = hypothesis.split()
    errors: List[Dict] = []

    matcher = SequenceMatcher(None, ref_words, hyp_words)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        elif tag == "replace":
            ref_span = ref_words[i1:i2]
            hyp_span = hyp_words[j1:j2]
            span_len = max(len(ref_span), len(hyp_span))
            for offset in range(span_len):
                has_ref = offset < len(ref_span)
                has_hyp = offset < len(hyp_span)
                if has_ref and has_hyp:
                    errors.append({
                        "type":    "substitution",
                        "ref":     ref_span[offset],
                        "hyp":     hyp_span[offset],
                        "ref_pos": i1 + offset,
                        "hyp_pos": j1 + offset,
                    })
                elif has_ref and not has_hyp:
                    errors.append({
                        "type":    "deletion",
                        "ref":     ref_span[offset],
                        "hyp":     "",
                        "ref_pos": i1 + offset,
                        "hyp_pos": None,
                    })
                else:
                    errors.append({
                        "type":    "insertion",
                        "ref":     "",
                        "hyp":     hyp_span[offset],
                        "ref_pos": None,
                        "hyp_pos": j1 + offset,
                    })

        elif tag == "delete":
            for ri in range(i1, i2):
                errors.append({
                    "type":    "deletion",
                    "ref":     ref_words[ri] if ri < len(ref_words) else "",
                    "hyp":     "",
                    "ref_pos": ri,
                    "hyp_pos": None,
                })

        elif tag == "insert":
            for hi in range(j1, j2):
                errors.append({
                    "type":    "insertion",
                    "ref":     "",
                    "hyp":     hyp_words[hi] if hi < len(hyp_words) else "",
                    "ref_pos": None,
                    "hyp_pos": hi,
                })

    return errors


def count_devanagari(text: str) -> int:
    return sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)


def has_roman(text: str) -> bool:
    return any("a" <= c.lower() <= "z" for c in text)


def is_repetition_loop(hypothesis: str, threshold: int = 8) -> bool:
    """
    Detect hallucination loops. Threshold raised to 8 (from 5) because
    preprocessing has already removed the worst short-utterance cases.
    A token appearing 8+ times total is unambiguously a generation loop.
    """
    words = hypothesis.split()
    if not words:
        return False
    freq: Dict[str, int] = defaultdict(int)
    for w in words:
        freq[w] += 1
    return max(freq.values()) >= threshold


# ============================================================================
# Model loading
# ============================================================================

def load_model(gpu_id: int = 0):
    best = MODEL_DIR / "best"
    model_path = best
    if not best.exists():
        ckpts = sorted(
            [p for p in MODEL_DIR.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1])
            if p.name.split("-")[-1].isdigit()
            else -1,
            reverse=True,
        )
        if not ckpts:
            raise FileNotFoundError(
                f"No fine-tuned model found under {MODEL_DIR}. "
                "Run the training script first."
            )
        model_path = ckpts[0]
        logger.warning(f"'best' dir not found. Using: {model_path}")

    logger.info(f"Loading fine-tuned model from {model_path}")
    processor = WhisperProcessor.from_pretrained(str(model_path))
    model     = WhisperForConditionalGeneration.from_pretrained(str(model_path))

    decoder_prompt_ids = processor.get_decoder_prompt_ids(
        language="Hindi", task="transcribe"
    )
    model.config.forced_decoder_ids            = decoder_prompt_ids
    model.generation_config.forced_decoder_ids = decoder_prompt_ids
    model.config.suppress_tokens               = []
    model.generation_config.suppress_tokens    = []

    device = get_device(gpu_id)
    model  = model.to(device).eval()
    logger.info("Fine-tuned model loaded.")
    return model, processor, device


# ============================================================================
# Inference
# ============================================================================

def run_inference_all(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    dataset,
    device: torch.device,
    batch_size: int = 8,
) -> List[Dict]:
    """
    Run full inference over the validation set.
    Returns a list of per-utterance dicts with reference, hypothesis, WER,
    word errors, and metadata.
    """
    results: List[Dict] = []
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    with torch.no_grad():
        for bi in tqdm(range(num_batches), desc="Inference on validation"):
            s = bi * batch_size
            e = min(s + batch_size, len(dataset))
            batch = dataset[s:e]

            audio_arrays = [
                a["array"] if isinstance(a, dict) else a
                for a in batch["audio"]
            ]

            inputs = processor(
                audio_arrays,
                sampling_rate=16_000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            input_features = inputs.input_features.to(device)
            attention_mask = inputs.attention_mask.to(device)

            generated_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                task="transcribe",
                language="Hindi",
                max_new_tokens=225,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )

            predictions = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for i, pred in enumerate(predictions):
                reference = batch["sentence"][i]
                ref_norm  = normalize_for_wer(reference)
                pred_norm = normalize_for_wer(pred)
                wer       = compute_wer_utterance(ref_norm, pred_norm)
                word_errs = get_word_errors(ref_norm, pred_norm)

                results.append({
                    "recording_id": batch["recording_id"][i],
                    "segment_idx":  batch["segment_idx"][i],
                    "reference":    reference,
                    "hypothesis":   pred,
                    "wer":          wer,
                    "duration":     batch["duration"][i],
                    "speaker_id":   batch["speaker_id"][i],
                    "word_errors":  word_errs,
                })

    return results


# ============================================================================
# Systematic error sampling
# ============================================================================

def systematic_error_sampling(
    results: List[Dict],
    target_samples: int = 25,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Stratified every-Nth sampling across the WER severity spectrum.

    Each stratum is built independently from the full unsorted error list and
    sorted ascending within itself so every-Nth picks are uniformly spaced
    across that stratum's WER range. No global sort before splitting strata —
    that would bias the 'low' stratum toward higher-WER items.
    """
    errors = [r for r in results if r["wer"] > 0]

    sampling_meta: Dict[str, Any] = {
        "total_validation_samples": len(results),
        "total_error_samples":      len(errors),
        "target_samples":           target_samples,
        "strata": {
            "low":    {"range": "0 < WER < 0.3",   "count": 0, "sampled": 0, "step": 0},
            "medium": {"range": "0.3 ≤ WER ≤ 1.0", "count": 0, "sampled": 0, "step": 0},
            "high":   {"range": "WER > 1.0",        "count": 0, "sampled": 0, "step": 0},
        },
        "fill_remaining": {"count": 0, "step": 0, "sampled": 0},
    }

    if not errors:
        logger.warning("No errors found in validation set.")
        return [], sampling_meta

    logger.info(f"Total utterances with WER > 0: {len(errors)}")

    low_sev  = sorted([e for e in errors if e["wer"] < 0.3],           key=lambda x: x["wer"])
    med_sev  = sorted([e for e in errors if 0.3 <= e["wer"] <= 1.0],   key=lambda x: x["wer"])
    high_sev = sorted([e for e in errors if e["wer"] > 1.0],           key=lambda x: x["wer"])

    sampling_meta["strata"]["low"]["count"]    = len(low_sev)
    sampling_meta["strata"]["medium"]["count"] = len(med_sev)
    sampling_meta["strata"]["high"]["count"]   = len(high_sev)

    logger.info(f"  Low    (WER < 0.3)  : {len(low_sev)}")
    logger.info(f"  Medium (0.3–1.0)    : {len(med_sev)}")
    logger.info(f"  High   (WER > 1.0)  : {len(high_sev)}")

    sampled: List[Dict] = []

    def sample_stratum(name: str, stratum: List[Dict], min_target: int = 5) -> List[Dict]:
        if not stratum:
            return []
        if len(stratum) >= min_target:
            step   = max(1, len(stratum) // min_target)
            picked = stratum[::step][:min_target]
        else:
            step   = 1
            picked = stratum[:]
        sampling_meta["strata"][name]["step"]    = step
        sampling_meta["strata"][name]["sampled"] = len(picked)
        return picked

    sampled.extend(sample_stratum("low",    low_sev,  5))
    sampled.extend(sample_stratum("medium", med_sev,  10))
    sampled.extend(sample_stratum("high",   high_sev, 10))

    if len(sampled) < target_samples:
        remaining    = target_samples - len(sampled)
        sampled_keys = {(e["recording_id"], e["segment_idx"]) for e in sampled}
        unsampled    = sorted(
            [e for e in errors
             if (e["recording_id"], e["segment_idx"]) not in sampled_keys],
            key=lambda x: x["wer"],
        )
        if unsampled:
            step = max(1, len(unsampled) // remaining)
            fill = unsampled[::step][:remaining]
            sampling_meta["fill_remaining"]["count"]   = len(unsampled)
            sampling_meta["fill_remaining"]["step"]    = step
            sampling_meta["fill_remaining"]["sampled"] = len(fill)
            sampled.extend(fill)

    sampled = sampled[:target_samples]
    sampling_meta["final_sample_count"] = len(sampled)

    wer_vals = [e["wer"] for e in sampled]
    logger.info(f"Selected {len(sampled)} error samples.")
    logger.info(
        f"  WER range: {min(wer_vals):.3f} – {max(wer_vals):.3f}  "
        f"(median {float(np.median(wer_vals)):.3f})"
    )
    return sampled, sampling_meta


# ============================================================================
# Error taxonomy
# ============================================================================

FUNCTION_WORDS = {
    "है", "का", "को", "में", "यह", "वह", "से", "की", "के",
    "पर", "तो", "भी", "ने", "ही", "तक", "पे", "ना", "हैं",
    "था", "थे", "थी", "एक", "और", "कि", "जो", "इस", "उस",
}

FILLERS = {
    "uh", "um", "hmm", "हम्म", "अ", "आ", "ऊं", "ह", "म्म",
    "हाँ", "ओह", "अच्छा",
}

CASUAL_MARKERS = {
    "मेको", "तेको", "बोहोत", "बहोत", "यार", "मतलब",
    "करना", "करो", "बताओ", "देखो", "सुनो", "भाई",
}


def analyze_error_patterns(error_samples: List[Dict]) -> Dict[str, List]:
    """
    Classify each sampled error into a data-driven category.

    Category priority (first match wins):
      1. repetition_loop          — token repeated >= 8 times in hypothesis
      2. number_quantifier        — digit in reference
      3. code_switching           — script mismatch in a substitution pair
      4. casual_colloquial        — informal Hindi marker
      5. proper_nouns             — title-cased word in a substitution
      6. deletion_function_words  — grammatical particle dropped
      7. insertion_noise          — filler word inserted
      8. other
    """
    categories: Dict[str, List] = {
        "repetition_loop":         [],
        "code_switching":          [],
        "casual_colloquial":       [],
        "proper_nouns":            [],
        "number_quantifier":       [],
        "deletion_function_words": [],
        "insertion_noise":         [],
        "other":                   [],
    }

    def classify(sample: Dict) -> str:
        hyp    = sample.get("hypothesis", "")
        ref    = sample.get("reference",  "")
        errors = sample.get("word_errors", [])

        if is_repetition_loop(hyp):
            return "repetition_loop"

        if any(ch.isdigit() for ch in ref):
            return "number_quantifier"

        for err in errors:
            et       = err.get("type", "")
            ref_word = err.get("ref",  "")
            hyp_word = err.get("hyp",  "")

            if et == "substitution":
                if (count_devanagari(ref_word) > 0 and has_roman(hyp_word)) or \
                   (has_roman(ref_word) and count_devanagari(hyp_word) > 0):
                    return "code_switching"
                if ref_word in CASUAL_MARKERS or hyp_word in CASUAL_MARKERS:
                    return "casual_colloquial"
                if ref_word.istitle() or hyp_word.istitle():
                    return "proper_nouns"

            if et == "deletion" and (
                ref_word in FUNCTION_WORDS or (ref_word and len(ref_word) <= 2)
            ):
                return "deletion_function_words"

            if et == "insertion" and hyp_word.lower() in FILLERS:
                return "insertion_noise"

        return "other"

    for sample in error_samples:
        categories[classify(sample)].append(sample)

    return categories


# ============================================================================
# Per-example reasoning
# ============================================================================

def generate_example_reasoning(sample: Dict, category: str) -> str:
    errors = sample.get("word_errors", [])

    if category == "repetition_loop":
        hyp_words = sample.get("hypothesis", "").split()
        freq: Dict[str, int] = defaultdict(int)
        for w in hyp_words:
            freq[w] += 1
        if freq:
            top_token, top_count = max(freq.items(), key=lambda x: x[1])
            return (
                f"Model entered a generation loop, repeating '{top_token}' "
                f"{top_count}x out of {len(hyp_words)} total tokens. "
                f"Known Whisper hallucination on short/low-energy audio. "
                f"Fix: repetition_penalty >= 1.3 and no_repeat_ngram_size=3 in generate()."
            )

    if not errors:
        return (
            "No aligned token-level mismatch found; error likely stems from "
            "broad phrase-level mismatch not captured by word alignment."
        )

    if category == "code_switching":
        for err in errors:
            rw, hw = err.get("ref", ""), err.get("hyp", "")
            if (count_devanagari(rw) > 0 and has_roman(hw)) or \
               (has_roman(rw) and count_devanagari(hw) > 0):
                return (
                    f"Script switch: ref='{rw}' vs hyp='{hw}'. Transcription "
                    f"guideline requires English loanwords in Devanagari; "
                    f"model output Roman script instead."
                )

    if category == "casual_colloquial":
        for err in errors:
            if err.get("type") == "substitution":
                return (
                    f"Informal spoken form: ref='{err.get('ref','')}' vs "
                    f"hyp='{err.get('hyp','')}' — colloquial pronunciation or "
                    f"dialectal variant not well represented in training data."
                )

    if category == "deletion_function_words":
        for err in errors:
            if err.get("type") == "deletion":
                pos = err.get("ref_pos")
                pos_s = f" at position {pos}" if pos is not None else ""
                return (
                    f"Grammatical particle '{err.get('ref','')}' dropped{pos_s}. "
                    f"Function-word deletions indicate the model is under-trained "
                    f"on formal Hindi style or the particle is acoustically weak."
                )

    if category == "insertion_noise":
        for err in errors:
            if err.get("type") == "insertion":
                return (
                    f"Spurious filler/noise token '{err.get('hyp','')}' inserted. "
                    f"Model is transcribing background noise or disfluency that "
                    f"the human annotator silently ignored."
                )

    if category == "number_quantifier":
        for err in errors:
            if err.get("type") in {"substitution", "deletion", "insertion"}:
                return (
                    f"Number/quantifier confusion: ref='{err.get('ref','')}' vs "
                    f"hyp='{err.get('hyp','')}' — similar-sounding numerals or "
                    f"digit-vs-word surface form mismatch."
                )

    if category == "proper_nouns":
        for err in errors:
            if err.get("type") == "substitution":
                return (
                    f"Named entity confusion: '{err.get('ref','')}' transcribed as "
                    f"'{err.get('hyp','')}' — low-frequency name or place not "
                    f"well represented in Whisper's pre-training vocabulary."
                )

    first = errors[0]
    return (
        f"Primary mismatch is a {first.get('type','?')} "
        f"(ref='{first.get('ref','')}', hyp='{first.get('hyp','')}'), "
        f"suggesting local lexical confusion."
    )


# ============================================================================
# Output helpers
# ============================================================================

def save_error_samples(error_samples: List[Dict]) -> None:
    with open(ERROR_SAMPLES_FILE, "w", encoding="utf-8") as f:
        for sample in error_samples:
            out = {
                "recording_id": sample["recording_id"],
                "segment_idx":  sample["segment_idx"],
                "reference":    sample["reference"],
                "hypothesis":   sample["hypothesis"],
                "wer":          round(sample["wer"], 4),
                "duration":     round(float(sample["duration"]), 2),
                "speaker_id":   sample["speaker_id"],
                "word_errors":  sample["word_errors"],  # full list, no truncation
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    logger.info(f"Error samples saved -> {ERROR_SAMPLES_FILE}")


def generate_taxonomy_report(
    error_samples: List[Dict],
    categories: Dict[str, List],
    sampling_meta: Dict[str, Any],
) -> None:

    category_display = {
        "repetition_loop":         "Repetition Loop / Hallucination",
        "code_switching":          "Code-switching (Devanagari <-> Roman)",
        "casual_colloquial":       "Casual / Colloquial Hindi",
        "proper_nouns":            "Proper Nouns & Named Entities",
        "number_quantifier":       "Number / Quantifier Errors",
        "deletion_function_words": "Function Word Deletions",
        "insertion_noise":         "Insertion of Noise / Fillers",
        "other":                   "Other / Uncategorised",
    }

    with open(ERROR_TAXONOMY_FILE, "w", encoding="utf-8") as f:
        f.write("# Error Taxonomy Analysis\n\n")
        f.write(f"**Total analysed errors**: {len(error_samples)}\n\n")

        f.write("## Sampling Strategy\n\n")
        f.write(
            f"Total validation samples evaluated: "
            f"{sampling_meta.get('total_validation_samples', 0)}\n\n"
        )
        f.write(
            f"Total utterances with WER > 0: "
            f"{sampling_meta.get('total_error_samples', 0)}\n\n"
        )
        f.write(
            "Stratified every-Nth sampling across three severity bands "
            "(built independently per stratum):\n\n"
        )
        for k, v in sampling_meta.get("strata", {}).items():
            f.write(
                f"- **{k.capitalize()}** ({v.get('range','')}): "
                f"pool = {v.get('count',0)}, "
                f"every-N step = {v.get('step',0)}, "
                f"drawn = {v.get('sampled',0)}\n"
            )
        fr = sampling_meta.get("fill_remaining", {})
        if fr.get("sampled", 0):
            f.write(
                f"- **Fill-remaining**: pool = {fr.get('count',0)}, "
                f"step = {fr.get('step',0)}, "
                f"drawn = {fr.get('sampled',0)}\n"
            )
        f.write(
            f"\n**Final sample count**: "
            f"{sampling_meta.get('final_sample_count', len(error_samples))}\n\n"
        )
        f.write(
            "_No cherry-picking: samples drawn mechanically using every-Nth "
            "within each WER severity stratum._\n\n"
        )

        f.write("## Error Category Overview\n\n")
        f.write("| Category | Count | % of sample |\n")
        f.write("|---|---|---|\n")
        for cat_key, display in category_display.items():
            n = len(categories.get(cat_key, []))
            if n == 0:
                continue
            pct = 100 * n / len(error_samples) if error_samples else 0
            f.write(f"| {display} | {n} | {pct:.1f}% |\n")
        f.write("\n")

        f.write("## Error Categories - Detailed Examples\n\n")
        for cat_key, display in category_display.items():
            samples = categories.get(cat_key, [])
            if not samples:
                continue
            count = len(samples)
            pct   = 100 * count / len(error_samples) if error_samples else 0

            f.write(f"### {display}\n\n")
            f.write(f"**Count**: {count} ({pct:.1f}% of sampled errors)\n\n")
            f.write("**Examples** (reference -> hypothesis, WER, cause):\n\n")

            n_examples = min(5, max(3, count))
            for sample in samples[:n_examples]:
                reasoning = generate_example_reasoning(sample, cat_key)
                f.write("```\n")
                f.write(f"Reference:   {sample['reference']}\n")
                f.write(f"Hypothesis:  {sample['hypothesis']}\n")
                f.write(f"WER:         {sample['wer']:.4f}\n")
                f.write(f"Duration:    {sample['duration']:.1f}s\n")
                f.write(f"Cause:       {reasoning}\n")
                f.write("```\n\n")

            if count < 3:
                f.write("_Note: fewer than 3 examples in this category._\n\n")

        f.write("## Top Recommendations (Q1-f)\n\n")

        sorted_cats = sorted(
            [(k, len(v)) for k, v in categories.items() if v],
            key=lambda x: x[1],
            reverse=True,
        )

        rec_map = {
            "repetition_loop": (
                "**Repetition Penalty in `model.generate()`**\n\n"
                "Set `repetition_penalty=1.3` and `no_repeat_ngram_size=3`. "
                "One-line inference change; no retraining needed. Directly "
                "prevents stuck-token hallucination loops.\n\n"
                "```python\n"
                "model.generate(\n"
                "    input_features,\n"
                "    attention_mask=attention_mask,\n"
                "    language='Hindi',\n"
                "    task='transcribe',\n"
                "    max_new_tokens=225,\n"
                "    repetition_penalty=1.3,\n"
                "    no_repeat_ngram_size=3,\n"
                ")\n"
                "```\n"
            ),
            "code_switching": (
                "**Post-processing Transliteration Normalisation**\n\n"
                "Apply a token-level lookup table (or `indic-transliteration`) "
                "to map Roman-script output back to Devanagari. Respects the "
                "transcription guideline and requires no retraining.\n"
            ),
            "deletion_function_words": (
                "**Language-model Rescoring / Beam Search**\n\n"
                "Use a Hindi n-gram or neural LM (e.g. IndicBERT) to rescore "
                "Whisper's beam candidates and promote hypotheses that retain "
                "grammatical particles. Alternatively increase `num_beams` from "
                "1 to 4 during inference.\n"
            ),
            "casual_colloquial": (
                "**Training Data Augmentation with Colloquial Hindi**\n\n"
                "The JoshTalks corpus is conversational. Add more examples of "
                "casual / dialectal Hindi to the fine-tuning set, or normalise "
                "reference transcriptions to a consistent spoken-form style.\n"
            ),
            "proper_nouns": (
                "**Named-entity Biasing in the Decoder**\n\n"
                "Pass a list of expected proper nouns as `prefix_tokens` or via "
                "hotword biasing. Whisper's `forced_decoder_ids` can be extended "
                "to boost named-entity tokens.\n"
            ),
            "number_quantifier": (
                "**Number Normalisation Post-processing**\n\n"
                "Apply a Hindi number-word-to-digit converter as a post-processing "
                "step to standardise surface forms.\n"
            ),
            "insertion_noise": (
                "**Voice Activity Detection (VAD) Pre-processing**\n\n"
                "Run Silero-VAD before feeding audio to Whisper to trim leading/"
                "trailing silence and low-energy frames that trigger filler-word "
                "insertions.\n"
            ),
            "other": (
                "**Data-driven Augmentation**\n\n"
                "Inspect `error_samples.jsonl` to identify emerging sub-patterns "
                "and add targeted training examples.\n"
            ),
        }

        written = 0
        for cat_key, count in sorted_cats:
            if written >= 3:
                break
            if cat_key in rec_map:
                f.write(f"{written + 1}. {rec_map[cat_key]}\n")
                written += 1

    logger.info(f"Error taxonomy report saved -> {ERROR_TAXONOMY_FILE}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Error analysis for fine-tuned Whisper Hindi model (Q1-d/e)"
    )
    parser.add_argument("--gpu_id",        type=int, default=0)
    parser.add_argument("--batch_size",    type=int, default=8)
    parser.add_argument("--target_samples", type=int, default=25)
    args = parser.parse_args()

    try:
        setup_gpu(args.gpu_id)

        logger.info("\n" + "=" * 70)
        logger.info("ERROR ANALYSIS PIPELINE (Q1-d / Q1-e)")
        logger.info("=" * 70)

        logger.info("\nStep 1: Loading validation dataset...")
        if not VAL_DIR.exists():
            raise FileNotFoundError(
                f"Validation set not found at {VAL_DIR}. "
                "Run 03_make_hf_dataset.py first."
            )
        val_dataset = load_from_disk(str(VAL_DIR))
        logger.info(f"Loaded {len(val_dataset):,} validation samples.")

        logger.info("\nStep 2: Loading fine-tuned model...")
        model, processor, device = load_model(args.gpu_id)

        logger.info("\nStep 3: Running inference over full validation set...")
        results = run_inference_all(
            model, processor, val_dataset, device,
            batch_size=args.batch_size,
        )
        n_errors = sum(1 for r in results if r["wer"] > 0)
        logger.info(
            f"Inference complete: {len(results)} utterances, "
            f"{n_errors} with WER > 0."
        )

        logger.info(f"\nStep 4: Systematic error sampling (target={args.target_samples})...")
        error_samples, sampling_meta = systematic_error_sampling(
            results, target_samples=args.target_samples
        )

        logger.info("\nStep 5: Building error taxonomy...")
        categories = analyze_error_patterns(error_samples)
        for cat, items in categories.items():
            if items:
                logger.info(f"  {cat:<30}: {len(items)}")

        logger.info("\nStep 6: Saving outputs...")
        save_error_samples(error_samples)
        generate_taxonomy_report(error_samples, categories, sampling_meta)

        logger.info("\n" + "=" * 70)
        logger.info("ERROR ANALYSIS COMPLETE")
        logger.info(f"  Samples  -> {ERROR_SAMPLES_FILE}")
        logger.info(f"  Taxonomy -> {ERROR_TAXONOMY_FILE}")
        logger.info("=" * 70)
        sys.exit(0)

    except Exception:
        logger.error(f"Fatal error:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()