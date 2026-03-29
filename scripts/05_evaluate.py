import os
import sys
import re
import logging
import argparse
import csv
import unicodedata
from pathlib import Path
from typing import List, Tuple, Optional
import traceback

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, load_from_disk
import jiwer
from tqdm import tqdm



def get_repo_root() -> Path:
    return Path(__file__).parent.parent


REPO_ROOT   = get_repo_root()
VAL_DIR     = REPO_ROOT / "datasets_hf" / "val"
MODEL_DIR   = REPO_ROOT / "models" / "whisper-small-hindi"
RESULTS_DIR = REPO_ROOT / "results"
WER_CSV     = RESULTS_DIR / "wer_table.csv"
EVAL_LOG    = RESULTS_DIR / "logs" / "evaluation.log"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(EVAL_LOG), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def setup_gpu(gpu_id: int = 0) -> None:
    if torch.cuda.is_available():
        total = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= total:
            gpu_id = 0
        props = torch.cuda.get_device_properties(gpu_id)
        logger.info(f"GPU {gpu_id}: {props.name}  {props.total_memory / 1e9:.1f} GB")
    else:
        logger.warning("CUDA not available.")


def get_device(gpu_id: int = 0) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    total = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= total:
        gpu_id = 0
    return torch.device(f"cuda:{gpu_id}")


def normalize_for_wer(text: str) -> str:
    
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[।\.]{2,}', '।', text)
    text = text.rstrip('।. ').strip()
    while "  " in text:
        text = text.replace("  ", " ")
    return text



def load_baseline_model(gpu_id: int = 0):
    logger.info("Loading baseline: openai/whisper-small")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="Hindi", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.config.suppress_tokens = []
    device = get_device(gpu_id)
    model = model.to(device).eval()
    logger.info("Baseline loaded.")
    return model, processor, device


def load_finetuned_model(gpu_id: int = 0):
    best = MODEL_DIR / "best"
    model_path = best
    if not best.exists():
        ckpts = sorted(
            [p for p in MODEL_DIR.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1])
            if p.name.split("-")[-1].isdigit() else -1,
            reverse=True,
        )
        if not ckpts:
            raise FileNotFoundError(f"No fine-tuned model found under {MODEL_DIR}.")
        model_path = ckpts[0]
        logger.warning(f"'best' not found. Using: {model_path}")

    logger.info(f"Loading fine-tuned model from {model_path}")
    processor = WhisperProcessor.from_pretrained(str(model_path))
    model = WhisperForConditionalGeneration.from_pretrained(str(model_path))
    # Leave forced_decoder_ids as saved — clearing and re-passing language/task
    # as kwargs gives identical output but is inconsistent with training config.
    model.generation_config.suppress_tokens = []
    model.config.suppress_tokens = []
    device = get_device(gpu_id)
    model = model.to(device).eval()
    logger.info("Fine-tuned model loaded.")
    return model, processor, device

def run_inference(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    dataset,
    device: torch.device,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
   
    text_key = None
    for candidate in ("sentence", "raw_transcription", "transcription", "text"):
        if candidate in dataset.column_names:
            text_key = candidate
            break
    if text_key is None:
        raise KeyError(f"No text column found. Columns: {dataset.column_names}")

    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    predictions: List[str] = []
    references:  List[str] = []
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    with torch.no_grad():
        for bi in tqdm(range(num_batches), desc="Inference"):
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
                language="Hindi",
                task="transcribe",
                max_new_tokens=225,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )

            batch_preds = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            predictions.extend(normalize_for_wer(p) for p in batch_preds)
            references.extend(normalize_for_wer(r) for r in batch[text_key])

    return predictions, references


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """Corpus-level WER. Empty references are filtered before computation."""
    pairs = [(r, p) for r, p in zip(references, predictions) if r.strip()]
    if not pairs:
        return 0.0
    filtered_refs, filtered_preds = zip(*pairs)
    return jiwer.wer(list(filtered_refs), list(filtered_preds))


def evaluate_on_dataset(
    model, processor, dataset, device,
    dataset_name: str,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
) -> float:
    logger.info(f"\nEvaluating on {dataset_name} ...")
    preds, refs = run_inference(
        model, processor, dataset, device,
        batch_size=batch_size,
        max_samples=max_samples,
    )
    wer = compute_wer(preds, refs)
    logger.info(f"  {dataset_name}  WER: {wer * 100:.2f}%")
    return wer

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper-small baseline and fine-tuned model"
    )
    parser.add_argument("--gpu_id",     type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    try:
        setup_gpu(args.gpu_id)

        logger.info("\n" + "=" * 70)
        logger.info("STARTING EVALUATION")
        logger.info("=" * 70)

        # ── Load datasets ──────────────────────────────────────────────── #
        logger.info("\nLoading FLEURS Hindi test set...")
        try:
            fleurs_test      = load_dataset("google/fleurs", "hi_in", split="test")
            fleurs_available = True
            logger.info(f"FLEURS: {len(fleurs_test):,} samples")
        except Exception as exc:
            logger.warning(f"Could not load FLEURS: {exc}")
            fleurs_available = False

        logger.info("Loading JoshTalks validation set...")
        if not VAL_DIR.exists():
            raise FileNotFoundError(f"Val set not found at {VAL_DIR}.")
        val_dataset = load_from_disk(str(VAL_DIR))
        logger.info(f"JoshTalks val: {len(val_dataset):,} samples")

        results = []

        # ── Baseline ───────────────────────────────────────────────────── #
        logger.info("\n" + "=" * 70)
        logger.info("BASELINE: openai/whisper-small")
        logger.info("=" * 70)
        baseline_model, baseline_proc, device = load_baseline_model(args.gpu_id)

        if fleurs_available:
            wer = evaluate_on_dataset(
                baseline_model, baseline_proc, fleurs_test, device,
                "FLEURS Hi test", batch_size=args.batch_size,
            )
            results.append({
                "model":   "Whisper-small (baseline)",
                "dataset": "FLEURS Hi test",
                "wer":     wer,
            })

        wer = evaluate_on_dataset(
            baseline_model, baseline_proc, val_dataset, device,
            "JoshTalks val", batch_size=args.batch_size,
        )
        results.append({
            "model":   "Whisper-small (baseline)",
            "dataset": "JoshTalks val",
            "wer":     wer,
        })

        del baseline_model
        torch.cuda.empty_cache()

        # ── Fine-tuned ─────────────────────────────────────────────────── #
        logger.info("\n" + "=" * 70)
        logger.info("FINE-TUNED MODEL")
        logger.info("=" * 70)
        ft_model, ft_proc, device = load_finetuned_model(args.gpu_id)

        if fleurs_available:
            wer = evaluate_on_dataset(
                ft_model, ft_proc, fleurs_test, device,
                "FLEURS Hi test", batch_size=args.batch_size,
            )
            results.append({
                "model":   "Whisper-small (fine-tuned)",
                "dataset": "FLEURS Hi test",
                "wer":     wer,
            })

        wer = evaluate_on_dataset(
            ft_model, ft_proc, val_dataset, device,
            "JoshTalks val", batch_size=args.batch_size,
        )
        results.append({
            "model":   "Whisper-small (fine-tuned)",
            "dataset": "JoshTalks val",
            "wer":     wer,
        })

        del ft_model
        torch.cuda.empty_cache()

        # ── Results table ──────────────────────────────────────────────── #
        logger.info("\n" + "=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 70)

        with open(WER_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Model", "Dataset", "WER (%)"])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "Model":   r["model"],
                    "Dataset": r["dataset"],
                    "WER (%)": f"{r['wer'] * 100:.2f}",
                })
        logger.info(f"Results written to {WER_CSV}\n")

        hdr = f"{'Model':<42} {'Dataset':<25} {'WER':>9}"
        logger.info(hdr)
        logger.info("-" * len(hdr))
        for r in results:
            logger.info(
                f"{r['model']:<42} {r['dataset']:<25} "
                f"{r['wer'] * 100:>8.2f}%"
            )

        logger.info("\n" + "=" * 70)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 70)
        sys.exit(0)

    except Exception:
        logger.error(f"Fatal error:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()