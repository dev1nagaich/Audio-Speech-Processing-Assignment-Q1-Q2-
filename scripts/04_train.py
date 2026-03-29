import os
import sys
import logging
import argparse
import unicodedata
import shutil
from pathlib import Path
from typing import Dict, List
import traceback

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import numpy as np
from dataclasses import dataclass
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_from_disk
import evaluate


def get_repo_root() -> Path:
    return Path(__file__).parent.parent


REPO_ROOT       = get_repo_root()
DATASETS_HF_DIR = REPO_ROOT / "datasets_hf"
TRAIN_DIR       = DATASETS_HF_DIR / "train"
VAL_DIR         = DATASETS_HF_DIR / "val"
MODEL_DIR       = REPO_ROOT / "models" / "whisper-small-hindi"
TRAIN_LOG       = REPO_ROOT / "results" / "logs" / "train.log"
TENSORBOARD_DIR = REPO_ROOT / "results" / "logs" / "tensorboard"

TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(TRAIN_LOG),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

FORCED_PREFIX_LEN = 4


def setup_gpu(gpu_ids: str = "0") -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    num_gpus = torch.cuda.device_count()
    logger.info(f"CUDA_VISIBLE_DEVICES={gpu_ids}  |  GPUs visible: {num_gpus}")
    if torch.cuda.is_available():
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name}  {props.total_memory / 1e9:.1f} GB")
    else:
        logger.warning("CUDA not available.")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict]) -> Dict:
        audio_features = [
            {
                "input_features": self.processor(
                    f["audio"]["array"],
                    sampling_rate=16_000,
                    return_tensors="pt",
                ).input_features[0]
            }
            for f in features
        ]

        batch = self.processor.feature_extractor.pad(
            audio_features,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        label_ids = []
        for f in features:
            text = unicodedata.normalize("NFC", f["sentence"])
            ids  = self.processor.tokenizer(text).input_ids
            label_ids.append(ids)

        padded_labels = self.processor.tokenizer.pad(
            {"input_ids": label_ids},
            return_tensors="pt",
            padding=True,
        )

        labels = padded_labels["input_ids"].masked_fill(
            padded_labels["input_ids"] == self.processor.tokenizer.pad_token_id,
            -100,
        )

        labels[:, :FORCED_PREFIX_LEN] = -100

        return {
            "input_features": batch["input_features"],
            "attention_mask": batch["attention_mask"],
            "labels":         labels,
        }


def compute_metrics(pred, processor: WhisperProcessor) -> Dict:
    wer_metric = evaluate.load("wer")

    pred_ids  = pred.predictions
    label_ids = pred.label_ids

    label_ids = np.where(
        label_ids == -100,
        processor.tokenizer.pad_token_id,
        label_ids,
    )

    pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    pred_str  = [unicodedata.normalize("NFC", s) for s in pred_str]
    label_str = [unicodedata.normalize("NFC", s) for s in label_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def train_whisper(
    batch_size:    int   = 4,
    num_epochs:    int   = 3,
    learning_rate: float = 1e-5,
    warmup_steps:  int   = 500,
    gpu_ids:       str   = "0",
) -> None:
    setup_gpu(gpu_ids)

    if MODEL_DIR.exists():
        logger.info(f"Removing stale model directory: {MODEL_DIR}")
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Step 1: Loading model")
    logger.info("=" * 70)

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="Hindi", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    model.config.forced_decoder_ids            = None
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens               = []
    model.generation_config.suppress_tokens    = []
    model.config.use_cache                     = True

    logger.info(f"Forced prefix length: {FORCED_PREFIX_LEN} tokens")
    logger.info(f"Model parameters: {model.num_parameters():,}")

    logger.info("=" * 70)
    logger.info("Step 2: Loading datasets")
    logger.info("=" * 70)

    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError("Train/val datasets not found. Run 03_make_hf_dataset.py first.")

    train_dataset = load_from_disk(str(TRAIN_DIR))
    val_dataset   = load_from_disk(str(VAL_DIR))

    logger.info(f"Train : {len(train_dataset):,} samples")
    logger.info(f"Val   : {len(val_dataset):,} samples")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    logger.info("=" * 70)
    logger.info("Step 3: Label sanity check (first 3 samples)")
    logger.info("=" * 70)
    for i in range(min(3, len(train_dataset))):
        sample    = train_dataset[i]
        batch     = data_collator([sample])
        label_ids = batch["labels"][0].clone()
        label_ids = torch.where(
            label_ids == -100,
            torch.tensor(processor.tokenizer.pad_token_id),
            label_ids,
        )
        decoded = processor.tokenizer.decode(label_ids, skip_special_tokens=False)
        logger.info(f"  Sample {i}: {decoded[:150]}")
    logger.info("(First 4 positions decoded as pad token — loss correctly ignored)")
    logger.info("=" * 70)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_DIR),

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(1, batch_size // 2),
        gradient_accumulation_steps=4,

        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",

        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        predict_with_generate=True,
        generation_max_length=225,
        generation_num_beams=1,

        logging_steps=25,
        save_total_limit=2,
        report_to=["tensorboard"],
        push_to_hub=False,

        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    logger.info(f"Per-device batch size  : {batch_size}")
    logger.info(f"Gradient accum steps   : 4")
    logger.info(f"Effective batch size   : {batch_size * 4}")
    logger.info(f"Epochs                 : {num_epochs}")
    logger.info(f"Learning rate          : {learning_rate}")
    logger.info(f"Warmup steps           : {warmup_steps}")

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        processing_class=processor,
    )

    trainer.model.generation_config.language = "hindi"
    trainer.model.generation_config.task     = "transcribe"

    logger.info("=" * 70)
    logger.info("Step 4: Training")
    logger.info("=" * 70)

    train_result = trainer.train()
    logger.info(f"Training loss: {train_result.training_loss:.4f}")

    best_dir = MODEL_DIR / "best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    logger.info(f"Best model saved → {best_dir}")

    eval_results = trainer.evaluate()
    logger.info(f"Final validation WER: {eval_results.get('eval_wer', float('nan')):.4f}")

    logger.info("=" * 70)
    logger.info("ALL DONE")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--num_epochs",    type=int,   default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps",  type=int,   default=500)
    parser.add_argument("--gpu_ids",       type=str,   default="0")
    args = parser.parse_args()

    try:
        train_whisper(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            gpu_ids=args.gpu_ids,
        )
        sys.exit(0)
    except Exception:
        logger.error(f"Fatal error:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()