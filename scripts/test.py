#!/usr/bin/env python3
"""
Debug: show top-10 worst WER samples from fine-tuned model on JoshTalks val.
Run from repo root.
"""
import torch, unicodedata, re, jiwer
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_from_disk

VAL_DIR   = Path("datasets_hf/val")
MODEL_DIR = Path("models/whisper-small-hindi/best")

def normalize(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[।\.]{2,}', '।', text)
    return text.rstrip('।. ').strip()

dataset   = load_from_disk(str(VAL_DIR))
processor = WhisperProcessor.from_pretrained(str(MODEL_DIR))
model     = WhisperForConditionalGeneration.from_pretrained(str(MODEL_DIR))
device    = torch.device("cuda:0")
model     = model.to(device).eval()

results = []
with torch.no_grad():
    for i, sample in enumerate(dataset):
        audio  = sample["audio"]["array"]
        inputs = processor(audio, sampling_rate=16000,
                           return_tensors="pt", return_attention_mask=True)
        ids = model.generate(
            inputs.input_features.to(device),
            attention_mask=inputs.attention_mask.to(device),
            language="Hindi", task="transcribe",
            max_new_tokens=225,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        pred = processor.decode(ids[0], skip_special_tokens=True)
        ref  = normalize(sample["sentence"])
        hyp  = normalize(pred)
        
        utt_wer = jiwer.wer(ref, hyp) if ref.strip() else 0.0
        results.append({
            "idx": i,
            "duration": sample["duration"],
            "ref": ref,
            "hyp": hyp,
            "wer": utt_wer,
            "ref_words": len(ref.split()),
            "hyp_words": len(hyp.split()),
        })
        
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(dataset)}")

# Sort by worst WER
results.sort(key=lambda x: x["wer"], reverse=True)

print("\n=== TOP 20 WORST SAMPLES ===")
for r in results[:20]:
    print(f"\nIDX={r['idx']}  dur={r['duration']:.1f}s  "
          f"WER={r['wer']:.2f}  "
          f"ref_words={r['ref_words']}  hyp_words={r['hyp_words']}")
    print(f"REF: {r['ref']}")
    print(f"HYP: {r['hyp'][:200]}")  # cap at 200 chars to avoid terminal flood

# Distribution
print("\n=== WER DISTRIBUTION ===")
bins = [(0,0.3),(0.3,0.5),(0.5,1.0),(1.0,2.0),(2.0,5.0),(5.0,float('inf'))]
for lo, hi in bins:
    count = sum(1 for r in results if lo <= r["wer"] < hi)
    print(f"  WER [{lo:.1f}, {hi:.1f}): {count} samples")

print(f"\nTotal samples: {len(results)}")
print(f"Mean WER: {sum(r['wer'] for r in results)/len(results):.3f}")
print(f"Median WER: {sorted(r['wer'] for r in results)[len(results)//2]:.3f}")