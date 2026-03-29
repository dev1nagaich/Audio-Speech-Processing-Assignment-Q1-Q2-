# Q2: ASR Cleanup Pipeline Report

## Pipeline Overview

Two post-processing operations applied sequentially to raw Whisper ASR output:

1. **Number Normalisation** — converts spoken Hindi number words to digits
2. **English Word Detection** — tags English loanwords with `[EN]...[/EN]`

> **Transcription guideline**: English words spoken in conversation are
> transcribed in Devanagari script. Detection identifies English-*origin*
> words — it does not treat them as errors.

## (a) Number Normalisation

### Algorithm

Greedy left-to-right span collection over number tokens, with idiom detection
as a pre-pass guard. Multipliers (सौ, हज़ार, लाख, करोड़) collapse partial
sums following standard Hindi place-value conventions.

### Correct conversions

| Example | Raw input | Normalised output | Value |
|---------|-----------|-------------------|-------|
| NUM-1 — Simple cardinal number — time duration | `तेरह` | `13` | 13 |
| NUM-2 — Compound number — travel time | (no number token in hypothesis) | — | — |
| NUM-3 — Large compound number — administrative/financial | `तीन सौ चौवन` | `354` | 354 |
| NUM-4 — Round thousands | `एक हज़ार पाँच सौ` | `1500` | 1500 |
| NUM-5 — Simple tens | `पच्चीस` | `25` | 25 |

### Edge cases — judgment calls

#### EDGE-1: Idiomatic paired numbers — should NOT convert

- **Input**: `दो-चार बातें करनी थीं`
- **Output**: `दो-चार बातें करनी थीं`
- **Reasoning**: दो-चार is a colloquial idiom meaning 'a few'. Converting to '2-4 बातें' would be wrong. Idiom pattern detected → kept as-is.

#### EDGE-2: एक बार — once (idiomatic)

- **Input**: `एक बार फिर से देख लो`
- **Output**: `एक बार फिर से देख लो`
- **Idiom guard triggered for**: `एक`
- **Reasoning**: एक बार = 'once'. Converting to '1 बार' breaks colloquial phrasing. Idiom rule: एक बार → kept.

#### EDGE-3: Repeated number in idiomatic emphasis

- **Input**: `नौ दो ग्यारह हो गया`
- **Output**: `नौ दो ग्यारह हो गया`
- **Idiom guard triggered for**: `नौ`, `दो`, `ग्यारह`
- **Reasoning**: नौ दो ग्यारह = idiom for 'fled/ran away'. Cannot convert (9-2-11 makes no sense). Idiom pattern block applied.

## (b) English Word Detection

### Algorithm

Three-tier classifier per token:

1. **Native Hindi exception list** — hard-coded common function words that
   could false-positive on phonological heuristics (e.g. भी, जी, नहीं)
2. **Lexicon lookup** — 100+ curated English loanwords in Devanagari,
   plus inflection stripping (removes ों, ी, ा etc.) before re-lookup
3. **Phonological heuristics** — patterns characteristic of English
   origin: ऑ (open-O), ज़/फ़ (dotted consonants), -शन / -मेंट / -इंग
   suffix clusters, retroflex consonant sequences

### Detection examples

#### EN-1: Job interview — common loanwords

- **Reference**: `मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई`
- **Tagged reference**: `मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया और मुझे [EN]जॉब[/EN] मिल गई`
- **Raw ASR**: `मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई`
- **Pipeline output**: `मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया और मुझे [EN]जॉब[/EN] मिल गई`
- **English words detected**: `इंटरव्यू`, `जॉब`
- **Analysis**: इंटरव्यू (interview) and जॉब (job) are English loanwords in Devanagari. Both correctly tagged.

#### EN-2: Mobile technology — loanwords from error_samples

- **Reference**: `हाँ मोबाइल होते ही नहीं थे`
- **Tagged reference**: `हाँ [EN]मोबाइल[/EN] होते ही नहीं थे`
- **Raw ASR**: `हां, मॉबैल अते ही नहीं सिर्क`
- **Pipeline output**: `हां, मॉबैल अते ही नहीं सिर्क`
- **English words detected**: (none in ASR output)
- **Analysis**: मोबाइल / मॉबैल (mobile) — English loanword. Tagged in both ref and hyp.

#### EN-3: School / education loanwords

- **Reference**: `नए स्कूल में एडमिशन ले रहे हैं`
- **Tagged reference**: `नए [EN]स्कूल[/EN] में [EN]एडमिशन[/EN] ले रहे हैं`
- **Raw ASR**: `नैस्कुल में अइन्मिशन लए राई है`
- **Pipeline output**: `नैस्कुल में अइन्मिशन लए राई है`
- **English words detected**: (none in ASR output)
- **Analysis**: स्कूल (school), एडमिशन (admission) — both lexicon hits. Demonstrates ASR mangling loanwords (नैस्कुल, अइन्मिशन) without changing their English-origin classification.

#### EN-4: Colloquial English adverb in Hindi conversation

- **Reference**: `अभी क्या कर रहे हैं आप रिसेंटली`
- **Tagged reference**: `अभी क्या कर रहे हैं आप [EN]रिसेंटली[/EN]`
- **Raw ASR**: `अभी क्या कर रहे हैं आप इसलिकुन समत्यों`
- **Pipeline output**: `अभी क्या कर रहे हैं आप इसलिकुन समत्यों`
- **English words detected**: (none in ASR output)
- **Analysis**: रिसेंटली (recently) is a Devanagari English loanword. Correctly tagged in reference. In hypothesis it becomes a nonsense word (इसलिकुन) — shows ASR failure on loanwords.

#### EN-5: Fruits & food loanwords (from error_samples)

- **Reference**: `हूं तो आप जंग फ्रूट्स ज्यादा खाते हैं या मतलब कि नार्मल खाते हैं`
- **Tagged reference**: `हूं तो आप जंग [EN]फ्रूट्स[/EN] ज्यादा खाते हैं या मतलब कि [EN]नार्मल[/EN] खाते हैं`
- **Raw ASR**: `हम्म तो आप चंकिल दूरी जाते हैं मनही को सस्या नार्बल खाता है`
- **Pipeline output**: `हम्म तो आप चंकिल दूरी जाते हैं मनही को सस्या नार्बल खाता है`
- **English words detected**: (none in ASR output)
- **Analysis**: फ्रूट्स (fruits), नार्मल (normal) — both English loanwords. नार्बल in hypothesis is ASR distortion of नार्मल.

#### EN-6: Combined number + English loanword

- **Reference**: `तेरह मिनट की मीटिंग थी`
- **Tagged reference**: `तेरह [EN]मिनट[/EN] की [EN]मीटिंग[/EN] थी`
- **Raw ASR**: `तेरह मिनट की मीटिंग थी`
- **Pipeline output**: `13 [EN]मिनट[/EN] की [EN]मीटिंग[/EN] थी`
- **English words detected**: `मिनट`, `मीटिंग`
- **Analysis**: तेरह → 13 (number). मीटिंग (meeting), मिनट (minute) — English loanwords. Pipeline applies both steps.

#### EN-7: Ekchually / adverb loanword

- **Reference**: `बट सर एक्चुली इस किताब के पीछे है ना`
- **Tagged reference**: `[EN]बट[/EN] सर [EN]एक्चुली[/EN] इस किताब के पीछे है ना`
- **Raw ASR**: `बट सर एक्सुली इस किताम के भी चैन हो लहां`
- **Pipeline output**: `[EN]बट[/EN] सर [EN]एक्सुली[/EN] इस किताम के भी चैन हो लहां`
- **English words detected**: `बट`, `एक्सुली`
- **Analysis**: एक्चुली (actually), बट (but) — both English. किताब is actually an Arabic/Urdu loanword, not English — correctly NOT tagged.

#### REAL-1: From error_samples — ludo/carom games

- **Reference**: `लूडो कैरम भी खेला है`
- **Tagged reference**: `[EN]लूडो[/EN] [EN]कैरम[/EN] भी खेला है`
- **Raw ASR**: `लूटो करम्यें भी खिला है`
- **Pipeline output**: `लूटो करम्यें भी खिला है`
- **English words detected**: (none in ASR output)
- **Analysis**: लूडो (Ludo) and कैरम (Carrom) are English board game names → loanwords. ASR transcribes them poorly but they remain classifiable.

#### REAL-2: From error_samples — school/college topic

- **Reference**: `कॉलेज में एडमिशन ले रहे हैं`
- **Tagged reference**: `[EN]कॉलेज[/EN] में [EN]एडमिशन[/EN] ले रहे हैं`
- **Raw ASR**: `कॉलेज मे अइन्मिशन लए राई है`
- **Pipeline output**: `[EN]कॉलेज[/EN] मे अइन्मिशन लए राई है`
- **English words detected**: `कॉलेज`
- **Analysis**: कॉलेज (college), एडमिशन (admission) — both tagged. ASR distorts एडमिशन to अइन्मिशन.

#### REAL-3: Topic / subject — education loanwords

- **Reference**: `ये सबजेक्ट कैसा है उसके बारे में पढ़ते हैं`
- **Tagged reference**: `ये [EN]सबजेक्ट[/EN] कैसा है उसके बारे में पढ़ते हैं`
- **Raw ASR**: `ये सब्येक कैसा है उज्की बारे भी पड़तें थे`
- **Pipeline output**: `ये सब्येक कैसा है उज्की बारे भी पड़तें थे`
- **English words detected**: (none in ASR output)
- **Analysis**: सबजेक्ट (subject) → English loanword. ASR output सब्येक is a distortion of the same word.

## WER Impact of Number Normalisation

Number normalisation is applied to the ASR *hypothesis*, not the reference.
The WER change shows whether converting number words to digits moves the
hypothesis closer to or further from the human reference.

| Example | WER before | WER hyp-only (↑ misleading) | WER **both** normalised (correct) | Outcome |
|---------|-----------|---------------------------|-----------------------------------|---------|
| NUM-1 | 0.0000 | 0.2000 (+0.2000) | 0.0000 (+0.0000) | — Unchanged |
| NUM-3 | 0.0000 | 0.5000 (+0.5000) | 0.0000 (+0.0000) | — Unchanged |
| NUM-4 | 0.0000 | 0.5714 (+0.5714) | 0.0000 (+0.0000) | — Unchanged |
| NUM-5 | 0.0000 | 0.2000 (+0.2000) | 0.0000 (+0.0000) | — Unchanged |
| EN-6 | 0.0000 | 0.2000 (+0.2000) | 0.0000 (+0.0000) | — Unchanged |

**Summary**: 0 improved, 5 unchanged, 0 regressed

### Why two 'after' columns?

'WER hyp-only' converts only the hypothesis to digits — the reference
still has word-forms. This **always looks worse** (e.g. ref=`तेरह मिनट`
vs hyp=`13 मिनट` — the digit `13` never matches the word `तेरह`).

'WER both normalised' converts **both** ref and hyp to digits first,
then measures edit distance. This is the **correct evaluation**:
if the ASR got the number right but in word form, WER drops to 0 after
normalisation. If the ASR got a completely wrong word (e.g. `टेड` instead
of `डेढ़`), normalisation has no effect — the error remains visible.

## Limitations & Future Work

### Number normalisation
- **Ordinals** (पहला, दूसरा, तीसरा) are not converted — they are
  semantically distinct and would require separate handling (1st, 2nd).
- **Fractions** (डेढ़ = 1.5, ढाई = 2.5, पौने = 0.75×) are not handled
  and are left as-is.
- **Idiom coverage**: the current idiom block list covers the most common
  patterns but is not exhaustive. A statistical co-occurrence model
  trained on the corpus would generalise better.

### English word detection
- **Lexicon recall**: the 100+ word lexicon covers high-frequency loanwords.
  Rare domain-specific terms (medical, legal) may be missed.
- **Phonological false positives**: some low-frequency native Hindi words
  (e.g. Sanskrit-origin) contain ऑ or ज़ and may be incorrectly tagged.
- **Script normalisation**: the pipeline currently only handles Devanagari-
  script loanwords. Roman-script fragments (transcription errors per
  guideline) are ignored — a separate correction pass is needed for those.
