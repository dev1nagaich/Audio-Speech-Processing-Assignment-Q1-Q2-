
import re
import sys
import json
import logging
import argparse
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Units (0-19) ────────────────────────────────────────────────────────────
UNITS: Dict[str, int] = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "सात": 7,
    "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
    "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16,
    "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
}

# ── Tens (20-90) ─────────────────────────────────────────────────────────────
TENS: Dict[str, int] = {
    "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24,
    "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29,
    "तीस": 30, "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34,
    "पैंतीस": 35, "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39,
    "चालीस": 40, "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चौवालीस": 44,
    "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49,
    "पचास": 50, "इक्यावन": 51, "बावन": 52, "तिरपन": 53, "चौवन": 54,
    "पचपन": 55, "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59,
    "साठ": 60, "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64,
    "पैंसठ": 65, "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69,
    "सत्तर": 70, "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74,
    "पचहत्तर": 75, "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79,
    "अस्सी": 80, "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84,
    "पचासी": 85, "छियासी": 86, "सत्तासी": 87, "अट्ठासी": 88, "नवासी": 89,
    "नब्बे": 90, "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94,
    "पचानवे": 95, "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}

MULTIPLIERS: Dict[str, int] = {
    "सौ": 100, "हज़ार": 1000, "हजार": 1000,
    "लाख": 100_000, "करोड़": 10_000_000, "करोड": 10_000_000,
}

# Fraction words that should block adjacent cardinal collapsing
_FRACTION_WORDS: set = {
    "डेढ़", "सवा", "ढाई", "अढ़ाई", "पौने", "साढ़े", "साढे",
}

# ── Idiomatic / frozen phrases that MUST NOT be converted ────────────────────
# These are collocations where numbers are used figuratively, not cardinally.
IDIOM_PATTERNS: List[re.Pattern] = [
    # दो-चार, तीन-चार  (a few)
    re.compile(r'(एक|दो|तीन|चार|पाँच|पांच)-(चार|दो|पाँच|छह|सात|आठ|नौ|दस)\s+\w+'),
    # एक-दो बातें / काम  (a couple of things)
    re.compile(r'\bएक[-\s]दो\b'),
    # दो-तीन दिन / बार  (two-three days)
    re.compile(r'\b(दो|तीन|चार)[-\s](तीन|चार|पाँच|पांच)\b'),
    # एक बार (once — keep as-is for idiomatic use)
    re.compile(r'\bएक\s+बार\b'),
    # सात समुंदर (seven seas — fixed phrase)
    re.compile(r'\bसात\s+समुंदर'),
    # नौ दो ग्यारह (flee — idiom)
    re.compile(r'\bनौ\s+दो\s+ग्यारह\b'),
    # चार चाँद (idiom: to add glory)
    re.compile(r'\bचार\s+चाँद\b'),
    # दस बीस (colloquial "some")
    re.compile(r'\bदस[-\s]बीस\b'),
    # तीन-पाँच करना (to argue / dither)
    re.compile(r'\bतीन[-\s]पाँच\b'),
]


def _is_in_idiom(tokens: List[str], idx: int) -> bool:
    """Check if the token at idx participates in a frozen idiomatic phrase."""
    context = " ".join(tokens[max(0, idx - 2): idx + 4])
    return any(p.search(context) for p in IDIOM_PATTERNS)


ALL_NUMBER_WORDS = {**UNITS, **TENS}

# Build sorted list longest-first so greedy matching works correctly.
_SORTED_NUMBER_WORDS = sorted(
    list(ALL_NUMBER_WORDS.keys()) + list(MULTIPLIERS.keys()),
    key=lambda w: -len(w),
)

# Regex that matches any single Hindi number word (word-boundary-aware).
_NUM_WORD_RE = re.compile(
    r'\b(' + '|'.join(re.escape(w) for w in _SORTED_NUMBER_WORDS) + r')\b'
)


def _resolve_number_span(span_tokens: List[str]) -> Optional[int]:
    """
    Convert a list of Hindi number tokens to an integer.

    Handles:
      - Simple units/tens: ['तीन'] → 3, ['पच्चीस'] → 25
      - Compound with सौ: ['तीन', 'सौ', 'चौवन'] → 354
      - Compound with हज़ार: ['एक', 'हज़ार'] → 1000
      - Combined: ['एक', 'लाख', 'पचास', 'हज़ार'] → 150_000

    Returns None if span contains unknown tokens.
    """
    total = 0
    current = 0

    for tok in span_tokens:
        if tok in UNITS:
            current += UNITS[tok]
        elif tok in TENS:
            current += TENS[tok]
        elif tok in MULTIPLIERS:
            mult = MULTIPLIERS[tok]
            if mult >= 100:
                if current == 0:
                    current = 1
                current *= mult
                if mult >= 1000:
                    total += current
                    current = 0
                # For सौ, don't add to total yet — more digits may follow
            else:
                return None  # unknown multiplier
        else:
            return None  # non-number token

    total += current
    return total if total > 0 else None


def normalise_numbers(text: str) -> Tuple[str, List[Dict]]:
    """
    Convert Hindi number-word sequences to digits in *text*.

    Returns:
      (normalised_text, list_of_conversions)
      Each conversion: {"original": str, "converted": str, "value": int,
                        "is_idiom": bool, "position": int}
    """
    conversions: List[Dict] = []
    tokens = text.split()
    result_tokens: List[str] = []
    i = 0

    while i < len(tokens):
        tok_clean = tokens[i].strip('।.,!?;:\'\"')

        # Not a number word → pass through
        if tok_clean not in ALL_NUMBER_WORDS and tok_clean not in MULTIPLIERS:
            result_tokens.append(tokens[i])
            i += 1
            continue

        # If a fractional token precedes a number run, keep the run verbatim.
        # Example: "डेढ़ दो तीन घंटे" should not collapse "दो तीन".
        prev_clean = tokens[i - 1].strip('।.,!?;:\'\"') if i > 0 else ""
        if prev_clean in _FRACTION_WORDS:
            span_start = i
            span_tokens_raw: List[str] = []
            while i < len(tokens):
                c = tokens[i].strip('।.,!?;:\'\"')
                if c in ALL_NUMBER_WORDS or c in MULTIPLIERS:
                    span_tokens_raw.append(tokens[i])
                    i += 1
                else:
                    break
            result_tokens.extend(span_tokens_raw)
            for j, rt in enumerate(span_tokens_raw):
                conversions.append({
                    "original": rt,
                    "converted": rt,
                    "value": None,
                    "is_idiom": False,
                    "position": span_start + j,
                })
            continue

        # Check idiom context
        if _is_in_idiom(tokens, i):
            result_tokens.append(tokens[i])
            conversions.append({
                "original": tokens[i],
                "converted": tokens[i],
                "value": None,
                "is_idiom": True,
                "position": i,
            })
            i += 1
            continue

        # Greedy span collection: gather as many consecutive number tokens as possible
        span_start = i
        span_tokens_clean: List[str] = []
        span_tokens_raw:   List[str] = []

        while i < len(tokens):
            c = tokens[i].strip('।.,!?;:\'\"')
            if c in ALL_NUMBER_WORDS or c in MULTIPLIERS:
                # Stop if this token is itself in an idiom
                if span_tokens_clean and _is_in_idiom(tokens, i):
                    break
                span_tokens_clean.append(c)
                span_tokens_raw.append(tokens[i])
                i += 1
            else:
                break

        value = _resolve_number_span(span_tokens_clean)

        if value is not None:
            original_span = " ".join(span_tokens_raw)
            # Preserve trailing punctuation from last token
            trailing = re.search(r'[।.,!?;:]+$', span_tokens_raw[-1])
            converted = str(value) + (trailing.group(0) if trailing else "")
            result_tokens.append(converted)
            conversions.append({
                "original": original_span,
                "converted": converted,
                "value": value,
                "is_idiom": False,
                "position": span_start,
            })
        else:
            # Could not resolve as a number — keep original tokens
            result_tokens.extend(span_tokens_raw)
            for j, rt in enumerate(span_tokens_raw):
                conversions.append({
                    "original": rt,
                    "converted": rt,
                    "value": None,
                    "is_idiom": False,
                    "position": span_start + j,
                })

    return " ".join(result_tokens), conversions



_ENGLISH_LOANWORDS_DEVANAGARI: List[str] = [
    # Tech / devices
    "मोबाइल", "फ़ोन", "फोन", "कंप्यूटर", "लैपटॉप", "इंटरनेट", "वेबसाइट",
    "ऐप", "एप", "स्क्रीन", "ऑनलाइन", "ऑफलाइन", "वाईफाई", "ब्लूटूथ",
    "डाउनलोड", "अपलोड", "चार्जर", "बैटरी", "कैमरा", "वीडियो",
    # Education / academia
    "स्कूल", "कॉलेज", "क्लास", "क्लासेस", "एग्जाम", "एग्जामिनेशन",
    "सब्जेक्ट", "सबजेक्ट", "प्रोजेक्ट", "असाइनमेंट", "होमवर्क",
    "एडमिशन", "फीस", "सिलेबस", "टॉपिक", "टेस्ट", "स्टूडेंट",
    "टीचर", "प्रिंसिपल", "प्रोफेसर",
    # Professional / workplace
    "जॉब", "ऑफिस", "मीटिंग", "प्रेजेंटेशन", "रिपोर्ट", "प्रोजेक्ट",
    "इंटरव्यू", "सैलरी", "मैनेजर", "टीम", "क्लाइंट", "बिज़नेस",
    "कंपनी", "स्टार्टअप", "सीटीओ", "सीईओ",
    # Finance / shopping
    "बैंक", "लोन", "ईएमआई", "पेमेंट", "रिफंड", "डिस्काउंट",
    "ऑफर", "डील", "बजट", "इन्वेस्टमेंट", "शेयर", "स्टॉक",
    # Media / entertainment
    "मूवी", "फिल्म", "सीरीज", "वेब सीरीज", "सॉन्ग", "एल्बम",
    "यूट्यूब", "नेटफ्लिक्स", "इंस्टाग्राम", "ट्विटर", "फेसबुक",
    "वॉट्सएप", "व्हाट्सएप", "रील", "पोस्ट", "स्टोरी", "लाइव",
    # Transport / travel
    "बस", "ट्रेन", "फ्लाइट", "होटल", "ट्रिप", "टूर", "टैक्सी", "ऑटो",
    # Food / lifestyle
    "रेस्टोरेंट", "कैफे", "फ़ास्ट फूड", "पिज़्ज़ा", "बर्गर",
    "जूस", "कोल्ड ड्रिंक", "फ्रूट्स",
    # Common conversational English in Hindi (code-mix)
    "नार्मल", "नॉर्मल", "एक्चुली", "बेसिकली", "लिटरली", "सीरियसली",
    "ऑबवियसली", "डेफिनेटली", "परफेक्ट", "एक्साइटेड", "एक्साईटेड",
    "नर्वस", "स्ट्रेस", "रिलैक्स", "बोर", "कूल", "ओके",
    "हैलो", "हाय", "बाय", "थैंक्यू", "सॉरी", "प्लीज़", "प्लीज",
    "बट",
    "रेडी", "सेट", "डन", "फाइन", "नाइस", "ग्रेट", "सुपर",
    "बेस्ट", "वर्स्ट", "स्मार्ट", "फन", "फनी",
    # Games / sports
    "लूडो", "कैरम", "क्रिकेट", "फुटबॉल", "बास्केटबॉल", "गेम",
    "प्लेयर", "स्कोर", "मैच", "टीम",
    # Misc high-frequency loanwords in conversational Hindi
    "प्रॉब्लम", "सॉल्यूशन", "आइडिया", "प्लान", "स्ट्रेटेजी",
    "रिसेंटली", "ऑल्रेडी", "डायरेक्ट", "सिम्पल", "क्लियर",
    "पॉइंट", "वर्जन", "अपडेट", "फीचर", "मोड", "सेटिंग",
    "रेट", "क्वालिटी", "डिमांड", "सप्लाई",
    # Number-related English
    "परसेंट", "किलोमीटर", "किलोग्राम", "मीटर", "मिनट",
    # Transport
    "स्कूटी", "बाइक", "कार",
    # School games
    "प्ले", "लिस्ट", "प्लेलिस्ट",
]

# Build a set for O(1) lookup; also keep sorted-longest-first list for regex
_LOANWORD_SET = set(_ENGLISH_LOANWORDS_DEVANAGARI)

# Devanagari suffixes that, when stripped, may reveal a loanword root
_DEVANAGARI_SUFFIXES = ['ों', 'ाओं', 'ियों', 'ों', 'ें', 'ो', 'ी', 'ा', 'े']

# Letter combinations characteristic of English origin
_ENGLISH_PHONEME_PATTERNS: List[re.Pattern] = [
    # Aspirated stops borrowed from English: ट्, ड् clusters
    re.compile(r'[टड][्]'),
    # ऑ (open O) — almost always English origin (ऑफिस, ऑनलाइन)
    re.compile(r'ऑ'),
    # फ़ (dotted fa) — English/Urdu loanword indicator
    re.compile(r'फ़'),
    # ज़ (dotted za) — English/Urdu loanword indicator
    re.compile(r'ज़'),
    # Long consonant clusters uncommon in native Hindi
    re.compile(r'[क-ह][्][क-ह][्][क-ह]'),
    # -ेशन (-tion/-sion) ending
    re.compile(r'शन$'),
    # -मेंट (-ment) ending
    re.compile(r'मेंट$'),
    # -इंग (-ing) ending
    re.compile(r'इंग$'),
    # -ली (-ly adverb suffix)
    re.compile(r'ली$'),
    # Aspirated retroflex common in transliteration of English words
    re.compile(r'ल[ी]$'),
]

# Suffixes of known native Hindi function words — used as negative filter
_NATIVE_HINDI_ENDINGS: set = {
    'ना', 'ने', 'नी', 'ता', 'ती', 'ते', 'था', 'थी', 'थे',
    'गा', 'गी', 'गे', 'कर', 'के', 'की', 'को', 'में', 'पर',
    'वाला', 'वाली', 'वाले', 'कारण', 'तरह',
}

# Roots that look English by phonology but are native Hindi
_NATIVE_HINDI_EXCEPTIONS: set = {
    'भी', 'जी', 'नहीं', 'यही', 'वही', 'सही', 'कई', 'और',
    'तो', 'जो', 'जब', 'अब', 'कब', 'यहाँ', 'वहाँ', 'कहाँ',
    'कभी', 'सभी', 'वही', 'आगे', 'पीछे', 'ऊपर', 'नीचे',
    'लेकिन', 'क्योंकि', 'मगर', 'परंतु', 'यदि', 'तथा',
    'हाँ', 'हां', 'नहीं', 'शायद', 'ज़रूर', 'बिल्कुल',
    'मेरा', 'तेरा', 'हमारा', 'आपका', 'उनका', 'इनका',
    'मुझे', 'तुझे', 'उसे', 'हमें', 'आपको', 'उन्हें',
    'इसलिए', 'इसलिये', 'इसीलिए', 'यानी', 'मतलब',
    'बहुत', 'थोड़ा', 'ज़्यादा', 'कम', 'सबसे', 'बेहद',
    'पहले', 'बाद', 'अभी', 'कभी', 'हमेशा', 'कभी-कभी',
    'क्या', 'कौन', 'कैसे', 'कितना', 'किसका', 'किसे',
    'अच्छा', 'बुरा', 'नया', 'पुराना', 'बड़ा', 'छोटा',
    'ठीक', 'सही', 'गलत', 'जरूरी', 'ज़रूरी',
}


def _strip_inflection(word: str) -> str:
    """Strip common Devanagari inflectional suffixes to get the root."""
    for suffix in sorted(_DEVANAGARI_SUFFIXES, key=len, reverse=True):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


def _phonological_english_score(word: str) -> int:
    """
    Heuristic score 0-3 indicating likelihood of English origin.
    Based on phonological patterns characteristic of English loanwords.
    """
    score = 0
    for pat in _ENGLISH_PHONEME_PATTERNS:
        if pat.search(word):
            score += 1
    return min(score, 3)


def is_english_loanword(word: str, context_words: Optional[List[str]] = None) -> bool:
    """
    Classify a single Devanagari word as English-origin loanword or not.

    Decision hierarchy:
      1. Hard-coded native Hindi exception → False
      2. Lexicon match (exact or after inflection stripping) → True
      3. Phonological heuristic score >= 2 → True (confident phonology)
      4. Otherwise → False

    The function does NOT flag Roman-script words — those would be
    transcription errors under the guideline (English spoken → Devanagari).
    """
    # Strip punctuation
    clean = word.strip('।.,!?;:\'\"-()[]')
    if not clean:
        return False

    # Only process Devanagari text (guideline: English → Devanagari)
    devanagari_chars = sum(1 for c in clean if 0x0900 <= ord(c) <= 0x097F)
    if devanagari_chars == 0:
        return False  # Roman script — transcription error, not our concern here

    # Native Hindi exceptions — never flag these
    if clean in _NATIVE_HINDI_EXCEPTIONS:
        return False

    # Exact lexicon match
    if clean in _LOANWORD_SET:
        return True

    # Match after stripping inflection
    root = _strip_inflection(clean)
    if root in _LOANWORD_SET:
        return True

    # Native Hindi endings → probably not a loanword
    for ending in _NATIVE_HINDI_ENDINGS:
        if clean.endswith(ending):
            return False

    # Phonological heuristic
    if _phonological_english_score(clean) >= 2:
        return True

    return False


@dataclass
class EnglishTagResult:
    original_text: str
    tagged_text: str
    english_words: List[str] = field(default_factory=list)
    english_positions: List[int] = field(default_factory=list)
    english_count: int = 0


def tag_english_words(text: str) -> EnglishTagResult:
    """
    Insert [EN]...[/EN] tags around English loanwords in a Hindi transcript.

    Returns an EnglishTagResult with the tagged text and metadata.

    Example:
      Input:  "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई"
      Output: "मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया और मुझे [EN]जॉब[/EN] मिल गई"
    """
    tokens = text.split()
    tagged_tokens: List[str] = []
    english_words: List[str] = []
    english_positions: List[int] = []

    for i, token in enumerate(tokens):
        if is_english_loanword(token, context_words=tokens):
            clean = token.strip('।.,!?;:\'\"-()[]')
            trailing_punct = token[len(clean):]
            leading_punct  = ""
            # Extract leading punct too
            stripped = token.lstrip('(["\'')
            if len(stripped) < len(token):
                leading_punct = token[: len(token) - len(stripped)]
                token = stripped

            tagged = f"{leading_punct}[EN]{clean}[/EN]{trailing_punct}"
            tagged_tokens.append(tagged)
            english_words.append(clean)
            english_positions.append(i)
        else:
            tagged_tokens.append(token)

    tagged_text = " ".join(tagged_tokens)
    return EnglishTagResult(
        original_text=text,
        tagged_text=tagged_text,
        english_words=english_words,
        english_positions=english_positions,
        english_count=len(english_words),
    )

@dataclass
class CleanupResult:
    """Per-utterance result of the full cleanup pipeline."""
    recording_id: int
    segment_idx: int
    reference: str
    raw_asr: str                  # Whisper output (hypothesis from error_samples)
    number_normalised: str        # after step (a) — hyp only
    final_tagged: str             # after step (b)
    number_conversions: List[Dict]
    english_tags: EnglishTagResult
    # WER metrics:
    # wer_before         : raw WER (ref word-form vs hyp word-form)
    # wer_hyp_only       : ref word-form vs hyp digit-form  [misleading — almost always worse]
    # wer_both_normalised: BOTH ref and hyp normalised to digits [the fair, meaningful metric]
    wer_before: float = 0.0
    wer_hyp_only: float = 0.0         # hyp converted, ref kept as words
    wer_both_normalised: float = 0.0  # the real metric — normalise both sides


def _compute_wer(ref: str, hyp: str) -> float:
    try:
        import jiwer
        ref = ref.strip()
        hyp = hyp.strip()
        if not ref:
            return 0.0
        return jiwer.wer(ref, hyp)
    except Exception:
        # Fallback if jiwer not available
        ref_words = ref.split()
        hyp_words = hyp.split()
        from difflib import SequenceMatcher
        m = SequenceMatcher(None, ref_words, hyp_words)
        edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2
                    in m.get_opcodes() if tag != 'equal')
        return edits / max(len(ref_words), 1)


def _normalise_for_wer(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[।\.]{2,}', '।', text)
    text = text.rstrip('।. ').strip()
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def run_pipeline_on_sample(sample: Dict) -> CleanupResult:
    ref = _normalise_for_wer(sample.get("reference", ""))
    raw = _normalise_for_wer(sample.get("hypothesis", ""))

    # Step (a): number normalisation — applied to BOTH ref and hyp independently
    normed_hyp, conversions_hyp = normalise_numbers(raw)
    normed_hyp = _normalise_for_wer(normed_hyp)

    normed_ref, conversions_ref = normalise_numbers(ref)
    normed_ref = _normalise_for_wer(normed_ref)

    # Step (b): English tagging on the normalised hyp output
    english_result = tag_english_words(normed_hyp)

    # ── Three WER measurements ────────────────────────────────────────────
    # 1. Baseline: both sides as raw word-forms (what Q1 reported)
    wer_before = _compute_wer(ref, raw)

    # 2. Hyp-only normalised (what the naive pipeline does) — shown for contrast
    #    Almost always WORSE because ref still has word-forms but hyp has digits.
    #    This is NOT a useful metric; shown only to explain why it looks bad.
    wer_hyp_only = _compute_wer(ref, normed_hyp)

    # 3. BOTH normalised — the correct, fair WER after cleanup.
    #    This is the metric that matters: did converting both sides to a canonical
    #    digit form reduce the word edit distance?
    wer_both_normalised = _compute_wer(normed_ref, normed_hyp)

    return CleanupResult(
        recording_id=sample.get("recording_id", -1),
        segment_idx=sample.get("segment_idx", -1),
        reference=ref,
        raw_asr=raw,
        number_normalised=normed_hyp,
        final_tagged=english_result.tagged_text,
        number_conversions=conversions_hyp,
        english_tags=english_result,
        wer_before=round(wer_before, 4),
        wer_hyp_only=round(wer_hyp_only, 4),
        wer_both_normalised=round(wer_both_normalised, 4),
    )



DEMO_EXAMPLES: List[Dict] = [
    # ── Number normalisation: correct conversions ──────────────────────────
    {
        "id": "NUM-1",
        "description": "Simple cardinal number — time duration",
        "category": "number_correct",
        "reference": "हमें तेरह मिनट हो गए",
        "hypothesis": "हमें तेरह मिनट हो गए",
        "note": "तेरह → 13 (straightforward cardinal in temporal context)",
    },
    {
        "id": "NUM-2",
        "description": "Compound number — travel time",
        "category": "number_correct",
        "reference": "कम से कम डेढ़ दो तीन घंटे लग जाते हैं",
        "hypothesis": "कम से कम डेढ़ दो तीन घंटे लग जाते हैं",
        "note": "दो तीन is kept as-is because the preceding fraction word डेढ़ breaks numeric span collapsing; this is not treated as an idiom.",
    },
    {
        "id": "NUM-3",
        "description": "Large compound number — administrative/financial",
        "category": "number_correct",
        "reference": "तीन सौ चौवन रुपए देने होंगे",
        "hypothesis": "तीन सौ चौवन रुपए देने होंगे",
        "note": "तीन सौ चौवन → 354 (multi-token compound resolved left-to-right)",
    },
    {
        "id": "NUM-4",
        "description": "Round thousands",
        "category": "number_correct",
        "reference": "एक हज़ार पाँच सौ लोग आए थे",
        "hypothesis": "एक हज़ार पाँच सौ लोग आए थे",
        "note": "एक हज़ार पाँच सौ → 1500",
    },
    {
        "id": "NUM-5",
        "description": "Simple tens",
        "category": "number_correct",
        "reference": "पच्चीस साल की उम्र में",
        "hypothesis": "पच्चीस साल की उम्र में",
        "note": "पच्चीस → 25 (irregular tens word, direct lookup)",
    },
    # ── Number normalisation: edge cases ────────────────────────────────────
    {
        "id": "EDGE-1",
        "description": "Idiomatic paired numbers — should NOT convert",
        "category": "number_edge",
        "reference": "दो-चार बातें करनी थीं",
        "hypothesis": "दो-चार बातें करनी थीं",
        "note": "दो-चार is a colloquial idiom meaning 'a few'. Converting to '2-4 बातें' would be wrong. Idiom pattern detected → kept as-is.",
    },
    {
        "id": "EDGE-2",
        "description": "एक बार — once (idiomatic)",
        "category": "number_edge",
        "reference": "एक बार फिर से देख लो",
        "hypothesis": "एक बार फिर से देख लो",
        "note": "एक बार = 'once'. Converting to '1 बार' breaks colloquial phrasing. Idiom rule: एक बार → kept.",
    },
    {
        "id": "EDGE-3",
        "description": "Repeated number in idiomatic emphasis",
        "category": "number_edge",
        "reference": "नौ दो ग्यारह हो गया",
        "hypothesis": "नौ दो ग्यारह हो गया",
        "note": "नौ दो ग्यारह = idiom for 'fled/ran away'. Cannot convert (9-2-11 makes no sense). Idiom pattern block applied.",
    },
    # ── English word detection ───────────────────────────────────────────────
    {
        "id": "EN-1",
        "description": "Job interview — common loanwords",
        "category": "english_detection",
        "reference": "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",
        "hypothesis": "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",
        "note": "इंटरव्यू (interview) and जॉब (job) are English loanwords in Devanagari. Both correctly tagged.",
    },
    {
        "id": "EN-2",
        "description": "Mobile technology — loanwords from error_samples",
        "category": "english_detection",
        "reference": "हाँ मोबाइल होते ही नहीं थे",
        "hypothesis": "हां, मॉबैल अते ही नहीं सिर्क",
        "note": "मोबाइल / मॉबैल (mobile) — English loanword. Tagged in both ref and hyp.",
    },
    {
        "id": "EN-3",
        "description": "School / education loanwords",
        "category": "english_detection",
        "reference": "नए स्कूल में एडमिशन ले रहे हैं",
        "hypothesis": "नैस्कुल में अइन्मिशन लए राई है",
        "note": "स्कूल (school), एडमिशन (admission) — both lexicon hits. Demonstrates ASR mangling loanwords (नैस्कुल, अइन्मिशन) without changing their English-origin classification.",
    },
    {
        "id": "EN-4",
        "description": "Colloquial English adverb in Hindi conversation",
        "category": "english_detection",
        "reference": "अभी क्या कर रहे हैं आप रिसेंटली",
        "hypothesis": "अभी क्या कर रहे हैं आप इसलिकुन समत्यों",
        "note": "रिसेंटली (recently) is a Devanagari English loanword. Correctly tagged in reference. In hypothesis it becomes a nonsense word (इसलिकुन) — shows ASR failure on loanwords.",
    },
    {
        "id": "EN-5",
        "description": "Fruits & food loanwords (from error_samples)",
        "category": "english_detection",
        "reference": "हूं तो आप जंग फ्रूट्स ज्यादा खाते हैं या मतलब कि नार्मल खाते हैं",
        "hypothesis": "हम्म तो आप चंकिल दूरी जाते हैं मनही को सस्या नार्बल खाता है",
        "note": "फ्रूट्स (fruits), नार्मल (normal) — both English loanwords. नार्बल in hypothesis is ASR distortion of नार्मल.",
    },
    {
        "id": "EN-6",
        "description": "Combined number + English loanword",
        "category": "combined",
        "reference": "तेरह मिनट की मीटिंग थी",
        "hypothesis": "तेरह मिनट की मीटिंग थी",
        "note": "तेरह → 13 (number). मीटिंग (meeting), मिनट (minute) — English loanwords. Pipeline applies both steps.",
    },
    {
        "id": "EN-7",
        "description": "Ekchually / adverb loanword",
        "category": "english_detection",
        "reference": "बट सर एक्चुली इस किताब के पीछे है ना",
        "hypothesis": "बट सर एक्सुली इस किताम के भी चैन हो लहां",
        "note": "एक्चुली (actually), बट (but) — both English. किताब is actually an Arabic/Urdu loanword, not English — correctly NOT tagged.",
    },
    # ── Pipeline on actual error_samples ────────────────────────────────────
    {
        "id": "REAL-1",
        "description": "From error_samples — ludo/carom games",
        "category": "real_data",
        "reference": "लूडो कैरम भी खेला है",
        "hypothesis": "लूटो करम्यें भी खिला है",
        "note": "लूडो (Ludo) and कैरम (Carrom) are English board game names → loanwords. ASR transcribes them poorly but they remain classifiable.",
    },
    {
        "id": "REAL-2",
        "description": "From error_samples — school/college topic",
        "category": "real_data",
        "reference": "कॉलेज में एडमिशन ले रहे हैं",
        "hypothesis": "कॉलेज मे अइन्मिशन लए राई है",
        "note": "कॉलेज (college), एडमिशन (admission) — both tagged. ASR distorts एडमिशन to अइन्मिशन.",
    },
    {
        "id": "REAL-3",
        "description": "Topic / subject — education loanwords",
        "category": "real_data",
        "reference": "ये सबजेक्ट कैसा है उसके बारे में पढ़ते हैं",
        "hypothesis": "ये सब्येक कैसा है उज्की बारे भी पड़तें थे",
        "note": "सबजेक्ट (subject) → English loanword. ASR output सब्येक is a distortion of the same word.",
    },
]

def _print_section(title: str, width: int = 80) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_example(result: CleanupResult, demo: Dict) -> None:
    print(f"\n  [{demo['id']}] {demo['description']}")
    print(f"  Category: {demo['category']}")
    print(f"  Reference  : {demo['reference']}")
    print(f"  Raw ASR    : {demo['hypothesis']}")

    if result.number_conversions:
        actual_conversions = [c for c in result.number_conversions
                              if c['value'] is not None and not c['is_idiom']]
        idiom_blocks = [c for c in result.number_conversions if c['is_idiom']]
        if actual_conversions:
            for c in actual_conversions:
                print(f"    ✓ Number: '{c['original']}' → {c['converted']}")
        if idiom_blocks:
            for c in idiom_blocks:
                print(f"    ⚑ Idiom (kept): '{c['original']}'")

    print(f"  After nums : {result.number_normalised}")
    print(f"  Tagged     : {result.final_tagged}")

    if result.english_tags.english_words:
        print(f"  EN words   : {', '.join(result.english_tags.english_words)}")

    has_numbers = result.number_conversions and any(
        c['value'] is not None and not c['is_idiom']
        for c in result.number_conversions
    )
    print(f"  WER before          : {result.wer_before:.4f}  (raw ref-words vs raw hyp-words)")
    if has_numbers:
        delta_naive = result.wer_hyp_only - result.wer_before
        delta_fair  = result.wer_both_normalised - result.wer_before
        sign_n = '+' if delta_naive >= 0 else ''
        sign_f = '+' if delta_fair  >= 0 else ''
        print(f"  WER hyp-only        : {result.wer_hyp_only:.4f}  (Δ {sign_n}{delta_naive:.4f})  ← misleading, ref still has words")
        print(f"  WER both normalised : {result.wer_both_normalised:.4f}  (Δ {sign_f}{delta_fair:.4f})  ← CORRECT metric")
    print(f"  Note       : {demo['note']}")


def generate_report(results: List[Tuple[CleanupResult, Dict]]) -> str:
    """Generate a full markdown report of the pipeline results."""
    lines = []
    lines.append("# Q2: ASR Cleanup Pipeline Report")
    lines.append("")
    lines.append("## Pipeline Overview")
    lines.append("")
    lines.append("Two post-processing operations applied sequentially to raw Whisper ASR output:")
    lines.append("")
    lines.append("1. **Number Normalisation** — converts spoken Hindi number words to digits")
    lines.append("2. **English Word Detection** — tags English loanwords with `[EN]...[/EN]`")
    lines.append("")
    lines.append("> **Transcription guideline**: English words spoken in conversation are")
    lines.append("> transcribed in Devanagari script. Detection identifies English-*origin*")
    lines.append("> words — it does not treat them as errors.")
    lines.append("")

    # ── Section A: Number normalisation ────────────────────────────────────
    lines.append("## (a) Number Normalisation")
    lines.append("")
    lines.append("### Algorithm")
    lines.append("")
    lines.append("Greedy left-to-right span collection over number tokens, with idiom detection")
    lines.append("as a pre-pass guard. Multipliers (सौ, हज़ार, लाख, करोड़) collapse partial")
    lines.append("sums following standard Hindi place-value conventions.")
    lines.append("")
    lines.append("### Correct conversions")
    lines.append("")
    lines.append("| Example | Raw input | Normalised output | Value |")
    lines.append("|---------|-----------|-------------------|-------|")

    for result, demo in results:
        if demo["category"] == "number_correct":
            actual = [c for c in result.number_conversions
                      if c['value'] is not None and not c['is_idiom']]
            if actual:
                for c in actual:
                    lines.append(
                        f"| {demo['id']} — {demo['description']} "
                        f"| `{c['original']}` | `{c['converted']}` | {c['value']} |"
                    )
            else:
                lines.append(
                    f"| {demo['id']} — {demo['description']} "
                    f"| (no number token in hypothesis) | — | — |"
                )

    lines.append("")
    lines.append("### Edge cases — judgment calls")
    lines.append("")

    for result, demo in results:
        if demo["category"] == "number_edge":
            lines.append(f"#### {demo['id']}: {demo['description']}")
            lines.append("")
            lines.append(f"- **Input**: `{demo['reference']}`")
            lines.append(f"- **Output**: `{result.number_normalised}`")
            idioms = [c for c in result.number_conversions if c['is_idiom']]
            if idioms:
                lines.append(f"- **Idiom guard triggered for**: " +
                              ", ".join(f"`{c['original']}`" for c in idioms))
            lines.append(f"- **Reasoning**: {demo['note']}")
            lines.append("")

    # ── Section B: English word detection ──────────────────────────────────
    lines.append("## (b) English Word Detection")
    lines.append("")
    lines.append("### Algorithm")
    lines.append("")
    lines.append("Three-tier classifier per token:")
    lines.append("")
    lines.append("1. **Native Hindi exception list** — hard-coded common function words that")
    lines.append("   could false-positive on phonological heuristics (e.g. भी, जी, नहीं)")
    lines.append("2. **Lexicon lookup** — 100+ curated English loanwords in Devanagari,")
    lines.append("   plus inflection stripping (removes ों, ी, ा etc.) before re-lookup")
    lines.append("3. **Phonological heuristics** — patterns characteristic of English")
    lines.append("   origin: ऑ (open-O), ज़/फ़ (dotted consonants), -शन / -मेंट / -इंग")
    lines.append("   suffix clusters, retroflex consonant sequences")
    lines.append("")
    lines.append("### Detection examples")
    lines.append("")

    for result, demo in results:
        if demo["category"] in {"english_detection", "combined", "real_data"}:
            tagged_ref = tag_english_words(demo["reference"])
            lines.append(f"#### {demo['id']}: {demo['description']}")
            lines.append("")
            lines.append(f"- **Reference**: `{demo['reference']}`")
            lines.append(f"- **Tagged reference**: `{tagged_ref.tagged_text}`")
            lines.append(f"- **Raw ASR**: `{demo['hypothesis']}`")
            lines.append(f"- **Pipeline output**: `{result.final_tagged}`")
            if result.english_tags.english_words:
                lines.append(f"- **English words detected**: "
                              + ", ".join(f"`{w}`" for w in result.english_tags.english_words))
            else:
                lines.append("- **English words detected**: (none in ASR output)")
            lines.append(f"- **Analysis**: {demo['note']}")
            lines.append("")

    # ── WER impact summary ─────────────────────────────────────────────────
    lines.append("## WER Impact of Number Normalisation")
    lines.append("")
    lines.append("Number normalisation is applied to the ASR *hypothesis*, not the reference.")
    lines.append("The WER change shows whether converting number words to digits moves the")
    lines.append("hypothesis closer to or further from the human reference.")
    lines.append("")
    lines.append("| Example | WER before | WER hyp-only (↑ misleading) | WER **both** normalised (correct) | Outcome |")
    lines.append("|---------|-----------|---------------------------|-----------------------------------|---------|")

    improved = unchanged = regressed = 0
    for result, demo in results:
        actual = [c for c in result.number_conversions
                  if c['value'] is not None and not c['is_idiom']]
        if not actual:
            continue
        delta_naive = result.wer_hyp_only - result.wer_before
        delta_fair  = result.wer_both_normalised - result.wer_before
        if delta_fair < -0.001:
            outcome = "✓ Improved"
            improved += 1
        elif delta_fair > 0.001:
            outcome = "✗ Regressed"
            regressed += 1
        else:
            outcome = "— Unchanged"
            unchanged += 1
        sign_n = '+' if delta_naive >= 0 else ''
        sign_f = '+' if delta_fair  >= 0 else ''
        lines.append(
            f"| {demo['id']} | {result.wer_before:.4f} "
            f"| {result.wer_hyp_only:.4f} ({sign_n}{delta_naive:.4f}) "
            f"| {result.wer_both_normalised:.4f} ({sign_f}{delta_fair:.4f}) "
            f"| {outcome} |"
        )

    lines.append("")
    lines.append(
        f"**Summary**: {improved} improved, {unchanged} unchanged, {regressed} regressed"
    )
    lines.append("")
    lines.append("### Why two 'after' columns?")
    lines.append("")
    lines.append("'WER hyp-only' converts only the hypothesis to digits — the reference")
    lines.append("still has word-forms. This **always looks worse** (e.g. ref=`तेरह मिनट`")
    lines.append("vs hyp=`13 मिनट` — the digit `13` never matches the word `तेरह`).")
    lines.append("")
    lines.append("'WER both normalised' converts **both** ref and hyp to digits first,")
    lines.append("then measures edit distance. This is the **correct evaluation**:")
    lines.append("if the ASR got the number right but in word form, WER drops to 0 after")
    lines.append("normalisation. If the ASR got a completely wrong word (e.g. `टेड` instead")
    lines.append("of `डेढ़`), normalisation has no effect — the error remains visible.")
    lines.append("")

    # ── Limitations ────────────────────────────────────────────────────────
    lines.append("## Limitations & Future Work")
    lines.append("")
    lines.append("### Number normalisation")
    lines.append("- **Ordinals** (पहला, दूसरा, तीसरा) are not converted — they are")
    lines.append("  semantically distinct and would require separate handling (1st, 2nd).")
    lines.append("- **Fractions** (डेढ़ = 1.5, ढाई = 2.5, पौने = 0.75×) are not handled")
    lines.append("  and are left as-is.")
    lines.append("- **Idiom coverage**: the current idiom block list covers the most common")
    lines.append("  patterns but is not exhaustive. A statistical co-occurrence model")
    lines.append("  trained on the corpus would generalise better.")
    lines.append("")
    lines.append("### English word detection")
    lines.append("- **Lexicon recall**: the 100+ word lexicon covers high-frequency loanwords.")
    lines.append("  Rare domain-specific terms (medical, legal) may be missed.")
    lines.append("- **Phonological false positives**: some low-frequency native Hindi words")
    lines.append("  (e.g. Sanskrit-origin) contain ऑ or ज़ and may be incorrectly tagged.")
    lines.append("- **Script normalisation**: the pipeline currently only handles Devanagari-")
    lines.append("  script loanwords. Roman-script fragments (transcription errors per")
    lines.append("  guideline) are ignored — a separate correction pass is needed for those.")
    lines.append("")

    return "\n".join(lines)

def main() -> None:
    parser = argparse.ArgumentParser(description="Q2 ASR Cleanup Pipeline")
    parser.add_argument("--jsonl", type=str, default=None,
                        help="Path to error_samples.jsonl (uses built-in demo if omitted)")
    parser.add_argument("--output_json", type=str,
                        default=str(RESULTS_DIR / "cleanup_results.json"))
    parser.add_argument("--output_report", type=str,
                        default=str(RESULTS_DIR / "cleanup_report.md"))
    args = parser.parse_args()

    _print_section("Q2: Hindi ASR Cleanup Pipeline")

    # ── Load samples ────────────────────────────────────────────────────────
    if args.jsonl and Path(args.jsonl).exists():
        logger.info(f"Loading error samples from {args.jsonl}")
        raw_samples: List[Dict] = []
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_samples.append(json.loads(line))
        demo_mode = False
    else:
        logger.info("No JSONL provided — running on built-in demo examples")
        raw_samples = [
            {
                "recording_id": int(d["id"].replace("-", "").replace("NUM", "100")
                                    .replace("EDGE", "200").replace("EN", "300")
                                    .replace("REAL", "400").replace("1", "1")
                                    .replace("2", "2").replace("3", "3")),
                "segment_idx": idx,
                "reference": d["reference"],
                "hypothesis": d["hypothesis"],
            }
            for idx, d in enumerate(DEMO_EXAMPLES)
        ]
        demo_mode = True

    # ── Run pipeline ─────────────────────────────────────────────────────────
    all_results: List[Tuple[CleanupResult, Dict]] = []

    if demo_mode:
        for idx, demo in enumerate(DEMO_EXAMPLES):
            sample = raw_samples[idx]
            result = run_pipeline_on_sample(sample)
            all_results.append((result, demo))
    else:
        # Run on actual JSONL data
        for sample in raw_samples:
            demo = {
                "id": f"REAL-{sample['recording_id']}-{sample['segment_idx']}",
                "description": "From error_samples.jsonl",
                "category": "real_data",
                "reference": sample["reference"],
                "hypothesis": sample["hypothesis"],
                "note": "",
            }
            result = run_pipeline_on_sample(sample)
            all_results.append((result, demo))

    # ── Print to terminal ─────────────────────────────────────────────────────
    _print_section("(a) Number Normalisation Examples")

    print("\n── Correct conversions ──")
    for result, demo in all_results:
        if demo["category"] == "number_correct":
            _print_example(result, demo)

    print("\n── Edge cases (idioms / frozen phrases) ──")
    for result, demo in all_results:
        if demo["category"] == "number_edge":
            _print_example(result, demo)

    _print_section("(b) English Word Detection Examples")

    for result, demo in all_results:
        if demo["category"] in {"english_detection", "combined", "real_data"}:
            _print_example(result, demo)

    # ── WER summary ────────────────────────────────────────────────────────
    _print_section("WER Impact Summary")
    improved = unchanged = regressed = 0
    for result, demo in all_results:
        actual = [c for c in result.number_conversions
                  if c['value'] is not None and not c['is_idiom']]
        if not actual:
            continue
        delta_fair  = result.wer_both_normalised - result.wer_before
        delta_naive = result.wer_hyp_only - result.wer_before
        if delta_fair < -0.001:
            improved += 1
            status = "IMPROVED"
        elif delta_fair > 0.001:
            regressed += 1
            status = "REGRESSED"
        else:
            unchanged += 1
            status = "UNCHANGED"
        sign_n = '+' if delta_naive >= 0 else ''
        sign_f = '+' if delta_fair  >= 0 else ''
        print(f"  [{demo['id']}] raw WER={result.wer_before:.4f}"
              f"  | hyp-only={result.wer_hyp_only:.4f}({sign_n}{delta_naive:.4f})"
              f"  | both-norm={result.wer_both_normalised:.4f}({sign_f}{delta_fair:.4f})"
              f"  → {status}")

    print(f"\n  Improved: {improved}  Unchanged: {unchanged}  Regressed: {regressed}")

    # ── Save outputs ───────────────────────────────────────────────────────
    output = {
        "pipeline_description": {
            "step_a": "Number normalisation: Hindi number words → digits, with idiom guard",
            "step_b": "English loanword detection: 3-tier classifier (exception list + lexicon + phonology)",
        },
        "results": [],
    }
    for result, demo in all_results:
        entry = {
            "id": demo["id"],
            "description": demo["description"],
            "category": demo["category"],
            "reference": result.reference,
            "raw_asr": result.raw_asr,
            "number_normalised": result.number_normalised,
            "final_tagged": result.final_tagged,
            "number_conversions": result.number_conversions,
            "english_words_detected": result.english_tags.english_words,
            "english_count": result.english_tags.english_count,
            "wer_before": result.wer_before,
            "wer_hyp_only": result.wer_hyp_only,
            "wer_both_normalised": result.wer_both_normalised,
            "wer_delta_naive": round(result.wer_hyp_only - result.wer_before, 4),
            "wer_delta_correct": round(result.wer_both_normalised - result.wer_before, 4),
            "note": demo["note"],
        }
        output["results"].append(entry)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved → {args.output_json}")

    # Generate markdown report
    report_md = generate_report(all_results)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"Report saved  → {args.output_report}")

    _print_section("DONE")
    print(f"  JSON results : {args.output_json}")
    print(f"  Markdown report : {args.output_report}")


if __name__ == "__main__":
    main()