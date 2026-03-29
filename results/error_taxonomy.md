# Error Taxonomy Analysis

**Total analysed errors**: 25

## Sampling Strategy

Total validation samples evaluated: 386

Total utterances with WER > 0: 385

Stratified every-Nth sampling across three severity bands (built independently per stratum):

- **Low** (0 < WER < 0.3): pool = 2, every-N step = 1, drawn = 2
- **Medium** (0.3 ≤ WER ≤ 1.0): pool = 343, every-N step = 34, drawn = 10
- **High** (WER > 1.0): pool = 40, every-N step = 4, drawn = 10
- **Fill-remaining**: pool = 363, step = 121, drawn = 3

**Final sample count**: 25

_No cherry-picking: samples drawn mechanically using every-Nth within each WER severity stratum._

## Error Category Overview

| Category | Count | % of sample |
|---|---|---|
| Casual / Colloquial Hindi | 4 | 16.0% |
| Function Word Deletions | 11 | 44.0% |
| Insertion of Noise / Fillers | 3 | 12.0% |
| Other / Uncategorised | 7 | 28.0% |

## Error Categories - Detailed Examples

### Casual / Colloquial Hindi

**Count**: 4 (16.0% of sampled errors)

**Examples** (reference -> hypothesis, WER, cause):

```
Reference:   और नेक्स्ट टॉपिक है उत्साहित लेकिन नर्वस महसूस करना तो जब हम नए स्कूल में जाते है तो हम उत्साहित तो बहुत रहते है बहुत ज्यादा एक्साईटेड रहते है कि हम नए स्कूल में आए है बट आ.
Hypothesis:   और नेक्स टॉपिक है, ऊत साहींत मे कोनले कि नर्वस मेयसूस करना. तो जब हम नैस्कुल में चाहते हैं तौ हम। अथसाहित तو बहुत रहतے है बखुछ ज्यादे एक्शाइड़ेट रेते यहें कि हमनैस आई स्कोल मैं آए है ।बट।
WER:         0.7895
Duration:    12.4s
Cause:       Informal spoken form: ref='नेक्स्ट' vs hyp='नेक्स' — colloquial pronunciation or dialectal variant not well represented in training data.
```

```
Reference:   बताइए मेरे पास अगर टॉपिक मतलब कुछ होता ऐसा कहानी तो मैं बताती आपको पर आप अगर आपके पास है तो बताइए मैं सुन लूंगी
Hypothesis:   बताइये हैं मिर्स सहली कुछ तोपकेख मनलग कूछ होता है, ऐसा कहान तुम बदताती है आठको. तर आकर अगर एक आ थो ब टाईए में स्विन्दूंगी ौजा!
WER:         0.9600
Duration:    14.3s
Cause:       Informal spoken form: ref='बताइए' vs hyp='बताइये' — colloquial pronunciation or dialectal variant not well represented in training data.
```

```
Reference:   हा मैं यही कहना चाहूंगा कि मतलब प्लूटो और यह तो नहीं कैरम तो नहीं मतलब इसके अलावा जैसे अ।
Hypothesis:   हमें यही काना चाउंगा कि मतलब लूटो और ये तो नहीं कैर्यम थो रही है मतلब सके अलावा जैसे आााा एक प्रुछा इस देशा फिल्डिया खुल भी ऐसे हैं जो ठौर्टी घर्मारे ऑई जाए जी जेजा अपने जुर्रेंट जर्हा है ज़े धाला ओर ज्यार बारा थे
WER:         1.9500
Duration:    13.7s
Cause:       Informal spoken form: ref='हा' vs hyp='हमें' — colloquial pronunciation or dialectal variant not well represented in training data.
```

```
Reference:   बट सर एक्चुली इस किताब के पीछे है ना बहुत ज़्यादा,आ मतलब म कोई ऐसी कहानी है जिन्होने ने वो सोच समझ के लिखी है, सोचिये और अमीर बनि अमीर बनिये इस किताब का मतलब हिंदी में ये है
Hypothesis:   बट सर एक्सुली इस किताम के भी चैन हो लहां बहुत जयादा मतलब कोई यासी कहानी आगी जिनाने और सोफ समझके लिखिया सो जी है औड अमीर पनी हां, इس किتा को मतلب हिंदी मी थे है
WER:         0.7436
Duration:    13.1s
Cause:       Informal spoken form: ref='एक्चुली' vs hyp='एक्सुली' — colloquial pronunciation or dialectal variant not well represented in training data.
```

### Function Word Deletions

**Count**: 11 (44.0% of sampled errors)

**Examples** (reference -> hypothesis, WER, cause):

```
Reference:   तो उसका हम परिचय लेते हैं उनके बारे में हम जानते हैं हा ये सबजेक्ट कैसा है उसके बारे में पढ़ते है समझते हैं कि हाँ या फिर जब हम कॉलेज में एडमिशन ले रहे हैं
Hypothesis:   उसका हम परिचेल लेते हैं, उन्ही को बारे में हम सुझानते एं है कि हां ये सब्येक कैसा है उज्की बாरे भी पड़तें थे समछते कि रहा ही आ या फिर जब हम कॉलेज मे अइन्मिशन लے राई है
WER:         0.7222
Duration:    11.5s
Cause:       Grammatical particle 'तो' dropped at position 0. Function-word deletions indicate the model is under-trained on formal Hindi style or the particle is acoustically weak.
```

```
Reference:   हा तो अब इस बात की एक एक बात ओर में बताना चाहूंगा तो हमारा हमारे एक मित्र थे उनके पास एक पहले स्कूटी चला करती थीं स्कूल के टाइम पे आपने देखा होगा पैक करके हां
Hypothesis:   तो आप इस बात में ही एक बहात नुर्य मैं बतना चाल कि समझा थो हमारा, हम हमरे एग मित्र थे उनके पास ये फिले ट्स्कूडी चला घरती थी. इशकौर्वें टाई पे आ अपने देखा होगा पैभ्ठ करके हाँ
WER:         0.7568
Duration:    13.9s
Cause:       Grammatical particle 'हा' dropped at position 0. Function-word deletions indicate the model is under-trained on formal Hindi style or the particle is acoustically weak.
```

```
Reference:   हा ये क्या होता है कि स्कूल से आने के बाद कभी कभार हम लोग भी खेल लेते हैं और मतलब छुपण छुपाई भी खेलते थे पेटे पाई पीड़्यू खेलते थे अलग अलग तरीके का खेल खेलते थे
Hypothesis:   यह क्या होते हैं कि सकूलसे आने करबात कमी के भार हम लोग में चेलने थे और अ, मतलब शुप्छी फाईडी खेल ते ठे फैटी दु खालते रही घरीकी का ख्येल केलतے थो.
WER:         0.8158
Duration:    14.0s
Cause:       Grammatical particle 'ये' dropped at position 1. Function-word deletions indicate the model is under-trained on formal Hindi style or the particle is acoustically weak.
```

```
Reference:   हूं तो आप जंग फ्रूट्स ज्यादा खाते हैं। या मतलब कि नार्मल खाते हैं।
Hypothesis:   हम्म तो आप चंकिल दूरी जाते हैं मनही को सस्या नार्बल खाता है
WER:         0.8571
Duration:    5.0s
Cause:       Grammatical particle 'खाते' dropped at position 6. Function-word deletions indicate the model is under-trained on formal Hindi style or the particle is acoustically weak.
```

```
Reference:   अ बहुत ज्यादा भीड़ भी बहुत सी आती थी जो भी जाते हैं ना गंगा आरती लिए जरूरी ठहरते हैं जी
Hypothesis:   बहुत ज्यादा, भी-विरे मोस्ला करने हैं जो अभी जाता है ना गमका आर्टिक ले जिल्डिए साथा ही धेंगे. जे!
WER:         0.9048
Duration:    7.8s
Cause:       Grammatical particle 'अ' dropped at position 0. Function-word deletions indicate the model is under-trained on formal Hindi style or the particle is acoustically weak.
```

### Insertion of Noise / Fillers

**Count**: 3 (12.0% of sampled errors)

**Examples** (reference -> hypothesis, WER, cause):

```
Reference:   थोड़ा सा अलगअलग प्ले लिस्ट में है थोड़ा सा अलगअलग तरीका से किए है।
Hypothesis:   अला सह्ट हुरे कोलक मैंनी लिस्त में है तो थोब जो चाहा शम्या है, अलोग अ लोक से डिका सेती हैं हां
WER:         1.3571
Duration:    3.8s
Cause:       Spurious filler/noise token 'लिस्त' inserted. Model is transcribing background noise or disfluency that the human annotator silently ignored.
```

```
Reference:   जी जी बोलिए आप
Hypothesis:   जीजि मुल्या है आप हमें करते हो अनहीं सकता हो है, यहसे बोली आ आता.
WER:         3.5000
Duration:    1.3s
Cause:       Spurious filler/noise token 'हमें' inserted. Model is transcribing background noise or disfluency that the human annotator silently ignored.
```

```
Reference:   एक संसार में रहते हुए हर कार्य को इन्होंने काफी अच्छे से किया है और इनकी जो जीवन में अह जो समस्याएं आई तो ये कभी उस समस्याएं को खुद से
Hypothesis:   एक संसार में रहती हुई, हर कारि को इन्होने काफी अच्ये से किया है और इलमकी जीवन मे आ जो समस्सेाया आई तो ये कभी उस समس्स्याये को ूछुत से.
WER:         0.3871
Duration:    14.9s
Cause:       Spurious filler/noise token 'जीवन' inserted. Model is transcribing background noise or disfluency that the human annotator silently ignored.
```

### Other / Uncategorised

**Count**: 7 (28.0% of sampled errors)

**Examples** (reference -> hypothesis, WER, cause):

```
Reference:   अभी क्या कर रहे हैं आप रिसेंटली
Hypothesis:   अभी क्या कर रहे हैं आप इसलिकुन समत्यों
WER:         0.2857
Duration:    2.2s
Cause:       Primary mismatch is a substitution (ref='रिसेंटली', hyp='इसलिकुन'), suggesting local lexical confusion.
```

```
Reference:   तो उसमें और कुछ आगे नहीं रहा
Hypothesis:   तो, उसमें और कुछ हागी नहीं रहा.
WER:         0.2857
Duration:    3.8s
Cause:       Primary mismatch is a substitution (ref='तो', hyp='तो,'), suggesting local lexical confusion.
```

```
Reference:   तो ये चीज कि मै हूं
Hypothesis:   तो ये चीज कि मुर्ह हम्म
WER:         0.3333
Duration:    3.7s
Cause:       Primary mismatch is a substitution (ref='मै', hyp='मुर्ह'), suggesting local lexical confusion.
```

```
Reference:   लूडो कैरम भी खेला है
Hypothesis:   लूटो करम्यें भी खिला है.
WER:         0.6000
Duration:    1.3s
Cause:       Primary mismatch is a substitution (ref='लूडो', hyp='लूटो'), suggesting local lexical confusion.
```

```
Reference:   हाँ मोबाइल होते ही नहीं थे
Hypothesis:   हां, मॉबैल अते ही नहीं सिर्क.
WER:         0.6667
Duration:    1.5s
Cause:       Primary mismatch is a substitution (ref='हाँ', hyp='हां,'), suggesting local lexical confusion.
```

## Top Recommendations (Q1-f)

1. **Language-model Rescoring / Beam Search**

Use a Hindi n-gram or neural LM (e.g. IndicBERT) to rescore Whisper's beam candidates and promote hypotheses that retain grammatical particles. Alternatively increase `num_beams` from 1 to 4 during inference.

2. **Data-driven Augmentation**

Inspect `error_samples.jsonl` to identify emerging sub-patterns and add targeted training examples.

3. **Training Data Augmentation with Colloquial Hindi**

The JoshTalks corpus is conversational. Add more examples of casual / dialectal Hindi to the fine-tuning set, or normalise reference transcriptions to a consistent spoken-form style.

