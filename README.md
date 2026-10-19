# Dataset creation

## Speakers
| Code | Speaker | Recording format |
|---|---|---|
| spk01 | Justin | 550 individual WAV files (one per sentence) |
| spk02 | Venkatesh | 501 individual WAV files (one per sentence) |
| spk03 | Hari Prakash | one long WAV file (all sentences in a single take) |
| spk04 | Job | one long WAV file (all sentences in a single take) |

## Sentence prompts: `master_sentences.csv`
- Source: the **Harvard sentences** (IEEE "Recommended Practice for Speech Quality Measurements", 1969). These are phonetically balanced English sentences widely used in speech research.
- **550 sentences**, numbered `0001`–`0550`:
  - `0001`–`0500` (Harvard Lists 1–50): **training** sentences. `0001`–`0100` (Lists 1–10) form the **100-sentence subset**.
  - `0501`–`0550` (Harvard Lists 51–55): **test** sentences.
- Columns: `id`, `split`, `harvard_list`, `in_100_subset`, `filename`, `word_count`, `text`.
- Sentence length: 5–11 words (mean 7.9). The lists were used unfiltered, so some sentences exceed the 8-word guideline.

## Recording instructions given to all speakers
- Quiet room, laptop microphone, same setup for every session.
- Read through the sentences once before recording, to avoid hesitations.
- Leave about **1 s of silence** before and after each sentence.
- Speak naturally at a normal pace, and re-record any sentence with a stumble.
- `record.py` saves WAV, mono, 16-bit, 48 kHz. The GMM pipeline resamples all audio to 16 kHz.

## Tools

### `record.py`: one sentence at a time
Shows each sentence, records it from the microphone, and saves it as `spkXX/spkXX_NNNN.wav`. It warns about clipping or very quiet takes, allows playback and re-recording, and resumes where you left off.
```bash
pip install sounddevice soundfile numpy
python record.py spk01            # or run it and type the speaker code when asked
python record.py spk01 --start 0250
python record.py --list-devices
```

### `split_long.py`: split a single long recording
For speakers who recorded everything in one take. It finds the silent pauses between sentences (frame energy below a threshold relative to the speech level for at least `--min-silence` seconds) and saves each sentence as a separate file.
```bash
python split_long.py long_recording.wav spk03 [--min-silence 0.6] [--threshold 0.1]
```
*(The GMM experiments do not need this step: the GMM pipeline reads long files directly and splits them by duration.)*

## Data statistics (after silence removal, from the GMM pipeline)
| Speaker | Total audio | Speech |
|---|---|---|
| Justin | 35.7 min | 18.1 min |
| Venkatesh | 53.7 min | 18.4 min |
| Hari Prakash | 41.7 min | 25.6 min |
| Job | 49.3 min | 25.5 min |

Average speech per sentence: **2.09 s**, measured on the one-file-per-sentence speakers.
