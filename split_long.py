"""
Automatically split one long recording into one WAV file per sentence,
by cutting at the silent pauses between sentences.

Setup (once):
    pip install numpy soundfile

Run:
    python split_long.py long_recording.wav spk03
    -> creates folder spk03/ with spk03_0001.wav, spk03_0002.wav, ...

If the number of pieces is far from the number of sentences read, adjust:
    --min-silence 0.6   (seconds of quiet that counts as a gap; raise it if
                         sentences are being cut in the middle, lower it if
                         sentences are being merged together)
    --threshold 0.1     (how quiet "silence" is, relative to the speech level;
                         raise it in a noisy room)
"""
import argparse
import os

import numpy as np
import soundfile as sf


def find_segments(y, sr, threshold, min_silence, min_speech, frame_ms=20):
    frame = int(sr * frame_ms / 1000)
    n = len(y) // frame
    rms = np.sqrt(np.mean(y[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)

    # speech level = loud frames; silence = below threshold * speech level
    speech_level = np.percentile(rms, 95)
    voiced = rms > threshold * speech_level

    min_sil_frames = int(min_silence * 1000 / frame_ms)
    min_sp_frames = int(min_speech * 1000 / frame_ms)

    segments, start, quiet = [], None, 0
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= min_sil_frames:
                end = i - quiet + 1
                if end - start >= min_sp_frames:
                    segments.append((start, end))
                start, quiet = None, 0
    if start is not None and n - start >= min_sp_frames:
        segments.append((start, n))

    return [(s * frame, e * frame) for s, e in segments]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav", help="the long recording")
    p.add_argument("speaker", help="speaker code, e.g. spk03")
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--min-silence", type=float, default=0.6)
    p.add_argument("--min-speech", type=float, default=0.8,
                   help="ignore sounds shorter than this (coughs, clicks)")
    p.add_argument("--pad", type=float, default=0.2,
                   help="seconds of silence kept before/after each sentence")
    args = p.parse_args()

    y, sr = sf.read(args.wav, always_2d=True)
    y = y.mean(axis=1)  # to mono
    print(f"Loaded {args.wav}: {len(y) / sr / 60:.1f} min at {sr} Hz")

    segs = find_segments(y, sr, args.threshold, args.min_silence, args.min_speech)
    pad = int(args.pad * sr)

    os.makedirs(args.speaker, exist_ok=True)
    lengths = []
    for k, (s, e) in enumerate(segs, 1):
        s, e = max(0, s - pad), min(len(y), e + pad)
        sf.write(os.path.join(args.speaker, f"{args.speaker}_{k:04d}.wav"),
                 y[s:e], sr, subtype="PCM_16")
        lengths.append((e - s) / sr)

    lengths = np.array(lengths)
    print(f"Saved {len(segs)} pieces to {args.speaker}/")
    if len(segs):
        print(f"Piece length: min {lengths.min():.1f}s, "
              f"median {np.median(lengths):.1f}s, max {lengths.max():.1f}s")
        long_ones = np.where(lengths > 2.5 * np.median(lengths))[0] + 1
        if len(long_ones):
            print(f"Possibly merged sentences (unusually long): {list(long_ones[:10])}"
                  f"{' ...' if len(long_ones) > 10 else ''}")


if __name__ == "__main__":
    main()
