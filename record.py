"""
Recording script for the speech assignment.

Shows one sentence at a time from master_sentences.csv, records it from the
microphone and saves it as <speaker>/<speaker>_<id>.wav (e.g. spk01/spk01_0001.wav).

Setup (once):
    pip install sounddevice soundfile numpy

Run:
    python record.py spk01
    python record.py spk01 --start 0250     # jump to a specific sentence
    python record.py spk01 --list-devices   # see available microphones
    python record.py spk01 --device 2       # use a specific microphone

It resumes automatically: sentences that already have a WAV file are skipped,
so you can stop (q) and continue later.
"""
import argparse
import csv
import os
import sys
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 48000      # record high quality; downsample later per task
CHANNELS = 1             # mono
SUBTYPE = "PCM_16"       # 16-bit WAV
CSV_PATH = "master_sentences.csv"


def load_sentences(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def record_until_enter(device):
    """Record from the mic until the user presses Enter. Returns a numpy array."""
    chunks = []
    stop = threading.Event()

    def callback(indata, frames, time, status):
        if status:
            print(f"  [audio warning: {status}]", file=sys.stderr)
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", device=device, callback=callback):
        input()  # blocks until Enter
        stop.set()

    if not chunks:
        return np.zeros((0, CHANNELS), dtype="float32")
    return np.concatenate(chunks, axis=0)


def check_audio(audio):
    """Return a list of warnings about the recording."""
    warnings = []
    seconds = len(audio) / SAMPLE_RATE
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if seconds < 1.0:
        warnings.append(f"very short ({seconds:.1f}s)")
    if peak >= 0.99:
        warnings.append("CLIPPING - too loud, move back from the mic")
    elif peak < 0.05:
        warnings.append("very quiet - move closer or raise mic volume")
    return warnings, seconds, peak


def main():
    p = argparse.ArgumentParser(description="Record sentences one by one.")
    p.add_argument("speaker", nargs="?", help="speaker code, e.g. spk01")
    p.add_argument("--start", help="sentence id to start from, e.g. 0250")
    p.add_argument("--device", type=int, default=None, help="input device index")
    p.add_argument("--csv", default=CSV_PATH, help="path to master_sentences.csv")
    p.add_argument("--list-devices", action="store_true", help="list microphones and exit")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return
    if not args.speaker:
        args.speaker = input("Enter your speaker code (e.g. spk01): ").strip()
        if not args.speaker:
            sys.exit("No speaker code given.")

    spk = args.speaker.strip().lower()
    rows = load_sentences(args.csv)
    os.makedirs(spk, exist_ok=True)

    start_idx = 0
    if args.start:
        ids = [r["id"] for r in rows]
        if args.start not in ids:
            sys.exit(f"Sentence id {args.start} not found in {args.csv}")
        start_idx = ids.index(args.start)

    total = len(rows)
    done = sum(os.path.exists(os.path.join(spk, f"{spk}_{r['id']}.wav")) for r in rows)
    print(f"\nSpeaker {spk}: {done}/{total} sentences already recorded.")
    print("Controls: Enter = start recording, Enter again = stop.")
    print("After each take: Enter = keep & next | r = redo | p = play back | q = quit\n")

    i = start_idx
    while i < total:
        row = rows[i]
        out_path = os.path.join(spk, f"{spk}_{row['id']}.wav")

        if os.path.exists(out_path) and not args.start:
            i += 1
            continue
        args.start = None  # only force the first one when jumping

        print("=" * 70)
        print(f"[{row['id']}]  ({row['split']})   {i + 1}/{total}")
        print(f"\n    {row['text']}\n")
        cmd = input("Press Enter to START (s = skip, q = quit): ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            i += 1
            continue

        print("  ● Recording... press Enter to STOP")
        audio = record_until_enter(args.device)
        warnings, seconds, peak = check_audio(audio)
        print(f"  Recorded {seconds:.1f}s, peak level {peak:.2f}")
        for w in warnings:
            print(f"  ⚠ {w}")

        while True:
            choice = input("  Enter = keep | r = redo | p = play | q = quit: ").strip().lower()
            if choice == "p":
                sd.play(audio, SAMPLE_RATE)
                sd.wait()
                continue
            break

        if choice == "r":
            continue  # same sentence again
        if choice == "q":
            print("Not saved. Quitting.")
            break

        sf.write(out_path, audio, SAMPLE_RATE, subtype=SUBTYPE)
        print(f"  ✓ saved {out_path}")
        i += 1

    done = sum(os.path.exists(os.path.join(spk, f"{spk}_{r['id']}.wav")) for r in rows)
    print(f"\n{spk}: {done}/{total} recorded.")
    if done == total:
        print("All done! Zip the folder and upload it.")


if __name__ == "__main__":
    main()
