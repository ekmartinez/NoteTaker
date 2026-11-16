import argparse
import queue
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

RATE = 16000
BLOCK = int(RATE * 0.1)
THRESHOLD = 0.05
SILENCE_BLOCKS = 8


def load_model():
    print("Loading model...", flush=True)
    return WhisperModel("base.en", device="cpu", compute_type="int8")


def live(device, out):
    model = load_model()
    q = queue.Queue()

    def callback(indata, frames, t, status):
        if status:
            print("Audio status:", status, flush=True)
        q.put(indata[:, 0].copy())

    preroll = deque(maxlen=3)
    buf, speaking, silent = [], False, 0

    with sd.InputStream(device=device, samplerate=RATE, channels=1,
                        dtype="float32", blocksize=BLOCK, callback=callback):
        print(f"Listening... notes go to {out}. Press Ctrl+C to stop", flush=True)
        try:
            while True:
                block = q.get()
                loud = float(np.sqrt(np.mean(block ** 2))) > THRESHOLD

                if not speaking:
                    preroll.append(block)
                    if loud:
                        speaking, silent = True, 0
                        buf = list(preroll)
                    continue

                buf.append(block)
                silent = 0 if loud else silent + 1

                if silent >= SILENCE_BLOCKS:
                    audio = np.concatenate(buf)
                    segments, _ = model.transcribe(audio, language="en")
                    text = " ".join(s.text.strip() for s in segments).strip()
                    if text:
                        print(text, flush=True)
                        with open(out, "a") as f:
                            f.write(text + "\n")
                    buf, speaking, silent = [], False, 0
                    preroll.clear()
        except KeyboardInterrupt:
            print("\nStopped.")


def transcribe_file(path):
    src = Path(path)
    if not src.exists():
        sys.exit(f"File not found: {src}")
    model = load_model()
    print(f"Transcribing {src} ...", flush=True)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src),
         "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio, language="en")
    out = src.with_suffix(".txt")
    with open(out, "w") as f:
        for s in segments:
            text = s.text.strip()
            print(text, flush=True)
            f.write(text + "\n")
    print(f"Saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Offline speech to text")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_live = sub.add_parser("live", help="note taker: listen to the microphone")
    p_live.add_argument("--device", type=int, default=2,
                        help="microphone device number (default 2)")
    p_live.add_argument("--out", default="notes.txt",
                        help="notes file (default notes.txt)")

    p_file = sub.add_parser("file", help="transcribe an audio or video file")
    p_file.add_argument("path", help="mp3, wav, mp4, etc.")

    args = parser.parse_args()
    if args.mode == "live":
        live(args.device, args.out)
    else:
        transcribe_file(args.path)


if __name__ == "__main__":
    main()
