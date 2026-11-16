# NoteTaker

Offline speech-to-text for the Raspberry Pi, built on faster-whisper.

It has two modes:

- **live**: listens to the microphone and appends each phrase to a notes file
- **file**: transcribes an audio or video file (mp3, wav, mp4, ...) to a text file

Everything runs locally. Internet is only needed once per model, to download it.

## Requirements

```
sudo apt install libportaudio2 ffmpeg
```

Python dependencies are managed by uv (faster-whisper, sounddevice, numpy).

## Setup

```
git clone <repo-url>
cd offline-notetaker
uv sync
```

The first run downloads the `base.en` model (about 145 MB) into
`~/.cache/huggingface`. Let it finish and do not press Ctrl+C. It may show
no progress bar. After that, no internet is needed.

## Usage

Note taker (microphone):

```
uv run notetaker.py live
uv run notetaker.py live --device 3 --out meeting.txt
```

Wait for "Listening...", speak, and pause for about a second. Each phrase
is printed and appended to `notes.txt`. Stop with Ctrl+C.

Transcribe a file:

```
uv run notetaker.py file lecture.mp3
```

The text is printed and saved next to the input file (`lecture.txt`).

Help:

```
uv run notetaker.py -h
```

## Choosing the microphone

List the devices Python can see:

```
uv run python -c "import sounddevice as sd; print(sd.query_devices())"
```

Pick the number of your microphone (it needs at least 1 input channel).
The default in the script is device 2 (Logitech USB). Use `--device N` to
change it for a single run, or edit the default in `notetaker.py`.

## Choosing a model

| Model | Download | Speed on Pi 5 | Accuracy |
|-------|----------|---------------|----------|
| `tiny.en` | about 75 MB | fastest | lowest |
| `base.en` (default) | about 145 MB | fast | good |
| `small.en` | about 480 MB | slower | noticeably better with names, accents and technical vocabulary (GAAP, tax terms) |

The `.en` models are English only. Each model is downloaded once and cached
in `~/.cache/huggingface`, so you can switch back and forth without
downloading again.

### Download `small.en` for testing

Run this once, with internet, and do not press Ctrl+C. It may show no
progress bar:

```
uv run python -c "from faster_whisper import WhisperModel; WhisperModel('small.en')"
```

To watch progress, open a second terminal and run:

```
watch -n 5 du -sh ~/.cache/huggingface/hub/models--Systran--faster-whisper-small.en
```

The download is finished when the first command returns to the prompt and
the folder is around 480 MB. If an interrupted attempt left partial files,
delete them and try again:

```
find ~/.cache/huggingface -name "*.incomplete" -delete
```

### Switch the script to `small.en`

Open `notetaker.py` and find `load_model()`:

```python
def load_model():
    print("Loading model...", flush=True)
    return WhisperModel("base.en", device="cpu", compute_type="int8")
```

Change `"base.en"` to `"small.en"`:

```python
    return WhisperModel("small.en", device="cpu", compute_type="int8")
```

Save the file and run it as usual. To go back, change it back to `"base.en"`.

### Compare speed and accuracy

Time the same recording with each model (`time` works in fish):

```
time uv run notetaker.py file test.wav
```

For live mode, the thing to check is the delay between the end of a phrase
and the text appearing. If `small.en` feels too slow for dictation, keep
`base.en` for live mode and use `small.en` only for file transcription.

## Tuning (constants at the top of notetaker.py)

- `THRESHOLD = 0.05`: volume level that counts as speech. Raise it if
  background noise is picked up, lower it if your voice is missed. To
  measure your levels, print the RMS of the input for a few seconds while
  talking and while quiet, then set the threshold between the two.
- `SILENCE_BLOCKS = 8`: 0.8 seconds of silence ends a phrase. Use 5 for
  shorter chunks.
- Model: see "Choosing a model" above.

## Troubleshooting

- **Nothing is transcribed in live mode**: the threshold is probably below
  the background noise, or the wrong device is selected.
- **Download looks stuck**: check progress in another terminal with
  `du -sh ~/.cache/huggingface/hub/models--Systran--faster-whisper-base.en`
  (or the `small.en` folder). Delete leftovers from an interrupted download
  with `find ~/.cache/huggingface -name "*.incomplete" -delete`, then try
  again.
- **`metadata_errors` error from PyAV**: that is why file mode decodes with
  ffmpeg directly instead of letting faster-whisper open the file.
- **Float16 warning at startup**: harmless, the model is converted
  automatically.
- **Long files take a while**: the whole file is decoded into memory first,
  then transcribed. Expect a good part of the audio's length on the Pi,
  more with `small.en`.
