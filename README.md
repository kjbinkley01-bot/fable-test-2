# StemDeck

A stem-based remix and DJ tool for Windows. **Milestone 1: the engine core**, with a simple
window to test it.

Drop a song on the window and StemDeck will:

1. split it into four stems (vocals, drums, bass, other) on your AMD GPU,
2. detect BPM, musical key (standard and Camelot, e.g. `A minor / 8A`) and the beat grid
   with downbeats,
3. show four stacked, colour-coded waveforms with the grid on top,
4. let you play, pause and seek, with a volume fader, mute and solo per stem,
5. remember everything, so the same song opens almost instantly next time,
6. export any stem, or a custom mix of stems, as WAV.

---

## Setup (once)

You need Windows 11 and an up-to-date AMD Adrenalin driver. Do the steps in order.

**1. Update the graphics driver.** Install the latest *AMD Software: Adrenalin Edition*
(26.10 or newer) from amd.com. AMD's GPU computing layer (ROCm) is delivered partly through
the driver, so an old driver is the most common reason the GPU is not found.

**2. Install Python 3.12.** Open *Terminal* (right-click the Start button) and run:

```bat
winget install -e --id Python.Python.3.12
```

Close and reopen the terminal afterwards.

**3. Get the code** and go into its folder. If `git` is not installed yet, run
`winget install -e --id Git.Git` first and reopen the terminal.

```bat
git clone https://github.com/kjbinkley01-bot/fable-test-2.git StemDeck
```

```bat
cd StemDeck
```

```bat
git checkout claude/eager-cray-9rj8qi
```

**4. Run setup.** This creates a private Python environment in the `.venv` folder (nothing is
installed system-wide), installs AMD's GPU build of PyTorch (a download of several GB),
installs the other packages, downloads the AI models (about 500 MB) and puts a StemDeck
shortcut on your Desktop.

```bat
setup.bat
```

At the end it prints what separation will run on. You want to see a line like
`GPU: AMD Radeon RX 9070 XT - 16.0 GB VRAM - self-test passed`.

**5. Run the benchmark** (see below) to confirm the GPU is fast enough:

```bat
benchmark.bat
```

**6. Start the app** with the **StemDeck** shortcut on your Desktop, or by double-clicking
`StemDeck.bat`. You can also drag a song onto `StemDeck.bat`.

### If the GPU is not detected

`setup.bat` and `benchmark.bat` print the reason. In order of likelihood:

| What it says | What to do |
|---|---|
| "PyTorch ... is CPU-only" | The AMD PyTorch install did not happen. Run `setup.bat` again and send me the output. |
| "has ROCm support but found no usable GPU" | Update the Adrenalin driver, reboot, run `setup.bat` again. |
| "GPU self-test ..." failed | The GPU was found but gives wrong results. Send me the output; do not ignore it. |
| A DLL is missing | Install the *Visual C++ 2015-2022 Redistributable (x64)* from Microsoft. |

AMD's documentation also says *Smart App Control* and *Windows Defender Application Guard*
"can interfere" with ROCm. **Do not turn Smart App Control off pre-emptively**: Windows does
not let you turn it back on without reinstalling. Only consider it if the steps above fail,
and tell me first.

The app still works without the GPU. It separates on the CPU instead and says so in a
yellow banner; it never falls back silently.

---

## Benchmark

```bat
benchmark.bat
```

separates a generated four-minute test signal. To use a real song, drag it onto
`benchmark.bat`, or:

```bat
benchmark.bat "C:\Music\some song.mp3"
```

It reports:

- **which device ran the model**, GPU or CPU, and the reason if it was not the GPU;
- a **cold** run and a **warm** run. The first run after installing is slower because AMD's
  libraries compile their GPU routines the first time they are used, then keep them;
- peak GPU memory and speed as a multiple of real time;
- a **GPU-versus-CPU check** on a short excerpt, to prove the GPU produces the same audio;
- **PASS / FAIL** against the target: a four-minute song in under 60 seconds on the GPU.

If it works but is slower than the target, let it try several settings and keep the fastest:

```bat
benchmark.bat "C:\Music\some song.mp3" --sweep
```

and to see how the faster model does:

```bat
benchmark.bat --model htdemucs
```

**Please send me the full output of the first benchmark run.** I could not run the GPU path
myself (see *What was and was not tested*).

---

## Checking BPM and key against Beatport / Mixed In Key

1. Pick five songs from different genres and look up their BPM and key on Beatport or in
   Mixed In Key.
2. Copy `test_songs.example.csv` to `test_songs.csv` and fill in the path, BPM and key of
   each song. The key can be written the way the tool shows it: `A minor`, `Am`, `8A`,
   `F# maj` all work.
3. Run:

```bat
check_analysis.bat test_songs.csv
```

For each song it prints StemDeck's values next to the reference and, at the end, how many
matched and whether the "at least 4 of 5" target is met. When a key does not match it says
how it differs (for example "relative major", which means the same notes with a different
home note), because that is where key detectors usually disagree with each other.

---

## Using the app

| Action | How |
|---|---|
| Load a song | Drag a file onto the window, or **Open…** (`Ctrl+O`) |
| Play / pause | **Play** button or `Space` |
| Seek | Click or drag on the waveform; `Left` / `Right` jump one bar; `Home` goes to the start |
| Zoom | Mouse wheel over the waveform, or **–** / **+** / **Fit** |
| Scroll when zoomed | `Shift` + mouse wheel, or the scroll bar |
| Stem volume | Fader on each strip; double-click resets to 0 dB |
| Mute / solo | **M** / **S** on each strip, or `1`-`4` (mute) and `Shift+1`-`4` (solo) |
| Export | **Export…** (`Ctrl+E`): one mixed file, or a separate file per stem |
| Half / double tempo | **÷2** / **×2** next to the BPM; the choice is remembered for that song |
| Model | **Best quality** (default) or **Fast**, top right; applies to songs loaded afterwards |

Thin lines on the waveform are beats, brighter lines are downbeats (bar starts), and the
numbers along the top are bar numbers. Hover over BPM or KEY for details: whether the tempo
is fixed, how confident the key detection is and what the runner-up keys were.

**Export** offers the stems you can currently hear by default. "Use the current fader
levels" makes the file match what you hear; turn it off for stems at their original level.
If a mix would clip in a 16- or 24-bit file it is turned down just enough to avoid that, and
the status line tells you by how much.

---

## How it works, in plain language

### Stem separation and the GPU

The separation model is **HTDemucs (fine-tuned)** from Meta: four neural networks, one
specialised for each stem. It is the cleanest of the Demucs family, which matters for bleed
such as hi-hats leaking into vocals. **Fast** uses a single network that does all four stems
at once, about four times quicker with slightly more bleed.

The model runs on **PyTorch with AMD's ROCm**, AMD's equivalent of NVIDIA's CUDA, which now
runs natively on Windows and officially supports the RX 9070 XT. I chose it over the
alternatives because it runs the original, unmodified model code, so quality is exactly what
the authors published:

| Option | Why not |
|---|---|
| ONNX Runtime + DirectML | Works on any DirectX 12 card, but Microsoft has put DirectML in maintenance mode and it needs a community-converted copy of the model. **This is the planned backup** if ROCm disappoints on your card. |
| ONNX Runtime + MIGraphX | AMD's newer route, but no ready-made Windows packages yet. |
| CPU | Always works; used automatically if the GPU fails. Minutes per song instead of seconds. |

A song is cut into overlapping windows of about eight seconds; the network processes them
and the results are cross-faded. Reference Demucs feeds the windows one at a time, which
leaves a modern GPU mostly idle, so StemDeck processes several at once. The result is the
same audio (a test checks it matches reference Demucs to within rounding error), just faster.

Three safety nets guard the GPU path: a self-test at start-up that compares small GPU
calculations with the CPU, a check that the four stems add back up to the original song, and
a rejection of any output containing invalid numbers. If any fails, the song is redone on
the CPU and the app tells you why.

### BPM and beat grid

A neural network (**Beat This!**, 2024) marks where it hears beats and downbeats. StemDeck
then checks whether those beats line up with an evenly spaced ruler. Most dance music was
made against a click at one fixed tempo, so if a ruler fits, StemDeck uses it: one BPM,
accurate to a hundredth, and a grid that cannot wobble. If no ruler fits (a live drummer, a
tempo change) the grid follows the detected beats and the BPM shown is the typical value.

### Key

StemDeck measures how much energy sits on each note across six octaves, correcting for
tracks that are not tuned to standard pitch, and a small neural network trained on dance
music with human-checked key labels scores all 24 keys. A classic music-theory key profile
is blended in at a small weight: the network learned from dance music and is the more
accurate of the two there, while the profile knows textbook harmony, which dance music rarely
uses. Together they scored the same on dance music and fixed plain chord progressions the
network alone got wrong.

In the plan I said key detection would run on the song with the drums removed. I measured
that on 40 held-out tracks and it made no difference (77.5% correct either way), so the
detector listens to the full mix, which is also what it was trained on.

### Playback

The audio runs in **its own process**, separate from the window. A busy or even frozen
window therefore cannot make the sound stutter. Stems sit fully in memory, shared between the
two processes. Every volume change, mute, solo, play, pause and seek is applied as a
10-millisecond glide or cross-fade rather than an instant jump, because instant jumps are
what you hear as clicks.

### Cache

A song is recognised by a fingerprint of the file's contents, not its name, so renaming or
moving a file still finds the cached result. Stems are stored losslessly (24-bit FLAC, roughly
150 MB for a four-minute song) under `%LOCALAPPDATA%\StemDeck\library`. Delete that folder to
free the space; songs are simply separated again next time.

---

## What was and was not tested

I built this on a Linux machine with **no AMD GPU, no sound card and no Windows**, so:

**Tested here**

- Separation matches reference Demucs numerically; CPU fallback; rejection of bad output.
- BPM and key accuracy against public labelled datasets (numbers below).
- The mixer, with thousands of random fader moves, mutes, solos, seeks and pauses at several
  buffer sizes, checking the output never jumps (a jump is a click).
- The audio engine process end to end, including through the real audio library against a
  dummy output device.
- The cache: lossless round trip, instant reload, survives renames and damaged entries.
- Export never clips and matches what the mixer plays.
- The window, rendered off-screen and inspected in screenshots.

**Only you can test**

- GPU detection and speed on the RX 9070 XT (run `benchmark.bat`).
- Real audio output through Windows (WASAPI), and that it is glitch-free by ear.
- Drag and drop from Explorer, the Desktop shortcut and the `.bat` files themselves.
- Stem quality by ear on your own music.

### Measured results

**Key.** Scored on 566 dance tracks the model never saw in training (the public GiantSteps+
set; labels originate from Beatport and were corrected by musicians):

| | Exact key | Weighted score* |
|---|---|---|
| StemDeck | **72%** | **79** |
| Classic template matching on the same features | 61% | 70 |

\* The standard research score, which gives part credit for near misses.

Of the misses, roughly a quarter each are the relative key (same notes, e.g. A minor vs
C major), the parallel key (A minor vs A major) and a neighbour on the Camelot wheel; the
remaining quarter are unrelated keys. The confidence shown in the app is meaningful: when it is above 80% the key was right
89% of the time; below 40%, only 43%.

Two cautions. This test set is deliberately hard, and it is all electronic dance music. For
other genres I have no labelled data, so accuracy there is unmeasured; a music-theory
safeguard is blended in (see *How it works*) but treat pop, rock or hip-hop results with more
suspicion and use `check_analysis.bat` to find out. "At least 4 of 5" is realistic, not
guaranteed.

**BPM.** Scored on 661 annotated dance tracks (GiantSteps tempo set):

| | |
|---|---|
| Within 4% of the annotated BPM | **89%** |
| Right tempo, possibly counted at half or double time | **97%** |
| Same, on the 85% of tracks where StemDeck commits to a fixed-tempo grid | **99.6%** |

The gap between the two rows is the half-time/double-time question (is drum & bass 87 or
174?), which is a convention rather than a fact; annotators disagree with each other on it.
StemDeck doubles anything it reads below 78 BPM, flags readings of 78-100 that look like
half-time, and gives you **÷2** / **×2** buttons. On tracks where StemDeck and the
annotation disagreed by 1-2 BPM, an independent method confirmed StemDeck's value in all seven
cases I checked (the annotations were tapped by hand).

**Playback engine.** With the engine under a stream of random fader moves, mutes and seeks
while the controlling program hogged its own CPU, the slowest audio block took 0.7 ms
against a budget of 5.8 ms, with no buffer underruns. Mixing one block takes about 0.04 ms.

**Cache.** A four-minute song takes about 150 MB on disk and reloads in under a second.

**Separation.** Output matches reference Demucs to within rounding error, so stem quality
is exactly that of the published model. Speed on this build machine's 4-core CPU: about
2x real time for the Fast model. GPU speed is the open question the benchmark answers.



---

## Project layout

```
stemdeck/
  config.py            sample rate, stem names and colours, data folders
  audio_io.py          read MP3/WAV/FLAC, write WAV
  settings.py          saved settings (the benchmark stores the fastest GPU setup here)
  library.py           the cache: load a song once, reuse the results
  separation/          GPU detection + self-test, Demucs runner, CPU fallback
  analysis/            beats.py (BPM, grid), key.py + keyfeatures.py, camelot.py, waveform.py
  engine/              mixer.py (the click-free maths), process.py (audio process),
                       client.py (what the window talks to), backends.py (sound card), export.py
  ui/                  the window: waveform view, mixer strips, export dialog
  bench.py, cli.py     benchmark and command line
scripts/               benchmark.py, check_analysis.py
tools/                 how the key model was trained and scored (not needed to run the app)
tests/                 automated tests
```

### Ready for milestone 2

- **Second deck.** The mixer already holds a list of decks and sums them; every engine
  command carries a deck number; the status block has room for two. The window only drives
  deck 0 today.
- **Tempo sync.** A fixed-tempo grid is stored as "first downbeat + BPM", which is exactly
  what sync needs: beat N is at `anchor + N * 60 / bpm`.
- **Tempo change and key shift.** A deck fetches audio through one function
  (`Deck._read` in `engine/mixer.py`). A time-stretch / pitch-shift stage slots in there,
  between "fetch stems at the playhead" and "apply faders", without touching the fader,
  transport or seek logic.
- **Other separation engines** (for example DirectML) plug in next to the Demucs runner in
  `separation/` behind the same two methods.

## Command line

```bat
.venv\Scripts\python -m stemdeck devices
```

```bat
.venv\Scripts\python -m stemdeck separate "song.mp3" -o stems_folder
```

```bat
.venv\Scripts\python -m stemdeck analyze "song.mp3"
```

## Running the tests

```bat
.venv\Scripts\python -m pip install pytest
```

```bat
.venv\Scripts\python -m pytest tests -q
```
