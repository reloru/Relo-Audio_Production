General-purpose audio production and analysis workspace — Claude Code cloud environment `Audio Production`.

## Scripts

All scripts are plain Python 3 and need `ffmpeg`/`ffprobe` on `PATH` plus `numpy`, `scipy`, `soundfile`, `numba` and `mutagen` (all preinstalled in the `Audio Production` environment).

### `scripts/master.py` — mastering chain

```
python3 scripts/master.py INPUT [--preset presets/NAME.json] [--target-lufs -10] [--ceiling-dbtp -1] \
    [--out-dir output] [--name NAME] [--title TITLE] [--no-mp3] [--codec-check]
```

Takes any file ffmpeg can decode (MP3, WAV, FLAC, …) at its native sample rate and writes:

- `NAME.wav` — 24-bit PCM with TPDF dither, title/artist tags
- `NAME.mp3` — 320 kbps CBR, ID3 tags (lyrics, cover art, comments, source URL) copied from the source when it is an MP3, title replaced by `--title`. Encoder/length frames and C2PA manifests are not copied: a C2PA manifest's hash binding covers the original audio bytes and would fail validation on the re-encoded file.
- `NAME_report.json` — settings, resolved thresholds, before/after loudness, ffmpeg cross-check, per-stage gain-reduction statistics

Processing order:

1. Subsonic high-pass (20 Hz, 4th-order Butterworth)
2. Static EQ from the preset (RBJ peak / shelf biquads)
3. Low-band compressor below 85 Hz on a zero-phase complementary split (inactive bands reconstruct the input exactly)
4. Dynamic cut on 5–11 kHz (zero-phase band; only acts above its threshold — sibilance and hi-hat spikes)
5. Glue compressor (broadband, stereo-linked, 100 Hz sidechain high-pass)
6. Gain search to the integrated-loudness target, optional 4x-oversampled soft clipper, then a lookahead true-peak limiter (4x-oversampled detection, hold, exponential release)
7. Raised-cosine fade-in/out (default 5 ms / 50 ms anti-click; a preset can lengthen the fade-out to tidy a cut-off tail)

Compressor thresholds are percentiles of each detector's level over the non-silent part of the track, so presets transfer between songs; the dBFS values actually used are in the report. Everything not set in a preset comes from `DEFAULTS` in `master.py`.

`--codec-check` encodes the master to AAC 256k, Vorbis 160k, Opus 128k and MP3 320k, decodes each, and reports the decoded true peak (ffmpeg, 192 kHz) so the ceiling can be confirmed for streaming transcodes.

Outputs go to `output/` by default, which is git-ignored: rendered audio is delivered, not committed.

### `scripts/measure.py` — loudness meter

```
python3 scripts/measure.py FILE [FILE ...]
```

ITU-R BS.1770-4 integrated loudness, EBU R128 loudness range, max momentary/short-term loudness, 4x-oversampled true peak, sample peak and PLR, printed next to ffmpeg's `ebur128` results as an independent check.

### `scripts/selftest.py` — checks

```
python3 scripts/selftest.py
```

Synthetic-signal checks: EBU Tech 3341 sine levels, agreement with ffmpeg on gated noise, true-peak accuracy, EQ gains, exact split reconstruction, gain-computer curve, limiter ceiling guarantee, clipper transparency, 24-bit write round trip.

## Presets

`presets/*.json` hold per-track overrides of `DEFAULTS` (any subset of keys). `presets/proportion.json` is the setting used for "Proportion".
