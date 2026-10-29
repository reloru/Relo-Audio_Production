#!/usr/bin/env python3
"""Stereo mastering chain for finished mixes (built for Suno renders).

Chain, in order:
  1. Subsonic high-pass (Butterworth, minimum phase)
  2. Static EQ (RBJ biquads: peak / lowshelf / highshelf)
  3. Low-band compressor (zero-phase complementary split; tames sub swells
     before they drive the limiter)
  4. High-band dynamic cut (zero-phase band split; tames sibilance and
     hi-hat spikes, inactive below threshold)
  5. Glue compressor (broadband, stereo-linked, high-passed sidechain)
  6. Gain to hit the integrated-loudness target, then a lookahead
     true-peak limiter (4x oversampled detection, hold, exponential release)
  7. TPDF dither to 24-bit PCM WAV, optional 320 kbps MP3 with source tags

Compressor thresholds are set as percentiles of each detector's level over
the active (non-silent) part of the track, so a preset carries over between
songs; the resolved dBFS values are written to the report.

Usage:
    python3 scripts/master.py INPUT [--preset presets/x.json]
        [--target-lufs -10] [--ceiling-dbtp -1] [--out-dir output]
        [--name NAME] [--title TITLE] [--no-mp3] [--codec-check]
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numba
import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "target_lufs": -10.0,
    "ceiling_dbtp": -1.0,
    "highpass": {"enabled": True, "freq": 20.0, "order": 4},
    "eq": [],
    "low_band": {"enabled": True, "crossover": 85.0, "rms_ms": 30.0, "attack_ms": 40.0,
                 "release_ms": 300.0, "ratio": 2.0, "knee_db": 6.0,
                 "threshold_pct": 75.0, "threshold_offset_db": 0.0, "max_gr_db": 6.0},
    "high_band": {"enabled": True, "low": 5000.0, "high": 11000.0, "rms_ms": 5.0,
                  "attack_ms": 1.0, "release_ms": 60.0, "ratio": 3.0, "knee_db": 4.0,
                  "threshold_pct": 98.0, "threshold_offset_db": 0.0, "max_gr_db": 4.0},
    "glue": {"enabled": True, "sidechain_hp": 100.0, "rms_ms": 10.0, "attack_ms": 30.0,
             "release_ms": 200.0, "ratio": 1.5, "knee_db": 6.0,
             "threshold_pct": 90.0, "threshold_offset_db": 0.0, "max_gr_db": 6.0},
    "clipper": {"enabled": False, "ceiling_offset_db": 1.0, "knee_db": 1.0},
    "limiter": {"lookahead_ms": 2.0, "hold_ms": 8.0, "release_ms": 40.0},
    "fade": {"in_ms": 5.0, "out_ms": 50.0},
    "report_block_s": 30.0,
}


def merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def coef(ms, sr):
    return float(np.exp(-1000.0 / (ms * sr)))


# ---------------------------------------------------------------- filters

def biquad(kind, freq, gain_db, q, sr):
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    cw, alpha = np.cos(w0), np.sin(w0) / (2 * q)
    if kind == "peak":
        b = [1 + alpha * a, -2 * cw, 1 - alpha * a]
        den = [1 + alpha / a, -2 * cw, 1 - alpha / a]
    elif kind == "lowshelf":
        sa = 2 * np.sqrt(a) * alpha
        b = [a * ((a + 1) - (a - 1) * cw + sa), 2 * a * ((a - 1) - (a + 1) * cw),
             a * ((a + 1) - (a - 1) * cw - sa)]
        den = [(a + 1) + (a - 1) * cw + sa, -2 * ((a - 1) + (a + 1) * cw),
               (a + 1) + (a - 1) * cw - sa]
    elif kind == "highshelf":
        sa = 2 * np.sqrt(a) * alpha
        b = [a * ((a + 1) + (a - 1) * cw + sa), -2 * a * ((a - 1) + (a + 1) * cw),
             a * ((a + 1) + (a - 1) * cw - sa)]
        den = [(a + 1) - (a - 1) * cw + sa, 2 * ((a - 1) - (a + 1) * cw),
               (a + 1) - (a - 1) * cw - sa]
    else:
        raise ValueError(f"unknown EQ type {kind!r}")
    b, den = np.array(b) / den[0], np.array(den) / den[0]
    return np.r_[b, den][None, :]


def eq_sos(bands, sr):
    if not bands:
        return None
    return np.vstack([biquad(b["type"], b["freq"], b["gain_db"], b.get("q", 0.707), sr)
                      for b in bands])


def split_low(x, freq, sr):
    """Zero-phase complementary split: low has |LR4| magnitude, low + high == x exactly."""
    low = signal.sosfiltfilt(signal.butter(2, freq, "low", fs=sr, output="sos"), x, axis=0)
    return low, x - low


def band_zero_phase(x, lo, hi, sr):
    return signal.sosfiltfilt(signal.butter(2, [lo, hi], "band", fs=sr, output="sos"), x, axis=0)


# ---------------------------------------------------------------- dynamics

@numba.njit(cache=True)
def _smooth_gr(gr, a_att, a_rel):
    out = np.empty_like(gr)
    y = 0.0
    for n in range(gr.shape[0]):
        g = gr[n]
        if g > y:
            y = a_att * y + (1.0 - a_att) * g
        else:
            y = a_rel * y + (1.0 - a_rel) * g
        out[n] = y
    return out


def detector_db(sc, rms_ms, sr):
    """Stereo-linked mean-square level (one-pole, rms_ms time constant) in dB."""
    a = coef(rms_ms, sr)
    p = (sc ** 2).mean(axis=1)
    ms = signal.lfilter([1 - a], [1, -a], p)
    return 10 * np.log10(np.maximum(ms, 1e-20))


def static_gr(level, thr, ratio, knee):
    """Soft-knee gain computer (Giannoulis, Massberg & Reiss, JAES 2012); returns GR dB >= 0."""
    over = level - thr
    gr = np.zeros_like(level)
    hard = 2 * over > knee
    gr[hard] = over[hard] * (1 - 1 / ratio)
    if knee > 0:
        soft = np.abs(2 * over) <= knee
        gr[soft] = (1 - 1 / ratio) * (over[soft] + knee / 2) ** 2 / (2 * knee)
    return gr


def active_mask(x, sr, frame_s=0.05, floor_db=30.0):
    """Per-sample mask of frames within floor_db of the track's loud (p90) frame level."""
    n = int(frame_s * sr)
    nb = len(x) // n
    rms = 10 * np.log10(np.maximum((x[: nb * n] ** 2).mean(axis=1).reshape(nb, n).mean(axis=1), 1e-20))
    keep = rms > np.percentile(rms, 90) - floor_db
    mask = np.zeros(len(x), dtype=bool)
    mask[: nb * n] = np.repeat(keep, n)
    return mask


def compressor_gr(level, mask, p, sr):
    thr = float(np.percentile(level[mask], p["threshold_pct"]) + p["threshold_offset_db"])
    gr = np.minimum(static_gr(level, thr, p["ratio"], p["knee_db"]), p["max_gr_db"])
    gr = _smooth_gr(gr, coef(p["attack_ms"], sr), coef(p["release_ms"], sr))
    return gr, thr


def db_to_gain(gr_db):
    return 10 ** (-gr_db / 20)


# ---------------------------------------------------------------- limiter

@numba.njit(cache=True)
def _window_min(r, back, fwd):
    """out[n] = min(r[n-back .. n+fwd]) clipped to the array, O(N) monotonic deque."""
    n_total = r.shape[0]
    out = np.empty(n_total)
    dq = np.empty(n_total, dtype=np.int64)
    head = 0
    tail = 0
    j = 0
    for n in range(n_total):
        hi = min(n_total - 1, n + fwd)
        while j <= hi:
            while tail > head and r[dq[tail - 1]] >= r[j]:
                tail -= 1
            dq[tail] = j
            tail += 1
            j += 1
        lo = n - back
        while dq[head] < lo:
            head += 1
        out[n] = r[dq[head]]
    return out


@numba.njit(cache=True)
def _release(q, c):
    """Instant attack, exponential recovery of gain reduction in the dB domain."""
    out = np.empty_like(q)
    e = 1.0
    for n in range(q.shape[0]):
        e = e ** c
        if q[n] < e:
            e = q[n]
        out[n] = e
    return out


def limiter_gain(tp_env, ceiling_lin, sr, lim):
    """Gain curve g with g[n] * tp_env[n] <= ceiling_lin for every n.

    q = min of required gain over [n - hold, n + lookahead - 1]; release recovery;
    then a normalized Hann kernel of `lookahead` samples applied causally. Every
    value inside the kernel at the critical sample is already <= the requirement,
    so the smoothed gain cannot overshoot it.
    """
    w = max(1, int(round(lim["lookahead_ms"] * sr / 1000)))
    hold = int(round(lim["hold_ms"] * sr / 1000))
    req = np.minimum(1.0, ceiling_lin / np.maximum(tp_env, 1e-12))
    q = _window_min(req, hold, w - 1)
    e = _release(q, coef(lim["release_ms"], sr))
    k = np.hanning(w + 2)[1:-1]
    k /= k.sum()
    g = np.convolve(np.r_[np.ones(w - 1), e], k, mode="valid")
    return np.minimum(g, 1.0)


def soft_clip(z, knee_start, clip):
    """Identity below knee_start, tanh curve above it, never exceeding clip (linear)."""
    a = np.abs(z)
    out = z.copy()
    m = a > knee_start
    span = clip - knee_start
    out[m] = np.sign(z[m]) * (knee_start + span * np.tanh((a[m] - knee_start) / span))
    return out


def oversampled_clip(x, sr, ceiling_dbtp, clip_cfg, chunk=1 << 20, pad=256):
    """Soft clip at 4x oversampling (keeps harmonics from aliasing), in overlapping chunks.

    The clip level sits ceiling_offset_db above the limiter ceiling, so the clipper
    only shaves the tops of transients and the limiter catches what remains. Only the
    clipping difference is band-limited and added back, so samples away from clip
    events are returned unchanged (no resampling round-trip error).
    """
    up = measure.oversample_factor(sr)
    clip = 10 ** ((ceiling_dbtp + clip_cfg["ceiling_offset_db"]) / 20)
    knee_start = clip * 10 ** (-clip_cfg["knee_db"] / 20)
    out = x.copy()
    n = len(x)
    for s in range(0, n, chunk):
        a, b = max(0, s - pad), min(n, s + chunk + pad)
        seg = x[a:b]
        if np.abs(seg).max() * 1.5 < knee_start:
            continue
        hi = signal.resample_poly(seg, up, 1, axis=0)
        delta = soft_clip(hi, knee_start, clip) - hi
        if not delta.any():
            continue
        delta = signal.resample_poly(delta, 1, up, axis=0)
        e = min(n, s + chunk)
        out[s:e] += delta[s - a:e - a]
    return out


def apply_fades(x, sr, fade):
    """Raised-cosine fade-in/out reaching exactly zero at the file edges."""
    x = x.copy()
    for key, sl in (("in_ms", slice(None)), ("out_ms", slice(None, None, -1))):
        n = min(int(round(fade.get(key, 0.0) * sr / 1000)), len(x))
        if n > 0:
            ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
            x[sl][:n] *= ramp[:, None]
    return x


# ---------------------------------------------------------------- output

def write_wav24(path, x, sr, title=None, artist=None, seed=0):
    rng = np.random.default_rng(seed)
    scale = 2 ** 23
    tpdf = rng.random(x.shape) - rng.random(x.shape)
    q = np.clip(np.round(x * scale + tpdf), -scale, scale - 1).astype(np.int32)
    with sf.SoundFile(str(path), "w", samplerate=sr, channels=x.shape[1],
                      subtype="PCM_24", format="WAV") as f:
        if title:
            f.title = title
        if artist:
            f.artist = artist
        f.write(q << 8)


def stale_frame(frame):
    """Source frames that would be false on a re-encoded file: encoder/length fields, and
    C2PA manifests (their hard binding hashes the original audio bytes, so a copied
    manifest fails validation). Plain-text provenance (comments, source URL) is kept."""
    if frame.FrameID in ("TSSE", "TLEN"):
        return True
    return frame.FrameID == "GEOB" and getattr(frame, "mime", "") == "application/c2pa"


def encode_mp3(wav, mp3, src, title):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-map", "0:a",
                    "-map_metadata", "-1", "-c:a", "libmp3lame", "-b:a", "320k",
                    str(mp3)], check=True)
    from mutagen.id3 import ID3, TIT2, ID3NoHeaderError
    tags = ID3()
    try:
        for frame in ID3(str(src)).values():
            if not stale_frame(frame):
                tags.add(frame)
    except ID3NoHeaderError:
        pass
    if title:
        tags.delall("TIT2")
        tags.add(TIT2(encoding=3, text=title))
    tags.save(str(mp3), v2_version=3)


def ffmpeg_true_peak(path):
    """Independent true peak: ffmpeg resample to 192 kHz, then astats peak.

    The chain is pinned to double precision; left to negotiate, aresample/astats can
    settle on an integer format and clip overs at 0 dBFS, hiding codec overshoot.
    """
    err = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0", "-af",
         "aformat=sample_fmts=dbl,aresample=192000,aformat=sample_fmts=dbl,"
         "astats=measure_perchannel=none:measure_overall=Peak_level",
         "-f", "null", "-"], check=True, capture_output=True, text=True).stderr
    vals = [ln.split(":")[-1] for ln in err.splitlines() if "Peak level dB" in ln]
    return float(vals[-1])


CODECS = [("aac_256k", ["-c:a", "aac", "-b:a", "256k"], "m4a"),
          ("aac_128k", ["-c:a", "aac", "-b:a", "128k"], "m4a"),
          ("vorbis_160k", ["-c:a", "libvorbis", "-b:a", "160k"], "ogg"),
          ("vorbis_96k", ["-c:a", "libvorbis", "-b:a", "96k"], "ogg"),
          ("opus_128k", ["-c:a", "libopus", "-b:a", "128k"], "opus"),
          ("mp3_320k", ["-c:a", "libmp3lame", "-b:a", "320k"], "mp3")]


def codec_check(wav, tmpdir):
    res = {}
    for name, args, ext in CODECS:
        enc = Path(tmpdir) / f"_codec_{name}.{ext}"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), *args, str(enc)], check=True)
        res[name] = ffmpeg_true_peak(enc)
        enc.unlink()
    return res


# ---------------------------------------------------------------- chain

def pre_limiter(x, sr, cfg, log):
    stats = {}
    mask = active_mask(x, sr)
    hp = cfg["highpass"]
    if hp["enabled"]:
        x = signal.sosfilt(signal.butter(hp["order"], hp["freq"], "high", fs=sr, output="sos"), x, axis=0)
        log(f"high-pass {hp['freq']} Hz, order {hp['order']}")
    sos = eq_sos(cfg["eq"], sr)
    if sos is not None:
        x = signal.sosfilt(sos, x, axis=0)
        log("EQ " + ", ".join(f"{b['type']} {b['freq']} Hz {b['gain_db']:+} dB Q{b.get('q', 0.707)}"
                              for b in cfg["eq"]))
    lb = cfg["low_band"]
    if lb["enabled"]:
        low, high = split_low(x, lb["crossover"], sr)
        gr, thr = compressor_gr(detector_db(low, lb["rms_ms"], sr), mask, lb, sr)
        x = high + low * db_to_gain(gr)[:, None]
        stats["low_band"] = {"threshold_dbfs": thr, "gr": gr}
        log(f"low band <{lb['crossover']} Hz: threshold {thr:.1f} dBFS, ratio {lb['ratio']}, "
            f"max GR {gr.max():.2f} dB")
    hb = cfg["high_band"]
    if hb["enabled"]:
        band = band_zero_phase(x, hb["low"], hb["high"], sr)
        gr, thr = compressor_gr(detector_db(band, hb["rms_ms"], sr), mask, hb, sr)
        x = x + band * (db_to_gain(gr) - 1)[:, None]
        stats["high_band"] = {"threshold_dbfs": thr, "gr": gr}
        log(f"high band {hb['low']}-{hb['high']} Hz: threshold {thr:.1f} dBFS, ratio {hb['ratio']}, "
            f"max GR {gr.max():.2f} dB")
    gl = cfg["glue"]
    if gl["enabled"]:
        sc = signal.sosfilt(signal.butter(2, gl["sidechain_hp"], "high", fs=sr, output="sos"), x, axis=0)
        gr, thr = compressor_gr(detector_db(sc, gl["rms_ms"], sr), mask, gl, sr)
        x = x * db_to_gain(gr)[:, None]
        stats["glue"] = {"threshold_dbfs": thr, "gr": gr}
        log(f"glue: threshold {thr:.1f} dBFS, ratio {gl['ratio']}, max GR {gr.max():.2f} dB")
    return x, stats


def limit_to_target(y, sr, cfg, log):
    """Find pre-gain so the clipped + limited output hits target_lufs.

    Returns (out, gain_db, limiter_gain_curve, clipper_stats_or_None).
    """
    clip_cfg = cfg["clipper"]
    tp_y = None if clip_cfg["enabled"] else measure.true_peak_envelope(y, sr)
    target = cfg["target_lufs"]
    ceiling = cfg["ceiling_dbtp"] - 0.05
    base = measure.loudness(y, sr)["integrated"]

    for attempt in range(6):
        c_lin = 10 ** (ceiling / 20)

        def run(gdb):
            gl = 10 ** (gdb / 20)
            z = y * gl
            if clip_cfg["enabled"]:
                z = oversampled_clip(z, sr, ceiling, clip_cfg)
                tp = measure.true_peak_envelope(z, sr)
            else:
                tp = tp_y * gl
            g = limiter_gain(tp, c_lin, sr, cfg["limiter"])
            out = z * g[:, None]
            return out, g, measure.loudness(out, sr)["integrated"], z

        lo, hi = target - base - 0.5, target - base + 20.0
        gdb = target - base
        out, g, lufs, z = run(gdb)
        for _ in range(30):
            if abs(lufs - target) <= 0.02:
                break
            if lufs < target:
                lo = gdb
            else:
                hi = gdb
            gdb = min(max(gdb + (target - lufs) * 1.5, lo + 1e-3), hi - 1e-3)
            if not lo < gdb < hi:
                gdb = (lo + hi) / 2
            out, g, lufs, z = run(gdb)
        tp_out = measure.true_peak_dbtp(out, sr)
        log(f"pre-gain {gdb:+.2f} dB -> {lufs:.2f} LUFS, true peak {tp_out:.3f} dBTP "
            f"(internal ceiling {ceiling:.2f})")
        if tp_out <= cfg["ceiling_dbtp"] - 0.03:
            clip_stats = None
            if clip_cfg["enabled"]:
                clean = y * 10 ** (gdb / 20)
                resid = z - clean
                active = np.abs(resid).max(axis=1) > 10 ** (-100 / 20)
                clip_stats = {
                    "residual_rel_db": float(10 * np.log10(max(np.sum(resid ** 2), 1e-30) / np.sum(clean ** 2))),
                    "pct_samples_active": float(active.mean() * 100),
                    "peak_reduction_db": float(measure.true_peak_dbtp(clean, sr) - measure.true_peak_dbtp(z, sr)),
                }
                log(f"clipper: residual {clip_stats['residual_rel_db']:.1f} dB rel. signal, active on "
                    f"{clip_stats['pct_samples_active']:.2f}% of samples, peak reduction "
                    f"{clip_stats['peak_reduction_db']:.2f} dB")
            return out, gdb, g, clip_stats
        ceiling -= tp_out - (cfg["ceiling_dbtp"] - 0.05) + 0.02
    raise RuntimeError("limiter could not meet the true-peak ceiling")


def block_stats(gr, sr, block_s):
    n = int(block_s * sr)
    rows = []
    for i in range(0, len(gr), n):
        seg = gr[i:i + n]
        rows.append({"start_s": i / sr, "max": float(seg.max()), "mean": float(seg.mean()),
                     "pct_over_1db": float((seg > 1).mean() * 100),
                     "pct_over_3db": float((seg > 3).mean() * 100)})
    return rows


def summarize_gr(name, gr, sr, block_s):
    return {"name": name, "max": float(gr.max()), "mean": float(gr.mean()),
            "pct_over_1db": float((gr > 1).mean() * 100), "pct_over_3db": float((gr > 3).mean() * 100),
            "blocks": block_stats(gr, sr, block_s)}


def master(src, out_dir, name, title, cfg, mp3=True, check_codecs=False, log=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    x, sr = measure.load_audio(src)
    log(f"source: {src} ({len(x) / sr:.2f} s, {sr} Hz, {x.shape[1]} ch)")
    before = measure.loudness(x, sr)
    before.update(true_peak=measure.true_peak_dbtp(x, sr))
    y, stats = pre_limiter(x, sr, cfg, log)
    out, gain_db, g, clip_stats = limit_to_target(y, sr, cfg, log)
    lim_gr = -20 * np.log10(np.maximum(g, 1e-12))
    out = apply_fades(out, sr, cfg["fade"])
    log(f"fades: in {cfg['fade'].get('in_ms', 0)} ms, out {cfg['fade'].get('out_ms', 0)} ms")

    wav = out_dir / f"{name}.wav"
    artist = None
    try:
        from mutagen import File as MFile
        tags = MFile(str(src), easy=True)
        artist = (tags.get("artist") or [None])[0] if tags else None
    except Exception:
        pass
    write_wav24(wav, out, sr, title=title, artist=artist)
    log(f"wrote {wav}")
    files = {"wav": str(wav)}
    if mp3:
        mp3_path = out_dir / f"{name}.mp3"
        encode_mp3(wav, mp3_path, src, title)
        files["mp3"] = str(mp3_path)
        log(f"wrote {mp3_path}")

    after = measure.loudness(out, sr)
    after.update(true_peak=measure.true_peak_dbtp(out, sr))
    report = {
        "source": str(src), "files": files, "config": cfg, "pre_gain_db": gain_db,
        "before": before, "after": after,
        "verify": {"wav": measure.ffmpeg_ebur128(wav), "wav_true_peak_192k": ffmpeg_true_peak(wav)},
        "dynamics": {k: {"threshold_dbfs": v["threshold_dbfs"],
                         **summarize_gr(k, v["gr"], sr, cfg["report_block_s"])}
                     for k, v in stats.items()},
    }
    report["dynamics"]["limiter"] = summarize_gr("limiter", lim_gr, sr, cfg["report_block_s"])
    if clip_stats:
        report["clipper"] = clip_stats
    if mp3:
        report["verify"]["mp3"] = measure.ffmpeg_ebur128(files["mp3"])
        report["verify"]["mp3_true_peak_192k"] = ffmpeg_true_peak(files["mp3"])
    if check_codecs:
        report["verify"]["codec_true_peak_192k"] = codec_check(wav, out_dir)
    rep_path = out_dir / f"{name}_report.json"
    rep_path.write_text(json.dumps(report, indent=2))
    log(f"wrote {rep_path}")
    return report


def print_summary(r, log=print):
    b, a, v = r["before"], r["after"], r["verify"]
    log(f"\n{'':18}{'source':>10}{'master':>10}{'ffmpeg':>10}")
    log(f"{'integrated LUFS':18}{b['integrated']:10.2f}{a['integrated']:10.2f}{v['wav']['integrated']:10.1f}")
    log(f"{'true peak dBTP':18}{b['true_peak']:10.2f}{a['true_peak']:10.2f}{v['wav_true_peak_192k']:10.2f}")
    log(f"{'LRA LU':18}{b['lra']:10.2f}{a['lra']:10.2f}{v['wav']['lra']:10.1f}")
    log(f"{'PLR dB':18}{b['true_peak'] - b['integrated']:10.2f}{a['true_peak'] - a['integrated']:10.2f}")
    log(f"{'max short-term':18}{b['max_short_term']:10.2f}{a['max_short_term']:10.2f}")
    if "mp3_true_peak_192k" in v:
        log(f"mp3: {v['mp3']['integrated']:.1f} LUFS, true peak {v['mp3_true_peak_192k']:.2f} dBTP")
    if "codec_true_peak_192k" in v:
        log("decoded true peak: " + ", ".join(f"{k} {p:.2f}" for k, p in v["codec_true_peak_192k"].items()))
    if "clipper" in r:
        c = r["clipper"]
        log(f"clipper: residual {c['residual_rel_db']:.1f} dB, active {c['pct_samples_active']:.2f}% of samples, "
            f"peak reduction {c['peak_reduction_db']:.2f} dB")
    log("\ngain reduction (dB): max / mean / % time >1 dB / % time >3 dB")
    for k, d in r["dynamics"].items():
        log(f"  {k:10} {d['max']:5.2f} / {d['mean']:4.2f} / {d['pct_over_1db']:5.1f}% / {d['pct_over_3db']:5.1f}%")
    log("limiter per block (start s: max GR, % >3 dB):")
    log("  " + "  ".join(f"{int(bk['start_s'])}:{bk['max']:.1f},{bk['pct_over_3db']:.1f}%"
                         for bk in r["dynamics"]["limiter"]["blocks"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--preset", help="JSON file overriding DEFAULTS")
    ap.add_argument("--target-lufs", type=float)
    ap.add_argument("--ceiling-dbtp", type=float)
    ap.add_argument("--out-dir", default=str(REPO / "output"))
    ap.add_argument("--name", help="output basename (default: <input stem>_master)")
    ap.add_argument("--title", help="title tag for the outputs")
    ap.add_argument("--no-mp3", action="store_true")
    ap.add_argument("--codec-check", action="store_true",
                    help="encode AAC/Vorbis/Opus/MP3, decode, and report true peaks")
    a = ap.parse_args(argv)
    cfg = DEFAULTS
    if a.preset:
        cfg = merge(cfg, json.loads(Path(a.preset).read_text()))
    if a.target_lufs is not None:
        cfg = merge(cfg, {"target_lufs": a.target_lufs})
    if a.ceiling_dbtp is not None:
        cfg = merge(cfg, {"ceiling_dbtp": a.ceiling_dbtp})
    src = Path(a.input)
    name = a.name or f"{src.stem}_master"
    r = master(src, Path(a.out_dir), name, a.title, cfg, mp3=not a.no_mp3, check_codecs=a.codec_check)
    print_summary(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
