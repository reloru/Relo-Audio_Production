#!/usr/bin/env python3
"""Loudness and peak measurement (ITU-R BS.1770-4 / EBU R128 / EBU Tech 3342).

Usage:
    python3 scripts/measure.py FILE [FILE ...]

Prints, for each file, this module's own measurements next to ffmpeg's
`ebur128` filter as an independent cross-check.
"""
import json
import re
import subprocess
import sys

import numpy as np
from scipy import signal


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    return int(st["sample_rate"]), int(st["channels"])


def load_audio(path):
    """Decode any ffmpeg-readable file to float64 (frames, channels) at its native rate."""
    sr, ch = probe(path)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        check=True, capture_output=True).stdout
    x = np.frombuffer(raw, dtype="<f4").astype(np.float64).reshape(-1, ch)
    return x, sr


def kweighting_sos(sr):
    """BS.1770 K-weighting (pre-filter shelf + RLB high-pass), re-derived for any rate."""
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / sr)
    vh = 10 ** (gain_db / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = [(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0,
             1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / sr)
    a0 = 1 + k / q + k * k
    hp = [1.0, -2.0, 1.0, 1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    return np.array([shelf, hp])


def _block_power(z, sr, win_s, hop_s):
    w, h = int(round(win_s * sr)), int(round(hop_s * sr))
    if len(z) < w:
        return np.array([]), np.array([])
    cs = np.concatenate([np.zeros((1, z.shape[1])), np.cumsum(z, axis=0)])
    starts = np.arange(0, len(z) - w + 1, h)
    power = ((cs[starts + w] - cs[starts]) / w).sum(axis=1)
    return starts / sr, power


def _lufs(power):
    return -0.691 + 10 * np.log10(np.maximum(power, 1e-30))


def loudness(x, sr):
    """Integrated loudness, loudness range, max momentary and max short-term (LUFS / LU).

    Channel weights are 1.0 (mono/stereo; surround weighting is not implemented).
    """
    z = signal.sosfilt(kweighting_sos(sr), x, axis=0) ** 2
    _, p400 = _block_power(z, sr, 0.4, 0.1)
    _, p3 = _block_power(z, sr, 3.0, 0.1)
    res = {"integrated": float("-inf"), "lra": 0.0,
           "max_momentary": float("-inf"), "max_short_term": float("-inf")}
    if len(p400):
        l400 = _lufs(p400)
        res["max_momentary"] = float(l400.max())
        gated = p400[l400 > -70]
        if len(gated):
            rel = _lufs(gated.mean()) - 10
            gated = gated[_lufs(gated) > rel]
            res["integrated"] = float(_lufs(gated.mean()))
    if len(p3):
        l3 = _lufs(p3)
        res["max_short_term"] = float(l3.max())
        g = p3[l3 > -70]
        if len(g):
            rel = _lufs(g.mean()) - 20
            ls = _lufs(g[_lufs(g) > rel])
            res["lra"] = float(np.percentile(ls, 95) - np.percentile(ls, 10))
    return res


def short_term(x, sr, hop_s=1.0):
    """Short-term loudness (3 s window) every hop_s seconds; returns (times, LUFS)."""
    z = signal.sosfilt(kweighting_sos(sr), x, axis=0) ** 2
    t, p = _block_power(z, sr, 3.0, hop_s)
    return t, _lufs(p)


def oversample_factor(sr):
    return 4 if sr < 96000 else 2 if sr < 192000 else 1


def true_peak_envelope(x, sr, chunk=1 << 20, pad=256):
    """Per-sample inter-sample peak magnitude (max over channels and the oversampled
    points in [n, n+1)), using BS.1770-4 Annex 2 style oversampling. Processed in
    overlapping chunks so memory stays bounded on long files."""
    up = oversample_factor(sr)
    env = np.abs(x).max(axis=1)
    if up == 1:
        return env
    n = len(x)
    for s in range(0, n, chunk):
        a, b = max(0, s - pad), min(n, s + chunk + pad)
        y = np.abs(signal.resample_poly(x[a:b], up, 1, axis=0)).max(axis=1)
        y = y[(s - a) * up:(min(n, s + chunk) - a) * up]
        env[s:s + len(y) // up] = np.maximum(env[s:s + len(y) // up], y.reshape(-1, up).max(axis=1))
    return env


def true_peak_dbtp(x, sr):
    return float(20 * np.log10(max(true_peak_envelope(x, sr).max(), 1e-12)))


def sample_peak_dbfs(x):
    return float(20 * np.log10(max(np.abs(x).max(), 1e-12)))


def ffmpeg_ebur128(path):
    """Integrated loudness, LRA and true peak as reported by ffmpeg's ebur128 filter."""
    err = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0",
         "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"],
        check=True, capture_output=True, text=True).stderr
    summary = err[err.rfind("Summary:"):]

    def grab(label):
        m = re.search(label + r":\s+(-?inf|-?[\d.]+)", summary)
        return float(m.group(1)) if m else float("nan")

    return {"integrated": grab(r"I"), "lra": grab(r"LRA"), "true_peak": grab(r"Peak")}


def measure_file(path):
    x, sr = load_audio(path)
    res = loudness(x, sr)
    res.update(true_peak=true_peak_dbtp(x, sr), sample_peak=sample_peak_dbfs(x),
               duration=len(x) / sr, sample_rate=sr, channels=x.shape[1])
    res["plr"] = res["true_peak"] - res["integrated"]
    res["ffmpeg"] = ffmpeg_ebur128(path)
    return res


def format_report(path, r):
    f = r["ffmpeg"]
    return (f"{path}\n"
            f"  {r['duration']:.2f} s, {r['sample_rate']} Hz, {r['channels']} ch\n"
            f"  integrated {r['integrated']:7.2f} LUFS   (ffmpeg {f['integrated']:.1f})\n"
            f"  LRA        {r['lra']:7.2f} LU     (ffmpeg {f['lra']:.1f})\n"
            f"  true peak  {r['true_peak']:7.2f} dBTP   (ffmpeg {f['true_peak']:.1f})\n"
            f"  sample pk  {r['sample_peak']:7.2f} dBFS\n"
            f"  PLR        {r['plr']:7.2f} dB\n"
            f"  max short-term {r['max_short_term']:.2f} LUFS, max momentary {r['max_momentary']:.2f} LUFS")


def main(argv):
    if not argv:
        print(__doc__.strip())
        return 2
    for p in argv:
        print(format_report(p, measure_file(p)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
