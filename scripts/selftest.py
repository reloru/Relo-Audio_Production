#!/usr/bin/env python3
"""Self-checks for measure.py and master.py on synthetic signals.

Usage:
    python3 scripts/selftest.py

Exits non-zero on the first failed check. Needs ffmpeg on PATH (used as an
independent loudness reference).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import master as M  # noqa: E402
import measure  # noqa: E402

SR = 48000
FAILED = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def sine(freq, dbfs, seconds, phase=0.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    s = 10 ** (dbfs / 20) * np.sin(2 * np.pi * freq * t + phase)
    return np.column_stack([s, s])


def test_meter():
    for level in (-23.0, -33.0):
        lufs = measure.loudness(sine(1000, level, 20), SR)["integrated"]
        check(f"EBU Tech 3341 stereo 1 kHz sine at {level} dBFS reads {level} LUFS",
              abs(lufs - level) <= 0.1, f"{lufs:.3f}")
    edge = SR // 10
    for freq, phase in ((12000, np.pi / 4), (1000, 0.3), (18000, 0.3)):
        x = sine(freq, -6.0, 1.0, phase=phase)
        env = measure.true_peak_envelope(x, SR)[edge:-edge]
        tp, sp = 20 * np.log10(env.max()), measure.sample_peak_dbfs(x[edge:-edge])
        check(f"true peak of a {freq} Hz sine at -6 dBFS (away from the file edges)",
              abs(tp + 6.0) <= 0.1, f"true {tp:.3f}, sample {sp:.3f}")

    rng = np.random.default_rng(1)
    noise = signal.lfilter([1], [1, -0.95], rng.standard_normal((SR * 15, 2)), axis=0)
    noise *= 10 ** (-20 / 20) / np.sqrt(np.mean(noise ** 2))
    noise[SR * 5:SR * 7] *= 0.05
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "noise.wav"
        sf.write(p, noise, SR, subtype="FLOAT")
        own = measure.loudness(noise, SR)
        ff = measure.ffmpeg_ebur128(p)
        check("integrated loudness matches ffmpeg ebur128 on gated noise",
              abs(own["integrated"] - ff["integrated"]) <= 0.1, f"{own['integrated']:.2f} vs {ff['integrated']}")
        check("loudness range matches ffmpeg ebur128", abs(own["lra"] - ff["lra"]) <= 0.2,
              f"{own['lra']:.2f} vs {ff['lra']}")
        hot = Path(d) / "hot.wav"
        sf.write(hot, sine(1000, 1.5, 1.0), SR, subtype="FLOAT")
        ff_tp = M.ffmpeg_true_peak(hot)
        check("ffmpeg true-peak check reports overs above 0 dBFS instead of clipping them",
              abs(ff_tp - 1.5) <= 0.1, f"{ff_tp:.2f} dBTP for a +1.5 dBFS float sine")


def test_filters():
    for kind, probe_hz in (("peak", 1800.0), ("lowshelf", 10.0), ("highshelf", 23000.0)):
        sos = M.biquad(kind, 1800.0, 3.0, 0.9, SR)
        _, h = signal.sosfreqz(sos, worN=[probe_hz], fs=SR)
        g = 20 * np.log10(abs(h[0]))
        check(f"{kind} biquad reaches +3 dB", abs(g - 3.0) <= 0.05, f"{g:.3f} dB at {probe_hz} Hz")
    x = np.random.default_rng(2).standard_normal((SR, 2))
    low, high = M.split_low(x, 85.0, SR)
    err = np.abs(low + high - x).max()
    check("low-band split reconstructs exactly", err < 1e-12, f"max err {err:.1e}")
    _, h = signal.sosfreqz(signal.butter(2, 85.0, "low", fs=SR, output="sos"), worN=[85.0], fs=SR)
    check("zero-phase split crossover is -6 dB at the crossover",
          abs(40 * np.log10(abs(h[0])) + 6.02) <= 0.05, f"{40 * np.log10(abs(h[0])):.3f} dB")


def test_dynamics():
    level = np.array([-40.0, -23.0, -20.0, -17.0, 0.0])
    gr = M.static_gr(level, -20.0, 2.0, 6.0)
    expected = np.array([0.0, 0.0, 0.375, 1.5, 10.0])
    check("gain computer: zero below the knee, quadratic inside it, (1-1/R)*over above it",
          np.allclose(gr, expected, atol=1e-12), np.array2string(gr, precision=3))
    rng = np.random.default_rng(3)
    x = rng.standard_normal((SR * 4, 2)) * 0.1
    bursts = rng.integers(0, len(x) - 200, 40)
    for b in bursts:
        x[b:b + 100] *= 12
    ceiling = 10 ** (-1.0 / 20)
    tp = measure.true_peak_envelope(x, SR)
    g = M.limiter_gain(tp, ceiling, SR, M.DEFAULTS["limiter"])
    worst = (g * tp).max() / ceiling
    check("limiter gain never lets the detected true peak exceed the ceiling",
          worst <= 1 + 1e-9 and g.max() <= 1.0, f"max g*tp/ceiling {worst:.6f}")
    y = x * g[:, None]
    tp_out = measure.true_peak_dbtp(y, SR)
    check("limited output true peak is at or under -1 dBTP (+0.05 dB measurement slack)",
          tp_out <= -0.95, f"{tp_out:.3f} dBTP")
    quiet = np.ones(SR) * 0.1
    gq = M.limiter_gain(quiet, ceiling, SR, M.DEFAULTS["limiter"])
    check("limiter is unity below the ceiling", np.all(gq == 1.0))


def test_clipper():
    clip_cfg = {"ceiling_offset_db": 1.0, "knee_db": 1.0}
    x = sine(1000, -12.0, 1.0)
    check("clipper returns a signal below its knee bit-identical",
          np.array_equal(M.oversampled_clip(x, SR, -1.0, clip_cfg), x))
    x[24000:24048] *= 6.0
    out = M.oversampled_clip(x, SR, -1.0, clip_cfg)
    before, after = measure.true_peak_dbtp(x, SR), measure.true_peak_dbtp(out, SR)
    untouched = np.array_equal(out[:23900], x[:23900]) and np.array_equal(out[24150:], x[24150:])
    check("clipper shaves a +3.5 dBFS burst toward 0 dBTP and leaves distant samples unchanged",
          after < 0.5 and before - after > 2.5 and untouched,
          f"{before:.2f} -> {after:.2f} dBTP, untouched={untouched}")


def test_fades():
    x = np.ones((SR, 2))
    y = M.apply_fades(x, SR, {"in_ms": 10.0, "out_ms": 100.0})
    n_in, n_out = 480, 4800
    ok = (np.all(y[0] == 0) and np.all(y[-1] == 0) and np.all(np.diff(y[:n_in, 0]) > 0)
          and np.all(np.diff(y[-n_out:, 0]) < 0) and np.all(y[n_in:-n_out] == 1.0))
    check("fades reach zero at both edges, are monotonic, and leave the middle untouched", ok)


def test_wav24():
    x = sine(440, -3.0, 0.5)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.wav"
        M.write_wav24(p, x, SR, title="t", artist="a")
        info = sf.info(str(p))
        back, _ = sf.read(str(p), dtype="float64")
        err = np.abs(back - x).max() * 2 ** 23
        check("24-bit WAV write is PCM_24 and round-trips within dither (<= 1.5 LSB)",
              info.subtype == "PCM_24" and err <= 1.5, f"{info.subtype}, max err {err:.2f} LSB")


def main():
    test_meter()
    test_filters()
    test_dynamics()
    test_clipper()
    test_fades()
    test_wav24()
    print(f"\n{'all checks passed' if not FAILED else f'{len(FAILED)} check(s) failed'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
