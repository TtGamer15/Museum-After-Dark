#!/usr/bin/env python3
"""Clean up recorded voice lines for Museum After Dark.

Usage:
  process_voice.py OUT_DIR clip1 clip2 ...     one file per line, in script order
  process_voice.py OUT_DIR --split one_file    one long recording, lines split on pauses

Each clip: mono 48 kHz -> rumble/hum cut -> FFT noise reduction tuned to the clip's own
noise floor -> gentle compression -> soft gate between words -> trim silence -> level match
-> short fades -> 96 kbps mono MP3. The breathing clip gets a lighter touch (no gate).
"""
import json, os, re, subprocess, sys, tempfile
import numpy as np
from scipy.io import wavfile

ORDER = ['hmm1', 'hmm2', 'heard1', 'heard2', 'peek1', 'gotcha1', 'gotcha2', 'nothing1', 'nothing2', 'yawn1', 'dawn1', 'gems1', 'breath1']
SR = 48000

def run(args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])
    return r

def load(path):
    tmp = tempfile.mktemp(suffix='.wav')
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', path, '-ac', '1', '-ar', str(SR), '-c:a', 'pcm_f32le', tmp])
    sr, x = wavfile.read(tmp); os.remove(tmp)
    return x.astype(np.float32)

def save_wav(x, path):
    wavfile.write(path, SR, x.astype(np.float32))

def frames_db(x, win=0.03):
    n = int(SR * win); m = max(1, len(x) // n)
    f = x[:m * n].reshape(m, n)
    return 20 * np.log10(np.sqrt((f ** 2).mean(axis=1)) + 1e-9), n

def ffilter(x, chain):
    a, b = tempfile.mktemp(suffix='.wav'), tempfile.mktemp(suffix='.wav')
    save_wav(x, a)
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', a, '-af', chain, '-ac', '1', '-ar', str(SR), '-c:a', 'pcm_f32le', b])
    sr, y = wavfile.read(b); os.remove(a); os.remove(b)
    return y.astype(np.float32)

def trim(x, floor_db, pad_in=0.06, pad_out=0.16, over=14):
    db, n = frames_db(x)
    on = np.where(db > floor_db + over)[0]
    if not len(on):
        return x
    s = max(0, on[0] * n - int(pad_in * SR)); e = min(len(x), (on[-1] + 1) * n + int(pad_out * SR))
    return x[s:e]

def fades(x, fin=0.008, fout=0.08):
    a, b = int(fin * SR), int(fout * SR)
    if len(x) > a + b:
        x = x.copy(); x[:a] *= np.linspace(0, 1, a); x[-b:] *= np.linspace(1, 0, b) ** 2
    return x

def level(x, target_rms_db, peak_db=-1.5):
    db, n = frames_db(x)
    voiced = db[db > db.max() - 25]
    rms = 10 ** (voiced.mean() / 20) if len(voiced) else 1e-3
    g = 10 ** (target_rms_db / 20) / max(rms, 1e-6)
    pk = np.abs(x).max() * g
    if pk > 10 ** (peak_db / 20):
        g *= 10 ** (peak_db / 20) / pk
    return x * g

def clean(x, kind):
    db, _ = frames_db(x)
    floor = float(np.percentile(db, 8))
    nf = int(np.clip(floor, -80, -25))
    if kind == 'breath':
        chain = f'highpass=f=70,bandreject=f=60:width_type=h:w=8,afftdn=nr=12:nf={nf}:tn=1,lowpass=f=11000'
        y = ffilter(x, chain)
        y = trim(y, floor - 6, pad_in=.05, pad_out=.3, over=8)
        y = level(y, -26, -4)
        return fades(y, .12, .5), floor
    # Already-clean recordings get a lighter touch, so the voice doesn't turn watery.
    rel = float(np.percentile(db, 97)) - floor
    nr, rng = (24, .08) if rel < 30 else (16, .14) if rel < 40 else (10, .22)
    chain = (f'highpass=f=80,highpass=f=80,bandreject=f=60:width_type=h:w=8,afftdn=nr={nr}:nf={nf}:tn=1,lowpass=f=12000,'
             'acompressor=threshold=0.12:ratio=3:attack=6:release=120:makeup=1.5')
    y = ffilter(x, chain)
    ydb, _ = frames_db(y)
    yfloor = float(np.percentile(ydb, 8)); peak = float(ydb.max())
    gate = 10 ** ((yfloor + 0.25 * (peak - yfloor)) / 20)
    y = ffilter(y, f'agate=threshold={gate:.5f}:ratio=3:attack=3:release=250:range={rng}:knee=6')
    y = trim(y, yfloor)
    y = level(y, -18)
    return fades(y), floor

def to_mp3(x, path):
    w = tempfile.mktemp(suffix='.wav'); save_wav(x, w)
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', w, '-ac', '1', '-ar', '44100', '-c:a', 'libmp3lame', '-b:a', '96k', path])
    os.remove(w)

def split_long(x):
    db, n = frames_db(x, 0.02)
    floor = np.percentile(db, 10); thr = floor + 12
    loud = db > thr
    segs, start, quiet = [], None, 0
    for i, v in enumerate(loud):
        if v:
            if start is None: start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet * 0.02 >= 0.8:
                segs.append((start, i - quiet + 1)); start, quiet = None, 0
    if start is not None: segs.append((start, len(loud)))
    segs = [(s, e) for s, e in segs if (e - s) * 0.02 >= 0.25]
    return [x[max(0, s * n - int(.3 * SR)): min(len(x), e * n + int(.4 * SR))] for s, e in segs]

def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', os.path.basename(p))]

def main():
    out = sys.argv[1]; args = sys.argv[2:]
    os.makedirs(out, exist_ok=True)
    breath = None
    if '--breath' in args:
        k = args.index('--breath'); breath = load(args[k + 1]); args = args[:k] + args[k + 2:]
    if args and args[0] == '--split':
        clips = split_long(load(args[1]))
        print(f'found {len(clips)} lines in the long recording')
        if '--drop' in args:   # takes to throw away (1-based), e.g. a messed-up line said twice
            drop = {int(v) - 1 for v in args[args.index('--drop') + 1].split(',')}
            clips = [c for i, c in enumerate(clips) if i not in drop]
        if breath is not None:
            clips = clips[:len(ORDER) - 1] + [None] * max(0, len(ORDER) - 1 - len(clips)) + [breath]
    else:
        clips = [load(p) for p in args]
    report, done = [], []
    for i, x in enumerate(clips[:len(ORDER)]):
        cid = ORDER[i]
        if x is None: continue
        y, floor = clean(x, 'breath' if cid.startswith('breath') else 'line')
        to_mp3(y, os.path.join(out, cid + '.mp3'))
        ydb, _ = frames_db(y)
        report.append(f'{cid}: {len(x)/SR:.1f}s -> {len(y)/SR:.2f}s, noise floor {floor:.0f} dB -> {np.percentile(ydb, 8):.0f} dB')
        done.append(cid)
    json.dump({'clips': done}, open(os.path.join(out, 'list.json'), 'w'))
    print('\n'.join(report))
    if len([c for c in clips if c is not None]) != len(ORDER):
        print(f'NOTE: got {len([c for c in clips if c is not None])} clips, the script has {len(ORDER)} lines')

if __name__ == '__main__':
    main()
