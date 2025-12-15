import os, glob
import numpy as np
from pathlib import Path

from .utils import ensure_dir
from .config import EXPECTED_TRACES

_USE_OBSPY = False
try:
    import segyio
except Exception:
    _USE_OBSPY = True
    from obspy.io.segy.segy import _read_segy

def normalize_gather(g):
    m = np.mean(g, keepdims=True)
    s = np.std(g, keepdims=True) + 1e-8
    g = (g - m) / s
    g = np.clip(g, -3.0, 3.0) / 3.0
    return g.astype(np.float32)

def iter_gathers_from_segy(path, expected_traces=EXPECTED_TRACES, verbose=True):
    if not _USE_OBSPY:
        with segyio.open(path, "r", ignore_geometry=True) as f:
            key = segyio.TraceField.FieldRecord
            ffids = np.asarray([f.attributes(key)[i] for i in range(f.tracecount)])
            uids = np.unique(ffids)
            if verbose:
                print(f"[GATHER] {len(uids)} FFIDs in {Path(path).name}")
            for gid in uids:
                idx = np.where(ffids == gid)[0]
                traces = np.asarray([f.trace[i] for i in idx], dtype=np.float32).T
                if expected_traces is not None and traces.shape[1] != expected_traces:
                    continue
                yield int(gid), traces
    else:
        st = _read_segy(path, headonly=False)
        arr = np.stack([tr.data for tr in st.traces], axis=1).astype(np.float32)
        if expected_traces is None:
            yield 0, arr
        else:
            for j in range(0, arr.shape[1], expected_traces):
                g = arr[:, j:j+expected_traces]
                if g.shape[1] == expected_traces:
                    yield j // expected_traces, g

def cache_gathers(use_dir, single_path, dir_path, dir_glob, cache_dir, expected_traces, limit=None):
    ensure_dir(cache_dir)
    out = []

    if use_dir:
        files = sorted(glob.glob(os.path.join(dir_path, dir_glob)))
    else:
        files = [single_path]

    k = 0
    for fp in files:
        p = Path(fp)
        if not p.exists():
            continue
        for gid, g in iter_gathers_from_segy(fp, expected_traces, verbose=True):
            g = normalize_gather(g)
            npy = os.path.join(cache_dir, f"{p.stem}_gid{gid:06d}.npy")
            np.save(npy, g)
            out.append(npy)
            k += 1
            if (limit is not None) and (k >= limit):
                print("[CACHE] limit reached.")
                return out

    print(f"[CACHE] total cached: {len(out)}")
    return out
