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

def normalize_gather(g: np.ndarray) -> np.ndarray:
    # z-score per gather → clip 3σ → map to [-1,1]
    m = np.mean(g, keepdims=True)
    s = np.std(g, keepdims=True) + 1e-8
    g = (g - m) / s
    g = np.clip(g, -3.0, 3.0) / 3.0
    return g.astype(np.float32)

def iter_gathers_from_segy(path: str, expected_traces=EXPECTED_TRACES, verbose: bool = True):
    if not _USE_OBSPY:
        with segyio.open(path, "r", ignore_geometry=True) as f:
            key = segyio.TraceField.FieldRecord
            ffids = np.asarray([f.attributes(key)[i] for i in range(f.tracecount)])
            uids = np.unique(ffids)
            if verbose:
                print(f"[GATHER] FFIDs: {len(uids)} in {Path(path).name}")
            for gid in uids:
                idx = np.where(ffids == gid)[0]
                traces = np.asarray([f.trace[i] for i in idx], dtype=np.float32).T  # (ns, ntr)
                if expected_traces is not None and traces.shape[1] != expected_traces:
                    continue
                yield int(gid), traces
    else:
        st = _read_segy(path, headonly=False)
        all_traces = [tr.data for tr in st.traces]
        arr = np.stack(all_traces, axis=1).astype(np.float32)
        if expected_traces is None:
            yield 0, arr
        else:
            for j in range(0, arr.shape[1], expected_traces):
                g = arr[:, j:j + expected_traces]
                if g.shape[1] == expected_traces:
                    yield j // expected_traces, g

def cache_gathers(use_dir: bool, single_path: str, dir_path: str, dir_glob: str,
                 cache_dir: str, expected_traces, limit=None):
    ensure_dir(cache_dir)
    out = []

    if use_dir:
        files = sorted(glob.glob(os.path.join(dir_path, dir_glob)))
        files = [f for f in files if os.path.isfile(f)]
        if len(files) == 0:
            raise RuntimeError(f"No SEGY in {dir_path} with pattern {dir_glob}")
    else:
        files = [single_path]
        p = Path(single_path)
        if not p.exists() or not p.is_file():
            raise RuntimeError(f"SINGLE_FILE_PATH not found: {single_path}")
        print(f"[INFO] Single file: {p} | size≈{p.stat().st_size/(1024*1024):.2f} MB")

    k = 0
    for fp in files:
        for gid, g in iter_gathers_from_segy(fp, expected_traces, verbose=True):
            g = normalize_gather(g)
            npy = os.path.join(cache_dir, f"{Path(fp).stem}_gid{gid:06d}.npy")
            np.save(npy, g)
            out.append(npy)
            k += 1
            if (limit is not None) and (k >= limit):
                print(f"[CACHE] Limit reached: {limit} gathers.")
                return out

    print(f"[CACHE] Total cached: {len(out)} in {cache_dir}")
    return out
