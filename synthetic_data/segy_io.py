import numpy as np

_USE_OBSPY = False
try:
    import segyio
except Exception:
    _USE_OBSPY = True
    from obspy.io.segy.segy import _read_segy

def read_shot_gather(path, expected_traces=None, normalize=True, verbose=False):
    if not _USE_OBSPY:
        with segyio.open(path, "r", ignore_geometry=True) as f:
            traces = np.asarray([f.trace[i] for i in range(f.tracecount)], dtype=np.float32)
            g = traces.T  # (ns, ntr)
    else:
        st = _read_segy(path, headonly=False)
        traces = np.stack([tr.data for tr in st.traces]).astype(np.float32)
        g = traces.T

    if expected_traces is not None and g.shape[1] > expected_traces:
        g = g[:, :expected_traces]

    if normalize:
        m = np.mean(g, keepdims=True)
        s = np.std(g, keepdims=True) + 1e-8
        g = (g - m) / s
        g = np.clip(g, -3.0, 3.0) / 3.0
        g = g.astype(np.float32)

    return g
