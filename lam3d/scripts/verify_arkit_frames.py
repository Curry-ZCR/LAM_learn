"""Frame-only HTTP verification against an already-succeeded job."""
import hashlib
import io
import json
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

BASE = "http://127.0.0.1:8000"
JOB = "93ba48328fdc"


def req(method, path, body=None, timeout=120):
    data = None
    r = urllib.request.Request(BASE + path, data=data, method=method)
    if body is not None:
        data = json.dumps(body).encode()
        r = urllib.request.Request(BASE + path, data=data, method=method)
        r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def frame(arkit, size=320, tries=4):
    for _ in range(tries):
        t0 = time.perf_counter()
        code, body = req("POST", f"/api/jobs/{JOB}/arkit_frame",
                         {"arkit": [float(x) for x in arkit], "size": size})
        dt = (time.perf_counter() - t0) * 1e3
        if code == 200:
            return code, body, dt
        print("    retry after HTTP %s: %s" % (code, body[:160]))
        time.sleep(2)
    return code, body, dt


def main():
    code, body = req("GET", f"/api/jobs/{JOB}/arkit_session")
    print("session:", code, body.decode())

    z = [0.0] * 52
    nn = list(z); nn[43] = 0.8
    ones = [1.0] * 52
    cases = [("neutral", z), ("nonneutral", nn), ("allones", ones)]
    out = {}
    for name, v in cases:
        code, png, dt = frame(v)
        if code != 200:
            print("%-11s -> HTTP %s %s" % (name, code, png[:200]))
            continue
        md5 = hashlib.md5(png).hexdigest()
        a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        out[name] = (md5, a)
        with open(f"/tmp/final_{name}.png", "wb") as f:
            f.write(png)
        print("%-11s -> 200 %7.1f ms %6d B md5=%s img[min=%d max=%d mean=%.2f std=%.2f black=%d]"
              % (name, dt, len(png), md5, a.min(), a.max(), a.mean(), a.std(),
                 int((a.sum(axis=-1) == 0).sum())))

    print("\ncomparisons:")
    if "neutral" in out:
        print("  neutral md5 == dad49ea524c9c28e16b9195e0e66a7c1 : %s"
              % (out["neutral"][0] == "dad49ea524c9c28e16b9195e0e66a7c1"))
    for k in out:
        if k != "neutral":
            d = np.abs(out[k][1].astype(int) - out["neutral"][1].astype(int))
            print("  %-11s vs neutral: md5 differs=%s  changed_px=%d/%d (%.1f%%)  maxdiff=%d"
                  % (k, out[k][0] != out["neutral"][0],
                     int((d.sum(axis=-1) > 0).sum()), d.shape[0] * d.shape[1],
                     100.0 * (d.sum(axis=-1) > 0).mean(), int(d.max())))


if __name__ == "__main__":
    main()
