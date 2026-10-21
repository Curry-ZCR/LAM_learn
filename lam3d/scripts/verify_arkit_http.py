"""One-shot HTTP verification of the ARKit covariance-modulation fix.

Submits a job, waits for it, then exercises the three ARKit cases through the
real HTTP API and prints hashes + latencies.  Retries once if a concurrent job
from another process evicts the live session (HTTP 409).
"""
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from PIL import Image

BASE = "http://127.0.0.1:8000"
IMAGE = "/workspace/LAM/assets/sample_input/messi.png"


def req(method, path, body=None, files=None, timeout=120):
    if files:
        boundary = uuid.uuid4().hex
        buf = io.BytesIO()
        for name, val in files.items():
            if isinstance(val, tuple):          # file part
                fn, data, ctype = val
                buf.write(f"--{boundary}\r\n".encode())
                buf.write(
                    f'Content-Disposition: form-data; name="{name}"; filename="{fn}"\r\n'.encode()
                )
                buf.write(f"Content-Type: {ctype}\r\n\r\n".encode())
                buf.write(data)
                buf.write(b"\r\n")
            else:                               # plain form field
                buf.write(f"--{boundary}\r\n".encode())
                buf.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
                buf.write(val.encode() if isinstance(val, str) else val)
                buf.write(b"\r\n")
        buf.write(f"--{boundary}--\r\n".encode())
        data = buf.getvalue()
        r = urllib.request.Request(BASE + path, data=data, method=method)
        r.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    else:
        data = None
        if body is not None:
            data = json.dumps(body).encode()
        r = urllib.request.Request(BASE + path, data=data, method=method)
        if body is not None:
            r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def png_stats(png):
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    return a


def frame(job_id, arkit, size=320, tries=3):
    last = None
    for _ in range(tries):
        t0 = time.perf_counter()
        code, body = req("POST", f"/api/jobs/{job_id}/arkit_frame",
                         {"arkit": list(map(float, arkit)), "size": size})
        dt = (time.perf_counter() - t0) * 1e3
        last = (code, body, dt)
        if code == 200:
            return code, body, dt
        time.sleep(2)
    return last


def main():
    with open(IMAGE, "rb") as f:
        raw = f.read()
    print("== submitting job (messi.png + Anti_Drugs) ==", flush=True)
    code, body = req("POST", "/api/generate", files={
        "motion": "Anti_Drugs",
        "sample": "messi.png",
        "skip_video": "true",
        "image": ("messi.png", raw, "image/png"),
    })
    print("  submit ->", code, body[:200])
    job = json.loads(body)["id"]
    print("  job_id =", job, flush=True)

    t0 = time.time()
    while True:
        code, body = req("GET", f"/api/jobs/{job}")
        st = json.loads(body)["state"]
        if st in ("succeeded", "failed", "canceled"):
            print("  state = %s after %.1f s" % (st, time.time() - t0), flush=True)
            break
        if time.time() - t0 > 300:
            print("  TIMEOUT waiting for job")
            return
        time.sleep(3)

    code, body = req("GET", f"/api/jobs/{job}/arkit_session")
    print("== session ==", code, body.decode())

    z = [0.0] * 52
    nn = list(z); nn[43] = 0.8
    ones = [1.0] * 52
    cases = [("neutral (all-zero)", z), ("non-neutral v[43]=0.8", nn),
             ("all-ones (52x1.0)", ones)]
    hashes = {}
    for name, v in cases:
        code, png, dt = frame(job, v)
        if code != 200:
            print("  %-24s -> HTTP %s  %s" % (name, code, png[:200]))
            continue
        md5 = hashlib.md5(png).hexdigest()
        a = png_stats(png)
        hashes[name] = md5
        print("  %-24s -> 200  %7.1f ms  %6d bytes  md5=%s  "
              "img[min=%d max=%d mean=%.2f std=%.2f black=%d]"
              % (name, dt, len(png), md5, a.min(), a.max(), a.mean(), a.std(),
                 int((a.sum(axis=-1) == 0).sum())))
        with open("/tmp/final_%s.png" % name.split()[0], "wb") as f:
            f.write(png)

    print("\n== comparisons ==")
    if "neutral (all-zero)" in hashes:
        print("  neutral md5 == pre-fix baseline dad49ea524c9c28e16b9195e0e66a7c1 : %s"
              % (hashes["neutral (all-zero)"] == "dad49ea524c9c28e16b9195e0e66a7c1"))
    for k in hashes:
        if k != "neutral (all-zero)":
            print("  %-24s differs from neutral : %s"
                  % (k, hashes[k] != hashes["neutral (all-zero)"]))


if __name__ == "__main__":
    main()
