"""
verify.py -- measurement, not eyeballing.

  1. intro evenness      CV of per-tile reveal fraction during the fade (~0.05 good, ~0.7 patchy)
  2. straight boundaries fraction of band-boundary edges on long axis-aligned runs
  3. band / dot stats, ink coverage, file sizes, XML validity
  4. real browser frames (headless Chrome, SMIL paused via setCurrentTime) + numeric checks
"""
import os, subprocess, sys, xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

GW, GH = 300, 340
RNG = np.random.default_rng(0)
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def evenness(dots, groups, starts_order, tile=20, min_dots=20):
    tx, ty = dots[:, 0] // tile, dots[:, 1] // tile
    tid = ty * 100 + tx
    uniq, inv = np.unique(tid, return_inverse=True)
    tot = np.bincount(inv)
    ok = tot >= min_dots
    rank = np.empty(groups.max() + 1, int)
    rank[starts_order] = np.arange(len(starts_order))
    r = rank[groups]
    cvs = []
    n = len(starts_order)
    for k in range(int(n * 0.1), int(n * 0.9)):
        shown = np.bincount(inv, weights=(r < k), minlength=len(uniq))
        f = shown[ok] / tot[ok]
        cvs.append(f.std() / f.mean())
    return float(np.mean(cvs))


def straight_fraction(dots, labels, run=12):
    tree = cKDTree(dots)
    yy, xx = np.mgrid[0:GH, 0:GW]
    dist, idx = tree.query(np.column_stack([xx.ravel(), yy.ravel()]))
    L = labels[idx].reshape(GH, GW)
    near = (dist <= 2.0).reshape(GH, GW)      # only where dots actually are
    vb = (L[:, 1:] != L[:, :-1]) & near[:, 1:] & near[:, :-1]
    hb = (L[1:, :] != L[:-1, :]) & near[1:, :] & near[:-1, :]

    def long_runs(b):                   # b: boolean, runs along axis 0
        cnt = 0
        for col in b.T:
            padded = np.r_[0, col.astype(np.int8), 0]
            d = np.diff(padded)
            s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            lens = e - s
            cnt += lens[lens >= run].sum()
        return cnt

    total = vb.sum() + hb.sum()
    straight = long_runs(vb) + long_runs(hb.T)
    return straight / total


def grid_labels(dots, k):
    side = int(np.sqrt(k))
    cx = np.minimum(dots[:, 0] * side // GW, side - 1)
    cy = np.minimum(dots[:, 1] * side // GH, side - 1)
    return cy * side + cx


def frames(mode, times):
    svg = open(f"out/{mode}.svg", encoding="utf-8").read()
    bg = "#0d1117" if mode == "dark" else "#ffffff"
    html = (f"<!doctype html><html><body style='margin:0;background:{bg}'>{svg}"
            "<script>const s=document.querySelector('svg');s.pauseAnimations();"
            "s.setCurrentTime(parseFloat(location.hash.slice(1)));</script></body></html>")
    path = os.path.abspath(f"out/_frame_{mode}.html")
    open(path, "w", encoding="utf-8").write(html)
    outs = []
    for t in times:
        png = os.path.abspath(f"out/frame_{mode}_{t:05.2f}.png")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=1", f"--screenshot={png}",
                        "--window-size=1180,610", "--virtual-time-budget=1500",
                        f"file:///{path.replace(os.sep, '/')}#{t}"],
                       check=True, capture_output=True, timeout=90)
        outs.append(png)
    return outs


def portrait_crop(png, mode):
    a = np.asarray(Image.open(png).convert("RGB")).astype(float)
    crop = a[94:94 + 476, 34:34 + 420]
    ref = np.array([167, 139, 250] if mode == "dark" else [124, 58, 237], float)
    # "ink" = closeness to the portrait hue
    d = np.linalg.norm(crop - ref, axis=2)
    return d < 90


def expected_portrait(mode):
    dots = np.load(f"data/dots_{mode}.npy")
    g = np.zeros((GH, GW), bool)
    g[dots[:, 1], dots[:, 0]] = True
    return np.asarray(Image.fromarray(g).resize((420, 476), Image.NEAREST))


def blur_corr(a, b, k=7):
    from scipy.ndimage import uniform_filter
    a = uniform_filter(a.astype(float), k)
    b = uniform_filter(b.astype(float), k)
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main():
    sys.path.insert(0, ".")
    import build_banner as bb
    for mode in ("dark", "light"):
        dots = np.load(f"data/dots_{mode}.npy")
        bands = np.load(f"data/band_{mode}.npy")
        intro = np.load(f"data/intro_{mode}.npy")
        print(f"\n===== {mode} =====")
        print(f"dots={len(dots)}  coverage={len(dots) / (GW * GH):.3f}")
        sizes = np.bincount(bands)
        print(f"bands={len(sizes)}  size min/med/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}")

        # 1) evenness (group id == reveal order)
        n = intro.max() + 1
        order = np.arange(n)
        e = evenness(dots, intro, order)
        _, spatial = kmeans2(dots.astype(float), n, minit="++", seed=1)
        e_bad = evenness(dots, spatial, np.arange(n))
        e_rand = evenness(dots, RNG.integers(n, size=len(dots)), np.arange(n))
        print(f"intro evenness     = {e:.3f}   (pure random {e_rand:.3f}, spatial-group baseline {e_bad:.3f})")

        # 2) straight boundaries
        s = straight_fraction(dots, bands)
        s_grid = straight_fraction(dots, grid_labels(dots, 94))
        _, km0 = kmeans2(dots.astype(float), 94, minit="++", seed=11)
        s_km0 = straight_fraction(dots, km0)
        print(f"straight-boundary  = {s:.3f}   (quantized grid {s_grid:.3f}, k-means no noise {s_km0:.3f})")

        # 3) file
        p = f"out/{mode}.svg"
        ET.parse(p)
        print(f"svg valid XML, {os.path.getsize(p) / 1024:.0f} KB")

    # 4) browser frames
    I = bb.INTRO
    times = {
        "intro_0.6": 0.6, "intro_1.4": 1.4, "intro_end": 3.1,
        "portrait": I + 1.5, "dissolve_mid": I + 3.65,
        "L1": I + 5.3, "L1toL2": I + 6.95, "L2": I + 8.6, "L3": I + 11.9, "return": I + 13.55,
        "loop2_portrait": I + bb.LOOP + 1.5,
    }
    logos = np.load("data/logo_masks.npy")
    for mode in ("dark", "light"):
        print(f"\n----- browser frames: {mode} -----")
        pngs = frames(mode, list(times.values()))
        exp = expected_portrait(mode)
        for (name, t), png in zip(times.items(), pngs):
            ink = portrait_crop(png, mode)
            line = f"{name:15s} t={t:6.2f}s ink={ink.mean():.4f}"
            line += f"  corr(portrait)={blur_corr(ink, exp):+.3f}"
            for li, ln in zip(range(3), ("L1", "L2", "L3")):
                lm = np.asarray(Image.fromarray(logos[li]).resize((420, 476), Image.NEAREST))
                line += f"  corr({ln})={blur_corr(ink, lm, 15):+.2f}"
            print(line)


if __name__ == "__main__":
    main()
