"""
preprocess.py -- source of truth for the banner portrait + logo data.

Outputs (data/*.npy):
  dots_dark.npy / dots_light.npy   (N,2) int  grid coords (x,y) of 1-bit dots, 300x340 grid
  mask_dark.npy                    (340,300) bool subject mask used for dark mode
  band_dark.npy / band_light.npy   (N,) int  drift band id per dot (~94 bands)
  intro_dark.npy / intro_light.npy (N,) int  intro group id per dot (~60 groups)
  travellers.npy                   (3,900,2) float  traveller positions per logo, OT-matched
  logo_masks.npy                   (3,340,300) bool traced logo masks on the grid

Re-run this, then build_banner.py. Never hand-edit the SVG.
"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

GW, GH = 300, 340          # portrait grid
RNG = np.random.default_rng(7)
N_TRAVELLERS = 900
N_BANDS = 94
N_INTRO = 60
BAND_NOISE_SIGMA = 4.0
TARGET_DARK_DOTS = 17000

# head + shoulders crop of the 375x500 photo (full width, aspect 300:340)
CROP = (0, 20, 375, 20 + int(375 * GH / GW))


# ---------------------------------------------------------------- portrait
def load_portrait():
    im = Image.open("refs/photo.jpg").convert("RGB").crop(CROP)
    im = im.resize((GW, GH), Image.LANCZOS)
    return im


def tone(im):
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=1)
    g = ImageEnhance.Contrast(g).enhance(1.3)
    g = g.filter(ImageFilter.UnsharpMask(radius=3, percent=140, threshold=0))
    return np.asarray(g).astype(np.float64) / 255.0


def segment(im):
    """Subject mask: colour distance from wall colour -> close -> fill holes -> largest CC."""
    a = np.asarray(im).astype(np.float64)
    # The wall has a lighting gradient, so model it as a smooth (quadratic) colour field
    # fitted to wall seeds (top rows + side strips above the shoulders), then threshold
    # the colour distance from that field. Wall residual p99 ~6, shirt >= ~30.
    yy, xx = np.mgrid[0:GH, 0:GW]
    seed = np.zeros((GH, GW), bool)
    seed[:10] = True
    seed[:170, :8] = True
    seed[:170, -8:] = True
    basis = np.stack([np.ones_like(xx), xx, yy, xx ** 2, yy ** 2, xx * yy], -1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(basis[seed], a[seed], rcond=None)
    score = np.linalg.norm(a - basis @ coef, axis=2)
    # The head casts a shadow on the wall: same chroma as the wall but darker than the
    # quadratic predicts. Wall chroma ~(0.349, 0.333); shirt is neutral (r <= 0.341) but its
    # shadowed side gets warm, so the chroma rule only applies above the shoulder line.
    chroma = a[..., :2] / (a.sum(2, keepdims=True) + 1e-6)
    wall_c = np.median(chroma[seed], axis=0)
    shadow = ((np.linalg.norm(chroma - wall_c, axis=2) < 0.0045)
              & (a.mean(2) > 200) & (yy < 215))
    m = (score > 16) & ~shadow
    m = ndimage.binary_opening(m, iterations=1)
    m = ndimage.binary_closing(m, structure=np.ones((3, 3)), iterations=4)
    # subject touches the bottom edge -> pad bottom so fill_holes can close it
    pad = np.vstack([m, np.ones((1, GW), bool)])
    pad = ndimage.binary_fill_holes(pad)[:-1]
    lab, n = ndimage.label(pad)
    if n > 1:
        sizes = ndimage.sum(pad, lab, range(1, n + 1))
        pad = lab == (1 + int(np.argmax(sizes)))
    # strip thin wall slivers bridged by the closing, keep the main body
    disk = np.hypot(*np.mgrid[-3:4, -3:4]) <= 3.2
    pad = ndimage.binary_opening(pad, structure=disk)
    lab, n = ndimage.label(pad)
    if n > 1:
        sizes = ndimage.sum(pad, lab, range(1, n + 1))
        pad = lab == (1 + int(np.argmax(sizes)))
    return pad, score


def fs_dither(v, mask=None):
    """1-bit Floyd-Steinberg, serpentine. Error never diffuses into masked-out pixels."""
    f = v.copy()
    h, w = f.shape
    out = np.zeros((h, w), bool)
    for y in range(h):
        rev = y % 2 == 1
        d = -1 if rev else 1
        xs = range(w - 1, -1, -1) if rev else range(w)
        for x in xs:
            if mask is not None and not mask[y, x]:
                continue
            old = f[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = new > 0
            e = old - new
            for dx, dy, k in ((d, 0, 7), (-d, 1, 3), (0, 1, 5), (d, 1, 1)):
                xx, yy = x + dx, y + dy
                if 0 <= xx < w and yy < h and (mask is None or mask[yy, xx]):
                    f[yy, xx] += e * k / 16
    if mask is not None:
        out &= mask            # hard-clear any bleed at the mask edge
    return out


# ---------------------------------------------------------------- logos
def trace_logo(path, box=230, center=(150, 165)):
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im).astype(np.float64)
    rgb, alpha = a[..., :3], a[..., 3]
    corners = np.array([rgb[2, 2], rgb[2, -3], rgb[-3, 2], rgb[-3, -3]])
    bg = np.median(corners, axis=0)
    m = (np.linalg.norm(rgb - bg, axis=2) > 60) & (alpha > 128)
    # keep components bigger than 0.5% (drops jpeg/edge specks)
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.flatnonzero(sizes > 0.005 * m.sum()))
    ys, xs = np.nonzero(keep)
    keep = keep[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = keep.shape
    s = box / max(h, w)
    tw, th = max(1, round(w * s)), max(1, round(h * s))
    up = Image.fromarray((keep * 255).astype(np.uint8)).resize((tw, th), Image.BICUBIC)
    up = np.asarray(up) > 127
    grid = np.zeros((GH, GW), bool)
    x0, y0 = int(center[0] - tw / 2), int(center[1] - th / 2)
    grid[y0:y0 + th, x0:x0 + tw] = up
    return grid


def blue_noise_sample(mask, n, iters=25):
    """Lloyd relaxation of n points over the mask pixels -> even coverage."""
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float64) + 0.5
    c = pts[RNG.choice(len(pts), n, replace=False)]
    for _ in range(iters):
        _, idx = cKDTree(c).query(pts)
        cnt = np.bincount(idx, minlength=n)[:, None]
        sx = np.bincount(idx, pts[:, 0], minlength=n)
        sy = np.bincount(idx, pts[:, 1], minlength=n)
        ok = cnt[:, 0] > 0
        c[ok] = np.column_stack([sx, sy])[ok] / cnt[ok]
    return c


def ot_match(a, b):
    """Optimal transport (equal mass) = assignment on squared distance. Returns b reordered."""
    _, col = linear_sum_assignment(cdist(a, b, "sqeuclidean"))
    return b[col]


# ---------------------------------------------------------------- groups
def drift_bands(dots, k=N_BANDS, sigma=BAND_NOISE_SIGMA):
    noisy = dots + RNG.normal(0, sigma, dots.shape)
    _, lab = kmeans2(noisy, k, minit="++", seed=11, iter=30)
    # relabel compactly (kmeans2 can leave empty clusters)
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def intro_groups(dots, k=N_INTRO, tile=10):
    """Group id == reveal order (0 first). Assignment is per horizontal run (keeps runs
    mergeable) and stratified: inside every tile x tile patch the runs are spread evenly
    over all k reveal slots with a random phase, so each group is scattered over the whole
    portrait and every patch thickens at the same rate."""
    order = np.lexsort((dots[:, 0], dots[:, 1]))
    run_id = np.empty(len(dots), int)
    starts = []
    prev, r = None, -1
    for i in order:
        x, y = dots[i]
        if prev is None or y != prev[1] or x != prev[0] + 1:
            r += 1
            starts.append((x, y))
        run_id[i] = r
        prev = (x, y)
    starts = np.array(starts)
    # Hilbert-curve block stratification: walk runs along a Hilbert curve; every block of k
    # consecutive runs (a compact patch, larger where dots are sparse) gets each reveal slot
    # exactly once, in shuffled order. Even at every scale, any density.
    h = np.array([_hilbert_index(512, int(x), int(y)) for x, y in starts])
    seq = np.argsort(h, kind="stable")
    run_group = np.empty(len(starts), int)
    for b in range(0, len(seq), k):
        blk = seq[b:b + k]
        run_group[blk] = RNG.permutation(k)[:len(blk)]
    return run_group[run_id]


def _hilbert_index(n, x, y):
    d, s = 0, n // 2
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        s //= 2
    return d


def main():
    im = load_portrait()
    v = tone(im)
    mask, _ = segment(im)

    # light: dots draw the dark parts, background kept
    light = fs_dither(1.0 - v)
    # dark: dots draw the lit subject; density scaled to land near the dot budget
    lo, hi = 0.2, 1.0
    for _ in range(14):
        k = (lo + hi) / 2
        n = int(fs_dither(v * k, mask).sum())
        lo, hi = (k, hi) if n < TARGET_DARK_DOTS else (lo, k)
    dark = fs_dither(v * k, mask)
    print(f"dark density scale k={k:.3f}")

    logos = [trace_logo(f"refs/{n}.png") for n in
             ("logo_antigravity", "logo_streamlit", "logo_code")]
    L1 = blue_noise_sample(logos[0], N_TRAVELLERS)
    L2 = ot_match(L1, blue_noise_sample(logos[1], N_TRAVELLERS))
    L3 = ot_match(L2, blue_noise_sample(logos[2], N_TRAVELLERS))
    trav = np.stack([L1, L2, L3])

    np.save("data/mask_dark.npy", mask)
    np.save("data/logo_masks.npy", np.stack(logos))
    np.save("data/travellers.npy", trav)
    for name, bits in (("dark", dark), ("light", light)):
        ys, xs = np.nonzero(bits)
        dots = np.column_stack([xs, ys])
        np.save(f"data/dots_{name}.npy", dots)
        np.save(f"data/band_{name}.npy", drift_bands(dots.astype(float)))
        np.save(f"data/intro_{name}.npy", intro_groups(dots))
        print(f"{name}: {len(dots)} dots")

    # previews (numpy -> png), not used by the SVG
    Image.fromarray((~dark * 255).astype(np.uint8)).save("out/preview_dither_dark.png")
    Image.fromarray((~light * 255).astype(np.uint8)).save("out/preview_dither_light.png")
    ov = np.asarray(im).copy()
    ov[~mask] = (ov[~mask] * 0.25).astype(np.uint8)
    Image.fromarray(ov).save("out/preview_mask.png")
    lg = np.zeros((GH, GW * 3), np.uint8)
    for i, m in enumerate(logos):
        lg[:, i * GW:(i + 1) * GW] = m * 90
        for x, y in trav[i]:
            lg[int(y), i * GW + int(x)] = 255
    Image.fromarray(lg).save("out/preview_logos.png")


if __name__ == "__main__":
    main()
