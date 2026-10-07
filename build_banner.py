"""
build_banner.py -- generates out/dark.svg and out/light.svg from data/*.npy.
All geometry, leaders and textLength values are computed here. Never hand-edit the SVG.
"""
import numpy as np
from xml.sax.saxutils import escape

W, H = 1180, 610
GW, GH = 300, 340
SCALE = 1.4                 # grid unit -> px  (420 x 476 portrait)
PORTRAIT_XY = (34, 94)

# ---- timeline (seconds) --------------------------------------------------
INTRO = 3.2
LOOP = 14.2                 # portrait 3.0 | 1.3 | L1 2.0 | 1.3 | L2 2.0 | 1.3 | L3 2.0 | 1.3
P_HOLD, TR, L_HOLD = 3.0, 1.3, 2.0
t_l1_in = P_HOLD + TR                   # 4.3   travellers fully on L1
t_l1_out = t_l1_in + L_HOLD             # 6.3
t_l2_in = t_l1_out + TR                 # 7.6
t_l2_out = t_l2_in + L_HOLD             # 9.6
t_l3_in = t_l2_out + TR                 # 10.9
t_l3_out = t_l3_in + L_HOLD             # 12.9  then 1.3s back to portrait = 14.2
assert abs(t_l3_out + TR - LOOP) < 1e-9
DRIFT = 0.42
EASE = "0.45 0 0.25 1"

def kt(*ts):
    return ";".join(f"{t / LOOP:.4f}".rstrip("0").rstrip(".") if t else "0" for t in ts)

PALETTE = {
    "dark": dict(page="#0A101F", win="#0A101F", bar="#111a2e", text="#E2E8F0",
                 dim="#3b4a63", muted="#94a3b8", chrome="#22D3EE", portrait="#A78BFA",
                 accent="#10B981", pill_text="#0A101F", live="#EF4444"),
    "light": dict(page="#FFFFFF", win="#F8FAFC", bar="#EAF0F6", text="#0F172A",
                  dim="#b6c2d1", muted="#475569", chrome="#0891B2", portrait="#7C3AED",
                  accent="#10B981", pill_text="#FFFFFF", live="#DC2626"),
}

FONT = "'JetBrains Mono','Fira Code','Cascadia Code','SF Mono',Consolas,'DejaVu Sans Mono',monospace"

ROWS = [
    [("Subject", "Kundurthi Abhiram"),
     ("Role", "AI & Data Science Engineer | ML/DL Developer"),
     ("Origin", "Faridabad, India"),
     ("Education", "B.Tech AI & DS @ Amrita Vishwa Vidyapeetham"),
     ("Status", "Building + Learning + Shipping"),
     ("ToolChain", "VS Code · Jupyter · Colab · Git · Vertex AI · MATLAB")],
    [("Core.Lang", "Python · Java · C · C++ · R · HTML"),
     ("Core.Frontend", "HTML · CSS · Streamlit · Flutter"),
     ("Core.Backend", "Python · Flask · REST APIs"),
     ("Core.Database", "SQL · Relational Databases"),
     ("Core.Infra", "GitHub · Google Cloud · Vertex AI · MLOps")],
    [("Grid.Mail", "abhi8904876457@gmail.com"),
     ("Grid.Portfolio", "ka-dev-personal-portfolio.netlify.app"),
     ("Grid.LinkedIn", "in/kundurthi-abhiram-287b57291"),
     ("Grid.GitHub", "github.com/Abhiram-k1"),
     ("Grid.Facebook", "fb.com/profile.php?id=100024182708886")],
]

ROW_FS, HEAD_FS, LIVE_FS, PILL_FS, ROW_DY = 14, 13, 12, 14, 23
CW = 0.6                    # monospace advance, em


# ---- portrait paths ------------------------------------------------------
def runs_path(dots):
    """Dots -> one <path> d of horizontal runs, relative moves between subpath starts."""
    if len(dots) == 0:
        return ""
    d = dots[np.lexsort((dots[:, 0], dots[:, 1]))]
    out, px, py = [], 0, 0
    i, n = 0, len(d)
    while i < n:
        x, y = int(d[i, 0]), int(d[i, 1])
        j = i
        while j + 1 < n and d[j + 1, 1] == y and d[j + 1, 0] == d[j, 0] + 1:
            j += 1
        w = j - i + 1
        out.append(f"m{x - px} {y - py}h{w}v1h-{w}z")
        px, py = x, y
        i = j + 1
    s = "".join(out)
    return "M" + s[1:]  # first move absolute


def portrait_layers(dots, bands, intro, centroid):
    # --- intro layer: ~60 interleaved groups (id == reveal order) fading in over ~2s
    n_intro = intro.max() + 1
    starts = 0.15 + 1.75 * np.arange(n_intro) / (n_intro - 1)
    fade = 0.55
    intro_parts = []
    for g in range(n_intro):
        d = runs_path(dots[intro == g])
        t = starts[g]
        intro_parts.append(
            f'<path d="{d}"><animate attributeName="opacity" values="0;0;1" '
            f'keyTimes="0;{t / (t + fade):.3f};1" dur="{t + fade:.3f}s" fill="freeze"/></path>')
    intro_svg = ('<g id="portrait-intro">'
                 f'<set attributeName="opacity" to="0" begin="{INTRO}s" fill="freeze"/>'
                 + "".join(intro_parts) + "</g>")

    # --- loop layer: ~94 drift bands
    keys = kt(0, P_HOLD, t_l1_in, t_l3_out, LOOP)
    splines = ";".join([EASE] * 4)
    loop_parts = []
    for b in range(bands.max() + 1):
        sel = dots[bands == b]
        if len(sel) == 0:
            continue
        c = sel.mean(0) + 0.5
        dx, dy = DRIFT * (centroid - c)
        loop_parts.append(
            f'<path d="{runs_path(sel)}">'
            f'<animateTransform attributeName="transform" type="translate" '
            f'values="0 0;0 0;{dx:.1f} {dy:.1f};{dx:.1f} {dy:.1f};0 0" keyTimes="{keys}" '
            f'calcMode="spline" keySplines="{splines}" dur="{LOOP}s" begin="{INTRO}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="1;1;0;0;1" keyTimes="{keys}" '
            f'calcMode="spline" keySplines="{splines}" dur="{LOOP}s" begin="{INTRO}s" repeatCount="indefinite"/>'
            f'</path>')
    loop_svg = ('<g id="portrait-loop">'
                f'<set attributeName="opacity" to="0" begin="0s" dur="{INTRO}s"/>'
                + "".join(loop_parts) + "</g>")
    return intro_svg, loop_svg


def travellers_svg(trav, size=2.0):
    keys = kt(0, t_l1_out, t_l2_in, t_l2_out, t_l3_in, LOOP)
    splines = ";".join(["0 0 1 1", EASE, "0 0 1 1", EASE, "0 0 1 1"])
    h = size / 2
    parts = []
    for i in range(trav.shape[1]):
        (x1, y1), (x2, y2), (x3, y3) = trav[0, i], trav[1, i], trav[2, i]
        v = (f"{x1:.1f} {y1:.1f};{x1:.1f} {y1:.1f};{x2:.1f} {y2:.1f};"
             f"{x2:.1f} {y2:.1f};{x3:.1f} {y3:.1f};{x3:.1f} {y3:.1f}")
        parts.append(
            f'<rect x="-{h}" y="-{h}" width="{size}" height="{size}" transform="translate({x1:.1f} {y1:.1f})">'
            f'<animateTransform attributeName="transform" type="translate" values="{v}" '
            f'keyTimes="{keys}" calcMode="spline" keySplines="{splines}" dur="{LOOP}s" '
            f'begin="{INTRO}s" repeatCount="indefinite"/></rect>')
    okeys = kt(0, P_HOLD, t_l1_in, t_l3_out, LOOP)
    return ('<g id="travellers" opacity="0">'
            f'<animate attributeName="opacity" values="0;0;1;1;0" keyTimes="{okeys}" '
            f'dur="{LOOP}s" begin="{INTRO}s" repeatCount="indefinite"/>'
            + "".join(parts) + "</g>")


# ---- text helpers --------------------------------------------------------
def tlen(s, fs):
    return len(s) * CW * fs


def text(x, y, s, fs, fill, anchor="start", weight=None, extra=""):
    w = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{fs}" fill="{fill}" text-anchor="{anchor}"{w} '
            f'textLength="{tlen(s, fs):.1f}" lengthAdjust="spacingAndGlyphs"{extra}>{escape(s)}</text>')


def info_row(x0, x1, y, label, value, p):
    fs = ROW_FS
    cw = CW * fs
    out = []
    if "." in label:
        pre, post = label.split(".", 1)
        out.append(f'<text x="{x0}" y="{y}" font-size="{fs}" textLength="{tlen(label, fs):.1f}" '
                   f'lengthAdjust="spacingAndGlyphs"><tspan fill="{p["muted"]}">{pre}.</tspan>'
                   f'<tspan fill="{p["chrome"]}">{escape(post)}</tspan></text>')
    else:
        out.append(text(x0, y, label, fs, p["chrome"]))
    out.append(text(x1, y, value, fs, p["text"], anchor="end"))
    ls = x0 + tlen(label, fs) + cw
    le = x1 - tlen(value, fs) - cw
    n = int((le - ls) // cw)
    assert n >= 3, f"row too long: {label}={value}"
    lead = "." * n
    out.append(f'<text x="{ls:.1f}" y="{y}" font-size="{fs}" fill="{p["dim"]}" '
               f'textLength="{le - ls:.1f}" lengthAdjust="spacingAndGlyphs">{lead}</text>')
    return "".join(out)


# ---- assemble --------------------------------------------------------------
def build(mode):
    p = PALETTE[mode]
    dots = np.load(f"data/dots_{mode}.npy")
    bands = np.load(f"data/band_{mode}.npy")
    intro = np.load(f"data/intro_{mode}.npy")
    trav = np.load("data/travellers.npy")
    centroid = trav[0].mean(0)

    intro_svg, loop_svg = portrait_layers(dots, bands, intro, centroid)
    trav_svg = travellers_svg(trav)

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'font-family="{FONT}">')
    s.append(f'<title>Kundurthi Abhiram — profile.sh --live</title>')
    # window
    s.append(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="14" fill="{p["win"]}" '
             f'stroke="{p["chrome"]}" stroke-opacity="0.45"/>')
    s.append(f'<path d="M0.5 38V14.5a14 14 0 0 1 14-14h{W - 29}a14 14 0 0 1 14 14V38z" fill="{p["bar"]}"/>')
    s.append(f'<line x1="0.5" y1="38" x2="{W - 0.5}" y2="38" stroke="{p["chrome"]}" stroke-opacity="0.3"/>')
    for i, c in enumerate(("#FF5F57", "#FEBC2E", "#28C840")):
        s.append(f'<circle cx="{22 + 20 * i}" cy="19" r="6" fill="{c}"/>')
    s.append(text(W / 2, 24, "profile.sh --live", 13, p["muted"], anchor="middle"))

    # left: VISUAL.MAP frame
    fx, fy, fw, fh = 22, 54, 444, 536
    s.append(f'<rect x="{fx}" y="{fy}" width="{fw}" height="{fh}" rx="8" fill="none" '
             f'stroke="{p["chrome"]}" stroke-opacity="0.35"/>')
    s.append(f'<rect x="{fx + 14}" y="{fy - 8}" width="{tlen("VISUAL.MAP", 12) + 16:.1f}" height="16" fill="{p["win"]}"/>')
    s.append(text(fx + 22, fy + 4, "VISUAL.MAP", 12, p["chrome"]))
    px, py = PORTRAIT_XY
    s.append(f'<g transform="translate({px} {py}) scale({SCALE})" fill="{p["portrait"]}" '
             f'shape-rendering="crispEdges">')
    s.append(intro_svg)
    s.append(loop_svg)
    s.append(trav_svg)
    s.append("</g>")

    # right: SYSTEM.INFO
    rx0, rx1 = 500, 1144
    s.append(f'<rect x="484" y="{fy}" width="{W - 22 - 484}" height="{fh}" rx="8" fill="none" '
             f'stroke="{p["chrome"]}" stroke-opacity="0.35"/>')
    s.append(text(rx0, 84, "SYSTEM.INFO", HEAD_FS, p["chrome"], weight="700"))
    # LIVE badge (pulsing)
    lw = 66
    lx = rx1 - lw
    s.append(f'<rect x="{lx}" y="68" width="{lw}" height="22" rx="11" fill="{p["live"]}" fill-opacity="0.14" '
             f'stroke="{p["live"]}" stroke-opacity="0.7"/>')
    s.append(f'<circle cx="{lx + 15}" cy="79" r="4" fill="{p["live"]}">'
             f'<animate attributeName="opacity" values="1;0.2;1" dur="1.4s" repeatCount="indefinite"/>'
             f'<animate attributeName="r" values="4;5.5;4" dur="1.4s" repeatCount="indefinite"/></circle>')
    s.append(text(lx + 27, 83.5, "LIVE", LIVE_FS, p["live"], weight="700"))
    # handle pill
    handle = "@Abhiram-k1"
    pw = tlen(handle, PILL_FS) + 28
    s.append(f'<rect x="{rx0}" y="100" width="{pw:.1f}" height="26" rx="13" fill="{p["accent"]}"/>')
    s.append(text(rx0 + 14, 118, handle, PILL_FS, p["pill_text"], weight="700"))
    s.append(f'<line x1="{rx0}" y1="140" x2="{rx1}" y2="140" stroke="{p["chrome"]}" stroke-opacity="0.25" '
             f'stroke-dasharray="2 4"/>')

    y = 166
    for si, section in enumerate(ROWS):
        for label, value in section:
            s.append(info_row(rx0, rx1, y, label, value, p))
            y += ROW_DY
        if si < len(ROWS) - 1:
            s.append(f'<line x1="{rx0}" y1="{y - 9}" x2="{rx1}" y2="{y - 9}" stroke="{p["chrome"]}" '
                     f'stroke-opacity="0.18" stroke-dasharray="2 4"/>')
            y += 14
    # prompt + blinking cursor
    prompt = "abhiram@profile:~$"
    s.append(text(rx0, 572, prompt, ROW_FS, p["accent"]))
    cx = rx0 + tlen(prompt, ROW_FS) + 8
    s.append(f'<rect x="{cx:.1f}" y="560" width="8" height="15" fill="{p["accent"]}">'
             f'<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;0.5;0.5;1" dur="1.1s" '
             f'repeatCount="indefinite"/></rect>')
    s.append("</svg>")
    svg = "".join(s)
    with open(f"out/{mode}.svg", "w", encoding="utf-8") as f:
        f.write(svg)
    with open(f"assets/{mode}.svg", "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"{mode}.svg  {len(svg.encode()) / 1024:.0f} KB  dots={len(dots)}  bands={bands.max() + 1}  "
          f"intro_groups={intro.max() + 1}  last_row_y={y - ROW_DY}")


if __name__ == "__main__":
    build("dark")
    build("light")
