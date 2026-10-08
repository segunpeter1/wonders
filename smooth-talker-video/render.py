"""Render the Smooth Talker short in a hand-drawn style.

Drawings are held "on twos" (12 per second at 24 fps), every drawing gets a
slightly different line wobble ("boil"), dance poses flip on the song's beat,
and the only audio is the original track.

Usage: python3 render.py <original_audio.wav> <output.mp4>
"""
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
W, H, FPS = 1080, 1920, 24
END = 25.2
SRC_W, SRC_H = 1536, 2752

# Beat grid measured from the song (113 BPM).
BEAT0, BEAT = 14.01, 0.53
BEATS = [BEAT0 + k * BEAT for k in range(int((END - BEAT0) / BEAT) + 2)]

SHOTS = [  # (name, start, end)
    ("shot01", 0.00, 1.85), ("shot02", 1.85, 4.40), ("shot03", 4.40, 6.40), ("shot04", 6.40, 7.30),
    ("shot05", 7.30, 10.60), ("shot06", 10.60, 13.75), ("shot07", 13.75, 16.40),
    ("brush", 16.40, 17.76), ("dougie", 17.76, 20.00), ("point", 20.00, 21.00),
    ("jade", 21.00, 22.90), ("together", 22.90, END),
]

CAPTIONS = [  # (words with start times, hide time)
    ([("TISSUE?", 0.86)], 1.90),
    ([("NO,", 1.90), ("BABY,", 2.58)], 3.08),
    ([("I'LL", 3.08), ("MISS", 3.66), ("YOU.", 3.98)], 4.70),
    ([("MMM,", 6.46), ("BABY.", 7.18)], 7.36),
    ([("SOMEONE", 7.36), ("ASKED", 7.54), ("ME", 7.84), ("IF", 8.02), ("I", 8.14), ("KNEW", 8.20)], 8.32),
    ([("WHAT", 8.32), ("IT", 8.46), ("FELT", 8.62), ("LIKE", 8.72)], 8.96),
    ([("FOR", 8.96), ("A", 9.18), ("DREAM", 9.26), ("TO", 9.36), ("COME", 9.56), ("TRUE.", 9.78)], 10.70),
    ([("I", 11.48), ("DIDN'T", 11.58), ("KNOW", 11.80), ("THE", 11.88), ("ANSWER", 12.02)], 12.40),
    ([("UNTIL", 12.40), ("THE", 12.60), ("DAY", 12.80), ("I", 12.98), ("MET", 13.16), ("YOU.", 13.56)], 14.40),
    ([("THEY", 16.42), ("BE", 16.62), ("LIKE,", 16.86), ("SMOOTH.", 17.06)], 17.76),
    ([("CAN", 17.76), ("YOU", 18.18), ("TEACH", 18.48), ("ME", 18.62), ("HOW", 18.88), ("TO", 19.04), ("DOUGIE?", 19.26)], 20.04),
    ([("YOU", 20.04), ("KNOW", 20.22), ("WHY?", 20.46)], 21.10),
    ([("'CAUSE", 21.10), ("ALL", 21.30), ("THE", 21.48), ("GIRLS", 21.68), ("LOVE", 21.90), ("ME.", 22.20)], 23.02),
    ([("ALL", 23.02), ("I", 23.20), ("NEED", 23.48), ("IS", 23.74), ("A", 24.02), ("BEAT", 24.14)], 24.24),
    ([("THAT'S", 24.24), ("SUPER", 24.60), ("BUMPIN'.", 24.76)], END + 1),
]

INK = (34, 28, 24)
MARKERS = [(255, 92, 141), (255, 200, 60), (70, 200, 190), (170, 130, 230)]

# ---------------------------------------------------------------- helpers


def clamp(x, a, b):
    return max(a, min(b, x))


def lerp(a, b, t):
    return a + (b - a) * t


def ease_io(t):
    t = clamp(t, 0, 1)
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = clamp(t, 0, 1)
    return 1 - (1 - t) ** 3


def kick(t, t0, decay):
    return 0.0 if t < t0 else math.exp(-(t - t0) / decay)


def beat_index(t):
    return int(math.floor((t - BEAT0) / BEAT)) if t >= BEAT0 else -1


def since_beat(t):
    return (t - BEAT0) % BEAT if t >= BEAT0 else 99


# ------------------------------------------------------------ boil + paper


def make_boil_maps(n=3, amp=2.2, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    maps = []
    for _ in range(n):
        dx = cv2.resize(rng.standard_normal((40, 24)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        dy = cv2.resize(rng.standard_normal((40, 24)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        maps.append((xs + dx * amp, ys + dy * amp))
    return maps


def make_paper(seed=1):
    rng = np.random.default_rng(seed)
    fine = rng.normal(0, 1, (H, W)).astype(np.float32)
    coarse = cv2.resize(rng.normal(0, 1, (60, 34)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    tex = 1 + 0.012 * cv2.GaussianBlur(fine, (0, 0), 0.8) + 0.012 * coarse
    return tex[..., None]


# ---------------------------------------------------------------- doodles


def jit(rng, a=2.5):
    return rng.uniform(-a, a)


def sketch_poly(d, pts, rng, color, width=6, closed=True, passes=2):
    for p in range(passes):
        q = [(x + jit(rng), y + jit(rng)) for x, y in pts]
        if closed:
            q.append(q[0])
        d.line(q, fill=color, width=width if p == 0 else max(2, width // 2), joint="curve")


def doodle_heart(d, x, y, r, rng, color=(235, 70, 110), fill=True):
    pts = []
    for i in range(28):
        t = 2 * math.pi * i / 28
        pts.append((x + r * math.sin(t) ** 3, y - r * (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)) / 16))
    if fill:
        d.polygon([(px + jit(rng, 1.5), py + jit(rng, 1.5)) for px, py in pts], fill=color + (230,))
    sketch_poly(d, pts, rng, INK + (255,), width=max(3, int(r / 7)))


def doodle_star(d, x, y, r, rng, color=(255, 210, 60)):
    pts = []
    for i in range(10):
        rr = r if i % 2 == 0 else r * 0.45
        a = -math.pi / 2 + i * math.pi / 5
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    d.polygon([(px + jit(rng, 1.5), py + jit(rng, 1.5)) for px, py in pts], fill=color + (240,))
    sketch_poly(d, pts, rng, INK + (255,), width=max(3, int(r / 8)))


def doodle_sparkle(d, x, y, r, rng, color=INK):
    for a in (0, math.pi / 2):
        x0, y0 = x + r * math.cos(a), y + r * math.sin(a)
        x1, y1 = x - r * math.cos(a), y - r * math.sin(a)
        d.line((x0 + jit(rng), y0 + jit(rng), x1 + jit(rng), y1 + jit(rng)), fill=color + (255,), width=max(3, int(r / 6)))
    for a in (math.pi / 4, 3 * math.pi / 4):
        rr = r * 0.45
        d.line((x + rr * math.cos(a), y + rr * math.sin(a), x - rr * math.cos(a), y - rr * math.sin(a)),
               fill=color + (255,), width=max(2, int(r / 10)))


def doodle_note(d, x, y, r, rng):
    d.ellipse((x - r * 0.6 + jit(rng), y - r * 0.4, x + r * 0.6, y + r * 0.4 + jit(rng)), fill=INK + (255,))
    d.line((x + r * 0.55, y, x + r * 0.55 + jit(rng), y - r * 2), fill=INK + (255,), width=max(3, int(r / 4)))
    d.line((x + r * 0.55, y - r * 2, x + r * 1.3 + jit(rng), y - r * 1.5), fill=INK + (255,), width=max(3, int(r / 4)))


def doodle_lines(d, x, y, r, ang, rng, n=3):
    """Little motion/emphasis strokes radiating at angle ang."""
    for k in range(n):
        a = ang + (k - (n - 1) / 2) * 0.35
        r0, r1 = r, r * 1.7
        d.line((x + r0 * math.cos(a) + jit(rng), y + r0 * math.sin(a) + jit(rng),
                x + r1 * math.cos(a) + jit(rng), y + r1 * math.sin(a) + jit(rng)), fill=INK + (255,), width=6)


# ------------------------------------------------------------- dance stage


def key_green(img):
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    k = g - np.maximum(r, b)
    alpha = 1 - np.clip((k - 10) / 22.0, 0, 1)
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 0.8)
    rgb = a.copy()
    rgb[..., 1] = np.minimum(g, np.maximum(r, b) + 6)  # despill
    return np.dstack([rgb.clip(0, 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])


def stage_background(beat, t, rng):
    """Cream paper with a sketchy sunburst whose marker colour changes every beat."""
    col = MARKERS[beat % len(MARKERS)]
    col2 = MARKERS[(beat + 1) % len(MARKERS)]
    img = Image.new("RGB", (W, H), (250, 243, 228))
    d = ImageDraw.Draw(img, "RGBA")
    cx, cy = W / 2, H * 0.45
    rot = t * 0.25
    n = 16
    for i in range(n):
        a0 = rot + i * 2 * math.pi / n
        a1 = a0 + math.pi / n
        L = 2200
        pts = [(cx + jit(rng, 4), cy + jit(rng, 4)), (cx + L * math.cos(a0), cy + L * math.sin(a0)),
               (cx + L * math.cos(a1), cy + L * math.sin(a1))]
        d.polygon(pts, fill=(col if i % 2 == 0 else col2) + (150,))
    # hatching strokes over the burst, like marker texture
    for _ in range(40):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        d.line((x, y, x + rng.uniform(40, 120), y - rng.uniform(20, 60)), fill=(255, 255, 255, 60), width=6)
    # a sketchy circle spotlight behind the dancer
    d.ellipse((cx - 470, cy - 470, cx + 470, cy + 470), fill=(250, 243, 228, 200))
    for k in range(2):
        e = 470 + jit(rng, 8)
        d.ellipse((cx - e + jit(rng), cy - e + jit(rng), cx + e, cy + e), outline=INK + (255,), width=7 - k * 3)
    return np.asarray(img, np.float32)


def stage_doodles(layer, beat, rng, kinds=("star", "heart", "note", "sparkle")):
    d = ImageDraw.Draw(layer)
    spots = [(120, 300), (950, 260), (90, 900), (990, 820), (160, 1650), (930, 1600), (540, 140)]
    brng = np.random.default_rng(beat)
    for i, (x, y) in enumerate(spots):
        kind = kinds[(i + beat) % len(kinds)]
        x, y = x + brng.uniform(-40, 40), y + brng.uniform(-40, 40)
        r = brng.uniform(38, 60)
        if kind == "star":
            doodle_star(d, x, y, r, rng, MARKERS[(i + beat) % 4])
        elif kind == "heart":
            doodle_heart(d, x, y, r * 0.9, rng)
        elif kind == "note":
            doodle_note(d, x, y, r * 0.5, rng)
        else:
            doodle_sparkle(d, x, y, r, rng)


def place_cutout(frame, cut, t, scale=1.0, tilt=0.0, dx=0, dy=0):
    """Alpha-composite a keyed full-frame cutout with a beat squash."""
    sb = since_beat(t)
    sq = 0.06 * math.exp(-sb / 0.09) if sb < 0.4 else 0.0
    sx, sy = 0.703 * scale * (1 + sq * 0.6), 0.703 * scale * (1 - sq)
    # anchor at the feet (bottom centre of the source)
    fx, fy = SRC_W / 2, SRC_H * 0.93
    ox, oy = W / 2 + dx, H * 0.93 + dy
    a = math.radians(tilt)
    m = np.array([[sx * math.cos(a), -sy * math.sin(a), 0], [sx * math.sin(a), sy * math.cos(a), 0]], np.float32)
    m[:, 2] = np.array([ox, oy]) - m[:, :2] @ np.array([fx, fy])
    warped = cv2.warpAffine(cut, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    al = warped[..., 3:4].astype(np.float32) / 255
    frame[:] = frame * (1 - al) + warped[..., :3].astype(np.float32) * al


# ---------------------------------------------------------------- camera


def cam_frame(img, cx, cy, zoom, rot=0.0, sx=0.0, sy=0.0):
    vw = SRC_W / zoom
    vh = vw * H / W
    s = W / vw
    cx = clamp(cx - sx / s, vw / 2, SRC_W - vw / 2)
    cy = clamp(cy - sy / s, vh / 2, SRC_H - vh / 2)
    a = math.radians(rot)
    m = cv2.getRotationMatrix2D((cx, cy), rot, s)
    m[:, 2] += np.array([W / 2 - cx, H / 2 - cy])
    out = cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    def proj(px, py):
        v = m @ np.array([px, py, 1.0])
        return float(v[0]), float(v[1])

    return out.astype(np.float32), proj


def shake(t, t0, amp, decay, seed=0):
    k = kick(t, t0, decay) * amp
    return k * math.sin(t * 71 + seed), k * math.cos(t * 53 + seed)


# --------------------------------------------------------------- captions

FONT = ImageFont.truetype(str(HERE / "fonts" / "PermanentMarker.ttf"), 86)
CAP_Y = 1380
MAX_W = 930


def draw_caption(layer, t, rng):
    ci = None
    for i, (words, _) in enumerate(CAPTIONS):
        if t >= words[0][1]:
            ci = i
    if ci is None or t >= CAPTIONS[ci][1]:
        return
    words, _ = CAPTIONS[ci]
    active = max(i for i, (_, s) in enumerate(words) if t >= s)
    texts = [w for w, _ in words]
    space = FONT.getlength(" ")
    lines, cur, cw = [], [], 0
    for i, w in enumerate(texts):
        ww = FONT.getlength(w)
        if cur and cw + space + ww > MAX_W:
            lines.append(cur)
            cur, cw = [], 0
        cw += (space if cur else 0) + ww
        cur.append(i)
    lines.append(cur)
    lh = 104
    pop = ease_out((t - words[0][1]) / 0.12)
    y0 = CAP_Y - lh * len(lines) / 2 + (1 - pop) * 40
    d = ImageDraw.Draw(layer)
    for li, line in enumerate(lines):
        lw = sum(FONT.getlength(texts[i]) for i in line) + space * (len(line) - 1)
        x = (W - lw) / 2 + jit(rng, 1.5)
        y = y0 + li * lh + jit(rng, 1.5)
        for i in line:
            ww = FONT.getlength(texts[i])
            if i == active:
                k = ease_out((t - words[i][1]) / 0.1)
                hl = [(x - 14 + jit(rng, 3), y + 22 + jit(rng, 3)), (x - 10 + (ww + 28) * k, y + 14 + jit(rng, 3)),
                      (x - 8 + (ww + 28) * k + jit(rng, 3), y + 98), (x - 16, y + 104 + jit(rng, 3))]
                d.polygon(hl, fill=(255, 77, 141, 235))
            d.text((x, y), texts[i], font=FONT, fill=(255, 255, 255), stroke_width=7, stroke_fill=INK)
            x += ww + space


# ------------------------------------------------------------------- shots


def story(name, lt, gt, imgs, rng, layer):
    d = ImageDraw.Draw(layer)
    img = imgs[name]
    if name == "shot01":
        k = ease_io(lt / 1.85)
        sx, sy = shake(gt, 0.86, 30, 0.2, 1)
        f, proj = cam_frame(img, lerp(768, 900, k), lerp(1350, 1150, k), lerp(1.05, 1.18, k) + 0.08 * kick(gt, 0.86, 0.15), sx=sx, sy=sy)
        if gt >= 0.86:
            x, y = proj(1000, 830)
            doodle_lines(d, x, y, 90, -2.5, rng, 4)
            doodle_lines(d, x, y, 90, -1.0, rng, 3)
    elif name == "shot02":
        k = ease_io(lt / 2.55)
        f, proj = cam_frame(img, lerp(768, 880, k), lerp(1400, 1150, k), lerp(1.05, 1.3, k))
        for i, t0 in enumerate((3.66, 3.85, 4.05)):
            if gt >= t0:
                a = gt - t0
                x, y = proj(1050 + i * 70, 720)
                doodle_heart(d, x + 20 * math.sin(a * 4 + i), y - 160 * a - i * 30, 34 + 6 * i, rng)
    elif name == "shot03":
        k = ease_io(lt / 2.0)
        f, proj = cam_frame(img, 768, lerp(1300, 1120, k), lerp(1.08, 1.22, k))
        for i, t0 in enumerate((4.9, 5.3, 5.7)):
            if gt >= t0:
                d.ellipse((410 + i * 110 + jit(rng), 330 + jit(rng), 450 + i * 110, 370), fill=INK + (255,))
    elif name == "shot04":
        z = 1.12 + 0.05 * kick(gt, 6.46, 0.15) + 0.06 * kick(gt, 7.18, 0.15) + 0.04 * lt
        f, proj = cam_frame(img, 900, 1000, z)
        x, y = proj(1250, 650)
        doodle_sparkle(d, x, y, 46, rng)
        x, y = proj(520, 820)
        doodle_sparkle(d, x, y, 34, rng)
    elif name == "shot05":
        k = ease_io((lt - 0.4) / 2.0)
        f, proj = cam_frame(img, lerp(950, 860, k), lerp(1550, 900, k), lerp(1.35, 1.1, k))
        for i in range(4):
            a = (lt * 0.9 + i * 0.25) % 1.0
            x, y = proj(820 + 90 * math.sin(i * 2.1 + lt * 2), 1700 - 500 * a)
            if a > 0.05:
                doodle_heart(d, x, y, 24 + 6 * (i % 2), rng)
        if gt >= 9.26:
            x, y = proj(930, 560)
            for j in range(3):
                doodle_sparkle(d, x + (j - 1) * 260, y - 230 + 60 * (j % 2), 40 * ease_out((gt - 9.26) / 0.2), rng)
    elif name == "shot06":
        k = ease_io(lt / 3.15)
        f, proj = cam_frame(img, lerp(768, 820, k), lerp(1350, 1150, k), lerp(1.05, 1.28, k))
        if gt >= 13.16:
            a = gt - 13.16
            for i in range(5):
                x, y = proj(700 + i * 90, 820)
                doodle_heart(d, x + 25 * math.sin(a * 3 + i), y - 200 * a - (i % 2) * 50, 30 + 8 * (i % 3), rng)
        x, y = proj(700, 860)
        doodle_lines(d, x, y, 70, -2.2, rng, 3)
    else:  # shot07
        k = ease_io(lt / 2.65)
        z = lerp(1.05, 1.3, k) + (0.035 * math.exp(-since_beat(gt) / 0.1) if gt >= BEAT0 else 0)
        f, proj = cam_frame(img, lerp(850, 1050, k), lerp(1300, 1100, k), z)
        x, y = proj(500, 820)
        doodle_lines(d, x, y, 120, -1.9, rng, 3)
        if gt >= 15.6:
            x, y = proj(985, 960)
            doodle_sparkle(d, x, y, 55 * ease_out((gt - 15.6) / 0.15), rng, color=(255, 255, 255))
    return f


def dance(name, lt, gt, cuts, rng, layer):
    b = beat_index(gt)
    f = stage_background(b, gt, rng)
    if name == "brush":
        cut, flip = ("dance01", False) if b % 2 == 0 else ("dance02", True)
        place_cutout(f, cuts[cut + ("_m" if flip else "")], gt)
        stage_doodles(layer, b, rng, ("sparkle", "star"))
        doodle_lines(ImageDraw.Draw(layer), 700 if not flip else 380, 620, 80, -0.6 if not flip else -2.5, rng, 3)
    elif name == "dougie":
        seq = [("dance03", ""), ("dance04", "_m"), ("dance05", ""), ("dance04", "")]
        cut, suf = seq[b % len(seq)]
        place_cutout(f, cuts[cut + suf], gt)
        stage_doodles(layer, b, rng)
    elif name == "point":
        z = 1.0 + 0.12 * kick(gt, 20.46, 0.18)
        place_cutout(f, cuts["dance06"], gt, scale=z)
        stage_doodles(layer, b, rng, ("sparkle", "star"))
        if gt >= 20.46:
            doodle_sparkle(ImageDraw.Draw(layer), 640, 470, 60 * ease_out((gt - 20.46) / 0.12), rng)
    elif name == "jade":
        tilt = 4 if b % 2 == 0 else -4
        place_cutout(f, cuts["jade01"], gt, tilt=tilt)
        stage_doodles(layer, b, rng, ("heart", "heart", "sparkle"))
    else:
        cut = "together01" if b % 2 == 0 else "together02"
        place_cutout(f, cuts[cut], gt, scale=0.95)
        stage_doodles(layer, b, rng)
    return f


# ------------------------------------------------------------------ render


def load_all():
    imgs, cuts = {}, {}
    for n in range(1, 8):
        imgs[f"shot{n:02d}"] = cv2.cvtColor(cv2.imread(str(HERE / "images" / f"shot{n:02d}.jpg")), cv2.COLOR_BGR2RGB)
    for name in ("dance01", "dance02", "dance03", "dance04", "dance05", "dance06", "jade01", "together01", "together02"):
        cut = key_green(Image.open(HERE / "images" / f"{name}.jpg"))
        cuts[name] = cut
        cuts[name + "_m"] = np.ascontiguousarray(cut[:, ::-1])
    return imgs, cuts


def render_drawing(di, imgs, cuts, maps, paper):
    gt = di * 2 / FPS
    rng = np.random.default_rng(1000 + di)
    name, s0, _ = next((s for s in SHOTS if s[1] <= gt < s[2]), SHOTS[-1])
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if name.startswith("shot"):
        f = story(name, gt - s0, gt, imgs, rng, layer)
    else:
        f = dance(name, gt - s0, gt, cuts, rng, layer)
    # doodles on top of the art, then boil everything together
    lay = np.asarray(layer, np.float32)
    al = lay[..., 3:4] / 255
    f = f * (1 - al) + lay[..., :3] * al
    mx, my = maps[di % len(maps)]
    f = cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    f *= paper
    # captions boil a little but stay crisp
    cap = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_caption(cap, gt, rng)
    c = np.asarray(cap, np.float32)
    ca = c[..., 3:4] / 255
    f = f * (1 - ca) + c[..., :3] * ca
    if gt > END - 0.35:
        f *= clamp((END - gt) / 0.35, 0, 1)
    return np.clip(f, 0, 255).astype(np.uint8)


def render_video(out_path):
    imgs, cuts = load_all()
    maps, paper = make_boil_maps(), make_paper()
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", str(out_path)], stdin=subprocess.PIPE)
    nframes = int(round(END * FPS))
    for di in range((nframes + 1) // 2):
        buf = render_drawing(di, imgs, cuts, maps, paper).tobytes()
        for _ in range(2 if 2 * di + 1 < nframes else 1):
            ff.stdin.write(buf)
        if di % 48 == 0:
            print(f"drawing {di}/{nframes // 2}", flush=True)
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    audio, out = Path(sys.argv[1]), Path(sys.argv[2])
    tmp_v = out.with_suffix(".video.mp4")
    render_video(tmp_v)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp_v), "-i", str(audio), "-c:v", "copy",
                    "-af", f"afade=t=out:st={END - 0.35}:d=0.35", "-c:a", "aac", "-b:a", "192k", "-t", str(END),
                    "-movflags", "+faststart", str(out)], check=True)
    tmp_v.unlink()
    print("done", out)
