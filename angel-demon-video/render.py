"""Render the Angel & Demon short: animated shots, word-timed captions, SFX, voiceover.

Usage: python3 render.py <voiceover.wav> <output.mp4>
"""
import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).parent
W, H, FPS = 1080, 1920, 30
END = 26.3
SR = 44100

# ---------------------------------------------------------------- timeline

SHOTS = [  # (shot, start, end) in seconds
    (1, 0.00, 2.30), (2, 2.30, 4.15), (3, 4.15, 5.78), (4, 5.78, 7.40),
    (5, 7.40, 9.00), (6, 9.00, 10.94), (7, 10.94, 12.85), (8, 12.85, 15.75),
    (9, 15.75, 18.45), (10, 18.45, 19.77), (11, 19.77, 20.44), (12, 20.44, 22.38),
    (13, 22.38, END),
]

# Caption chunks: list of (word, start time). Times come from word-level transcription.
CAPTIONS = [
    [("YOU", 0.00), ("WAKE", 0.26), ("UP", 0.56), ("WITH", 0.74)],
    [("TWO", 0.94), ("SMALL", 1.18), ("LUMPS", 1.44), ("ON", 1.74), ("YOUR", 1.94), ("BACK", 2.06)],
    [("JUST", 2.30), ("AROUND", 2.60), ("YOUR", 2.90), ("SHOULDER", 3.10), ("BLADES", 3.40)],
    [("YOUR", 4.20), ("FRIEND", 4.30), ("HAS", 4.66), ("A", 4.92), ("SIMILAR", 5.06), ("DILEMMA", 5.32)],
    [("HOWEVER,", 5.78), ("THEIRS", 6.14), ("ARE", 6.62), ("ON", 6.70), ("THEIR", 6.86), ("FOREHEAD", 7.04)],
    [("AND", 7.40), ("THEY", 7.76), ("LOOK", 7.88), ("LIKE", 8.08), ("ZITS", 8.32)],
    [("SMALL", 9.04), ("HORNS", 9.42), ("PROTRUDE", 9.90), ("FROM", 10.52), ("THEIRS", 10.66)],
    [("WHILE", 10.94), ("FEATHERS", 11.22), ("COME", 11.66), ("FROM", 11.98), ("YOURS", 12.20)],
    [("WITHIN", 12.96), ("A", 13.16), ("MONTH,", 13.36)],
    [("YOU", 13.56), ("HAVE", 13.78), ("LARGE", 14.02), ("WHITE", 14.50), ("DOVE", 14.98), ("WINGS", 15.32)],
    [("WHILE", 15.78), ("YOUR", 16.04), ("FRIEND", 16.20), ("HAS", 16.52), ("LONG,", 16.80), ("CURLY", 17.16), ("HORNS", 17.68)],
    [("TURNS", 18.52), ("OUT", 18.74), ("YOU'RE", 19.00), ("AN", 19.24), ("ANGEL,", 19.36)],
    [("THEY'RE", 19.80), ("A", 20.04), ("DEMON", 20.12)],
    [("AND", 20.44), ("YOU'RE", 20.84), ("SUPPOSED", 21.10), ("TO", 21.44), ("FIGHT!", 21.76)],
    [("BUT", 22.42), ("YOU", 22.54), ("BOTH", 22.70), ("WOULD", 22.98), ("RATHER", 23.14)],
    [("JUST", 23.52), ("GO", 23.94), ("SEE", 24.24), ("A", 24.52), ("MOVIE", 24.66)],
]
CAPTION_OFF = 25.85

# --------------------------------------------------------------- helpers


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


def ease_in(t):
    t = clamp(t, 0, 1)
    return t ** 3


def back_out(t):
    t = clamp(t, 0, 1)
    c = 1.9
    return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2


def kick(t, t0, decay):
    """1 at t0, decaying exponentially afterwards, 0 before."""
    return 0.0 if t < t0 else math.exp(-(t - t0) / decay)


def shake(t, t0, amp, decay, seed=0):
    k = kick(t, t0, decay) * amp
    return (k * math.sin(t * 71 + seed) * math.cos(t * 23 + seed),
            k * math.sin(t * 59 + seed * 2.1) * math.cos(t * 31))


def radial(size, sigma):
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    return np.exp(-(x * x + y * y) / (2 * sigma * sigma)).astype(np.float32)


GLOW = radial(256, 0.35)
VIGNETTE = None


def vignette():
    global VIGNETTE
    if VIGNETTE is None:
        y, x = np.mgrid[-1:1:H * 1j, -1:1:W * 1j]
        VIGNETTE = np.clip((np.sqrt((x * 0.9) ** 2 + (y * 0.6) ** 2) - 0.45) * 1.6, 0, 1)[..., None].astype(np.float32)
    return VIGNETTE


def add_glow(frame, x, y, radius, color, strength):
    """Additive glow centred on (x, y) in output pixels."""
    if strength <= 0.003:
        return
    size = int(radius * 2)
    if size < 4:
        return
    g = np.asarray(Image.fromarray((GLOW * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR),
                   np.float32) / 255
    x0, y0 = int(x - radius), int(y - radius)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x0 + size), min(H, y0 + size)
    if x1c <= x0c or y1c <= y0c:
        return
    patch = g[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c), None]
    frame[y0c:y1c, x0c:x1c] += patch * np.array(color, np.float32) * strength


def tint(frame, color, amount, mask=None):
    if amount <= 0.003:
        return
    c = np.array(color, np.float32)
    if mask is None:
        frame[:] = frame * (1 - amount) + c * amount
    else:
        m = mask * amount
        frame[:] = frame * (1 - m) + c * m


# ----------------------------------------------------------------- assets


def load_shots():
    imgs = {}
    for n in range(1, 14):
        imgs[n] = Image.open(HERE / "images" / f"shot{n:02d}.jpg").convert("RGB")
    return imgs


def load_feather():
    im = np.asarray(Image.open(HERE / "images" / "feather.png").convert("RGB")).astype(np.int16)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    green = (g - np.maximum(r, b)).clip(0, 255)
    alpha = (255 - green * 3).clip(0, 255).astype(np.uint8)
    rgb = im.copy()
    rgb[..., 1] = np.minimum(g, np.maximum(r, b) + 10)  # despill
    out = Image.fromarray(np.dstack([rgb.clip(0, 255).astype(np.uint8), alpha]), "RGBA")
    out = out.crop(out.getbbox())
    h = 170
    return out.resize((int(out.width * h / out.height), h), Image.LANCZOS)


# Usable region of each image (x0, y0, x1, y1). Shots 2 and 7 have a blank strip at the bottom.
CROPS = {2: (250, 0, 1458, 2147), 7: (180, 0, 1388, 2147)}

# ----------------------------------------------------------------- camera


class Cam:
    def __init__(self, cx, cy, zoom, rot=0.0, sx=0.0, sy=0.0):
        self.cx, self.cy, self.zoom, self.rot, self.sx, self.sy = cx, cy, zoom, rot, sx, sy


def solve_cam(shot, img, cam):
    x0, y0, x1, y1 = CROPS.get(shot, (0, 0, img.width, img.height))
    cw = x1 - x0
    vw = cw / cam.zoom
    vh = vw * H / W
    s = W / vw
    # shake is in output px; convert to source units
    cx = cam.cx - cam.sx / s
    cy = cam.cy - cam.sy / s
    cx = clamp(cx, x0 + vw / 2, x1 - vw / 2)
    cy = clamp(cy, y0 + vh / 2, y1 - vh / 2)
    return cx, cy, s


def render_base(shot, img, cam):
    cx, cy, s = solve_cam(shot, img, cam)
    a = math.radians(cam.rot)
    ca, sa = math.cos(a) / s, math.sin(a) / s
    # src = c + R * (out - centre) / s
    data = (ca, -sa, cx - ca * W / 2 + sa * H / 2,
            sa, ca, cy - sa * W / 2 - ca * H / 2)
    out = img.transform((W, H), Image.AFFINE, data, Image.BICUBIC)
    proj = lambda px, py: (  # noqa: E731
        W / 2 + ((px - cx) * math.cos(a) + (py - cy) * math.sin(a)) * s,
        H / 2 + (-(px - cx) * math.sin(a) + (py - cy) * math.cos(a)) * s)
    return np.asarray(out, np.float32), proj, s


# -------------------------------------------------------------- particles


class Feathers:
    def __init__(self, sprite, n, seed, area, vy=(120, 220), scale=(0.45, 0.9)):
        rng = np.random.default_rng(seed)
        self.sprite = sprite
        self.p = [dict(x=rng.uniform(*area[0]), y=rng.uniform(*area[1]), vy=rng.uniform(*vy),
                       amp=rng.uniform(30, 80), w=rng.uniform(1.5, 3), ph=rng.uniform(0, 6.3),
                       sc=rng.uniform(*scale), rot0=rng.uniform(-40, 40)) for _ in range(n)]

    def draw(self, layer, t):
        for p in self.p:
            x = p["x"] + p["amp"] * math.sin(p["w"] * t + p["ph"])
            y = p["y"] + p["vy"] * t
            ang = p["rot0"] + 35 * math.sin(p["w"] * t + p["ph"] + 0.8)
            spr = self.sprite.resize((max(2, int(self.sprite.width * p["sc"])),
                                      max(2, int(self.sprite.height * p["sc"]))), Image.BILINEAR)
            spr = spr.rotate(ang, Image.BILINEAR, expand=True)
            layer.alpha_composite(spr, (int(x - spr.width / 2), int(y - spr.height / 2)))


def star(draw, x, y, r, alpha, color=(255, 255, 255)):
    if r < 1 or alpha <= 0:
        return
    pts = []
    for i in range(8):
        rr = r if i % 2 == 0 else r * 0.22
        a = i * math.pi / 4
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    draw.polygon(pts, fill=color + (int(255 * alpha),))


class Sparkles:
    def __init__(self, n, seed, area, size=(14, 34), color=(255, 255, 255)):
        rng = np.random.default_rng(seed)
        self.color = color
        self.p = [dict(x=rng.uniform(*area[0]), y=rng.uniform(*area[1]), r=rng.uniform(*size),
                       f=rng.uniform(2, 5), ph=rng.uniform(0, 6.3)) for _ in range(n)]

    def draw(self, layer, t, fade=1.0):
        d = ImageDraw.Draw(layer)
        for p in self.p:
            k = max(0.0, math.sin(p["f"] * t + p["ph"])) ** 2
            star(d, p["x"], p["y"], p["r"] * k, k * fade, self.color)


class Embers:
    def __init__(self, n, seed, area, color=(255, 140, 40)):
        rng = np.random.default_rng(seed)
        self.color = color
        self.p = [dict(x=rng.uniform(*area[0]), y=rng.uniform(*area[1]), vy=rng.uniform(150, 420),
                       amp=rng.uniform(10, 40), w=rng.uniform(2, 5), r=rng.uniform(8, 22),
                       life=rng.uniform(0.6, 1.6), t0=rng.uniform(-1.0, 0.8)) for _ in range(n)]

    def draw(self, frame, t):
        for p in self.p:
            age = (t - p["t0"]) % p["life"]
            k = math.sin(math.pi * age / p["life"])
            x = p["x"] + p["amp"] * math.sin(p["w"] * t)
            y = p["y"] - p["vy"] * age
            add_glow(frame, x, y, p["r"], self.color, 1.1 * k)


class Burst:
    """Debris flying out of a point, with gravity."""

    def __init__(self, n, seed, color):
        rng = np.random.default_rng(seed)
        self.color = color
        self.p = []
        for _ in range(n):
            a = rng.uniform(0, 2 * math.pi)
            v = rng.uniform(250, 750)
            self.p.append(dict(vx=v * math.cos(a), vy=v * math.sin(a) - 250, r=rng.uniform(5, 14)))

    def draw(self, layer, t, x0, y0, life=0.7):
        if t < 0 or t > life:
            return
        d = ImageDraw.Draw(layer)
        fade = 1 - t / life
        for p in self.p:
            x = x0 + p["vx"] * t
            y = y0 + p["vy"] * t + 900 * t * t
            r = p["r"] * (0.6 + 0.4 * fade)
            d.ellipse((x - r, y - r, x + r, y + r), fill=self.color + (int(230 * fade),),
                      outline=(30, 20, 15, int(230 * fade)), width=2)


def ring(layer, x, y, t, color=(255, 255, 255), r0=20, r1=150, life=0.4, width=10):
    if t < 0 or t > life:
        return
    k = ease_out(t / life)
    r = lerp(r0, r1, k)
    a = int(255 * (1 - t / life))
    d = ImageDraw.Draw(layer)
    d.ellipse((x - r, y - r, x + r, y + r), outline=color + (a,), width=max(2, int(width * (1 - k * 0.6))))


def light_rays(n=11):
    big = Image.new("L", (W * 2, H * 2), 0)
    d = ImageDraw.Draw(big)
    ox, oy = W, -200
    for i in range(n):
        a = math.radians(90 + (i - n / 2) * 9)
        w = math.radians(2.2)
        L = 5000
        d.polygon([(ox, oy), (ox + L * math.cos(a - w), oy + L * math.sin(a - w)),
                   (ox + L * math.cos(a + w), oy + L * math.sin(a + w))], fill=255)
    return big.filter(ImageFilter.GaussianBlur(30))


# --------------------------------------------------------------- captions

FONT = ImageFont.truetype(str(HERE / "fonts" / "Bangers.ttf"), 104)
CAP_Y = 1330
MAX_W = 940
_cap_cache = {}


def caption_image(ci, active):
    key = (ci, active)
    if key in _cap_cache:
        return _cap_cache[key]
    words = [w for w, _ in CAPTIONS[ci]]
    space = FONT.getlength(" ")
    lines, cur, cur_w = [], [], 0
    for i, w in enumerate(words):
        ww = FONT.getlength(w)
        if cur and cur_w + space + ww > MAX_W:
            lines.append(cur)
            cur, cur_w = [], 0
        cur_w += (space if cur else 0) + ww
        cur.append(i)
    lines.append(cur)
    lh = 112
    img = Image.new("RGBA", (W, lh * len(lines) + 60), (0, 0, 0, 0))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d, ds = ImageDraw.Draw(img), ImageDraw.Draw(shadow)
    for li, line in enumerate(lines):
        lw = sum(FONT.getlength(words[i]) for i in line) + space * (len(line) - 1)
        x = (W - lw) / 2
        y = 20 + li * lh
        for i in line:
            col = (255, 222, 60) if i == active else (255, 255, 255)
            ds.text((x + 4, y + 9), words[i], font=FONT, fill=(0, 0, 0, 150), stroke_width=9, stroke_fill=(0, 0, 0, 150))
            d.text((x, y), words[i], font=FONT, fill=col, stroke_width=8, stroke_fill=(0, 0, 0))
            x += FONT.getlength(words[i]) + space
    shadow = shadow.filter(ImageFilter.GaussianBlur(5))
    shadow.alpha_composite(img)
    _cap_cache[key] = shadow
    return shadow


def draw_caption(layer, t):
    if t >= CAPTION_OFF + 0.3:
        return
    ci = None
    for i, chunk in enumerate(CAPTIONS):
        if t >= chunk[0][1]:
            ci = i
    if ci is None:
        return
    chunk = CAPTIONS[ci]
    active = max(i for i, (_, s) in enumerate(chunk) if t >= s)
    img = caption_image(ci, active)
    since = t - chunk[0][1]
    sc = lerp(0.65, 1.0, back_out(since / 0.16))
    if sc != 1.0:
        img = img.resize((max(1, int(img.width * sc)), max(1, int(img.height * sc))), Image.BILINEAR)
    if t > CAPTION_OFF:
        a = 1 - (t - CAPTION_OFF) / 0.3
        img = img.copy()
        img.putalpha(img.getchannel("A").point(lambda v: int(v * a)))
    layer.alpha_composite(img, (int(W / 2 - img.width / 2), int(CAP_Y - img.height / 2)))


# ------------------------------------------------------------------ shots


def build_scene():
    imgs = load_shots()
    feather = load_feather()
    fx = dict(
        f7=Feathers(feather, 4, 7, ((600, 900), (700, 950)), vy=(90, 160), scale=(0.35, 0.6)),
        s7=Sparkles(7, 70, ((550, 950), (750, 1050))),
        f8=Feathers(feather, 9, 8, ((60, 1020), (-300, 700)), vy=(160, 260)),
        s8=Sparkles(10, 80, ((80, 1000), (150, 1200))),
        f10=Feathers(feather, 6, 10, ((60, 1020), (-200, 600)), vy=(120, 200), scale=(0.35, 0.7)),
        s10=Sparkles(14, 100, ((60, 1020), (100, 1400)), color=(255, 240, 180)),
        e11=Embers(70, 11, ((0, W), (900, 2100))),
        e12=Embers(40, 12, ((560, W), (700, 2100))),
        b9l=Burst(14, 91, (150, 115, 80)), b9r=Burst(14, 92, (150, 115, 80)),
        rays=light_rays(),
    )
    return imgs, fx


def shot_frame(n, lt, gt, img, fx):
    """Return float32 RGB frame and an RGBA overlay layer for shot n at local time lt."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    post = []  # callables(frame) run after base render

    if n == 1:
        k = ease_io(lt / 2.3)
        z = lerp(1.0, 1.12, k) + 0.035 * kick(gt, 1.44, 0.18) * math.sin((gt - 1.44) * 40)
        cam = Cam(771, lerp(1376, 1050, k), z)
    elif n == 2:
        k = ease_io(lt / 0.95)
        z = lerp(1.25, 1.45, k) + 0.04 * kick(gt, 3.10, 0.12) + 0.04 * kick(gt, 3.40, 0.12)
        cam = Cam(lerp(560, 1067, k), lerp(820, 1150, k), z)
    elif n == 3:
        z = lerp(1.28, 1.06, back_out(lt / 0.28)) + 0.03 * lt
        jx, jy = 5 * math.sin(gt * 47), 4 * math.cos(gt * 39)
        cam = Cam(791, 1000, z, sx=jx, sy=jy)
    elif n == 4:
        k = ease_io(lt / 1.62)
        z = lerp(1.05, 1.32, k) + 0.05 * kick(gt, 7.04, 0.15)
        cam = Cam(778, lerp(1150, 780, k), z)
    elif n == 5:
        k = ease_io(lt / 1.6)
        sx, sy = shake(gt, 8.32, 22, 0.14, 3)
        z = lerp(1.06, 1.16, k) + 0.05 * kick(gt, 8.32, 0.12)
        cam = Cam(853, 1150, z, sx=sx, sy=sy)
    elif n == 6:
        a1, b1 = shake(gt, 9.42, 26, 0.16, 1)
        a2, b2 = shake(gt, 9.90, 30, 0.18, 2)
        z = 1.1 + 0.025 * lt + 0.08 * kick(gt, 9.42, 0.12) + 0.1 * kick(gt, 9.90, 0.13)
        cam = Cam(798, 1000, z, sx=a1 + a2, sy=b1 + b2)
        red = 0.35 * kick(gt, 9.42, 0.25) + 0.45 * kick(gt, 9.90, 0.3) + 0.12
        post.append(lambda f: tint(f, (200, 20, 30), red, vignette()))
    elif n == 7:
        k = ease_io(lt / 1.9)
        cam = Cam(784, 1250, lerp(1.04, 1.12, k), rot=1.1 * math.sin(lt * 2.4))
    elif n == 8:
        z = lerp(1.32, 1.02, ease_out(lt / 1.1)) + 0.012 * lt
        sx, sy = shake(gt, 12.85, 14, 0.2, 4)
        cam = Cam(771, 1300, z, sx=sx, sy=sy)
        flash = kick(gt, 12.85, 0.12)
        post.append(lambda f: tint(f, (255, 255, 255), flash))
    elif n == 9:
        z = lerp(1.18, 1.03, ease_io(lt / 2.7))
        a1, b1 = shake(gt, 15.75, 34, 0.22, 5)
        a2, b2 = shake(gt, 17.68, 18, 0.15, 6)
        cam = Cam(768, lerp(1100, 1300, ease_io(lt / 2.7)), z + 0.04 * kick(gt, 17.68, 0.12), sx=a1 + a2, sy=b1 + b2)
    elif n == 10:
        z = lerp(1.0, 1.09, ease_io(lt / 1.32))
        cam = Cam(768, 1300, z)
        flash = 0.8 * kick(gt, 18.45, 0.12)
        post.append(lambda f: tint(f, (255, 245, 210), flash))
    elif n == 11:
        z = lerp(1.0, 1.18, ease_in(lt / 0.67) * 0.6 + lt / 0.67 * 0.4)
        cam = Cam(771, 1200, z)
        flick = 1 + 0.07 * math.sin(gt * 53) * math.sin(gt * 17)
        post.append(lambda f: f.__imul__(flick))
        post.append(lambda f: tint(f, (120, 0, 0), 0.35, vignette()))
        post.append(lambda f: tint(f, (255, 90, 20), 0.5 * kick(gt, 19.77, 0.1)))
    elif n == 12:
        a1, b1 = shake(gt, 20.44, 16, 0.18, 7)
        a2, b2 = shake(gt, 21.76, 40, 0.25, 8)
        z = lerp(1.06, 1.12, lt / 1.94) + 0.14 * kick(gt, 21.76, 0.18)
        cam = Cam(771, 1250, z, sx=a1 + a2, sy=b1 + b2)
        flash = 0.9 * kick(gt, 21.76, 0.1)
        post.append(lambda f: tint(f, (255, 250, 230), flash))
    else:  # 13
        k = ease_io(lt / 3.4)
        cam = Cam(lerp(771, 768, k), lerp(980, 1150, k), lerp(1.28, 1.06, k))

    frame, proj, s = render_base(n, img, cam)
    for p in post:
        p(frame)

    # overlays that depend on projected positions
    if n == 2:
        for bx, by in ((895, 1232), (1239, 1211)):
            x, y = proj(bx, by)
            for t0 in (3.10, 3.40):
                ring(layer, x, y, gt - t0, life=0.35)
    elif n == 4:
        for bx, by in ((725, 765), (870, 725)):
            x, y = proj(bx, by)
            add_glow(frame, x, y, 70, (255, 40, 40), 0.55 * kick(gt, 7.04, 0.3) + 0.12 * (1 + math.sin(gt * 9)))
            ring(layer, x, y, gt - 7.04, color=(255, 60, 60), r0=10, r1=90, life=0.35, width=8)
    elif n == 6:
        for bx, by in ((660, 647), (908, 647)):
            x, y = proj(bx, by)
            add_glow(frame, x, y, 110, (255, 30, 20), 0.5 * kick(gt, 9.90, 0.3) + 0.3 * kick(gt, 9.42, 0.25))
    elif n == 7:
        fx["f7"].draw(layer, lt)
        fx["s7"].draw(layer, lt, fade=clamp((gt - 11.2) / 0.3, 0, 1))
    elif n == 8:
        fx["f8"].draw(layer, lt)
        fx["s8"].draw(layer, lt)
    elif n == 9:
        for bx, by, b in ((165, 853, fx["b9l"]), (1376, 826, fx["b9r"])):
            x, y = proj(bx, by)
            b.draw(layer, gt - 15.75, x, y)
            b.draw(layer, gt - 17.68, x, y, life=0.5)
    elif n == 10:
        rays = fx["rays"].rotate(4 * math.sin(lt * 1.5), Image.BILINEAR, center=(W, 0))
        r = np.asarray(rays.crop((W // 2, 0, W // 2 + W, H)), np.float32)[..., None] / 255
        frame += r * np.array((255, 235, 170), np.float32) * (0.13 + 0.05 * math.sin(lt * 6))
        hx, hy = proj(768, 661)
        add_glow(frame, hx, hy, 220, (255, 220, 120), 0.22 + 0.1 * math.sin(lt * 8))
        fx["f10"].draw(layer, lt)
        fx["s10"].draw(layer, lt)
    elif n == 11:
        fx["e11"].draw(frame, lt)
    elif n == 12:
        cx, cy = proj(771, 1101)
        add_glow(frame, cx, cy, 300, (255, 230, 150), 0.35 + 0.2 * math.sin(lt * 14) + 1.2 * kick(gt, 21.76, 0.25))
        fx["e12"].draw(frame, lt)
    elif n == 13:
        fxp, fyp = proj(863, 905)
        add_glow(frame, fxp, fyp, 70 + 8 * math.sin(gt * 31), (255, 150, 40), 0.75 + 0.2 * math.sin(gt * 23))
        hx, hy = proj(523, 647)
        add_glow(frame, hx, hy, 140, (255, 220, 120), 0.3 + 0.08 * math.sin(gt * 4))
    return frame, layer


def render_video(out_path):
    imgs, fx = build_scene()
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", str(out_path)], stdin=subprocess.PIPE)
    nframes = int(round(END * FPS))
    for i in range(nframes):
        gt = i / FPS
        n, s0, _ = next(s for s in SHOTS if s[1] <= gt < s[2] or s is SHOTS[-1])
        frame, layer = shot_frame(n, gt - s0, gt, imgs[n], fx)
        if gt > END - 0.45:  # fade out
            frame *= clamp((END - gt) / 0.45, 0, 1)
        draw_caption(layer, gt)
        base = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
        base.alpha_composite(layer)
        ff.stdin.write(base.convert("RGB").tobytes())
        if i % 60 == 0:
            print(f"frame {i}/{nframes}", flush=True)
    ff.stdin.close()
    ff.wait()


# -------------------------------------------------------------------- SFX


def tone(freq, dur, amp=1.0, shape="sine"):
    n = int(dur * SR)
    f = np.broadcast_to(np.asarray(freq, np.float64), (n,)) if np.ndim(freq) else np.full(n, freq, np.float64)
    ph = 2 * np.pi * np.cumsum(f) / SR
    if shape == "saw":
        w = 2 * ((ph / (2 * np.pi)) % 1) - 1
    elif shape == "square":
        w = np.sign(np.sin(ph))
    else:
        w = np.sin(ph)
    return w * amp


def env(n, attack, release_tau):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-np.maximum(t - attack, 0) / release_tau)


def lowpass(x, cutoff):
    """One-pole low-pass; cutoff may be an array (sweep)."""
    c = np.broadcast_to(np.asarray(cutoff, np.float64), x.shape)
    a = 1 - np.exp(-2 * np.pi * c / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    return y


def noise(dur, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, int(dur * SR))


def build_sfx():
    out = np.zeros(int((END + 1) * SR))

    def put(t, sig, gain=1.0):
        i = int(t * SR)
        out[i:i + len(sig)] += sig[: len(out) - i] * gain

    # 1: morning birds + boing on "lumps"
    for k, t in enumerate((0.10, 0.32, 0.70, 0.86, 1.15)):
        n = int(0.07 * SR)
        f = np.linspace(2800 + 300 * k, 4300, n)
        put(t, tone(f, 0.07) * env(n, 0.005, 0.03), 0.10)
    n = int(0.5 * SR)
    tt = np.arange(n) / SR
    put(1.44, tone(lerp(330, 170, tt / 0.5) + 70 * np.sin(2 * np.pi * 16 * tt) * np.exp(-5 * tt), 0.5)
        * env(n, 0.005, 0.15), 0.35)
    # 2: bloops on the bumps
    for t in (3.10, 3.40):
        n = int(0.12 * SR)
        put(t, tone(np.geomspace(250, 950, n), 0.12) * env(n, 0.003, 0.05), 0.4)

    # record scratch
    def scratch(t, seed):
        n = int(0.32 * SR)
        tt = np.arange(n) / SR
        f = 700 + 500 * np.sin(2 * np.pi * 3.2 * tt)
        sig = 0.5 * tone(f, 0.32, shape="saw") + 0.5 * lowpass(noise(0.32, seed), f * 3)
        put(t, sig * env(n, 0.01, 0.12), 0.35)

    scratch(4.15, 1)
    # 4: dun-dun sting
    for t, f in ((6.86, 98), (7.10, 92)):
        n = int(0.32 * SR)
        sig = lowpass(tone(f, 0.32, shape="saw") + 0.6 * tone(f / 2, 0.32), 900)
        put(t, sig * env(n, 0.01, 0.12), 0.55)
    # 5: squish on "zits"
    n = int(0.22 * SR)
    sig = lowpass(noise(0.22, 2), np.linspace(2500, 300, n)) * 1.6 + 0.5 * tone(np.linspace(220, 80, n), 0.22)
    put(8.32, sig * env(n, 0.005, 0.07), 0.5)
    # 6: horn crack-pops
    for t, seed in ((9.42, 3), (9.90, 4)):
        n = int(0.3 * SR)
        click = noise(0.3, seed) * env(n, 0.0005, 0.008)
        thump = tone(np.geomspace(180, 55, n), 0.3) * env(n, 0.002, 0.08)
        put(t, click * 0.8 + thump, 0.6)
    # 7: poof + sparkle
    n = int(0.35 * SR)
    put(11.22, lowpass(noise(0.35, 5), 1500) * env(n, 0.02, 0.1) * 2, 0.4)
    for k, f in enumerate((2093, 2637, 3136, 4186)):
        n = int(0.25 * SR)
        put(11.32 + k * 0.06, tone(f, 0.25) * env(n, 0.002, 0.07), 0.12)
    # 8: big wing whoosh + shimmer
    n = int(0.9 * SR)
    tt = np.arange(n) / SR
    bell = np.sin(np.pi * np.clip(tt / 0.9, 0, 1)) ** 2
    put(12.80, lowpass(noise(0.9, 6), 300 + 2500 * bell) * bell * 2.2, 0.6)
    rng = np.random.default_rng(8)
    for k in range(16):
        n = int(0.4 * SR)
        put(13.0 + k * 0.11, tone(rng.uniform(1800, 4200), 0.4) * env(n, 0.003, 0.12), 0.06)
    # 9: thud + rumble
    for t, g in ((15.75, 0.8), (17.68, 0.5)):
        n = int(0.5 * SR)
        put(t, (tone(np.geomspace(110, 40, n), 0.5) + 0.4 * lowpass(noise(0.5, 9), 600)) * env(n, 0.002, 0.14), g)
    rum = lowpass(lowpass(noise(2.6, 10), 120), 120) * 6
    nr = len(rum)
    put(15.85, rum * np.minimum(1, np.minimum(np.arange(nr), nr - np.arange(nr)) / (0.3 * SR)), 0.35)
    # 10: heavenly chord
    n = int(1.5 * SR)
    tt = np.arange(n) / SR
    chord = sum(tone(f * (1 + 0.003 * np.sin(2 * np.pi * 5 * tt + i)), 1.5) for i, f in
                enumerate((440, 554.4, 659.3, 880, 1108.7)))
    put(18.45, chord * np.clip(tt / 0.12, 0, 1) * np.exp(-tt / 0.7) / 5, 0.45)
    n = int(1.0 * SR)
    put(18.47, tone(1760, 1.0) * env(n, 0.002, 0.3), 0.12)
    # 11: fire whoosh + crackle
    n = int(0.8 * SR)
    tt = np.arange(n) / SR
    put(19.74, lowpass(noise(0.8, 11), 200 + 1800 * tt / 0.8) * np.sin(np.pi * tt / 0.8) * 2.5, 0.55)
    rng = np.random.default_rng(12)
    for _ in range(18):
        m = int(0.015 * SR)
        put(19.8 + rng.uniform(0, 0.65), noise(0.015, int(rng.integers(1e6))) * env(m, 0.0005, 0.004), 0.25)

    # 12: impacts
    def impact(t, g, seed):
        n = int(1.0 * SR)
        boom = tone(np.geomspace(90, 32, n), 1.0) * env(n, 0.002, 0.3)
        crack = lowpass(noise(1.0, seed), 3000) * env(n, 0.001, 0.04)
        put(t, boom + crack * 0.9, g)

    impact(20.44, 0.45, 13)
    impact(21.76, 0.9, 14)
    # 13: record scratch, cinema room tone, popcorn crunch
    scratch(22.38, 15)
    hum = lowpass(noise(3.9, 16), 250) * 3
    nh = len(hum)
    put(22.45, hum * np.minimum(1, np.minimum(np.arange(nh), nh - np.arange(nh)) / (0.5 * SR)), 0.12)
    rng = np.random.default_rng(17)
    for t in (23.95, 24.55):
        for _ in range(6):
            m = int(0.02 * SR)
            put(t + rng.uniform(0, 0.12), lowpass(noise(0.02, int(rng.integers(1e6))), 4000) * env(m, 0.0005, 0.006), 0.3)
    return out[: int(END * SR)]


def read_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
        return a.reshape(-1, w.getnchannels())


def build_audio(voice_path, out_path):
    voice = read_wav(voice_path)
    if voice.shape[1] == 1:
        voice = np.repeat(voice, 2, axis=1)
    mix = np.zeros((int(END * SR), 2))
    mix[: min(len(voice), len(mix))] += voice[: len(mix)]
    mix += build_sfx()[:, None] * 0.85
    fade = int(0.45 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
    peak = np.abs(mix).max()
    if peak > 0.98:
        mix *= 0.98 / peak
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    voice, out = Path(sys.argv[1]), Path(sys.argv[2])
    tmp_v, tmp_a = out.with_suffix(".video.mp4"), out.with_suffix(".audio.wav")
    build_audio(voice, tmp_a)
    render_video(tmp_v)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp_v), "-i", str(tmp_a), "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)], check=True)
    tmp_v.unlink()
    tmp_a.unlink()
    print("done", out)
