"""Render the Two Guards Riddle short: animated shots, word-timed captions, SFX, voiceover.

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
END = 15.35
SR = 44100
UP = 2  # source images are 768x1376; upscale 2x before animating

SHOTS = [  # (shot, start, end) in seconds
    (1, 0.00, 1.50), (2, 1.50, 3.00), (3, 3.00, 4.80), (4, 4.80, 5.90),
    (5, 5.90, 7.15), (6, 7.15, 8.60), (7, 8.60, 9.65), (8, 9.65, 11.70),
    (9, 11.70, 12.60), (10, 12.60, 13.60), (11, 13.60, END),
]

# Caption chunks: (words with start times, end time or None, is_stage_direction)
CAPTIONS = [
    ([("ONE", 0.00), ("OF", 0.26), ("US", 0.32), ("SPEAKS", 0.50), ("NOTHING", 0.76), ("BUT", 0.96),
      ("THE", 1.16), ("TRUTH,", 1.34)], None, False),
    ([("THE", 1.56), ("OTHER", 1.66), ("NOTHING", 1.86), ("BUT", 2.18), ("LIES.", 2.52)], None, False),
    ([("OKAY,", 3.04), ("I", 3.14), ("KNOW", 3.28), ("THIS.", 3.44)], None, False),
    ([("WE", 3.74), ("HAVE", 3.80), ("TO", 3.96), ("ASK–", 4.18)], None, False),
    ([("*BARBARIAN", 4.80), ("TAKES", 5.06), ("AXE", 5.44)], None, True),
    ([("AND", 5.74), ("KILLS", 5.94), ("THE", 6.32), ("FIRST", 6.60), ("GUARD*", 6.86)], None, True),
    ([("WHAT", 7.20), ("THE", 7.48), ("HELL!", 7.96)], None, False),
    ([("*TO", 8.64), ("THE", 8.72), ("REMAINING", 8.78), ("GUARD*", 9.06)], None, True),
    ([("IS", 9.70), ("HE", 9.82), ("DEAD?", 10.04)], None, False),
    ([("NO.", 11.74)], None, False),
    ([("THIS", 12.66), ("ONE", 12.82), ("LIAR.", 12.98)], 13.60, False),
]

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


def back_out(t):
    t = clamp(t, 0, 1)
    c = 1.9
    return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2


def kick(t, t0, decay):
    return 0.0 if t < t0 else math.exp(-(t - t0) / decay)


def shake(t, t0, amp, decay, seed=0):
    k = kick(t, t0, decay) * amp
    return (k * math.sin(t * 71 + seed) * math.cos(t * 23 + seed),
            k * math.sin(t * 59 + seed * 2.1) * math.cos(t * 31))


def radial(size, sigma):
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    return np.exp(-(x * x + y * y) / (2 * sigma * sigma)).astype(np.float32)


GLOW = radial(256, 0.35)
_vig = None


def vignette():
    global _vig
    if _vig is None:
        y, x = np.mgrid[-1:1:H * 1j, -1:1:W * 1j]
        _vig = np.clip((np.sqrt((x * 0.9) ** 2 + (y * 0.6) ** 2) - 0.45) * 1.6, 0, 1)[..., None].astype(np.float32)
    return _vig


def add_glow(frame, x, y, radius, color, strength):
    if strength <= 0.003 or radius < 2:
        return
    size = int(radius * 2)
    g = np.asarray(Image.fromarray((GLOW * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR),
                   np.float32) / 255
    x0, y0 = int(x - radius), int(y - radius)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(W, x0 + size), min(H, y0 + size)
    if x1c <= x0c or y1c <= y0c:
        return
    frame[y0c:y1c, x0c:x1c] += g[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c), None] * np.array(color, np.float32) * strength


def tint(frame, color, amount, mask=None):
    if amount <= 0.003:
        return
    c = np.array(color, np.float32)
    m = amount if mask is None else mask * amount
    frame[:] = frame * (1 - m) + c * m


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


class Dust:
    """Debris flying out of a point, with gravity."""

    def __init__(self, n, seed, color, speed=(250, 750)):
        rng = np.random.default_rng(seed)
        self.color = color
        self.p = []
        for _ in range(n):
            a = rng.uniform(math.pi, 2 * math.pi)
            v = rng.uniform(*speed)
            self.p.append(dict(vx=v * math.cos(a), vy=v * math.sin(a), r=rng.uniform(6, 16)))

    def draw(self, layer, t, x0, y0, life=0.8):
        if t < 0 or t > life:
            return
        d = ImageDraw.Draw(layer)
        fade = 1 - t / life
        for p in self.p:
            x = x0 + p["vx"] * t
            y = y0 + p["vy"] * t + 1100 * t * t
            r = p["r"] * (0.5 + 0.5 * fade)
            d.ellipse((x - r, y - r, x + r, y + r), fill=self.color + (int(220 * fade),))


def ring(layer, x, y, t, color=(255, 255, 255), r0=20, r1=180, life=0.4, width=12):
    if t < 0 or t > life:
        return
    k = ease_out(t / life)
    r = lerp(r0, r1, k)
    d = ImageDraw.Draw(layer)
    d.ellipse((x - r, y - r, x + r, y + r), outline=color + (int(255 * (1 - t / life)),),
              width=max(2, int(width * (1 - k * 0.6))))


def speed_lines(layer, t, cx, cy, strength, seed=0):
    if strength <= 0.02:
        return
    rng = np.random.default_rng(seed + int(t * 30))
    d = ImageDraw.Draw(layer)
    for _ in range(26):
        a = rng.uniform(0, 2 * math.pi)
        r0 = rng.uniform(560, 760)
        r1 = r0 + rng.uniform(300, 700)
        w = int(rng.uniform(3, 9))
        d.line((cx + r0 * math.cos(a), cy + r0 * math.sin(a), cx + r1 * math.cos(a), cy + r1 * math.sin(a)),
               fill=(255, 255, 255, int(200 * strength)), width=w)


# ----------------------------------------------------------------- assets


def load_shots():
    imgs = {}
    for n in range(1, 12):
        im = Image.open(HERE / "images" / f"shot{n:02d}.jpg").convert("RGB")
        imgs[n] = im.resize((im.width * UP, im.height * UP), Image.LANCZOS)
    return imgs


def load_helmet():
    im = np.asarray(Image.open(HERE / "images" / "helmet.jpg").convert("RGB")).astype(np.int16)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    green = (g - np.maximum(r, b)).clip(0, 255)
    alpha = (255 - green * 3).clip(0, 255).astype(np.uint8)
    rgb = im.copy()
    rgb[..., 1] = np.minimum(g, np.maximum(r, b) + 10)
    out = Image.fromarray(np.dstack([rgb.clip(0, 255).astype(np.uint8), alpha]), "RGBA")
    return out.crop(out.getbbox())


# ----------------------------------------------------------------- camera

# Usable region of the upscaled image; shot 05 has a thin panel border.
CROPS = {5: (24, 24, 1512, 2728)}


class Cam:
    def __init__(self, cx, cy, zoom, rot=0.0, sx=0.0, sy=0.0):
        self.cx, self.cy, self.zoom, self.rot, self.sx, self.sy = cx * UP, cy * UP, zoom, rot, sx, sy


def render_base(shot, img, cam):
    x0, y0, x1, y1 = CROPS.get(shot, (0, 0, img.width, img.height))
    cw, ch = x1 - x0, y1 - y0
    vw = min(cw, ch * W / H) / cam.zoom
    vh = vw * H / W
    s = W / vw
    cx = clamp(cam.cx - cam.sx / s, x0 + vw / 2, x1 - vw / 2)
    cy = clamp(cam.cy - cam.sy / s, y0 + vh / 2, y1 - vh / 2)
    a = math.radians(cam.rot)
    ca, sa = math.cos(a) / s, math.sin(a) / s
    data = (ca, -sa, cx - ca * W / 2 + sa * H / 2, sa, ca, cy - sa * W / 2 - ca * H / 2)
    out = img.transform((W, H), Image.AFFINE, data, Image.BICUBIC)

    def proj(px, py):  # source coords at original 768x1376 scale -> output px
        px, py = px * UP, py * UP
        return (W / 2 + ((px - cx) * math.cos(a) + (py - cy) * math.sin(a)) * s,
                H / 2 + (-(px - cx) * math.sin(a) + (py - cy) * math.cos(a)) * s)

    return np.asarray(out, np.float32), proj


# --------------------------------------------------------------- captions

FONT = ImageFont.truetype(str(HERE.parent / "angel-demon-video" / "fonts" / "Bangers.ttf"), 104)
CAP_Y = 1330
MAX_W = 940
_cap_cache = {}


def caption_image(ci, active):
    key = (ci, active)
    if key in _cap_cache:
        return _cap_cache[key]
    chunk, _, stage = CAPTIONS[ci]
    words = [w for w, _ in chunk]
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
    base_col = (140, 215, 255) if stage else (255, 255, 255)
    for li, line in enumerate(lines):
        lw = sum(FONT.getlength(words[i]) for i in line) + space * (len(line) - 1)
        x = (W - lw) / 2
        y = 20 + li * lh
        for i in line:
            col = (255, 222, 60) if i == active else base_col
            ds.text((x + 4, y + 9), words[i], font=FONT, fill=(0, 0, 0, 150), stroke_width=9, stroke_fill=(0, 0, 0, 150))
            d.text((x, y), words[i], font=FONT, fill=col, stroke_width=8, stroke_fill=(0, 0, 0))
            x += FONT.getlength(words[i]) + space
    shadow = shadow.filter(ImageFilter.GaussianBlur(5))
    shadow.alpha_composite(img)
    _cap_cache[key] = shadow
    return shadow


def draw_caption(layer, t):
    ci = None
    for i, (chunk, _, _) in enumerate(CAPTIONS):
        if t >= chunk[0][1]:
            ci = i
    if ci is None:
        return
    chunk, end, _ = CAPTIONS[ci]
    if end is not None and t >= end:
        return
    active = max(i for i, (_, s) in enumerate(chunk) if t >= s)
    img = caption_image(ci, active)
    sc = lerp(0.65, 1.0, back_out((t - chunk[0][1]) / 0.16))
    if sc != 1.0:
        img = img.resize((max(1, int(img.width * sc)), max(1, int(img.height * sc))), Image.BILINEAR)
    layer.alpha_composite(img, (int(W / 2 - img.width / 2), int(CAP_Y - img.height / 2)))


# ------------------------------------------------------------------ shots

HEARTBEATS = [10.35, 10.85, 11.30]


def build_fx():
    return dict(
        helmet=load_helmet(),
        s2=Sparkles(3, 2, ((380, 620), (330, 520)), size=(30, 60)),
        s3=Sparkles(12, 3, ((80, 1000), (100, 900)), color=(230, 210, 255)),
        s10=Sparkles(6, 10, ((150, 500), (500, 800)), color=(255, 240, 150), size=(25, 55)),
        stars5=Sparkles(5, 5, ((620, 920), (1150, 1350)), color=(255, 230, 80), size=(20, 40)),
        dust5=Dust(18, 5, (170, 150, 120)),
        dust7=Dust(10, 7, (90, 85, 95), speed=(150, 400)),
        book6=Dust(10, 6, (235, 220, 180), speed=(200, 600)),
    )


def torch_flicker(gt, seed=0):
    return 0.5 + 0.18 * math.sin(gt * 19 + seed) * math.sin(gt * 7.3 + seed * 2)


def shot_frame(n, lt, gt, img, fx):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    post = []

    if n == 1:
        k = ease_io(lt / 1.5)
        cam = Cam(384, lerp(700, 640, k), lerp(1.0, 1.1, k))
    elif n == 2:
        k = ease_io(lt / 1.5)
        cam = Cam(lerp(420, 360, k), lerp(600, 480, k), lerp(1.05, 1.2, k) + 0.03 * kick(gt, 2.52, 0.15))
    elif n == 3:
        k = ease_io(lt / 1.8)
        cam = Cam(400, lerp(720, 620, k) + 8 * math.sin(lt * 3), lerp(1.04, 1.14, k) + 0.03 * kick(gt, 3.04, 0.15))
    elif n == 4:
        k = ease_io(lt / 1.1)
        sx, sy = shake(gt, 5.44, 14, 0.12, 4)
        cam = Cam(lerp(400, 430, k), lerp(720, 520, k), lerp(1.02, 1.22, k) + 0.06 * kick(gt, 5.44, 0.12), sx=sx, sy=sy)
        post.append(lambda f: tint(f, (0, 0, 0), 0.25 + 0.25 * k, vignette()))
    elif n == 5:
        sx, sy = shake(gt, 5.92, 46, 0.22, 5)
        z = lerp(1.22, 1.04, ease_out(lt / 0.5)) + 0.01 * lt
        cam = Cam(400, 720, z, sx=sx, sy=sy)
        flash = kick(gt, 5.90, 0.09)
        post.append(lambda f: tint(f, (255, 255, 255), flash))
    elif n == 6:
        sx, sy = shake(gt, 7.96, 26, 0.18, 6)
        z = lerp(1.3, 1.08, back_out(lt / 0.3)) + 0.08 * kick(gt, 7.96, 0.15)
        cam = Cam(400, 520, z, sx=sx + 4 * math.sin(gt * 50), sy=sy)
    elif n == 7:
        k = ease_io(lt / 1.05)
        a1, b1 = shake(gt, 8.65, 18, 0.12, 7)
        a2, b2 = shake(gt, 9.10, 22, 0.12, 8)
        cam = Cam(lerp(450, 330, k), lerp(700, 620, k), lerp(1.02, 1.15, k), sx=a1 + a2, sy=b1 + b2)
    elif n == 8:
        k = ease_io(lt / 2.05)
        beat = sum(0.025 * kick(gt, hb, 0.1) for hb in HEARTBEATS)
        cam = Cam(390, lerp(640, 520, k), lerp(1.0, 1.3, k) + beat, sx=2 * math.sin(gt * 60))
        post.append(lambda f: tint(f, (0, 0, 0), 0.15 + 0.4 * k, vignette()))
    elif n == 9:
        sx, sy = shake(gt, 11.74, 16, 0.15, 9)
        cam = Cam(400, 620, lerp(1.12, 1.04, ease_out(lt / 0.3)) + 0.05 * kick(gt, 11.74, 0.12), sx=sx, sy=sy)
    elif n == 10:
        z = lerp(1.25, 1.03, ease_out(lt / 0.35)) + 0.05 * kick(gt, 12.98, 0.15)
        cam = Cam(390, 640, z)
    else:  # 11
        k = ease_io(lt / 1.75)
        cam = Cam(384, lerp(700, 560, k), lerp(1.0, 1.2, k))

    frame, proj = render_base(n, img, cam)
    for p in post:
        p(frame)

    if n == 1:
        x, y = proj(384, 580)
        add_glow(frame, x, y, 420, (60, 200, 255), 0.35 + 0.12 * math.sin(gt * 5))
        for tx, ty in ((48, 470), (720, 470)):
            x, y = proj(tx, ty)
            add_glow(frame, x, y, 140, (255, 150, 50), torch_flicker(gt, tx))
    elif n == 2:
        fx["s2"].draw(layer, lt * 2, fade=clamp((gt - 1.6) / 0.2, 0, 1))
        x, y = proj(130, 470)
        add_glow(frame, x, y, 170, (255, 150, 50), torch_flicker(gt))
    elif n == 3:
        x, y = proj(200, 620)
        add_glow(frame, x, y, 260, (255, 220, 140), 0.35 + 0.15 * math.sin(gt * 6))
        fx["s3"].draw(layer, lt)
    elif n == 4:
        x, y = proj(680, 470)
        add_glow(frame, x, y, 170, (255, 140, 40), torch_flicker(gt) * 1.2)
        x, y = proj(450, 300)
        add_glow(frame, x, y, 40, (255, 60, 30), 0.6 * kick(gt, 5.44, 0.25))  # eye glint
    elif n == 5:
        ix, iy = proj(560, 690)
        add_glow(frame, ix, iy, 380, (255, 220, 120), 1.1 * kick(gt, 5.92, 0.18))
        ring(layer, ix, iy, gt - 5.92, color=(255, 240, 180), r0=40, r1=420, life=0.35, width=18)
        speed_lines(layer, gt, W / 2, H / 2, kick(gt, 5.92, 0.2), 5)
        gx, gy = proj(210, 900)
        fx["dust5"].draw(layer, gt - 5.92, gx, gy)
        fx["stars5"].draw(layer, lt * 1.5, fade=clamp((gt - 6.2) / 0.2, 0, 1))
        # helmet flies off the guard toward the camera, spinning
        ht = (gt - 5.95) / 0.8
        if 0 <= ht <= 1:
            hel = fx["helmet"]
            sc = lerp(0.18, 1.1, ht ** 1.6)
            spr = hel.resize((max(2, int(hel.width * sc)), max(2, int(hel.height * sc))), Image.BILINEAR)
            spr = spr.rotate(-720 * ht, Image.BILINEAR, expand=True)
            hx = lerp(ix, W + 200, ht)
            hy = lerp(iy - 50, -250, ht) - 500 * math.sin(math.pi * ht) * 0.4
            layer.alpha_composite(spr, (int(hx - spr.width / 2), int(hy - spr.height / 2)))
    elif n == 6:
        x, y = proj(345, 650)
        fx["book6"].draw(layer, gt - 7.25, x, y, life=0.9)
        x, y = proj(225, 470)
        add_glow(frame, x, y, 170, (255, 150, 50), torch_flicker(gt))
    elif n == 7:
        for t0 in (8.65, 9.10):
            x, y = proj(560, 960)
            fx["dust7"].draw(layer, gt - t0, x, y, life=0.5)
        x, y = proj(30, 440)
        add_glow(frame, x, y, 150, (255, 150, 50), torch_flicker(gt))
    elif n == 9:
        x, y = proj(690, 760)
        ring(layer, x, y, gt - 11.78, color=(255, 255, 255), r0=30, r1=200, life=0.35)
    elif n == 10:
        x, y = proj(240, 330)
        add_glow(frame, x, y, 260, (255, 230, 120), 0.9 * kick(gt, 12.98, 0.3))
        ring(layer, x, y, gt - 12.98, color=(255, 230, 100), r0=30, r1=260, life=0.45)
        fx["s10"].draw(layer, lt * 1.5, fade=clamp((gt - 12.98) / 0.15, 0, 1))
    return frame, layer


def render_video(out_path):
    imgs, fx = load_shots(), build_fx()
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", str(out_path)], stdin=subprocess.PIPE)
    nframes = int(round(END * FPS))
    for i in range(nframes):
        gt = i / FPS
        n, s0, _ = next((s for s in SHOTS if s[1] <= gt < s[2]), SHOTS[-1])
        frame, layer = shot_frame(n, gt - s0, gt, imgs[n], fx)
        if gt > END - 0.4:
            frame *= clamp((END - gt) / 0.4, 0, 1)
        draw_caption(layer, gt)
        base = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
        base.alpha_composite(layer)
        ff.stdin.write(base.convert("RGB").tobytes())
        if i % 60 == 0:
            print(f"frame {i}/{nframes}", flush=True)
    ff.stdin.close()
    ff.wait()


# -------------------------------------------------------------------- SFX


def tone(freq, dur, shape="sine"):
    n = int(dur * SR)
    f = np.broadcast_to(np.asarray(freq, np.float64), (n,))
    ph = 2 * np.pi * np.cumsum(f) / SR
    if shape == "saw":
        return 2 * ((ph / (2 * np.pi)) % 1) - 1
    return np.sin(ph)


def env(n, attack, release_tau):
    t = np.arange(n) / SR
    return np.clip(t / max(attack, 1e-4), 0, 1) * np.exp(-np.maximum(t - attack, 0) / release_tau)


def lowpass(x, cutoff):
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


def bell(freqs, dur, tau):
    n = int(dur * SR)
    return sum(tone(f, dur) * env(n, 0.002, tau / (1 + i * 0.6)) / (1 + i) for i, f in enumerate(freqs))


def build_sfx():
    out = np.zeros(int((END + 1) * SR))

    def put(t, sig, gain=1.0):
        i = int(t * SR)
        out[i:i + len(sig)] += sig[: len(out) - i] * gain

    def fade_edges(sig, sec):
        n = len(sig)
        r = np.arange(n)
        return sig * np.minimum(1, np.minimum(r, n - r) / (sec * SR))

    # torch crackle bed under the dungeon shots
    rng = np.random.default_rng(1)
    bed = lowpass(noise(13.6, 2), 400) * 1.2
    for _ in range(140):
        i = int(rng.uniform(0, 13.5) * SR)
        m = int(0.01 * SR)
        bed[i:i + m] += noise(0.01, int(rng.integers(1e6))) * env(m, 0.0005, 0.003) * rng.uniform(0.3, 1)
    put(0.0, fade_edges(bed, 0.4), 0.10)
    # 1: magic door hum
    n = int(3.0 * SR)
    tt = np.arange(n) / SR
    hum = (tone(82, 3.0) + 0.5 * tone(123 + 2 * np.sin(2 * np.pi * 0.7 * tt), 3.0) + 0.3 * tone(330, 3.0)
           * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * tt)))
    put(0.0, fade_edges(hum, 0.5), 0.10)
    # 2: sly ting
    put(1.62, bell((2093, 3136, 4186), 0.8, 0.35), 0.2)
    # 3: thinking chime
    for k, f in enumerate((1047, 1319, 1568, 2093)):
        put(3.04 + k * 0.07, bell((f, f * 2.01), 0.5, 0.18), 0.12)
    # 4: metal scrape as he grabs the axe
    n = int(0.55 * SR)
    tt = np.arange(n) / SR
    scrape = lowpass(noise(0.55, 4), 1500 + 3000 * tt / 0.55) * (1 + 0.6 * np.sin(2 * np.pi * 37 * tt))
    put(4.85, scrape * np.sin(np.pi * tt / 0.55) * 1.5 + bell((1180, 2650), 0.55, 0.25) * 0.4, 0.3)
    put(5.44, (tone(np.geomspace(120, 50, int(0.3 * SR)), 0.3) * env(int(0.3 * SR), 0.002, 0.08)), 0.5)
    # 5: whoosh + CLANG + boom, helmet whoosh, birdie tweets
    n = int(0.3 * SR)
    tt = np.arange(n) / SR
    put(5.64, lowpass(noise(0.3, 5), 400 + 4000 * tt / 0.3) * (tt / 0.3) ** 2 * 2.5, 0.5)
    put(5.92, bell((523, 1410, 2213, 3320, 4410), 1.4, 0.5), 0.8)
    n = int(0.8 * SR)
    put(5.92, tone(np.geomspace(100, 35, n), 0.8) * env(n, 0.002, 0.25) + lowpass(noise(0.8, 6), 2500) * env(n, 0.001, 0.05), 0.8)
    n = int(0.7 * SR)
    tt = np.arange(n) / SR
    put(6.0, lowpass(noise(0.7, 7), 300 + 2500 * np.sin(np.pi * tt / 0.7)) * np.sin(np.pi * tt / 0.7) * 2, 0.35)
    for k in range(4):
        m = int(0.12 * SR)
        put(6.3 + k * 0.18, tone(np.linspace(2600, 3600, m), 0.12) * env(m, 0.005, 0.04), 0.07)
    # 6: shock stab + book thud
    n = int(0.9 * SR)
    stab = sum(tone(f, 0.9, "saw") for f in (98, 147, 196, 233))
    put(7.18, lowpass(stab, 1800) * env(n, 0.005, 0.25) / 3, 0.45)
    m = int(0.3 * SR)
    put(7.55, (tone(np.geomspace(160, 60, m), 0.3) + lowpass(noise(0.3, 8), 900)) * env(m, 0.002, 0.06), 0.45)
    # 7: heavy footsteps
    for t in (8.65, 9.10):
        m = int(0.4 * SR)
        put(t, (tone(np.geomspace(90, 40, m), 0.4) + 0.5 * lowpass(noise(0.4, 9), 500)) * env(m, 0.002, 0.1), 0.7)
    # 8: heartbeat + gulp
    for hb in HEARTBEATS:
        for dt, g in ((0, 1.0), (0.16, 0.7)):
            m = int(0.25 * SR)
            put(hb + dt, tone(np.geomspace(70, 45, m), 0.25) * env(m, 0.004, 0.06) * g, 0.7)
    m = int(0.18 * SR)
    put(11.45, tone(np.concatenate([np.linspace(500, 180, m // 2), np.linspace(180, 420, m - m // 2)]), 0.18)
        * env(m, 0.005, 0.06), 0.3)
    # 9: squeak on "No."
    m = int(0.15 * SR)
    put(11.76, tone(np.linspace(1300, 1900, m), 0.15) * np.sin(np.pi * np.arange(m) / m), 0.15)
    # 10: correct-answer ding
    put(12.98, bell((1319, 2637, 3956), 1.5, 0.6), 0.35)
    put(13.10, bell((1760, 3520), 1.3, 0.5), 0.25)
    # 11: crickets
    for k in range(8):
        t0 = 13.75 + k * 0.2
        m = int(0.12 * SR)
        tt = np.arange(m) / SR
        put(t0, tone(4300, 0.12) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 30 * tt))) * np.sin(np.pi * tt / 0.12), 0.06)
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
    n = min(len(voice), len(mix))
    mix[:n] += voice[:n]
    mix += build_sfx()[:, None] * 0.85
    fade = int(0.4 * SR)
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
