"""Render the Engagement Swords short: animated shots, word-timed captions, SFX, voiceover.

Reuses the effect/audio helpers from ../guards-riddle-video/render.py.
Usage: python3 render.py <voiceover.wav> <output.mp4>
"""
import importlib.util
import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("guards_render", HERE.parent / "guards-riddle-video" / "render.py")
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)

clamp, lerp, ease_io, ease_out, back_out, kick, shake = g.clamp, g.lerp, g.ease_io, g.ease_out, g.back_out, g.kick, g.shake
add_glow, tint, vignette, ring, star, Sparkles = g.add_glow, g.tint, g.vignette, g.ring, g.star, g.Sparkles
tone, env, lowpass, noise, bell = g.tone, g.env, g.lowpass, g.noise, g.bell
FONT = g.FONT

W, H, FPS = 1080, 1920, 30
END = 23.1
SR = 44100

SHOTS = [  # (shot, start, end)
    (1, 0.00, 1.62), (2, 1.62, 2.85), (3, 2.85, 5.40), (4, 5.40, 7.90), (5, 7.90, 8.76),
    (6, 8.76, 10.70), (7, 10.70, 12.30), (8, 12.30, 15.54), (9, 15.54, 17.85),
    (10, 17.85, 18.95), (11, 18.95, 20.85), (12, 20.85, END),
]

CAPTIONS = [
    [("REPLACE", 0.00), ("THE", 0.62), ("ENGAGEMENT", 1.02), ("RING", 1.40)],
    [("WITH", 1.62), ("AN", 1.84), ("ENGAGEMENT", 1.98), ("SWORD.", 2.38)],
    [("THE", 3.44), ("VIKINGS", 3.50), ("LITERALLY", 3.62)],
    [("EXCHANGED", 4.08), ("SWORDS", 4.52), ("AT", 4.78), ("A", 4.96), ("WEDDING.", 4.98)],
    [("THE", 5.48), ("MAN", 5.58), ("WAS", 5.66), ("SUPPOSED", 5.86), ("TO", 5.96)],
    [("BREAK", 6.24), ("INTO", 6.54), ("HIS", 6.88), ("FATHER'S", 7.16), ("CRYPT,", 7.54)],
    [("STEAL", 7.96), ("HIS", 8.30), ("SWORD", 8.58)],
    [("AND", 8.76), ("GIVE", 9.14), ("IT", 9.32), ("TO", 9.40), ("HIS", 9.56), ("WIFE", 9.72)],
    [("AND", 9.92), ("GET", 10.02), ("MARRIED.", 10.22)],
    [("THE", 10.76), ("WOMAN", 10.86), ("GAVE", 11.00), ("HER", 11.20), ("HUSBAND", 11.34)],
    [("A", 11.56), ("BRAND", 11.66), ("NEW", 11.92), ("SWORD", 12.12)],
    [("AND", 12.30), ("LIKE,", 12.50)],
    [("THE", 12.76), ("WOMAN", 13.02), ("WAS", 13.14), ("APPARENTLY", 13.34), ("SUPPOSED", 13.66), ("TO", 13.98)],
    [("HANG", 14.14), ("ON", 14.32), ("TO", 14.50), ("THIS", 14.66), ("ANCIENT", 14.82), ("SWORD", 15.28)],
    [("AND", 15.54), ("MAKE", 15.70), ("SURE", 15.84), ("THAT", 16.00), ("IT", 16.14), ("WAS", 16.24)],
    [("BURIED", 16.42), ("WITH", 16.66), ("HER", 16.82), ("HUSBAND", 16.98), ("WHEN", 17.14), ("HE", 17.34), ("DIED.", 17.46)],
    [("SO", 17.96), ("YEAH...", 18.04)],
    [("ENGAGEMENT", 19.00), ("SWORDS...", 19.15)],
    [("OR", 19.66), ("AT", 19.70), ("LEAST", 19.76), ("WEDDING", 20.02), ("SWORDS", 20.42)],
    [("SHOULD", 20.90), ("DEFINITELY", 21.10), ("BE", 21.58), ("A", 21.94), ("THING!", 22.10)],
]
CAPTION_OFF = 22.75
LABEL = ("FUN FACT:", 2.92, 5.40)  # orange label above the caption

# Artwork region of each image (some have a blank strip at the bottom).
CROPS = {3: 1838, 4: 1838, 5: 1838, 6: 1838, 7: 1812, 8: 1812, 9: 1722, 11: 1960}

# --------------------------------------------------------------- camera


class Cam:
    def __init__(self, cx, cy, zoom, rot=0.0, sx=0.0, sy=0.0):
        self.cx, self.cy, self.zoom, self.rot, self.sx, self.sy = cx, cy, zoom, rot, sx, sy


def render_base(n, img, cam):
    x0, y0, x1, y1 = 0, 0, img.width, CROPS.get(n, img.height)
    vw = min(x1 - x0, (y1 - y0) * W / H) / cam.zoom
    vh = vw * H / W
    s = W / vw
    cx = clamp(cam.cx - cam.sx / s, x0 + vw / 2, x1 - vw / 2)
    cy = clamp(cam.cy - cam.sy / s, y0 + vh / 2, y1 - vh / 2)
    a = math.radians(cam.rot)
    ca, sa = math.cos(a) / s, math.sin(a) / s
    data = (ca, -sa, cx - ca * W / 2 + sa * H / 2, sa, ca, cy - sa * W / 2 - ca * H / 2)
    out = img.transform((W, H), Image.AFFINE, data, Image.BICUBIC)

    def proj(px, py):
        return (W / 2 + ((px - cx) * math.cos(a) + (py - cy) * math.sin(a)) * s,
                H / 2 + (-(px - cx) * math.sin(a) + (py - cy) * math.cos(a)) * s)

    return np.asarray(out, np.float32), proj, s


# ------------------------------------------------------------- particles


class Snow:
    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self.p = [dict(x=rng.uniform(0, W), y=rng.uniform(-H, H), vy=rng.uniform(60, 190), r=rng.uniform(3, 9),
                       amp=rng.uniform(10, 40), w=rng.uniform(0.8, 2), ph=rng.uniform(0, 6.3)) for _ in range(n)]

    def draw(self, layer, t):
        d = ImageDraw.Draw(layer)
        for p in self.p:
            y = (p["y"] + p["vy"] * t) % (H + 40) - 20
            x = p["x"] + p["amp"] * math.sin(p["w"] * t + p["ph"])
            r = p["r"]
            d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, int(120 + 15 * r)))


def heart(draw, x, y, r, alpha, color=(235, 70, 90)):
    if r < 2 or alpha <= 0:
        return
    pts = []
    for i in range(40):
        t = 2 * math.pi * i / 40
        pts.append((x + r * 16 * math.sin(t) ** 3 / 16,
                    y - r * (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)) / 16))
    draw.polygon(pts, fill=color + (int(255 * alpha),), outline=(60, 10, 20, int(255 * alpha)))


class Hearts:
    def __init__(self, n, seed, area):
        rng = np.random.default_rng(seed)
        self.p = [dict(x=rng.uniform(*area[0]), y=rng.uniform(*area[1]), r=rng.uniform(16, 32), vy=rng.uniform(80, 160),
                       t0=rng.uniform(0, 1.2), ph=rng.uniform(0, 6.3)) for _ in range(n)]

    def draw(self, layer, t):
        d = ImageDraw.Draw(layer)
        for p in self.p:
            a = t - p["t0"]
            if a < 0 or a > 1.6:
                continue
            k = math.sin(math.pi * a / 1.6)
            heart(d, p["x"] + 15 * math.sin(3 * a + p["ph"]), p["y"] - p["vy"] * a, p["r"] * (0.6 + 0.4 * k), k)


class Embers:
    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self.p = [dict(x=rng.uniform(0, W), y=rng.uniform(300, H), vy=rng.uniform(40, 120), r=rng.uniform(6, 16),
                       f=rng.uniform(1, 3), ph=rng.uniform(0, 6.3)) for _ in range(n)]

    def draw(self, frame, t):
        for p in self.p:
            k = 0.5 + 0.5 * math.sin(p["f"] * t + p["ph"])
            add_glow(frame, p["x"] + 20 * math.sin(t + p["ph"]), p["y"] - p["vy"] * t, p["r"], (255, 150, 60), 0.9 * k)


class Motes:
    def __init__(self, n, seed, area):
        rng = np.random.default_rng(seed)
        self.p = [dict(x=rng.uniform(*area[0]), y=rng.uniform(*area[1]), r=rng.uniform(2, 5),
                       vx=rng.uniform(-15, 15), vy=rng.uniform(-10, 20), ph=rng.uniform(0, 6.3)) for _ in range(n)]

    def draw(self, layer, t):
        d = ImageDraw.Draw(layer)
        for p in self.p:
            a = int(110 + 90 * math.sin(2 * t + p["ph"]))
            x, y, r = p["x"] + p["vx"] * t, p["y"] + p["vy"] * t, p["r"]
            d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 240, 210, max(0, a)))


def slash(layer, x0, y0, x1, y1, k, width=46):
    if k <= 0:
        return
    k = min(1, k)
    xe, ye = lerp(x0, x1, k), lerp(y0, y1, k)
    d = ImageDraw.Draw(layer)
    d.line((x0, y0, xe, ye), fill=(0, 0, 0, 255), width=width + 14)
    d.line((x0, y0, xe, ye), fill=(230, 30, 40, 255), width=width)


# ------------------------------------------------------------- captions

CAP_Y = 1330
_cache = {}


def caption_image(ci, active):
    key = (ci, active)
    if key not in _cache:
        g.CAPTIONS = [(c, None, False) for c in CAPTIONS]
        g._cap_cache.clear()
        _cache[key] = g.caption_image(ci, active)
    return _cache[key]


def label_image():
    if "label" not in _cache:
        f = FONT.font_variant(size=120)
        img = Image.new("RGBA", (W, 170), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        tw = f.getlength(LABEL[0])
        d.text(((W - tw) / 2, 20), LABEL[0], font=f, fill=(255, 150, 40), stroke_width=9, stroke_fill=(0, 0, 0))
        _cache["label"] = img
    return _cache["label"]


def draw_caption(layer, t):
    if LABEL[1] <= t < LABEL[2]:
        img = label_image()
        sc = lerp(0.5, 1.0, back_out((t - LABEL[1]) / 0.18))
        img = img.resize((int(img.width * sc), int(img.height * sc)), Image.BILINEAR)
        img = img.rotate(-4, Image.BILINEAR, expand=True)
        layer.alpha_composite(img, (int(W / 2 - img.width / 2), int(CAP_Y - 170 - img.height / 2)))
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
    sc = lerp(0.65, 1.0, back_out((t - chunk[0][1]) / 0.16))
    if sc != 1.0:
        img = img.resize((max(1, int(img.width * sc)), max(1, int(img.height * sc))), Image.BILINEAR)
    if t > CAPTION_OFF:
        img = img.copy()
        a = 1 - (t - CAPTION_OFF) / 0.3
        img.putalpha(img.getchannel("A").point(lambda v: int(v * a)))
    layer.alpha_composite(img, (int(W / 2 - img.width / 2), int(CAP_Y - img.height / 2)))


# ----------------------------------------------------------------- shots


def build_fx():
    return dict(
        snow=Snow(70, 1), snow2=Snow(50, 2),
        hearts=Hearts(9, 6, ((560, 1000), (250, 650))),
        embers=Embers(40, 9),
        motes=Motes(60, 8, ((0, W), (0, 1500))),
        sp1=Sparkles(6, 1, ((380, 700), (300, 650)), size=(25, 55)),
        sp2=Sparkles(10, 2, ((380, 700), (100, 1100)), size=(20, 45)),
        sp11=Sparkles(14, 11, ((60, 1020), (200, 1300)), size=(18, 40)),
        sp12=Sparkles(16, 12, ((250, 830), (150, 1150)), size=(18, 50), color=(255, 245, 210)),
        sp7=Sparkles(5, 7, ((450, 800), (250, 650)), size=(20, 45)),
    )


def flicker(gt, seed=0):
    return 0.55 + 0.2 * math.sin(gt * 19 + seed) * math.sin(gt * 7.3 + seed * 2)


def shot_frame(n, lt, gt, img, fx):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    post = []
    if n == 1:
        k = ease_io(lt / 1.62)
        sx, sy = shake(gt, 1.40, 18, 0.12, 1)
        cam = Cam(768, lerp(1300, 1150, k), lerp(1.05, 1.25, k), sx=sx, sy=sy)
    elif n == 2:
        k = ease_out(lt / 1.0)
        cam = Cam(768, lerp(1050, 1250, k), lerp(1.5, 1.08, k))
        post.append(lambda f: tint(f, (255, 255, 255), 0.7 * kick(gt, 1.62, 0.1)))
    elif n == 3:
        k = ease_io(lt / 2.55)
        cam = Cam(lerp(580, 880, k), 919, lerp(1.0, 1.08, k))
    elif n == 4:
        k = ease_io(lt / 2.5)
        cam = Cam(lerp(560, 1000, k), lerp(880, 950, k), lerp(1.02, 1.12, k) + 0.012 * math.sin(lt * 9))
    elif n == 5:
        tug = math.sin(lt * 2 * math.pi * 3.5)
        sx, sy = shake(gt, 8.58, 26, 0.12, 5)
        cam = Cam(770 + 35 * tug, 900, 1.08 + 0.08 * kick(gt, 8.58, 0.12), rot=1.6 * tug, sx=sx, sy=sy)
    elif n == 6:
        k = ease_io(lt / 1.94)
        cam = Cam(lerp(600, 960, k), 919, lerp(1.0, 1.1, k))
    elif n == 7:
        z = lerp(1.25, 1.02, back_out(lt / 0.35)) + 0.04 * lt / 1.6
        cam = Cam(830, 900, z)
    elif n == 8:
        k = ease_io(lt / 3.24)
        cam = Cam(lerp(1060, 880, k), lerp(800, 950, k), lerp(1.03, 1.15, k) + 0.04 * kick(gt, 15.28, 0.15))
    elif n == 9:
        k = ease_io(lt / 2.31)
        z = lerp(1.25, 1.0, k) + 0.12 * kick(gt, 17.46, 0.2)
        cam = Cam(lerp(860, 700, k), 900, z)
    elif n == 10:
        bounce = -abs(math.sin(lt * math.pi * 2.6)) * 30 * math.exp(-lt * 1.5)
        cam = Cam(768, 1150, 1.0 + 0.04 * kick(gt, 17.96, 0.15) + 0.04 * kick(gt, 18.04, 0.15), sy=bounce)
    elif n == 11:
        k = ease_io(lt / 1.9)
        cam = Cam(lerp(560, 990, k), lerp(900, 1100, k), lerp(1.05, 1.12, k))
    else:  # 12
        k = ease_io(lt / 2.25)
        cam = Cam(768, lerp(1150, 950, k), lerp(1.0, 1.15, k) + 0.04 * kick(gt, 22.10, 0.2))

    frame, proj, s = render_base(n, img, cam)
    for p in post:
        p(frame)

    if n == 1:
        x, y = proj(757, 1060)
        add_glow(frame, x, y, 200, (255, 240, 200), 0.35 + 0.2 * math.sin(gt * 9))
        fx["sp1"].draw(layer, lt * 1.5)
        cx, cy = proj(757, 1180)
        L = 330 * s / 1.0
        slash(layer, cx - L, cy - L, cx + L, cy + L, (gt - 1.40) / 0.09)
        slash(layer, cx + L, cy - L, cx - L, cy + L, (gt - 1.50) / 0.09)
    elif n == 2:
        # a glint travelling up the blade
        k = clamp((gt - 1.75) / 0.5, 0, 1)
        bx, by = proj(777, lerp(1500, 210, k))
        add_glow(frame, bx, by, 150, (255, 255, 255), 0.9 * math.sin(math.pi * k))
        tx, ty = proj(777, 206)
        add_glow(frame, tx, ty, 160, (255, 250, 220), 0.9 * kick(gt, 2.25, 0.3))
        fx["sp2"].draw(layer, lt * 1.5)
    elif n == 3:
        fx["snow"].draw(layer, gt)
        x, y = proj(880, 1100)
        add_glow(frame, x, y, 160, (255, 250, 230), 0.5 * kick(gt, 4.52, 0.35))
    elif n == 4:
        x, y = proj(1011, 523)
        add_glow(frame, x, y, 260, (255, 150, 50), flicker(gt))
        add_glow(frame, x, y, 110, (255, 200, 120), flicker(gt, 3) * 0.6)
    elif n == 5:
        x, y = proj(1020, 400)
        add_glow(frame, x, y, 240, (255, 150, 50), flicker(gt))
        x, y = proj(715, 880)
        ring(layer, x, y, gt - 8.58, color=(255, 255, 255), r0=30, r1=240, life=0.3)
    elif n == 6:
        fx["snow2"].draw(layer, gt)
        fx["hearts"].draw(layer, lt)
    elif n == 7:
        x, y = proj(165, 963)
        add_glow(frame, x, y, 300, (255, 140, 40), flicker(gt) * 0.8)
        x, y = proj(825, 578)
        add_glow(frame, x, y, 120, (255, 255, 255), 0.5 + 0.3 * math.sin(gt * 7))
        fx["sp7"].draw(layer, lt * 1.5)
    elif n == 8:
        fx["motes"].draw(layer, lt)
        x, y = proj(560, 300)
        add_glow(frame, x, y, 280, (255, 235, 190), 0.25 + 0.05 * math.sin(gt * 3))
        x, y = proj(468, 1568)
        ring(layer, x, y, gt - 14.82, color=(255, 255, 255), r0=15, r1=90, life=0.35, width=6)
    elif n == 9:
        fx["embers"].draw(frame, lt)
        x, y = proj(894, 454)
        add_glow(frame, x, y, 320, (255, 170, 80), 0.3 + 0.05 * math.sin(gt * 2))
        x, y = proj(660, 1080)
        ring(layer, x, y, gt - 17.46, color=(255, 255, 255), r0=15, r1=110, life=0.3, width=7)
    elif n == 10:
        fx["snow"].draw(layer, gt)
    elif n == 11:
        fx["sp11"].draw(layer, lt * 1.6)
    elif n == 12:
        x, y = proj(766, 344)
        add_glow(frame, x, y, 330, (255, 245, 210), 0.18 + 0.08 * math.sin(gt * 5) + 0.5 * kick(gt, 22.10, 0.35))
        ring(layer, x, y, gt - 22.10, color=(255, 245, 200), r0=40, r1=420, life=0.5, width=14)
        fx["sp12"].draw(layer, lt * 1.5)
    return frame, layer


def render_video(out_path):
    imgs = {n: Image.open(HERE / "images" / f"shot{n:02d}.jpg").convert("RGB") for n in range(1, 13)}
    fx = build_fx()
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", str(out_path)], stdin=subprocess.PIPE)
    nframes = int(round(END * FPS))
    for i in range(nframes):
        gt = i / FPS
        n, s0, _ = next((s for s in SHOTS if s[1] <= gt < s[2]), SHOTS[-1])
        frame, layer = shot_frame(n, gt - s0, gt, imgs[n], fx)
        if gt > END - 0.45:
            frame *= clamp((END - gt) / 0.45, 0, 1)
        draw_caption(layer, gt)
        base = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
        base.alpha_composite(layer)
        ff.stdin.write(base.convert("RGB").tobytes())
        if i % 90 == 0:
            print(f"frame {i}/{nframes}", flush=True)
    ff.stdin.close()
    ff.wait()


# ------------------------------------------------------------------- SFX


def build_sfx():
    out = np.zeros(int((END + 1) * SR))

    def put(t, sig, gain=1.0):
        i = int(t * SR)
        out[i:i + len(sig)] += sig[: len(out) - i] * gain

    def fade_edges(sig, sec):
        r = np.arange(len(sig))
        return sig * np.minimum(1, np.minimum(r, len(sig) - r) / (sec * SR))

    def swish(t, dur, f0, f1, gain, seed):
        n = int(dur * SR)
        tt = np.arange(n) / SR
        put(t, lowpass(noise(dur, seed), np.linspace(f0, f1, n)) * np.sin(np.pi * tt / dur) * 3, gain)

    # 1: twinkles, then two slashes
    for k, t in enumerate((0.15, 0.55, 0.95)):
        put(t, bell((2637 + 300 * k, 3951), 0.5, 0.15), 0.12)
    swish(1.38, 0.13, 1500, 7000, 0.45, 1)
    swish(1.48, 0.13, 1500, 7000, 0.45, 2)
    # 2: sword "shing"
    swish(1.62, 0.25, 2000, 8000, 0.35, 3)
    put(1.66, bell((1870, 3120, 4630, 6050), 1.4, 0.55), 0.35)
    # snowy wind beds
    for t0, dur, seed in ((2.85, 2.55, 4), (8.76, 1.94, 5), (17.85, 1.1, 6)):
        n = int(dur * SR)
        tt = np.arange(n) / SR
        wind = lowpass(noise(dur, seed), 300 + 250 * np.sin(2 * np.pi * 0.6 * tt) ** 2) * 2.5
        put(t0, fade_edges(wind, 0.25), 0.18)
    # 3: viking horn
    n = int(1.6 * SR)
    tt = np.arange(n) / SR
    f = 110 * (1 + 0.03 * np.minimum(tt / 0.3, 1)) * (1 + 0.004 * np.sin(2 * np.pi * 5 * tt))
    horn = lowpass(tone(f, 1.6, "saw") + 0.5 * tone(f * 1.5, 1.6, "saw"), 900)
    put(2.88, horn * np.clip(tt / 0.25, 0, 1) * np.clip((1.6 - tt) / 0.4, 0, 1), 0.4)
    # 4: creaky door, tiptoes, torch crackle
    n = int(0.9 * SR)
    tt = np.arange(n) / SR
    creak = tone(180 + 120 * np.sin(2 * np.pi * 1.3 * tt) + 40 * np.sin(2 * np.pi * 23 * tt), 0.9, "saw")
    put(5.42, lowpass(creak, 1400) * np.sin(np.pi * tt / 0.9) * (0.6 + 0.4 * np.sign(np.sin(2 * np.pi * 31 * tt))), 0.22)
    for t in (6.35, 6.75, 7.15, 7.55):
        m = int(0.08 * SR)
        put(t, lowpass(noise(0.08, int(t * 100)), 900) * env(m, 0.002, 0.02) * 2, 0.35)
    rng = np.random.default_rng(7)
    for _ in range(40):
        m = int(0.01 * SR)
        put(rng.uniform(5.4, 8.7), noise(0.01, int(rng.integers(1e6))) * env(m, 0.0005, 0.003), 0.15 * rng.uniform(0.3, 1))
    # 5: bone rattle and the pop
    for _ in range(22):
        m = int(0.03 * SR)
        put(rng.uniform(7.92, 8.55), lowpass(noise(0.03, int(rng.integers(1e6))), 3000) * env(m, 0.001, 0.008), 0.35)
    m = int(0.15 * SR)
    put(8.58, tone(np.geomspace(300, 1100, m), 0.15) * env(m, 0.002, 0.05), 0.45)
    swish(8.60, 0.2, 800, 5000, 0.3, 8)
    # 6: "aww" harp
    for k, f in enumerate((523, 659, 784, 1047, 1319)):
        put(8.85 + k * 0.08, bell((f, f * 2), 1.0, 0.4), 0.13)
    # 7: ta-da
    for t, chord in ((10.72, (262, 330, 392)), (10.95, (392, 494, 587, 784))):
        n = int(0.6 * SR) if t > 10.9 else int(0.18 * SR)
        d = n / SR
        sig = lowpass(sum(tone(f, d, "saw") for f in chord), 2500) / len(chord)
        put(t, sig * env(n, 0.01, d * 0.6), 0.35)
    # 8: dusty poof, spider skitter
    n = int(0.4 * SR)
    put(12.32, lowpass(noise(0.4, 9), 900) * env(n, 0.02, 0.12) * 2, 0.35)
    for k in range(10):
        m = int(0.012 * SR)
        put(14.80 + k * 0.025, noise(0.012, 100 + k) * env(m, 0.0005, 0.003), 0.18)
    # 9: soft sad horn, then a peek pop
    n = int(1.6 * SR)
    tt = np.arange(n) / SR
    f = 220 * (1 + 0.006 * np.sin(2 * np.pi * 5 * tt))
    put(15.56, lowpass(tone(f, 1.6, "saw"), 700) * np.clip(tt / 0.3, 0, 1) * np.clip((1.6 - tt) / 0.5, 0, 1), 0.18)
    m = int(0.1 * SR)
    put(17.46, tone(np.geomspace(600, 1500, m), 0.1) * env(m, 0.002, 0.03), 0.35)
    # 10: boops on "so" / "yeah"
    for t in (17.96, 18.06):
        m = int(0.14 * SR)
        put(t, tone(np.geomspace(700, 350, m), 0.14) * env(m, 0.003, 0.05), 0.3)
    # 11: shop bell + sparkles
    put(18.97, bell((2637, 3951), 1.0, 0.35), 0.3)
    put(19.10, bell((2349, 3520), 1.0, 0.35), 0.25)
    for k in range(8):
        put(19.4 + k * 0.17, bell((3000 + 250 * (k % 4), 4500), 0.4, 0.1), 0.06)
    # 12: heavenly chord + ding on "thing"
    n = int(2.2 * SR)
    tt = np.arange(n) / SR
    chord = sum(tone(f * (1 + 0.003 * np.sin(2 * np.pi * 5 * tt + i)), 2.2) for i, f in
                enumerate((523, 659, 784, 1047, 1319)))
    put(20.88, chord * np.clip(tt / 0.2, 0, 1) * np.exp(-tt / 1.2) / 5, 0.35)
    put(22.10, bell((1568, 3136, 4704), 1.0, 0.5), 0.3)
    return out[: int(END * SR)]


def build_audio(voice_path, out_path):
    voice = g.read_wav(voice_path)
    if voice.shape[1] == 1:
        voice = np.repeat(voice, 2, axis=1)
    mix = np.zeros((int(END * SR), 2))
    n = min(len(voice), len(mix))
    mix[:n] += voice[:n]
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
