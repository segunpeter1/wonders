"""Render The Human Lesson short: 3D-style stills with mouth lip-sync, camera moves and captions.

Lip-sync: for each talking shot only a soft oval around the mouth is swapped
between the closed and open images, driven by the voice loudness.
Audio is the original voiceover only.

Usage: python3 render.py <voiceover.wav> <output.mp4>
"""
import importlib.util
import math
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("guards_render", HERE.parent / "guards-riddle-video" / "render.py")
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)
clamp, lerp, ease_io, ease_out, back_out, kick, shake = g.clamp, g.lerp, g.ease_io, g.ease_out, g.back_out, g.kick, g.shake
add_glow, Sparkles = g.add_glow, g.Sparkles

W, H, FPS = 1080, 1920, 30
END = 7.47
SRC_W, SRC_H = 940, 1672

SHOTS = [("shot01", 0.00, 1.00), ("shot02", 1.00, 2.50), ("shot03", 2.50, 4.30),
         ("shot04", 4.30, 5.65), ("shot05", 5.65, END)]

# Mouth centre (source px) and oval radii for the talking shots.
MOUTHS = {"shot02": (554, 380, 105, 95), "shot03": (586, 482, 100, 80), "shot05": (648, 460, 115, 105)}

CAPTIONS = [
    [("REMEMBER,", 0.00)],
    [("A", 1.06), ("HUMAN", 1.24), ("COMES", 1.46), ("OUT", 1.74)],
    [("OF", 2.00), ("A", 2.14), ("HUMAN", 2.32)],
    [("BECAUSE", 2.50), ("A", 2.84), ("HUMAN", 3.10)],
    [("CAME", 3.30), ("IN", 3.60), ("A", 3.88), ("HUMAN.", 4.04)],
    [("UNTIL", 5.74), ("WE", 6.12), ("MEET", 6.48), ("AGAIN!", 6.76)],
]
CAPTION_HIDE = [(4.55, 5.74)]  # no caption during the parents' silent stare

FONT = ImageFont.truetype(str(HERE.parent / "angel-demon-video" / "fonts" / "LuckiestGuy.ttf"), 96)
CAP_Y = 1560

# ------------------------------------------------------------------ audio


def read_wav(path):
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        return a.reshape(-1, w.getnchannels()).mean(1), sr


def mouth_track(voice, sr):
    """Per-frame mouth openness 0..1 from voice loudness, with a little hold so it doesn't chatter."""
    hop = sr // FPS
    n = int(END * FPS) + 1
    rms = np.array([np.sqrt(np.mean(voice[i * hop:(i + 1) * hop] ** 2)) if i * hop < len(voice) else 0 for i in range(n)])
    lo, hi = 0.025, 0.11
    o = np.clip((rms - lo) / (hi - lo), 0, 1)
    # alternate open/close on sustained sound so long vowels still "flap"
    state = np.zeros(n)
    for i in range(n):
        state[i] = 1.0 if o[i] > 0.5 else (0.55 if o[i] > 0.15 else 0.0)
    for i in range(2, n):
        if state[i] == 1.0 and state[i - 1] == 1.0 and state[i - 2] == 1.0 and i % 4 == 0:
            state[i] = 0.55
    return state


# ---------------------------------------------------------------- images


def load_images():
    def rd(name):
        im = Image.open(HERE / "images" / f"{name}.png").convert("RGB").resize((SRC_W, SRC_H), Image.LANCZOS)
        return np.asarray(im, np.float32)

    imgs = {n: rd(n) for n in ("shot01", "shot04")}
    for s, (mx, my, rx, ry) in MOUTHS.items():
        closed, opened = rd(f"{s}-closed"), rd(f"{s}-open")
        mask = np.zeros((SRC_H, SRC_W), np.float32)
        cv2.ellipse(mask, (mx, my), (rx, ry), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 18)[..., None]
        imgs[s] = (closed, opened, mask)
    return imgs


def compose_mouth(entry, openness):
    closed, opened, mask = entry
    if openness <= 0:
        return closed
    return closed * (1 - mask * openness) + opened * (mask * openness)


def cam(img, cx, cy, zoom, rot=0.0, sx=0.0, sy=0.0):
    vw = SRC_W / zoom
    vh = vw * H / W
    s = W / vw
    cx = clamp(cx - sx / s, vw / 2, SRC_W - vw / 2)
    cy = clamp(cy - sy / s, vh / 2, SRC_H - vh / 2)
    m = cv2.getRotationMatrix2D((cx, cy), rot, s)
    m[:, 2] += np.array([W / 2 - cx, H / 2 - cy])
    out = cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    def proj(px, py):
        v = m @ np.array([px, py, 1.0])
        return float(v[0]), float(v[1])

    return out, proj


# ---------------------------------------------------------------- captions

_cache = {}


def caption_img(ci, active):
    key = (ci, active)
    if key in _cache:
        return _cache[key]
    words = [w for w, _ in CAPTIONS[ci]]
    space = FONT.getlength(" ")
    tw = sum(FONT.getlength(w) for w in words) + space * (len(words) - 1)
    scale = min(1.0, 980 / tw)
    f = FONT.font_variant(size=int(96 * scale))
    space = f.getlength(" ")
    tw = sum(f.getlength(w) for w in words) + space * (len(words) - 1)
    img = Image.new("RGBA", (W, 170), (0, 0, 0, 0))
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d, ds = ImageDraw.Draw(img), ImageDraw.Draw(sh)
    x = (W - tw) / 2
    for i, w in enumerate(words):
        col = (255, 214, 0) if i == active else (255, 255, 255)
        ds.text((x + 5, 40), w, font=f, fill=(0, 0, 0, 170), stroke_width=10, stroke_fill=(0, 0, 0, 170))
        d.text((x, 30), w, font=f, fill=col, stroke_width=9, stroke_fill=(20, 14, 40))
        x += f.getlength(w) + space
    sh = sh.filter(ImageFilter.GaussianBlur(6))
    sh.alpha_composite(img)
    _cache[key] = sh
    return sh


def draw_caption(layer, t):
    if any(a <= t < b for a, b in CAPTION_HIDE):
        return
    ci = None
    for i, c in enumerate(CAPTIONS):
        if t >= c[0][1]:
            ci = i
    if ci is None:
        return
    words = CAPTIONS[ci]
    active = max(i for i, (_, s) in enumerate(words) if t >= s)
    img = caption_img(ci, active)
    sc = lerp(0.6, 1.0, back_out((t - words[0][1]) / 0.14))
    # each new word gives a tiny bump
    sc *= 1 + 0.05 * kick(t, words[active][1], 0.08)
    if abs(sc - 1) > 0.002:
        img = img.resize((max(1, int(img.width * sc)), max(1, int(img.height * sc))), Image.BILINEAR)
    layer.alpha_composite(img, (int(W / 2 - img.width / 2), int(CAP_Y - img.height / 2)))


# ------------------------------------------------------------------- shots

STARS = Sparkles(14, 5, ((450, 1060), (0, 1300)), size=(14, 34))


def shot_frame(name, lt, gt, imgs, openness):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    post = []
    if name == "shot01":
        k = ease_out(lt / 1.0)
        f, proj = cam(imgs[name], lerp(470, 560, k), lerp(836, 560, k), lerp(1.0, 1.35, k))
        x, y = proj(565, 420)
        add_glow(f, x, y, 420, (255, 200, 110), 0.55 * kick(gt, 0.0, 0.35) + 0.25)
        post.append(lambda fr: fr.__iadd__(np.float32(90 * kick(gt, 0.0, 0.12))))
    elif name == "shot02":
        src = compose_mouth(imgs[name], openness)
        k = ease_io(lt / 1.5)
        z = lerp(1.0, 1.22, k) + 0.04 * kick(gt, 1.24, 0.12) + 0.04 * kick(gt, 2.32, 0.12)
        sx, sy = shake(gt, 1.0, 10, 0.1, 2)
        f, proj = cam(src, 520, lerp(800, 560, k), z, sx=sx, sy=sy)
        x, y = proj(200, 170)
        add_glow(f, x, y, 120, (255, 240, 200), 0.35 + 0.15 * math.sin(gt * 9))
    elif name == "shot03":
        src = compose_mouth(imgs[name], openness)
        k = ease_io(lt / 1.8)
        z = lerp(1.05, 1.15, k) + 0.22 * ease_out((gt - 3.30) / 0.12) * (1 if gt >= 3.30 else 0)
        sx, sy = shake(gt, 3.30, 22, 0.15, 3)
        f, proj = cam(src, lerp(600, 600, k), lerp(820, 600, k) if gt < 3.30 else 560, z, sx=sx, sy=sy)
    elif name == "shot04":
        k = ease_io(lt / 1.35)
        sx, sy = shake(gt, 4.30, 24, 0.14, 4)
        f, proj = cam(imgs[name], 470, lerp(700, 620, k), lerp(1.02, 1.2, k), sx=sx, sy=sy)
        post.append(lambda fr: fr.__iadd__(np.float32(70 * kick(gt, 4.30, 0.08))))
    else:
        src = compose_mouth(imgs[name], openness)
        k = ease_out(lt / 0.5)
        bob = 18 * math.sin(lt * 3.2)
        z = lerp(1.35, 1.05, k) + 0.02 * lt
        f, proj = cam(src, lerp(640, 560, k), lerp(520, 760, k), z, rot=1.5 * math.sin(lt * 2.4), sy=bob)
        x, y = proj(720, 380)
        add_glow(f, x, y, 380, (220, 230, 255), 0.25 + 0.06 * math.sin(gt * 3))
        STARS.draw(layer, gt * 1.4)
        post.append(lambda fr: fr.__iadd__(np.float32(80 * kick(gt, 5.65, 0.1))))
    for p in post:
        p(f)
    return f, layer


def render_video(out_path, openness):
    imgs = load_images()
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", str(out_path)], stdin=subprocess.PIPE)
    n = int(round(END * FPS))
    for i in range(n):
        gt = i / FPS
        name, s0, _ = next((s for s in SHOTS if s[1] <= gt < s[2]), SHOTS[-1])
        f, layer = shot_frame(name, gt - s0, gt, imgs, openness[i] if name in MOUTHS else 0)
        if gt > END - 0.3:
            f *= clamp((END - gt) / 0.3, 0, 1)
        draw_caption(layer, gt)
        base = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).convert("RGBA")
        base.alpha_composite(layer)
        ff.stdin.write(base.convert("RGB").tobytes())
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    voice_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    voice, sr = read_wav(voice_path)
    tmp_v = out.with_suffix(".video.mp4")
    render_video(tmp_v, mouth_track(voice, sr))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp_v), "-i", str(voice_path), "-c:v", "copy",
                    "-af", f"afade=t=out:st={END - 0.3}:d=0.3", "-c:a", "aac", "-b:a", "192k", "-t", str(END),
                    "-movflags", "+faststart", str(out)], check=True)
    tmp_v.unlink()
    print("done", out)
